"""Build an irregular grid world that mimics the structural properties of the
Manhattan graph used by Vafa et al. (2024).

The point of the irregularity is narrow and specific.  On a plain N/S/E/W
lattice the current position is a closed-form function of the *counts* of each
direction token seen so far:

    position = origin + (#E - #W, #N - #S)

A model can therefore ace next-token prediction, routing and the state probe
with two bounded counters and never represent the graph at all.  Manhattan does
not admit that solution.  Note that one-way streets and deleted edges do *not*
break it either -- they change which moves are *legal*, but a legal walk still
lands at origin + net displacement.  The only thing that actually breaks the
shortcut is edges whose displacement is not determined by their direction
label, i.e. variable-length edges: an "N" that sometimes means +1 row and
sometimes +3.  That is the mechanism this builder is organised around; one-ways
and deletions are layered on top for realism (they make legality
state-dependent and give the graph Manhattan-like out-degree).

Invariants held by construction:
  * every node has at most one outgoing edge per direction (required by the
    (node, direction) -> neighbor map that the whole Vafa pipeline is built on)
  * the digraph is strongly connected over all size*size nodes, so every
    (origin, destination) pair is realisable
"""
import argparse
import json
import math
import pickle
import random
from collections import deque

DIRECTIONS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# (d_row, d_col) unit steps, with row increasing southwards.
_STEP = {
    "N": (-1, 0), "NE": (-1, 1), "E": (0, 1), "SE": (1, 1),
    "S": (1, 0), "SW": (1, -1), "W": (0, -1), "NW": (-1, -1),
}


def bearing(src, dst):
    """Quantise the bearing from src to dst to one of 8 compass points.

    Mirrors Vafa et al., who convert real street bearings to 8 cardinal
    directions.  Returns None if src == dst.
    """
    d_north = src[0] - dst[0]
    d_east = dst[1] - src[1]
    if d_north == 0 and d_east == 0:
        return None
    degrees = math.degrees(math.atan2(d_east, d_north)) % 360
    return DIRECTIONS[int(round(degrees / 45.0)) % 8]


def is_strongly_connected(out_edges, nodes):
    """Two BFS passes: reachability from an arbitrary root, forwards and back."""
    if not nodes:
        return True
    root = next(iter(nodes))
    forward = {u: set(v for v in d.values()) for u, d in out_edges.items()}
    backward = {u: set() for u in nodes}
    for u, targets in forward.items():
        for v in targets:
            backward[v].add(u)

    def reached(adj):
        seen, queue = {root}, deque([root])
        while queue:
            for nxt in adj.get(queue.popleft(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    return len(reached(forward)) == len(nodes) and len(reached(backward)) == len(nodes)


class MapBuilder:
    def __init__(self, size, rng):
        self.size = size
        self.rng = rng
        self.coords = {r * size + c: (r, c) for r in range(size) for c in range(size)}
        self.nodes = set(self.coords)
        # out_edges[u][direction] = v
        self.out_edges = {u: {} for u in self.nodes}

    def node_at(self, r, c):
        if 0 <= r < self.size and 0 <= c < self.size:
            return r * self.size + c
        return None

    def add_edge(self, u, v):
        """Add u -> v if it respects the one-edge-per-direction invariant."""
        if u == v or v is None:
            return False
        direction = bearing(self.coords[u], self.coords[v])
        if direction is None or direction in self.out_edges[u]:
            return False
        self.out_edges[u][direction] = v
        return True

    def try_remove(self, u, direction):
        """Remove an edge only if the graph stays strongly connected."""
        if direction not in self.out_edges[u]:
            return False
        victim = self.out_edges[u].pop(direction)
        if is_strongly_connected(self.out_edges, self.nodes):
            return True
        self.out_edges[u][direction] = victim
        return False

    def build_lattice(self):
        for u, (r, c) in self.coords.items():
            for direction in ("N", "E", "S", "W"):
                dr, dc = _STEP[direction]
                self.add_edge(u, self.node_at(r + dr, c + dc))

    def add_long_edges(self, fraction, max_skip):
        """Re-point a fraction of lattice edges at a node 2..max_skip cells away.

        This is the shortcut-breaking step: after it, a direction token no
        longer implies a fixed displacement.
        """
        candidates = [(u, d) for u in sorted(self.nodes) for d in sorted(self.out_edges[u])]
        self.rng.shuffle(candidates)
        upgraded = 0
        for u, direction in candidates[: int(fraction * len(candidates))]:
            if direction not in self.out_edges[u]:
                continue
            r, c = self.coords[u]
            dr, dc = _STEP[direction]
            skip = self.rng.randint(2, max_skip)
            target = self.node_at(r + dr * skip, c + dc * skip)
            if target is None:
                continue
            original = self.out_edges[u][direction]
            self.out_edges[u][direction] = target
            if is_strongly_connected(self.out_edges, self.nodes):
                upgraded += 1
            else:
                self.out_edges[u][direction] = original
        return upgraded

    def add_diagonals(self, count, max_skip):
        """Lay 'Broadway'-style diagonal avenues with variable-length hops."""
        added = 0
        # Cycle the four diagonal headings so every direction token gets used.
        headings = [d for d in ("NE", "SE", "SW", "NW") for _ in range(2)]
        for i in range(count):
            direction = headings[i % len(headings)]
            dr, dc = _STEP[direction]
            # Start in the corner the avenue runs away from, so it can cross the
            # whole grid instead of falling off the edge immediately.
            r = self.rng.randrange(self.size // 2, self.size) if dr < 0 else self.rng.randrange(self.size // 2)
            c = self.rng.randrange(self.size // 2) if dc > 0 else self.rng.randrange(self.size // 2, self.size)
            for _ in range(self.size):
                node = self.node_at(r, c)
                skip = self.rng.randint(1, max_skip)
                target = self.node_at(r + dr * skip, c + dc * skip)
                if node is None or target is None:
                    break
                added += self.add_edge(node, target)
                r, c = self.coords[target]
        return added

    def make_one_way(self, probability):
        """Drop the reverse of some bidirectional pairs, keeping connectivity."""
        pairs = set()
        for u in self.nodes:
            for v in self.out_edges[u].values():
                pairs.add((min(u, v), max(u, v)))
        pairs = sorted(pairs)
        self.rng.shuffle(pairs)
        removed = 0
        for u, v in pairs:
            if self.rng.random() >= probability:
                continue
            a, b = (u, v) if self.rng.random() < 0.5 else (v, u)
            direction = bearing(self.coords[a], self.coords[b])
            if self.out_edges[a].get(direction) == b and self._has_reverse(a, b):
                removed += self.try_remove(a, direction)
        return removed

    def _has_reverse(self, a, b):
        return self.out_edges[b].get(bearing(self.coords[b], self.coords[a])) == a

    def delete_edges(self, probability):
        candidates = [(u, d) for u in sorted(self.nodes) for d in sorted(self.out_edges[u])]
        self.rng.shuffle(candidates)
        return sum(
            self.try_remove(u, d)
            for u, d in candidates
            if self.rng.random() < probability
        )

    def export(self):
        valid_turns = {u: sorted(self.out_edges[u]) for u in sorted(self.nodes)}
        node_and_direction_to_neighbor = {
            (u, d): v for u in sorted(self.nodes) for d, v in sorted(self.out_edges[u].items())
        }
        return valid_turns, node_and_direction_to_neighbor


def build(size=10, seed=0, p_long=0.50, p_oneway=0.40, p_delete=0.18,
          n_diagonals=8, max_skip=3, vanilla=False):
    builder = MapBuilder(size, random.Random(seed))
    builder.build_lattice()
    stats = {"size": size, "seed": seed, "vanilla": vanilla}
    if not vanilla:
        stats["diagonal_edges"] = builder.add_diagonals(n_diagonals, max_skip)
        stats["long_edges"] = builder.add_long_edges(p_long, max_skip)
        stats["oneway_removals"] = builder.make_one_way(p_oneway)
        stats["deletions"] = builder.delete_edges(p_delete)

    valid_turns, node_and_direction_to_neighbor = builder.export()
    assert is_strongly_connected(builder.out_edges, builder.nodes), "graph must be strongly connected"

    degrees = [len(v) for v in valid_turns.values()]
    stats.update(
        num_nodes=len(valid_turns),
        num_edges=len(node_and_direction_to_neighbor),
        mean_out_degree=round(sum(degrees) / len(degrees), 3),
        min_out_degree=min(degrees),
        directions_used=sorted({d for _, d in node_and_direction_to_neighbor}),
        distinct_turn_sets=len({tuple(v) for v in valid_turns.values()}),
    )
    return builder.coords, valid_turns, node_and_direction_to_neighbor, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--p-long", type=float, default=0.50,
                        help="fraction of edges re-pointed to a 2..max-skip hop")
    parser.add_argument("--p-oneway", type=float, default=0.40)
    parser.add_argument("--p-delete", type=float, default=0.18)
    parser.add_argument("--n-diagonals", type=int, default=8)
    parser.add_argument("--max-skip", type=int, default=3)
    parser.add_argument("--vanilla", action="store_true",
                        help="plain bidirectional lattice (degenerate control arm)")
    parser.add_argument("--out", type=str, required=True, help="output directory")
    args = parser.parse_args()

    coords, valid_turns, n2n, stats = build(
        size=args.size, seed=args.seed, p_long=args.p_long, p_oneway=args.p_oneway,
        p_delete=args.p_delete, n_diagonals=args.n_diagonals, max_skip=args.max_skip,
        vanilla=args.vanilla)

    import os
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/valid_turns.pkl", "wb") as f:
        pickle.dump(valid_turns, f)
    with open(f"{args.out}/node_and_direction_to_neighbor.pkl", "wb") as f:
        pickle.dump(n2n, f)
    with open(f"{args.out}/coords.pkl", "wb") as f:
        pickle.dump(coords, f)
    with open(f"{args.out}/map_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
