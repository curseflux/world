"""Degeneracy checks for a generated grid world.

The headline check is the *counter shortcut*.  On a plain lattice the current
node is determined by the origin plus the per-direction token counts, so a model
can solve the task with a pair of bounded counters and never build a graph.  If
that shortcut survives, every downstream world-model metric measures the wrong
thing -- and worse for representation work, the counter solution is inherently
geometric, so activations would look like a tidy 2-D lattice for reasons that
have nothing to do with the model having learned a map.

`counter_model_accuracy` is how often the best possible two-counter model lands
on the correct node.  It is 1.0 for a vanilla lattice by construction; a usable
irregular map should push it far enough down that counting cannot substitute for
tracking the graph.
"""
import argparse
import collections
import json
import pickle
import random

from build_map import DIRECTIONS


def load(map_dir):
    with open(f"{map_dir}/valid_turns.pkl", "rb") as f:
        valid_turns = pickle.load(f)
    with open(f"{map_dir}/node_and_direction_to_neighbor.pkl", "rb") as f:
        n2n = pickle.load(f)
    return valid_turns, n2n


def counter_accuracy(valid_turns, n2n, coords, walks_per_node=400, max_len=12, seed=0):
    """Accuracy of the best two-counter model, i.e. the shortcut we want dead.

    A counter model assigns each direction token a fixed displacement and
    predicts `origin + sum(counts[d] * delta[d])`.  We give it the most
    favourable possible delta (the modal displacement of each token) and then
    measure how often it lands on the true node.  On a plain lattice this is
    1.0 by construction; the lower it is, the more the task actually requires
    tracking the graph.
    """
    rng = random.Random(seed)
    modal = {}
    per_direction = collections.defaultdict(collections.Counter)
    for (node, direction), neighbor in n2n.items():
        r1, c1 = coords[node]
        r2, c2 = coords[neighbor]
        per_direction[direction][(r2 - r1, c2 - c1)] += 1
    for direction, counter in per_direction.items():
        modal[direction] = counter.most_common(1)[0][0]

    position = {v: k for k, v in coords.items()}
    correct = total = 0
    for origin in valid_turns:
        for _ in range(walks_per_node):
            node, counts = origin, collections.Counter()
            for _ in range(rng.randint(1, max_len)):
                turns = valid_turns[node]
                if not turns:
                    break
                move = rng.choice(turns)
                counts[move] += 1
                node = n2n[(node, move)]
            r, c = coords[origin]
            for direction, n in counts.items():
                r += modal[direction][0] * n
                c += modal[direction][1] * n
            total += 1
            correct += position.get((r, c)) == node
    return {
        "walks_sampled": total,
        "counter_model_accuracy": round(correct / max(total, 1), 4),
    }


def displacement_spread(n2n, coords):
    """How many distinct (d_row, d_col) displacements does each token cover?

    1 for every direction means the token implies a fixed displacement, which is
    exactly the condition that makes the counter shortcut work.
    """
    spread = collections.defaultdict(set)
    for (node, direction), neighbor in n2n.items():
        r1, c1 = coords[node]
        r2, c2 = coords[neighbor]
        spread[direction].add((r2 - r1, c2 - c1))
    return {d: len(v) for d, v in sorted(spread.items())}


def turn_set_partition(valid_turns):
    """Paper 2's next-token coarsening: states grouped by legal-move set."""
    buckets = collections.Counter(tuple(v) for v in valid_turns.values())
    return {
        "num_states": len(valid_turns),
        "num_turn_set_classes": len(buckets),
        "largest_turn_set_class": max(buckets.values()),
    }


def path_lengths(valid_turns, n2n):
    """All-pairs BFS hop counts; sets the context-length budget."""
    lengths, unreachable = [], 0
    nodes = list(valid_turns)
    for src in nodes:
        dist, queue = {src: 0}, collections.deque([src])
        while queue:
            u = queue.popleft()
            for d in valid_turns[u]:
                v = n2n[(u, d)]
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for dst in nodes:
            if dst == src:
                continue
            if dst in dist:
                lengths.append(dist[dst])
            else:
                unreachable += 1
    lengths.sort()
    return {
        "num_od_pairs": len(lengths),
        "unreachable_pairs": unreachable,
        "median_shortest_path": lengths[len(lengths) // 2],
        "p95_shortest_path": lengths[int(0.95 * len(lengths))],
        "diameter": lengths[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--walks-per-node", type=int, default=400)
    args = parser.parse_args()

    valid_turns, n2n = load(args.map_dir)
    with open(f"{args.map_dir}/coords.pkl", "rb") as f:
        coords = pickle.load(f)

    report = {}
    report.update(counter_accuracy(valid_turns, n2n, coords, args.walks_per_node))
    report["displacements_per_direction"] = displacement_spread(n2n, coords)
    report.update(turn_set_partition(valid_turns))
    report.update(path_lengths(valid_turns, n2n))
    print(json.dumps(report, indent=2))

    if report["counter_model_accuracy"] > 0.5:
        print(f"\nWARNING: a two-counter model already scores "
              f"{report['counter_model_accuracy']:.1%} on state. The graph is close to "
              "degenerate -- raise --p-long / --max-skip / --n-diagonals.")


if __name__ == "__main__":
    main()
