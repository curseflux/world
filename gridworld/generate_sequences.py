"""Generate turn-by-turn sequence datasets over a grid world.

Emits exactly the artifacts the Vafa et al. pipeline consumes, so the eval
scripts in world-model-evaluation-main/ run unmodified against a 10x10 map:

    data/<name>/train_sequences.txt          "<origin> <dest> <dir> ... end"
    data/<name>/heldout_sequences.txt
    data/<name>/valid_turns.pkl
    data/<name>/node_and_direction_to_neighbor.pkl
    data/<name>/all_pairs.pkl                reachable (origin, dest) pairs
    data/<name>/shortest_paths.pkl           (o, d) -> number of nodes on the path
    data/<name>/model_config.json            layers/width for the size sweep
    data/<name>/tokenizer.pt                 (only if torch is importable)

The three traversal distributions follow Appendix F of the paper:
  shortest        Dijkstra over Euclidean edge length
  noisy-shortest  Dijkstra over W + eps, eps ~ Gamma(W, 1), 50 resampled maps
  random-walks    uniform origin, uniform length, uniform legal edge; the
                  destination is wherever the walk happens to end

Train/heldout are split by (origin, destination) pair, never by sequence, so no
OD pair appears on both sides -- the same discipline the paper uses.  On a
100-node graph there are only 9,900 OD pairs in total, which is worth keeping in
mind: see the coverage warning printed at the end of a run.
"""
import argparse
import heapq
import json
import math
import os
import pickle
import random
from collections import defaultdict


def load_map(map_dir):
    with open(f"{map_dir}/valid_turns.pkl", "rb") as f:
        valid_turns = pickle.load(f)
    with open(f"{map_dir}/node_and_direction_to_neighbor.pkl", "rb") as f:
        n2n = pickle.load(f)
    with open(f"{map_dir}/coords.pkl", "rb") as f:
        coords = pickle.load(f)
    return valid_turns, n2n, coords


def edge_lengths(valid_turns, n2n, coords):
    """Euclidean length of each edge, so 'shortest path' is a real tradeoff.

    Without this the long skip edges would be free and shortest paths would
    always prefer them.
    """
    lengths = {}
    for node, turns in valid_turns.items():
        for direction in turns:
            neighbor = n2n[(node, direction)]
            (r1, c1), (r2, c2) = coords[node], coords[neighbor]
            lengths[(node, direction)] = math.hypot(r2 - r1, c2 - c1)
    return lengths


def dijkstra(origin, valid_turns, n2n, weights):
    """Single-source shortest paths; returns predecessor direction per node."""
    dist = {origin: 0.0}
    prev = {}
    queue = [(0.0, origin)]
    while queue:
        d, node = heapq.heappop(queue)
        if d > dist.get(node, math.inf):
            continue
        for direction in valid_turns[node]:
            neighbor = n2n[(node, direction)]
            nd = d + weights[(node, direction)]
            if nd < dist.get(neighbor, math.inf):
                dist[neighbor] = nd
                prev[neighbor] = (node, direction)
                heapq.heappush(queue, (nd, neighbor))
    return dist, prev


def reconstruct(origin, destination, prev):
    directions = []
    node = destination
    while node != origin:
        if node not in prev:
            return None
        node, direction = prev[node]
        directions.append(direction)
    return directions[::-1]


def hop_counts(valid_turns, n2n):
    """(o, d) -> node count along the fewest-hop path, matching the paper's
    shortest_paths.pkl which stores len(node_list)."""
    from collections import deque
    counts = {}
    for origin in valid_turns:
        dist, queue = {origin: 0}, deque([origin])
        while queue:
            u = queue.popleft()
            for direction in valid_turns[u]:
                v = n2n[(u, direction)]
                if v not in dist:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        for node, d in dist.items():
            counts[(origin, node)] = d + 1
    return counts


def format_sequence(origin, destination, directions):
    return " ".join([str(origin), str(destination)] + list(directions) + ["end"])


def gen_shortest(pairs, valid_turns, n2n, weights, noisy_rounds, rng):
    """Deterministic shortest paths, or noisy_rounds resampled weightings."""
    sequences = []
    rounds = max(noisy_rounds, 1)
    for _ in range(rounds):
        if noisy_rounds:
            w = {k: v + rng.gammavariate(max(v, 1e-3), 1.0) for k, v in weights.items()}
        else:
            w = weights
        by_origin = defaultdict(list)
        for origin, destination in pairs:
            by_origin[origin].append(destination)
        for origin, destinations in by_origin.items():
            _, prev = dijkstra(origin, valid_turns, n2n, w)
            for destination in destinations:
                directions = reconstruct(origin, destination, prev)
                if directions:
                    sequences.append(format_sequence(origin, destination, directions))
    return sequences


def gen_random_walks(count, valid_turns, n2n, min_len, max_len, rng):
    """Uniform origin, uniform length, uniform legal edge (paper Appendix F)."""
    nodes = [n for n in valid_turns if valid_turns[n]]
    sequences = []
    for _ in range(count):
        node = rng.choice(nodes)
        origin = node
        directions = []
        for _ in range(rng.randint(min_len, max_len)):
            turns = valid_turns[node]
            if not turns:
                break
            direction = rng.choice(turns)
            directions.append(direction)
            node = n2n[(node, direction)]
        if directions and node != origin:
            sequences.append(format_sequence(origin, node, directions))
    return sequences


def split_by_pair(sequences, heldout_pairs):
    train, heldout = [], []
    for seq in sequences:
        parts = seq.split(" ")
        pair = (int(parts[0]), int(parts[1]))
        (heldout if pair in heldout_pairs else train).append(seq)
    return train, heldout


def save_tokenizer(out_dir, sequences, n2n, valid_turns):
    """Build the SimpleTokenizer the training code expects, if torch is around."""
    try:
        import sys
        import torch
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "world-model-evaluation-main"))
        from model import SimpleTokenizer
    except ImportError as exc:
        return f"skipped tokenizer.pt ({exc}); rerun this script where torch is installed"
    tokenizer = SimpleTokenizer(sequences, n2n, valid_turns)
    torch.save(tokenizer, f"{out_dir}/tokenizer.pt")
    return f"wrote tokenizer.pt (vocab {len(tokenizer.word_to_id)})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="data/<name>")
    parser.add_argument("--mode", choices=["shortest", "noisy-shortest", "random-walks"],
                        required=True)
    parser.add_argument("--noisy-rounds", type=int, default=50)
    parser.add_argument("--num-walks", type=int, default=500_000)
    parser.add_argument("--min-walk-len", type=int, default=3)
    parser.add_argument("--max-walk-len", type=int, default=40)
    parser.add_argument("--heldout-frac", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--n-layer", type=int, default=12)
    parser.add_argument("--n-embd", type=int, default=768)
    parser.add_argument("--n-head", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    valid_turns, n2n, coords = load_map(args.map_dir)
    weights = edge_lengths(valid_turns, n2n, coords)
    counts = hop_counts(valid_turns, n2n)

    # Pairs that fit in the context window (+3 for origin, destination, end).
    all_pairs = [pair for pair, n in counts.items()
                 if pair[0] != pair[1] and n + 3 < args.max_tokens]
    heldout_pairs = set(rng.sample(all_pairs, int(args.heldout_frac * len(all_pairs))))

    if args.mode == "random-walks":
        sequences = gen_random_walks(args.num_walks, valid_turns, n2n,
                                     args.min_walk_len, args.max_walk_len, rng)
    else:
        sequences = gen_shortest(all_pairs, valid_turns, n2n, weights,
                                 args.noisy_rounds if args.mode == "noisy-shortest" else 0,
                                 rng)

    sequences = [s for s in sequences if len(s.split(" ")) <= args.max_tokens]
    sequences = list(dict.fromkeys(sequences))  # dedupe, preserving order
    rng.shuffle(sequences)
    train, heldout = split_by_pair(sequences, heldout_pairs)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/train_sequences.txt", "w") as f:
        f.write("\n".join(train))
    with open(f"{args.out_dir}/heldout_sequences.txt", "w") as f:
        f.write("\n".join(heldout))
    for name, obj in [("valid_turns", valid_turns),
                      ("node_and_direction_to_neighbor", n2n),
                      ("all_pairs", all_pairs),
                      ("shortest_paths", counts),
                      ("coords", coords)]:
        with open(f"{args.out_dir}/{name}.pkl", "wb") as f:
            pickle.dump(obj, f)
    with open(f"{args.out_dir}/model_config.json", "w") as f:
        json.dump({"n_layer": args.n_layer, "n_embd": args.n_embd,
                   "n_head": args.n_head}, f, indent=2)

    note = save_tokenizer(args.out_dir, sequences, n2n, valid_turns)
    train_pairs = {tuple(s.split(" ")[:2]) for s in train}
    summary = {
        "mode": args.mode,
        "train_sequences": len(train),
        "heldout_sequences": len(heldout),
        "train_tokens": sum(len(s.split(" ")) for s in train),
        "distinct_train_od_pairs": len(train_pairs),
        "total_od_pairs": len(all_pairs),
        "heldout_od_pairs": len(heldout_pairs),
        "tokenizer": note,
    }
    print(json.dumps(summary, indent=2))

    coverage = len(train_pairs) / max(len(all_pairs) - len(heldout_pairs), 1)
    if coverage > 0.95 and len(train) < 50_000:
        print("\nWARNING: training data covers %.0f%% of all trainable OD pairs in only "
              "%d sequences. At this scale the model can memorise the pair table, so a "
              "good score here does not transfer to a claim about Manhattan. Prefer the "
              "random-walks arm, or scale the grid up." % (100 * coverage, len(train)))


if __name__ == "__main__":
    main()
