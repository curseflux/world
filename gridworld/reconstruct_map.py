"""Reconstruct a grid world's map from sequences a model generated, and score it.

The paper's `mapping/` pipeline is Manhattan-only: it downloads the street graph
from OpenStreetMap via osmnx and renders with folium onto real lat/long. None of
that applies to a synthetic grid, so this reuses the part that *is* generic --
`mapping/reconstruction.reconstruct_sequence` -- with a Euclidean neighbourhood
instead of a great-circle one, and renders plain SVG.

The payoff of a synthetic map is that scoring stops being visual. The paper
inspects maps by eye for "impossible orientations and flyovers"; on 100 nodes we
can compute edge precision and recall against ground truth exactly:

    precision = true_used / (true_used + invented)   of the edges the model's
                                                     sequences imply, how many
                                                     are real
    recall    = true_used / (true_used + never_used) of the real edges, how many
                                                     the model's sequences reach

Precision falls as more sequences are reconstructed (every extra sequence is
another chance to invent an edge) while recall rises, so --num-sequences must
match across any two runs you intend to compare.

Use --corrupt to run the paper's own control (Figure 3, middle panel): take
sequences that are valid under the true world model, corrupt a fraction of the
direction tokens, and reconstruct from those. A model whose map is merely noisy
scores like the control; one whose map is incoherent scores far worse.
"""
import argparse
import json
import math
import os
import pickle
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "world-model-evaluation-main", "mapping"))
import networkx as nx
import reconstruction

import render


def load_map(map_dir):
    with open(f"{map_dir}/valid_turns.pkl", "rb") as f:
        valid_turns = pickle.load(f)
    with open(f"{map_dir}/node_and_direction_to_neighbor.pkl", "rb") as f:
        n2n = pickle.load(f)
    with open(f"{map_dir}/coords.pkl", "rb") as f:
        coords = pickle.load(f)
    return valid_turns, n2n, coords


def build_true_graph(coords, n2n):
    """MultiDiGraph in the shape reconstruction.py expects."""
    graph = nx.MultiDiGraph()
    for node, position in coords.items():
        graph.add_node(node, lat_long=position)
    for (node, direction), neighbor in n2n.items():
        graph.add_edge(node, neighbor, direction=direction, edge_type="true_unused")
    return graph


def sample2sequence(sample):
    """Same contract as mapping/utils.sample2sequence: [src, dst] + directions."""
    tokens = sample.split(" ")
    if len(tokens) <= 3 or tokens[-1] != "end":
        return []
    return [int(tokens[0]), int(tokens[1])] + tokens[2:-1]


def corrupt_sequences(valid_turns, n2n, coords, count, rate, max_len, rng):
    """Valid random traversals with a fraction of direction tokens re-labelled."""
    directions = sorted({d for _, d in n2n})
    nodes = [n for n in valid_turns if valid_turns[n]]
    sequences = []
    for _ in range(count):
        node = rng.choice(nodes)
        origin, moves = node, []
        for _ in range(rng.randint(3, max_len)):
            turns = valid_turns[node]
            if not turns:
                break
            move = rng.choice(turns)
            moves.append(move)
            node = n2n[(node, move)]
        if not moves or node == origin:
            continue
        for i in range(len(moves)):
            if rng.random() < rate:
                moves[i] = rng.choice([d for d in directions if d != moves[i]])
        sequences.append([origin, node] + moves)
    return sequences


def score(reconstructed):
    counts = {"true": 0, "new": 0, "true_unused": 0}
    invented = []
    for u, v, _, data in reconstructed.out_edges(keys=True, data=True):
        counts[data["edge_type"]] = counts.get(data["edge_type"], 0) + 1
        if data["edge_type"] == "new":
            invented.append((u, v, data.get("direction")))
    used = counts["true"] + counts["new"]
    real = counts["true"] + counts["true_unused"]
    return {
        "true_edges_recovered": counts["true"],
        "false_edges_invented": counts["new"],
        "true_edges_never_used": counts["true_unused"],
        "edge_precision": round(counts["true"] / used, 4) if used else 0.0,
        "edge_recall": round(counts["true"] / real, 4) if real else 0.0,
    }, invented


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--samples", help="file of model samples, one sequence per line")
    parser.add_argument("--corrupt", type=float,
                        help="instead of --samples, corrupt this fraction of direction "
                             "tokens in true traversals (the paper's noise control)")
    parser.add_argument("--num-sequences", type=int, default=6400)
    parser.add_argument("--max-degree", type=int, default=0,
                        help="0 uses the true graph's own maximum out-degree")
    parser.add_argument("--max-distance", type=float, default=0.0,
                        help="0 uses the length of the longest true edge")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--show-unused", action="store_true",
                        help="also draw true edges the reconstruction never used. The "
                             "paper's make_map skips these, so the default does too.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if bool(args.samples) == bool(args.corrupt is not None):
        parser.error("pass exactly one of --samples or --corrupt")

    rng = random.Random(args.seed)
    valid_turns, n2n, coords = load_map(args.map_dir)
    true_graph = build_true_graph(coords, n2n)

    # Give the reconstruction the same benefit of the doubt the paper does: it may
    # not invent an edge longer than the longest real one, nor exceed the true
    # graph's own maximum out-degree.
    max_degree = args.max_degree or max(len(v) for v in valid_turns.values())
    max_distance = args.max_distance or max(
        math.dist(coords[node], coords[neighbor]) for (node, _), neighbor in n2n.items())

    if args.samples:
        with open(args.samples) as f:
            raw = [line for line in f.read().split("\n") if line.strip()]
        sequences = [s for s in (sample2sequence(line) for line in raw) if s]
        source = f"{len(sequences)} of {len(raw)} samples well-formed"
    else:
        sequences = corrupt_sequences(valid_turns, n2n, coords, args.num_sequences,
                                      args.corrupt, 40, rng)
        source = f"true traversals with {args.corrupt:.0%} of tokens corrupted"

    rng.shuffle(sequences)
    sequences = sequences[:args.num_sequences]
    if not sequences:
        raise SystemExit("no usable sequences; check the samples file format")

    for edge in true_graph.edges(keys=True):
        true_graph.edges[edge]["edge_type"] = "true"
    reconstructed = true_graph.copy()
    for edge in reconstructed.edges(keys=True):
        reconstructed.edges[edge]["edge_type"] = "true_unused"
    neighbours = lambda graph, node: [
        other for other in graph.nodes
        if math.dist(coords[node], coords[other]) <= max_distance]

    failed = 0
    for sequence in sequences:
        if not reconstruction.reconstruct_sequence(
                reconstructed, sequence, neighbours, max_degree=max_degree):
            failed += 1

    metrics, invented = score(reconstructed)
    metrics.update(sequences_used=len(sequences),
                   sequences_unreconstructable=failed,
                   max_degree=max_degree,
                   max_distance=round(max_distance, 3),
                   source=source)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/reconstruction.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.out_dir}/invented_edges.json", "w") as f:
        json.dump([{"from": u, "to": v, "direction": d} for u, v, d in invented], f, indent=2)
    with open(f"{args.out_dir}/reconstructed_graph.pkl", "wb") as f:
        pickle.dump(reconstructed, f)
    subtitle = (f"precision {metrics['edge_precision']:.3f}   "
                f"recall {metrics['edge_recall']:.3f}   "
                f"{metrics['false_edges_invented']} false edges   -   {source}")
    render.render(reconstructed, coords, f"{args.out_dir}/map.svg",
                  title="Reconstructed map", subtitle=subtitle,
                  show_unused=args.show_unused)
    render.render_pair(
        true_graph, reconstructed, coords, f"{args.out_dir}/true_vs_reconstructed.svg",
        left_title="True world model",
        left_sub=f"{true_graph.number_of_edges()} edges",
        right_title="Reconstructed from sequences", right_sub=subtitle,
        show_unused=args.show_unused)

    print(json.dumps(metrics, indent=2))
    print(f"-> {args.out_dir}/map.svg")
    print(f"-> {args.out_dir}/true_vs_reconstructed.svg")


if __name__ == "__main__":
    main()
