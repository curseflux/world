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


def render_svg(reconstructed, coords, metrics, title, path, cell=64, pad=44):
    rows = max(r for r, _ in coords.values()) + 1
    cols = max(c for _, c in coords.values()) + 1
    width, height = cols * cell + 2 * pad, rows * cell + 2 * pad + 34

    def xy(node):
        r, c = coords[node]
        return pad + c * cell, pad + r * cell + 34

    style = {"true": ("#1f2933", 1.7, 1.0),
             "true_unused": ("#c3cbd4", 1.4, 1.0),
             "new": ("#d7263d", 2.0, 0.95)}
    layer = {"true_unused": 0, "true": 1, "new": 2}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="ui-sans-serif,system-ui,sans-serif">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        '<defs>',
    ]
    for kind, (color, _, _) in style.items():
        parts.append(
            f'<marker id="a-{kind}" viewBox="0 0 10 10" refX="9" refY="5" '
            f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" fill="{color}"/></marker>')
    parts.append('</defs>')
    parts.append(f'<text x="{pad}" y="24" font-size="15" fill="#1f2933">{title}</text>')

    edges = sorted(reconstructed.out_edges(keys=True, data=True),
                   key=lambda e: layer.get(e[3]["edge_type"], 0))
    for u, v, _, data in edges:
        color, stroke, opacity = style.get(data["edge_type"], ("#999", 1, 1))
        x1, y1 = xy(u)
        x2, y2 = xy(v)
        # Stop short of the node so the arrowhead stays readable.
        dx, dy = x2 - x1, y2 - y1
        norm = math.hypot(dx, dy) or 1
        x1, y1 = x1 + 5 * dx / norm, y1 + 5 * dy / norm
        x2, y2 = x2 - 8 * dx / norm, y2 - 8 * dy / norm
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{stroke}" opacity="{opacity}" '
            f'marker-end="url(#a-{data["edge_type"]})"/>')
    for node in coords:
        x, y = xy(node)
        parts.append(f'<circle cx="{x}" cy="{y}" r="3.4" fill="#ffffff" '
                     f'stroke="#52606d" stroke-width="1.2"/>')

    legend = [("#1f2933", f"recovered true edge ({metrics['true_edges_recovered']})"),
              ("#d7263d", f"invented false edge ({metrics['false_edges_invented']})"),
              ("#c3cbd4", f"true edge never used ({metrics['true_edges_never_used']})")]
    for i, (color, label) in enumerate(legend):
        y = height - pad + 14 + i * 15
        parts.append(f'<line x1="{pad}" y1="{y - 4}" x2="{pad + 22}" y2="{y - 4}" '
                     f'stroke="{color}" stroke-width="2.4"/>')
        parts.append(f'<text x="{pad + 30}" y="{y}" font-size="11.5" '
                     f'fill="#52606d">{label}</text>')

    parts.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(parts))


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

    reconstructed = true_graph.copy()
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
    title = (f"precision {metrics['edge_precision']:.3f}  "
             f"recall {metrics['edge_recall']:.3f}  -  {source}")
    render_svg(reconstructed, coords, metrics, title, f"{args.out_dir}/map.svg")

    print(json.dumps(metrics, indent=2))
    print(f"-> {args.out_dir}/map.svg")


if __name__ == "__main__":
    main()
