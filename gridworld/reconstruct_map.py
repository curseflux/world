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

import chartkit
import render
from build_map import bearing


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


def token_error_rate(sequences, valid_turns, n2n):
    """Per-token probability that a generated direction is illegal.

    This is the quantity --corrupt takes, and the one the paper matches its
    control on: "with probability equal to the probability of an error for the
    random walks transformer, we randomly re-label an edge in a sequence".

    Each direction token is a Bernoulli trial. After the first illegal one the
    walk's true state is undefined, so later tokens in that sequence cannot be
    judged and the sequence stops contributing. Counting trials up to and
    including the first error is the MLE for the per-token rate.

    A per-*sequence* failure rate is not a substitute: over ~26-token sequences a
    per-token rate of 0.002 already makes ~5% of sequences invalid, so passing a
    sequence rate to --corrupt over-corrupts the control by more than an order of
    magnitude and flatters the model it is meant to be compared against.
    """
    trials = errors = 0
    for sequence in sequences:
        node = sequence[0]
        for direction in sequence[2:]:
            trials += 1
            if direction not in valid_turns.get(node, ()):
                errors += 1
                break
            node = n2n[(node, direction)]
    return errors / trials if trials else 0.0


def score(reconstructed, coords):
    """Edge precision/recall, plus the rate of physically impossible edges.

    reconstruct_sequence labels a new edge with the direction token that
    produced it and never checks that against the geometry, so an edge labelled
    NW can run east. Those are the paper's "streets whose orientations are
    physically impossible"; counting them turns its most vivid qualitative claim
    into a number.
    """
    counts = {"true": 0, "new": 0, "true_unused": 0}
    invented = []
    for u, v, _, data in reconstructed.out_edges(keys=True, data=True):
        counts[data["edge_type"]] = counts.get(data["edge_type"], 0) + 1
        if data["edge_type"] == "new":
            label = data.get("direction")
            actual = bearing(coords[u], coords[v])
            invented.append({"from": u, "to": v, "label": label,
                             "actual_bearing": actual, "impossible": actual != label})
    used = counts["true"] + counts["new"]
    real = counts["true"] + counts["true_unused"]
    union = counts["true"] + counts["new"] + counts["true_unused"]
    impossible = sum(1 for e in invented if e["impossible"])
    precision = counts["true"] / used if used else 0.0
    recall = counts["true"] / real if real else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_edges_recovered": counts["true"],
        "false_edges_invented": counts["new"],
        "true_edges_never_used": counts["true_unused"],
        "reconstructed_edge_count": used,
        "true_edge_count": real,
        "edge_count_ratio": round(used / real, 4) if real else 0.0,
        "edge_precision": round(precision, 4),
        "edge_recall": round(recall, 4),
        "edge_f1": round(f1, 4),
        # Jaccard is the "same edge set, no more and no less" number: it reaches
        # 1.0 only when the reconstructed edge set equals the true one exactly,
        # and every invented or missed edge drives it down.
        "edge_jaccard": round(counts["true"] / union, 4) if union else 0.0,
        "impossible_orientation_edges": impossible,
        "impossible_orientation_rate": round(impossible / counts["new"], 4) if counts["new"] else 0.0,
    }, invented


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--map-dir", required=True)
    parser.add_argument("--samples", help="file of model samples, one sequence per line")
    parser.add_argument("--corrupt", type=float,
                        help="instead of --samples, corrupt this fraction of direction "
                             "tokens in true traversals (the paper's noise control)")
    parser.add_argument("--num-sequences", type=int, default=0,
                        help="0 auto-scales to the graph: ~0.65 sequences per true "
                             "edge, the same ratio the paper used (6,400 sequences "
                             "for Manhattan's 9,846 edges), floored at 200. Copying "
                             "6,400 onto a 170-edge map reconstructs 38 sequences "
                             "per edge and saturates the metric.")
    parser.add_argument("--max-degree", type=int, default=0,
                        help="0 uses the true graph's own maximum out-degree")
    parser.add_argument("--max-distance", type=float, default=0.0,
                        help="0 uses the length of the longest true edge")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sweep", action="store_true",
                        help="report the metrics at doubling sequence budgets instead "
                             "of one number, showing whether edge invention saturates")
    parser.add_argument("--show-nodes", action="store_true",
                        help="draw a dot at each intersection. Off by default: the "
                             "paper's make_map draws edges only.")
    parser.add_argument("--highlight-false", action="store_true",
                        help="in the reconstructed panel, colour real streets black "
                             "and invented ones red. Off by default: the true map "
                             "sits alongside for comparison, so the reconstructed "
                             "panel is drawn as one map of what the model implies.")
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
    true_edges = sum(len(v) for v in valid_turns.values())
    if not args.num_sequences:
        args.num_sequences = max(200, round(0.65 * true_edges))
        if args.sweep:
            # A sweep is about the shape of the curve, so it needs range well
            # past the point-estimate budget; at the auto budget alone it would
            # be a single point.
            args.num_sequences = max(6400, 32 * args.num_sequences)
        print(f"--num-sequences auto: {args.num_sequences} "
              f"({true_edges} true edges x 0.65, floored at 200"
              f"{', x32 for the sweep' if args.sweep else ''})")
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
    neighbours = lambda graph, node: [
        other for other in graph.nodes
        if math.dist(coords[node], coords[other]) <= max_distance]

    def run(batch):
        graph = true_graph.copy()
        for edge in graph.edges(keys=True):
            graph.edges[edge]["edge_type"] = "true_unused"
        failures = 0
        for sequence in batch:
            if not reconstruction.reconstruct_sequence(
                    graph, sequence, neighbours, max_degree=max_degree):
                failures += 1
        return graph, failures

    if args.sweep:
        # Invented edges accumulate with every sequence reconstructed, so a
        # single precision number is a property of (model, budget) rather than
        # of the model. The curve says which: one that flattens means a bounded
        # map, even a wrong one; one still climbing means the model keeps
        # inventing streets for as long as you keep asking.
        budgets, size = [], 25
        while size < len(sequences):
            budgets.append(size)
            size *= 2
        budgets.append(len(sequences))
        curve = []
        for budget in budgets:
            partial, _ = run(sequences[:budget])
            point, _ = score(partial, coords)
            curve.append({"sequences": budget, **{
                k: point[k] for k in ("edge_precision", "edge_recall", "edge_f1",
                                      "edge_jaccard", "edge_count_ratio",
                                      "false_edges_invented")}})
        print(f"{'seqs':>7}  {'prec':>6}  {'recall':>6}  {'F1':>6}  "
              f"{'IoU':>6}  {'|E|/|E*|':>8}  {'invented':>8}")
        for point in curve:
            print(f"{point['sequences']:>7}  {point['edge_precision']:>6.3f}  "
                  f"{point['edge_recall']:>6.3f}  {point['edge_f1']:>6.3f}  "
                  f"{point['edge_jaccard']:>6.3f}  {point['edge_count_ratio']:>8.3f}  "
                  f"{point['false_edges_invented']:>8d}")
        os.makedirs(args.out_dir, exist_ok=True)
        saturation = true_edges / (len(valid_turns) * max_degree)
        with open(f"{args.out_dir}/sweep.json", "w") as f:
            json.dump({"source": source, "saturation_jaccard": round(saturation, 4),
                       "peak_jaccard_at": max(curve, key=lambda p: p["edge_jaccard"]),
                       "curve": curve}, f, indent=2)
        chartkit.line_chart(
            [("precision", [p["edge_precision"] for p in curve]),
             ("recall", [p["edge_recall"] for p in curve]),
             ("jaccard", [p["edge_jaccard"] for p in curve])],
            [p["sequences"] for p in curve],
            f"{args.out_dir}/sweep.svg",
            title="Reconstruction quality vs how many sequences you reconstruct",
            subtitle=(f"{source}. Every sequence is another chance to invent an edge, "
                      f"so precision and jaccard only fall; recall only rises. "
                      f"Compare two runs at the same budget."),
            y_label="score",
            reference=(saturation, f"saturation floor {saturation:.2f}"))
        # Jaccard peaks where recall has just saturated: before that the map is
        # still missing real streets, after it only false ones are being added.
        # That peak is the budget at which the metric best separates models.
        best = max(curve, key=lambda p: p["edge_jaccard"])
        print(f"\njaccard peaks at {best['edge_jaccard']:.3f} with "
              f"{best['sequences']} sequences (recall {best['edge_recall']:.3f}); "
              f"by {curve[-1]['sequences']} it has fallen to "
              f"{curve[-1]['edge_jaccard']:.3f} against a floor of {saturation:.3f}.")
        print(f"Compare runs at a fixed budget near {best['sequences']}, not at the "
              f"largest one you can afford.")
        print(f"-> {args.out_dir}/sweep.json")
        print(f"-> {args.out_dir}/sweep.svg")
        return

    reconstructed, failed = run(sequences)
    metrics, invented = score(reconstructed, coords)

    # Reconstruction can fill at most max_degree out-edges per node, so with enough
    # sequences ANY model with a non-zero error rate converges on this floor. Past
    # it the metric stops measuring the model and starts measuring the budget.
    ceiling_edges = len(valid_turns) * max_degree
    saturation = true_edges / ceiling_edges
    metrics["saturation_jaccard"] = round(saturation, 4)
    metrics.update(sequences_used=len(sequences),
                   sequences_unreconstructable=failed,
                   token_error_rate=round(
                       token_error_rate(sequences, valid_turns, n2n), 6),
                   max_degree=max_degree,
                   max_distance=round(max_distance, 3),
                   source=source)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/reconstruction.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{args.out_dir}/invented_edges.json", "w") as f:
        json.dump(invented, f, indent=2)
    with open(f"{args.out_dir}/reconstructed_graph.pkl", "wb") as f:
        pickle.dump(reconstructed, f)
    subtitle = (f"precision {metrics['edge_precision']:.3f}   "
                f"recall {metrics['edge_recall']:.3f}   "
                f"{metrics['false_edges_invented']} false edges, "
                f"{metrics['impossible_orientation_edges']} physically impossible"
                f"   -   {source}")
    render.render(reconstructed, coords, f"{args.out_dir}/map.svg",
                  title="Reconstructed map", subtitle=subtitle,
                  palette=render.RECONSTRUCTED, highlight_false=args.highlight_false,
                  show_unused=args.show_unused, show_nodes=args.show_nodes)
    render.render_pair(
        true_graph, reconstructed, coords, f"{args.out_dir}/true_vs_reconstructed.svg",
        left_title="True world model",
        left_sub=f"{true_graph.number_of_edges()} edges",
        right_title="Reconstructed from sequences", right_sub=subtitle,
        highlight_false=args.highlight_false,
        show_unused=args.show_unused, show_nodes=args.show_nodes)

    print(json.dumps(metrics, indent=2))
    if metrics["edge_jaccard"] < saturation * 1.25:
        print(f"\nWARNING: jaccard {metrics['edge_jaccard']:.3f} is close to the "
              f"saturation floor {saturation:.3f} ({true_edges} true edges over "
              f"{len(valid_turns)} nodes x {max_degree} slots). At this budget the "
              f"metric barely separates a good model from a bad one -- lower "
              f"--num-sequences, and read token_error_rate as the model-quality "
              f"number instead.")
    print(f"-> {args.out_dir}/map.svg")
    print(f"-> {args.out_dir}/true_vs_reconstructed.svg")


if __name__ == "__main__":
    main()
