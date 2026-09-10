"""Collect every run's results into one table.

Reads world-model-evaluation-main/results/<run>/*.json -- each record carries the
architecture it was produced with, so nothing here needs to open a checkpoint --
and prints one row per run, sorted by parameter count.

    python gridworld/compare_runs.py
    python gridworld/compare_runs.py --filter nyc10 --csv sweep.csv

The point of the sweep is the shape of the columns, not any single row. If
compression and edge_jaccard stay flat while parameters grow by an order of
magnitude, size was not the binding constraint -- which is what the paper's own
numbers imply and what this testbed exists to test.
"""
import argparse
import csv
import json
import os
import sys

# script filename -> (column heading, key inside the record)
COLUMNS = [
    ("evaluate_traversal_capabilities.json", "valid", "percent_valid_traversals"),
    ("next_token_test.json", "next-tok", "next_token_accuracy"),
    ("probe_test.json", "probe", "probe_accuracy"),
    ("compression_test.json", "compress", "compression_precision"),
    ("distinction_test.json", "dist-P", "distinction_precision"),
    ("distinction_test.json", "dist-R", "distinction_recall"),
    ("detour_analysis-p0.01.json", "detour.01", "valid_traversal_rate"),
    ("detour_analysis-p0.5.json", "detour.5", "valid_traversal_rate"),
    ("map/reconstruction.json", "jaccard", "edge_jaccard"),
    ("map-control/reconstruction.json", "ctrl-jac", "edge_jaccard"),
    ("map/reconstruction.json", "sat-floor", "saturation_jaccard"),
    ("map/reconstruction.json", "|E|/|E*|", "edge_count_ratio"),
    # The budget-independent quality number: unlike jaccard it does not drift
    # with how many sequences were reconstructed.
    ("error_analysis.json", "tok-err", "token_error_rate"),
    ("error_analysis.json", "arrived", "reached_destination_rate"),
]


def gpt2_parameters(n_layer, n_embd, n_head=None, vocab=110, n_positions=1024):
    """Parameter count for a GPT-2 body plus its embeddings.

    Per layer: 4*d^2 attention (q, k, v, out) + 8*d^2 feed-forward (4d up, 4d
    down). Close enough to rank runs by; biases and layernorms are noise at this
    scale, and the head is tied to the input embedding.
    """
    return 12 * n_layer * n_embd ** 2 + vocab * n_embd + n_positions * n_embd


def read(results_root, run, relative):
    path = os.path.join(results_root, run, relative)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def collect(results_root, name_filter):
    rows = []
    for run in sorted(os.listdir(results_root)):
        if not os.path.isdir(os.path.join(results_root, run)):
            continue
        if name_filter and name_filter not in run:
            continue
        row = {"run": run}
        architecture = {}
        for filename, heading, key in COLUMNS:
            record = read(results_root, run, filename)
            row[heading] = record.get(key) if record else None
            if record and not architecture:
                architecture = {k: record[k] for k in ("n_layer", "n_embd", "n_head")
                                if k in record}
        row.update(architecture)
        if {"n_layer", "n_embd"} <= architecture.keys():
            row["params"] = gpt2_parameters(architecture["n_layer"], architecture["n_embd"])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default=None,
                        help="results directory (default: "
                             "../world-model-evaluation-main/results)")
    parser.add_argument("--filter", default=None,
                        help="only runs whose name contains this substring")
    parser.add_argument("--csv", default=None, help="also write the table here")
    args = parser.parse_args()

    results_root = args.results or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "world-model-evaluation-main", "results")
    if not os.path.isdir(results_root):
        raise SystemExit(f"No results directory at {results_root}")

    rows = collect(results_root, args.filter)
    if not rows:
        raise SystemExit("No runs found. Train one, then run the metric scripts.")
    rows.sort(key=lambda r: (r.get("params") or 0, r["run"]))

    headings = ["run", "layers", "dims", "heads", "params"] + [c[1] for c in COLUMNS]
    keys = ["run", "n_layer", "n_embd", "n_head", "params"] + [c[1] for c in COLUMNS]
    widths = [max(len(h), 8) for h in headings]
    widths[0] = max(len(r["run"]) for r in rows) + 1

    def render(value, width):
        if value is None:
            return "-".rjust(width)
        if isinstance(value, float):
            return f"{value:.3f}".rjust(width)
        if isinstance(value, int) and value > 10_000:
            return f"{value / 1e6:.1f}M".rjust(width)
        return str(value).rjust(width)

    print("  ".join(h.rjust(w) for h, w in zip(headings, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(render(row.get(k), w) for k, w in zip(keys, widths)))

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n-> {args.csv}")

    gaps = [r for r in rows if r.get("jaccard") is not None
            and r.get("sat-floor") is not None
            and r["jaccard"] < r["sat-floor"] * 1.25]
    if gaps:
        print("\nSaturated (jaccard within 25% of the floor, so it is measuring the "
              "sequence budget rather than the model): "
              + ", ".join(r["run"] for r in gaps), file=sys.stderr)

    missing = [r["run"] for r in rows if r.get("compress") is None]
    if missing:
        print(f"\nNo compression_test.json yet for: {', '.join(missing)}",
              file=sys.stderr)


if __name__ == "__main__":
    main()
