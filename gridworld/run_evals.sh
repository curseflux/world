#!/usr/bin/env bash
# Run every evaluation for one trained dataset and persist the results.
#
#   ./run_evals.sh <data-name> <run-name> <map-dir> [num-sequences]
#
# <data-name> is the dataset under data/; <run-name> is the trained model under
# ckpts/, which is also where results are written. Pass the same run name you
# gave train.py as --model_name. They can be equal if you only train one
# architecture on a dataset.
#
# Writes results/<data>/<script>.json for each metric, results/<data>/map/ for
# the reconstruction, and results/<data>/map-control/ for the matched noise
# control -- the true world model corrupted at the model's own error rate, which
# is what makes the reconstruction numbers interpretable. Ends by printing the
# headline comparison.
#
# For the paper's untrained reference row, re-run the metric scripts by hand
# with --use-untrained-model.
set -uo pipefail

USAGE="usage: run_evals.sh <data-name> <run-name> <map-dir> [num-sequences]"
DATA="${1:?$USAGE}"
RUN="${2:?$USAGE}"
MAP_DIR="${3:?$USAGE}"
# 0 lets reconstruct_map.py scale the budget to the graph.
NSEQ="${4:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/../world-model-evaluation-main"
MAP_DIR="$(cd "$MAP_DIR" && pwd)"
cd "$REPO"

RESULTS="results/$RUN"
mkdir -p "$RESULTS"
FAILURES=0

run() {  # keep going if one step fails; the log and the tally say which
  echo; echo "===== $* ====="
  if ! "$@" 2>&1 | tee -a "$RESULTS/console.log"; then
    echo "FAILED: $*" | tee -a "$RESULTS/console.log"
    FAILURES=$((FAILURES + 1))
    return 1
  fi
}

run python evaluate_traversal_capabilities.py --data "$DATA" --run "$RUN"
run python next_token_test.py --data "$DATA" --run "$RUN"
run python probe_test.py --data "$DATA" --run "$RUN" --use-heldout
run python compression_test.py --data "$DATA" --run "$RUN"
run python distinction_test.py --data "$DATA" --run "$RUN"
for P in 0.01 0.10 0.50 0.75; do
  run python detour_analysis.py --data "$DATA" --run "$RUN" --detour-prob "$P"
done

# The reconstruction steps feed each other, so each one gates the next rather
# than letting a failure cascade into a later script with a missing input.
if run python "$HERE/sample_from_model.py" --data "$DATA" --run "$RUN" \
      --out "$RESULTS/samples.txt" --num-sequences "$NSEQ"; then

  run python "$HERE/reconstruct_map.py" --map-dir "$MAP_DIR" \
    --samples "$RESULTS/samples.txt" --num-sequences "$NSEQ" \
    --out-dir "$RESULTS/map"

  run python "$HERE/reconstruct_map.py" --map-dir "$MAP_DIR" \
    --samples "$RESULTS/samples.txt" --num-sequences "$NSEQ" --sweep \
    --out-dir "$RESULTS/map"

  # Where the errors fall, not just how many. A flat hazard means transcription
  # noise over a map the model basically has; a rising one means it loses track
  # of state as the path lengthens.
  run python "$HERE/analyze_errors.py" --map-dir "$MAP_DIR" \
    --samples "$RESULTS/samples.txt" --out-dir "$RESULTS" --label "$RUN"

  if [ -f "$RESULTS/map/reconstruction.json" ]; then
    # --corrupt is a per-TOKEN probability, so the control must be matched on the
    # model's per-token error rate. A per-sequence failure rate is roughly an
    # order of magnitude larger over 26-token sequences and would over-corrupt
    # the control, flattering the model it exists to be compared against.
    ERR=$(python -c "
import json
m = json.load(open('$RESULTS/map/reconstruction.json'))
print(m['token_error_rate'])")

    # Same error rate, same sequence budget: the only fair comparison, since
    # invented edges accumulate with both.
    run python "$HERE/reconstruct_map.py" --map-dir "$MAP_DIR" --corrupt "$ERR" \
      --num-sequences "$NSEQ" --out-dir "$RESULTS/map-control"
    run python "$HERE/reconstruct_map.py" --map-dir "$MAP_DIR" --corrupt "$ERR" \
      --num-sequences "$NSEQ" --sweep --out-dir "$RESULTS/map-control"
  else
    echo "No reconstruction.json; skipping the noise control." | tee -a "$RESULTS/console.log"
    FAILURES=$((FAILURES + 1))
  fi
else
  echo "Sampling failed; skipping reconstruction and control." | tee -a "$RESULTS/console.log"
fi

echo
echo "================ summary ================"
python - "$RESULTS" <<'PY' 2>&1 | tee -a "$RESULTS/console.log"
import json, os, sys

results = sys.argv[1]


def load(*parts):
    path = os.path.join(results, *parts)
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


rows = [
    ("valid traversals", "evaluate_traversal_capabilities.json", "percent_valid_traversals"),
    ("next-token accuracy", "next_token_test.json", "next_token_accuracy"),
    ("probe accuracy", "probe_test.json", "probe_accuracy"),
    ("compression precision", "compression_test.json", "compression_precision"),
    ("distinction precision", "distinction_test.json", "distinction_precision"),
    ("distinction recall", "distinction_test.json", "distinction_recall"),
]
for label, filename, key in rows:
    record = load(filename)
    value = record.get(key) if record else None
    print(f"  {label:<24} {value if value is None else f'{value:.3f}'}")

errors = load("error_analysis.json")
if errors:
    print(f"\n  token error rate         {errors['token_error_rate']:.4%}")
    print(f"  fully legal paths        {errors['legal_sequence_rate']:.3f}")
    print(f"  reached destination      {errors['reached_destination_rate']:.3f}")

model, control = load("map", "reconstruction.json"), load("map-control", "reconstruction.json")
if model:
    print(f"\n  reconstructed edges      {model['reconstructed_edge_count']} "
          f"vs {model['true_edge_count']} true "
          f"(ratio {model['edge_count_ratio']:.2f})")
    print(f"  edge jaccard             {model['edge_jaccard']:.3f} "
          f"(saturation floor {model.get('saturation_jaccard', float('nan')):.3f})")
    print(f"  impossible orientations  {model['impossible_orientation_rate']:.3f}")
if model and control:
    gap = model["edge_jaccard"] - control["edge_jaccard"]
    print(f"  control jaccard          {control['edge_jaccard']:.3f} "
          f"(true map corrupted at {model['token_error_rate']:.4f}/token, same budget)")
    print(f"\n  GAP vs control           {gap:+.3f}")
    print("  Near zero: the map is no worse than transcription noise at the same")
    print("  rate. Well below zero: the errors are structured, an incoherent map.")
PY

echo
if [ "$FAILURES" -gt 0 ]; then
  echo "$FAILURES step(s) failed -- see $REPO/$RESULTS/console.log"
fi
echo "All results under $REPO/$RESULTS/"
echo "Compare every run with: python gridworld/compare_runs.py"
