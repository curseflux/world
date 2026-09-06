#!/usr/bin/env bash
# Run every evaluation for one trained dataset and persist the results.
#
#   ./run_evals.sh <data-name> <map-dir> [num-sequences]
#
# Each metric script writes results/<data>/<script>.json; the reconstruction
# writes results/<data>/map/{reconstruction.json,map.svg,invented_edges.json}.
# A matched noise control is run at the model's own error rate, which is the
# comparison that makes the reconstruction numbers interpretable.
set -uo pipefail

DATA="${1:?usage: run_evals.sh <data-name> <map-dir> [num-sequences]}"
MAP_DIR="${2:?usage: run_evals.sh <data-name> <map-dir> [num-sequences]}"
NSEQ="${3:-6400}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/../world-model-evaluation-main"
MAP_DIR="$(cd "$MAP_DIR" && pwd)"
cd "$REPO"

RESULTS="results/$DATA"
mkdir -p "$RESULTS"

run() {  # keep going if one metric fails; the log says which
  echo; echo "===== $* ====="
  if ! "$@" 2>&1 | tee -a "$RESULTS/console.log"; then
    echo "FAILED: $*" | tee -a "$RESULTS/console.log"
  fi
}

run python evaluate_traversal_capabilities.py --data "$DATA"
run python next_token_test.py --data "$DATA"
run python probe_test.py --data "$DATA" --use-heldout
run python compression_test.py --data "$DATA"
run python distinction_test.py --data "$DATA"
for P in 0.01 0.10 0.50 0.75; do
  run python detour_analysis.py --data "$DATA" --detour-prob "$P"
done

echo; echo "===== map reconstruction ====="
python "$HERE/sample_from_model.py" --data "$DATA" \
  --out "$RESULTS/samples.txt" --num-sequences "$NSEQ" 2>&1 | tee -a "$RESULTS/console.log"
python "$HERE/reconstruct_map.py" --map-dir "$MAP_DIR" \
  --samples "$RESULTS/samples.txt" --num-sequences "$NSEQ" \
  --out-dir "$RESULTS/map" 2>&1 | tee -a "$RESULTS/console.log"

# Matched noise control: the true world model corrupted at the model's own error
# rate. Without it the reconstruction numbers have nothing to be compared against.
ERR=$(python - "$RESULTS/map/reconstruction.json" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
print(round(m["sequences_unreconstructable"] / max(m["sequences_used"], 1), 4))
PY
)
echo; echo "===== noise control at error rate ${ERR} ====="
python "$HERE/reconstruct_map.py" --map-dir "$MAP_DIR" --corrupt "$ERR" \
  --num-sequences "$NSEQ" --out-dir "$RESULTS/map-control" 2>&1 | tee -a "$RESULTS/console.log"

echo; echo "All results under $REPO/$RESULTS/"
ls -1 "$RESULTS" "$RESULTS/map" 2>/dev/null
