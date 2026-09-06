#!/usr/bin/env bash
# Build one grid world and all three traversal datasets, then validate them.
#
#   ./build_all.sh <tag> [size] [seed] [n_layer] [n_embd] [n_head]
#
# Writes the map to gridworld/maps/<tag>/ and datasets to
# world-model-evaluation-main/data/<tag>-{shortest,noisy-shortest,random-walks}/
# so the paper's own eval scripts pick them up via their --data flag.
set -euo pipefail

TAG="${1:?usage: build_all.sh <tag> [size] [seed] [n_layer] [n_embd] [n_head]}"
SIZE="${2:-10}"
SEED="${3:-0}"
# Depth over width: NextLat (App. F.1) reports depth helping state
# tracking substantially and width negligibly, and 768 dims is far more
# than a 100-node map needs.
N_LAYER="${4:-12}"
N_EMBD="${5:-256}"
N_HEAD="${6:-8}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_DIR="$HERE/maps/$TAG"
DATA_ROOT="$HERE/../world-model-evaluation-main/data"

echo "== building map: size=${SIZE} seed=${SEED} =="
python3 "$HERE/build_map.py" --size "$SIZE" --seed "$SEED" --out "$MAP_DIR"
python3 "$HERE/check_map.py" --map-dir "$MAP_DIR" | tee "$MAP_DIR/check_map.json"

for MODE in shortest noisy-shortest random-walks; do
  OUT="$DATA_ROOT/${TAG}-${MODE}"
  echo
  echo "== generating ${MODE} -> ${OUT} =="
  python3 "$HERE/generate_sequences.py" \
    --map-dir "$MAP_DIR" --out-dir "$OUT" --mode "$MODE" --seed "$SEED" \
    --n-layer "$N_LAYER" --n-embd "$N_EMBD" --n-head "$N_HEAD"
  python3 "$HERE/validate_dataset.py" --data-dir "$OUT"
done

echo
echo "Done. Train with, from world-model-evaluation-main/:"
echo "  python train.py --data ${TAG}-random-walks --model_name ${TAG}-random-walks \\"
echo "      --num_layers ${N_LAYER} --n_embd ${N_EMBD} --n_head ${N_HEAD} \\"
echo "      --batch_size_per_gpu 512 --eval_every 0.5 --early_stopping_patience 5"
echo
echo "The batch size matters most: the repo default of 6 is tuned for 1.5B models"
echo "on 8 A100s and gives ~85x more steps per epoch than this dataset needs."
