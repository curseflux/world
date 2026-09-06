# Grid world: a controlled stand-in for the Manhattan map

A generator for small, irregular street graphs plus the turn-by-turn sequence
datasets that go with them, in exactly the format the Vafa et al. (2024)
pipeline in `../world-model-evaluation-main/` consumes. The point is to make the
model-size question answerable: on a 100-node map you can sweep model size and
graph size against each other in hours instead of GPU-weeks, with ground truth
you control exactly.

## Why the map is irregular

A plain 10x10 lattice is **not** a graph-tracking problem. With N/S/E/W moves,

```
position = origin + (#E - #W, #N - #S)
```

so the current node is a closed-form function of the *counts* of the direction
tokens seen so far. A model can score ~100% on the next-token test, route
perfectly, and satisfy a state probe using two bounded counters, having never
represented the graph at all.

Two things follow, and the second is the one that bites hardest:

1. You would measure a shortcut Manhattan does not admit, and read the result as
   "model size was the problem."
2. The counter solution is **inherently geometric**. Activations would organise
   into a tidy 2-D lattice, so a representation-geometry experiment would come
   back positive for reasons having nothing to do with the model learning a map.

**One-way streets and deleted edges do not fix this.** They change which moves
are *legal*, but a legal walk still lands at `origin + net displacement`. The
only thing that breaks the shortcut is **edges whose displacement is not
determined by their direction label** — an `N` that sometimes means one row and
sometimes three. That is what `--p-long` and the diagonal avenues do; one-ways
and deletions are layered on for realism (they make legality state-dependent and
bring the out-degree near Manhattan's 2.15).

`check_map.py` measures this directly as `counter_model_accuracy`: how often the
best possible two-counter model lands on the right node. It is **1.0 on a
vanilla lattice by construction**, and roughly **0.14** at the defaults. Treat
anything above ~0.3 as a map that is still too close to degenerate.

## Usage

```bash
./build_all.sh nyc10 10 0 12 768 12      # tag, size, seed, n_layer, n_embd, n_head
```

That writes the map to `maps/<tag>/` and three datasets to
`../world-model-evaluation-main/data/<tag>-{shortest,noisy-shortest,random-walks}/`,
validating each. Then, from `world-model-evaluation-main/`:

```bash
python train.py --data nyc10-random-walks --model_name nyc10-random-walks \
    --num_layers 12 --n_embd 256 --n_head 8 \
    --batch_size_per_gpu 512 --eval_every 0.5 --early_stopping_patience 5
python next_token_test.py  --data nyc10-random-walks
python probe_test.py       --data nyc10-random-walks --use-heldout
python compression_test.py --data nyc10-random-walks
python distinction_test.py --data nyc10-random-walks
python detour_analysis.py  --data nyc10-random-walks
```

Model size is read per-dataset from `data/<name>/model_config.json`, so the same
map can be trained at several sizes by generating the dataset once per
configuration (or by editing that file).

## Training settings

The repo's defaults are tuned for Manhattan: 1.5B-parameter models, 4.7B tokens,
8 A100s. They are badly wrong for a 100-node map, and the batch size is the one
that hurts.

| knob | repo default | use instead | why |
| --- | --- | --- | --- |
| `--batch_size_per_gpu` | 6 | 256-512 | 6 gives ~62k steps/epoch on 370k random-walk sequences; 512 gives ~720. This is the ~85x, not the epoch count. |
| `--n_embd` | 768 | 256 | 12x768 is ~85M params against 9.5M training tokens. Cut width, keep depth. |
| `--eval_every` | 5000 | 0.5 | An int above the steps in an epoch is rejected by Lightning, so 5000 *crashes* once the batch size is sane. A fraction means "portion of an epoch". |
| `--max_epochs` | 25 | leave it | Set `--early_stopping_patience 5` instead and let val_loss decide. |

`--max_epochs 25` is not the reason training is slow, and lowering it is not the
fix. `ModelCheckpoint` already monitors `val_loss` with `save_top_k=1`, so extra
epochs never worsen the checkpoint you keep -- they only cost wall-clock. Early
stopping turns the question into one you do not have to answer in advance.

Sequences here are at most 43 tokens against GPT-2's 1024-position default, and
the vocabulary is ~110 tokens, so large batches are cheap.

## Where results go

The metric scripts originally reported only through `tqdm.set_description`, so
every number lived in a progress-bar line on stderr and vanished when the
terminal scrolled. They now also write JSON:

```
world-model-evaluation-main/results/<data>/
    next_token_test.json  probe_test.json  compression_test.json
    distinction_test.json  detour_analysis.json
    evaluate_traversal_capabilities.json
    samples.txt              sequences drawn from the trained model
    map/reconstruction.json  edge precision and recall
    map/map.svg              the reconstructed map
    map/invented_edges.json  every false edge, listed
    map-control/             the same, from the matched noise control
```

Run the lot with `./run_evals.sh <data-name> <map-dir>`.

## Reconstructing the map

The paper's `mapping/` pipeline is Manhattan-only -- it downloads the street
graph from OpenStreetMap through osmnx and renders with folium onto real
lat/long -- so it cannot draw a synthetic grid. `sample_from_model.py` and
`reconstruct_map.py` replace it, reusing the one generic piece,
`mapping/reconstruction.reconstruct_sequence`, with a Euclidean neighbourhood
and plain SVG output.

The payoff of a synthetic map is that scoring stops being visual. Where the
paper inspects maps by eye for "impossible orientations and flyovers", 100 nodes
can be scored exactly:

```
precision = true_used / (true_used + invented)     of the edges the model implies, how many are real
recall    = true_used / (true_used + never_used)   of the real edges, how many it reaches
```

Two things to hold fixed before comparing runs. Precision falls as more
sequences are reconstructed, since each one is another chance to invent an edge,
while recall rises -- so `--num-sequences` must match. And an absolute precision
number means little on its own: `--corrupt` reproduces the paper's Figure 3
control by corrupting a fraction of direction tokens in *true* traversals, and
`run_evals.sh` runs it automatically at the model's own error rate. A model whose
map is merely noisy scores like the control; one whose map is incoherent scores
far worse. Fed uncorrupted sequences the reconstruction returns precision and
recall of exactly 1.0, which is the pipeline's self-check.

## Scripts

| script | what it does |
| --- | --- |
| `build_map.py` | builds the graph; strongly connected over all `size^2` nodes and at most one outgoing edge per direction, both by construction |
| `check_map.py` | `counter_model_accuracy`, displacements per direction, legal-turn-set partition, path-length distribution |
| `generate_sequences.py` | shortest / noisy-shortest / random-walks datasets per Appendix F, split by OD pair |
| `validate_dataset.py` | every sequence legal, no OD-pair leakage, node and edge coverage |
| `sample_from_model.py` | draws traversals from a trained model, no osmnx |
| `reconstruct_map.py` | reconstructs the implied map, scores edge precision/recall, renders SVG |
| `run_evals.sh` | runs every metric plus reconstruction and the matched control |

## Two traps this does not remove

**Coverage.** A 100-node graph has only 9,900 OD pairs, versus ~21M for
Manhattan. The `shortest` arm is therefore about 7.9k sequences covering *every*
trainable pair — a memorisation test, not a world-model test.
`generate_sequences.py` prints a warning when it detects this. The
`random-walks` arm is the sound one, and it is also the arm on which the paper's
own models did best (compression 0.50 vs 0.10).

**A single grid size proves nothing about size.** If a small model nails a 10x10
map, that does not establish that capacity was Manhattan's problem — state-space
size and OD coverage both changed too. The informative run is the 2-D sweep:
graph size (5x5 ... 30x30) against model size, with depth and width varied
separately, since NextLat (App. F.1) reports depth helping substantially and
width negligibly.

## Expectation setting

Vafa et al. (2025) found a 12-layer, 768-dim transformer has weak inductive bias
toward state on a **1-D lattice with five states**. These maps have 100. A
well-sized model failing here is a live possibility and the more informative
outcome — it would be a much cleaner result than Manhattan's, because capacity is
ruled out by construction.
