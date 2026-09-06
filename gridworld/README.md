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
./build_all.sh nyc10 10 0 2.15 12 256 8   # tag, size, seed, density, n_layer, n_embd, n_head
```

`density` is the target mean out-degree. The default 2.15 is Manhattan's own
(9,846 edges over 4,580 intersections). Strong connectivity is a hard floor --
every node needs an incoming and an outgoing edge -- so a low enough target
cannot be met; `map_stats.json` reports the density actually achieved rather
than silently returning a disconnected graph.

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

## The maps are directed, and that is not cosmetic

Manhattan's graph is directed, and the paper's figures do show direction -- just
not with arrowheads. Four independent confirmations:

* Section 3.1 defines the labelling as `D: V x V -> {., N, S, E, W, NE, NW, SE, SW}`,
  a function on *ordered* pairs, and states "each intersection has at most one
  edge in each direction".
* Footnote 2: "if a turn is only valid from one direction, it is represented as
  two different nodes" -- a note that only makes sense for one-way streets.
* The code is directed throughout: `get_all_possible_pairs.py` builds an
  `nx.DiGraph`, `make_graphs.build_true_graph` an `nx.MultiDiGraph`, and the data
  comes from `osmnx.graph_from_place(..., network_type="drive")`, which respects
  one-way streets.
* The Figure 9 caption: false edges are "red with a darkening gradient
  indicating the directionality of the edge".

So direction stays. Making the grid undirected would delete one-way streets --
a defining feature of Manhattan -- collapse the DFA into a symmetric one, and
make detour re-routing trivial, since every wrong turn could simply be undone.

What was worth fixing is the *drawing*. Arrowheads on a few hundred edges bury
the signal, and the paper does not use them. `render.py` now follows
`mapping/make_maps.py` exactly:

| edge | drawn as |
| --- | --- |
| true, used by the reconstruction | straight, thin, black, no direction shown |
| false, invented by the reconstruction | **curved**, with a lightsalmon -> firebrick gradient running source to target |
| true, never used | skipped (`continue` in the paper's `make_map`) -- pass `--show-unused` to draw them |

The curve carries the other half of the paper's argument. Its Bezier control
point leans in the direction of the edge's own *label*, so an edge labelled NW
that actually runs east bulges northwest before swinging back. That is how
"streets with impossible physical orientations" and "flyovers above other
streets" become visible instead of merely counted.

## Rendering the maps

```bash
# the true map on its own
python render_map.py --map-dir maps/nyc10 --out maps/nyc10/true_map.svg

# reconstructed map, plus a Figure 3-style true-vs-reconstructed pair
python reconstruct_map.py --map-dir maps/nyc10 \
    --samples ../world-model-evaluation-main/results/<data>/samples.txt \
    --out-dir ../world-model-evaluation-main/results/<data>/map

# same thing for the noise control, no trained model needed
python reconstruct_map.py --map-dir maps/nyc10 --corrupt 0.02 --out-dir /tmp/control
```

`reconstruct_map.py` writes `map.svg` and `true_vs_reconstructed.svg` side by
side; `build_all.sh` renders the true map automatically.

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
| `reconstruct_map.py` | reconstructs the implied map, scores edge precision/recall, renders both SVGs |
| `render_map.py` | renders the true map on its own |
| `render.py` | shared renderer, following the paper's figure convention |
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
