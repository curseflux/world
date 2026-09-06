# Grid world: a controlled stand-in for the Manhattan map

A generator for small, irregular street graphs plus the turn-by-turn datasets
that go with them, in exactly the format the Vafa et al. (2024) pipeline in
`../world-model-evaluation-main/` consumes. The point is to make the model-size
question answerable: on a 100-node map you can sweep model size against graph
size in hours instead of GPU-weeks, with ground truth you control exactly.

---

## Fresh run, every command

Paths are relative to the repository root. Steps 1 and 4 run from the root;
steps 2 and 3 run from `world-model-evaluation-main/`, because the paper's
scripts resolve `data/` and `ckpts/` relative to the working directory.

### 1. Build the map and the datasets

```bash
./gridworld/build_all.sh nyc10 10 0 1.7 12 768 12
#                        tag  size seed density n_layer n_embd n_head
```

Writes `gridworld/maps/nyc10/` (map, stats, `true_map.svg`) and three datasets
under `world-model-evaluation-main/data/nyc10-{shortest,noisy-shortest,random-walks}/`,
validating each. At density 1.7 the map has 100 intersections and 170 one-way
streets, and the random-walks arm has ~306k sequences / ~8.7M tokens.

### 2. Train

```bash
cd world-model-evaluation-main

python train.py --data nyc10-random-walks --model_name nyc10-random-walks \
    --num_layers 12 --n_embd 768 --n_head 12 \
    --batch_size_per_gpu 1024 --eval_every 0.5 --early_stopping_patience 5
```

`--model_name` must match `--data`: checkpoints go to `ckpts/<model_name>/` and
the eval scripts look in `ckpts/<data>/`. At batch 1024 that is ~299 steps per
epoch. Early stopping ends the run when `val_loss` plateaus, so `--max_epochs`
does not need tuning.

### 3. All metrics

Each writes `results/nyc10-random-walks/<script>.json` and echoes a summary.

```bash
python evaluate_traversal_capabilities.py --data nyc10-random-walks
python next_token_test.py                 --data nyc10-random-walks
python probe_test.py                      --data nyc10-random-walks --use-heldout
python compression_test.py                --data nyc10-random-walks
python distinction_test.py                --data nyc10-random-walks

for P in 0.01 0.10 0.50 0.75; do
  python detour_analysis.py --data nyc10-random-walks --detour-prob $P
done
```

An untrained baseline, the reference row of the paper's Table 1, is the same
commands with `--use-untrained-model`.

### 4. Sample from the model and rebuild its map

```bash
cd ..

python gridworld/sample_from_model.py --data nyc10-random-walks \
    --out world-model-evaluation-main/results/nyc10-random-walks/samples.txt \
    --num-sequences 6400

python gridworld/reconstruct_map.py --map-dir gridworld/maps/nyc10 \
    --samples world-model-evaluation-main/results/nyc10-random-walks/samples.txt \
    --num-sequences 6400 \
    --out-dir world-model-evaluation-main/results/nyc10-random-walks/map
```

Writes `map.svg`, `true_vs_reconstructed.svg` (the Figure 3 layout),
`reconstruction.json` and `invented_edges.json`.

Then the same run as a curve, because a single precision or Jaccard number is a
property of (model, sequence budget) rather than of the model:

```bash
python gridworld/reconstruct_map.py --map-dir gridworld/maps/nyc10 \
    --samples world-model-evaluation-main/results/nyc10-random-walks/samples.txt \
    --num-sequences 6400 --sweep \
    --out-dir world-model-evaluation-main/results/nyc10-random-walks/map
```

### 5. The matched noise control

An absolute precision number means nothing on its own. Corrupt true traversals
at the model's own error rate and reconstruct from those; a model whose map is
merely noisy scores like the control, one whose map is incoherent scores far
worse. Take the rate from `sequences_unreconstructable / sequences_used` in
step 4's `reconstruction.json`:

```bash
python gridworld/reconstruct_map.py --map-dir gridworld/maps/nyc10 \
    --corrupt 0.02 --num-sequences 6400 \
    --out-dir world-model-evaluation-main/results/nyc10-random-walks/map-control
```

### Steps 3-5 in one go

```bash
./gridworld/run_evals.sh nyc10-random-walks gridworld/maps/nyc10 [num-sequences]
```

Runs every metric, samples from the model, reconstructs both point estimate and
sweep, derives the control's error rate from the model's own, runs the control at
the same rate *and* the same budget, and prints a summary ending in the number
that matters:

```
  reconstructed edges      498 vs 170 true (ratio 2.93)
  edge jaccard             0.290
  impossible orientations  0.620
  control jaccard          0.341 (same error rate and budget)

  GAP vs control           -0.051
```

Each step is guarded: one metric failing does not stop the rest, and the
reconstruction steps gate each other so a failed sampling run cannot cascade into
a control run with no error rate to use. Failures are tallied at the end and
recorded in `console.log`. For the paper's untrained reference row, re-run the
metric scripts by hand with `--use-untrained-model`.

### Rendering maps on their own

```bash
python gridworld/render_map.py --map-dir gridworld/maps/nyc10 --out true_map.svg
```

`--show-nodes` adds a dot per intersection; `--show-unused` (on
`reconstruct_map.py`) adds true edges the reconstruction never used. Both are
off by default, matching the paper.

---

## What the numbers mean

### Why the map is irregular

A plain 10x10 lattice is **not** a graph-tracking problem. With N/S/E/W moves,

```
position = origin + (#E - #W, #N - #S)
```

so the current node is a closed-form function of the *counts* of the direction
tokens. A model can score ~100% on the next-token test, route perfectly, and
satisfy a state probe with two bounded counters, having never represented the
graph. Worse for representation work, that shortcut is inherently geometric:
activations would organise into a tidy 2-D lattice for reasons having nothing to
do with learning a map.

**One-way streets and deleted edges do not fix this.** They change which moves
are *legal*, but a legal walk still lands at `origin + net displacement`. Only
**edges whose displacement is not determined by their direction label** break
it -- an `N` that sometimes means one row and sometimes three. That is what
`--p-long` and the diagonal avenues do; one-ways and deletions are layered on for
realism and Manhattan-like out-degree.

`check_map.py` measures it as `counter_model_accuracy`: how often the best
possible two-counter model lands on the right node. **1.0 on a vanilla lattice
by construction**, ~0.15 at the defaults. Treat anything above ~0.3 as still too
close to degenerate.

### Density

`--density` is the target mean out-degree. Strong connectivity is a hard floor,
so `map_stats.json` reports what was achieved rather than returning a
disconnected graph. Measured at seed 0:

| density | edges | counter model | turn-set classes | diameter | median path |
| --- | --- | --- | --- | --- | --- |
| 2.15 (Manhattan's own) | 215 | 0.156 | 28 | 15 | 6 |
| 1.9 | 190 | 0.157 | 27 | 18 | 7 |
| **1.7 (the runbook)** | 170 | 0.151 | 21 | 20 | 9 |
| 1.5 | 150 | 0.149 | 18 | 28 | 11 |
| 1.3 | 130 | 0.120 | 17 | 31 | 12 |

The shortcut stays dead all the way down and every OD pair stays reachable, but
thinning costs on two fronts: fewer distinct legal-turn-set classes, which is the
partition Vafa et al. (2025) found models collapsing onto, and a longer diameter,
which means longer sequences.

### The maps are directed, and that is not cosmetic

Manhattan's graph is directed and the paper's figures do show direction -- just
not with arrowheads:

* Section 3.1 defines the labelling as `D: V x V -> {., N, S, E, W, NE, NW, SE, SW}`,
  a function on *ordered* pairs, with "at most one edge in each direction".
* Footnote 2 splits an intersection in two "if a turn is only valid from one
  direction" -- only meaningful for one-way streets.
* The code is directed throughout: `nx.DiGraph` in `get_all_possible_pairs.py`,
  `nx.MultiDiGraph` in `build_true_graph`, and data from
  `osmnx.graph_from_place(..., network_type="drive")`, which respects one-ways.
* The Figure 9 caption: false edges are "red with a darkening gradient
  indicating the directionality of the edge".

Making the grid undirected would delete one-way streets, collapse the DFA into a
symmetric one, and make detour re-routing trivial, since every wrong turn could
be undone.

### Drawing convention

`render.py` follows `mapping/make_maps.py` exactly:

| element | drawn as |
| --- | --- |
| true edge, used | straight, thin, black, no direction shown |
| false edge, invented | **curved**, lightsalmon -> firebrick gradient toward the target |
| true edge, never used | skipped (`continue` in `make_map`) |
| intersections | **not drawn** -- `make_map` assigns a `radius` and never uses it |

No arrowheads, and no dots: everything on the page is a road. A dot at every
intersection makes a lattice read as graph paper, which is precisely the wrong
impression, since the regular-looking lines *are* the roads.

The curve carries half the argument. Its Bezier control point leans along the
edge's own *label*, so an edge labelled NW that runs east bulges northwest before
swinging back -- which is how "impossible physical orientations" and "flyovers"
become visible rather than merely counted.

### Every metric, and what it is worth

| metric | file | paper's Manhattan values | what it tells you |
| --- | --- | --- | --- |
| `next_token_accuracy` | `next_token_test.json` | 1.00 / 1.00 / 1.00 | **Smoke test only.** Saturates for every trained model. This is the Connect-4 point of Section 2.2: a model that ignores state entirely can still be a near-perfect next-token predictor. Below ~0.95 means something is broken. |
| `probe_accuracy` | `probe_test.json` | 0.91 / 0.92 / 0.99 (0.10 untrained) | **Not a world-model measure.** Linear decodability of the current intersection. Probe accuracy is invariant under any invertible linear map of the residual stream, so it constrains nothing about how states relate to each other. The untrained baseline of 0.10 over 4,580 classes shows how much comes free from the prefix. |
| `percent_valid_traversals` | `evaluate_traversal_capabilities.json` | 0.96 - 0.99 | Capability, not coherence. Can it route at all on unseen OD pairs. |
| `compression_precision` | `compression_test.json` | 0.10 / 0.05 / 0.50 | **The headline diagnostic.** Two prefixes reaching the *same* state must accept the same continuations. This is where the paper's models fail hardest while scoring 1.00 on next-token. |
| `distinction_precision`, `distinction_recall` | `distinction_test.json` | 0.35/0.20, 0.37/0.24, 0.99/1.00 | Two prefixes reaching *different* states must have distinguishing suffixes. Random walks pass this while still failing compression, which is why both are needed. |
| `valid_traversal_rate` | `detour_analysis-p<rate>.json` | 0.99 -> 0.69 at p=0.01 (shortest paths) | **The consequence.** An incoherent map cannot re-route. The shortest-paths model loses a third of its traversals at a 1% detour rate. |
| `edge_jaccard` | `map/reconstruction.json` | (figures only) | **The "same map, no more and no less" number.** Reaches 1.0 only when the reconstructed edge set equals the true one exactly; every invented *and* every missed edge drives it down. This is the one to quote. |
| `edge_count_ratio` | `map/reconstruction.json` | (figures only) | `\|E_reconstructed\| / \|E_true\|`. Says how badly and in which direction the edge budget is blown. 1.0 is right; 2.9 means the implied city has nearly three times the streets. |
| `edge_precision`, `edge_recall`, `edge_f1` | `map/reconstruction.json` | (figures only) | The components. Recall saturates at 1.0 almost immediately, so F1 and Jaccard are both driven by precision here. |
| `impossible_orientation_rate` | `map/reconstruction.json` | (figures only) | Share of invented edges whose direction label disagrees with their actual bearing -- the paper's "physically impossible orientations", counted. |
| `counter_model_accuracy` | `maps/<tag>/check_map.json` | n/a | A property of the *map*, not the model. Must stay low or the task is degenerate. |

### Reconstruction scoring: which number to look at

Write `R` for the edge set the model's sequences imply and `T` for the true one:

```
precision = |R n T| / |R|          of the edges the model implies, how many are real
recall    = |R n T| / |T|          of the real edges, how many it reaches
F1        = harmonic mean of the two
jaccard   = |R n T| / |R u T|      how close R is to being T exactly
ratio     = |R| / |T|              how badly the edge budget is blown, and which way
```

**`edge_jaccard` is the number you want.** "Correct edges, exactly as many as the
original, no more and no less" is set equality, `R == T`, and Jaccard is 1.0 if
and only if that holds. An invented edge and a missed edge cost the same. Fed
uncorrupted sequences the reconstruction returns exactly 1.0 with
`edge_count_ratio` 1.0, which is the pipeline's self-check.

Pair it with `edge_count_ratio` for the diagnosis. Recall saturates at 1.0 almost
immediately -- a few hundred sequences reach nearly every real street -- so all
the signal is in over-generation, and the ratio names it directly: 2.9 means the
implied city has nearly three times Manhattan's streets.

**Two things must be controlled before the number means anything.**

*Budget.* Every sequence is another chance to force a new edge, so precision and
Jaccard fall monotonically with `--num-sequences` while recall rises. A single
value is a property of (model, budget), not of the model. `--sweep` reports the
curve instead. At 2% corruption on the density-1.7 map:

```
   seqs    prec  recall      F1     IoU  |E|/|E*|  invented
    200   0.677   0.988   0.804   0.672     1.459        80
    800   0.445   1.000   0.616   0.445     2.247       212
   3200   0.353   1.000   0.522   0.353     2.829       311
   6307   0.341   1.000   0.509   0.341     2.929       328
```

Invention saturates -- the marginal cost per 1,000 sequences collapses from ~70
to ~8 -- so this is a bounded map that happens to be wrong, not a model
inventing streets without limit. A curve still climbing at the right edge means
the opposite, and is the more damning result. Always compare two runs at the
same budget.

*A baseline.* Corrupting just **2%** of direction tokens in *true* traversals
already drops Jaccard to ~0.34, so a model scoring 0.4 may be doing well. The
quantity that means something is the gap against the control at the model's own
error rate:

```
edge_jaccard(model) - edge_jaccard(control at the same error rate)
```

Near zero: the model's map is no worse than random transcription noise. Well
below zero: its errors are structured -- an incoherent map rather than a noisy
one. That is the comparison Figure 3 makes with its three panels, and
`run_evals.sh` runs the control automatically.

`impossible_orientation_rate` separates the two failure modes. Random corruption
puts it near 0.9, because a randomly relabelled edge is almost never
geometrically consistent. A model inventing *geometrically sensible* streets that
simply are not real would score much lower -- a coherent map of the wrong city
rather than noise.

---

## Where results go

```
world-model-evaluation-main/results/<data>/
    next_token_test.json  probe_test.json  compression_test.json
    distinction_test.json  detour_analysis-p<rate>.json
    evaluate_traversal_capabilities.json
    console.log              everything the run printed, including failures
    samples.txt              sequences drawn from the trained model
    map/reconstruction.json  every edge metric at the full budget
    map/sweep.json           the same metrics at doubling sequence budgets
    map/map.svg              the reconstructed map
    map/true_vs_reconstructed.svg
    map/invented_edges.json  every false edge, with its label and true bearing
    map-control/             the same pair, from the matched noise control
```

Detour results are one file per rate -- `detour_analysis-p0.01.json` and so on --
since the script is invoked once per probability.

## Training settings

The repo's defaults target Manhattan: 1.5B models, 4.7B tokens, 8 A100s.

| knob | repo default | use instead | why |
| --- | --- | --- | --- |
| `--batch_size_per_gpu` | 6 | 512-1024 | 6 gives ~51k steps/epoch on 306k sequences; 1024 gives ~299 |
| `--eval_every` | 5000 | 0.5 | an int above the steps in an epoch is rejected by Lightning, so 5000 *crashes* once the batch size is sane; a fraction means a portion of an epoch |
| `--max_epochs` | 25 | leave it | set `--early_stopping_patience 5` and let `val_loss` decide |
| `--n_embd` | 768 | 768, or 256 | 12x768 is ~85M parameters against 8.7M training tokens. The runbook keeps 768 to match the paper; 256 is the better-proportioned choice, and NextLat (App. F.1) found depth helps state tracking while width does not |

`--max_epochs 25` is not why training is slow. `ModelCheckpoint` already keeps the
best `val_loss` checkpoint, so extra epochs cost only wall-clock.

## Scripts

| script | what it does |
| --- | --- |
| `build_map.py` | builds the graph; strongly connected and at most one outgoing edge per direction, both by construction |
| `check_map.py` | `counter_model_accuracy`, displacements per direction, legal-turn-set partition, path lengths |
| `generate_sequences.py` | shortest / noisy-shortest / random-walks datasets per Appendix F, split by OD pair |
| `validate_dataset.py` | every sequence legal, no OD-pair leakage, node and edge coverage |
| `sample_from_model.py` | draws traversals from a trained model, no osmnx |
| `reconstruct_map.py` | reconstructs the implied map, scores it, renders both SVGs |
| `render_map.py` | renders the true map on its own |
| `render.py` | shared renderer following the paper's figure convention |
| `build_all.sh` | steps 1 above |
| `run_evals.sh` | steps 3-5 above |

## Two traps this does not remove

**Coverage.** A 100-node graph has 9,900 OD pairs against Manhattan's ~21M. The
`shortest` arm is ~7.9k sequences covering *every* trainable pair -- a
memorisation test, not a world-model test. `generate_sequences.py` warns when it
detects this. The `random-walks` arm is the sound one, and the arm the paper's
own models did best on (compression 0.50 vs 0.10).

**A single grid size proves nothing about model size.** If a small model nails a
10x10 map, that does not establish capacity was Manhattan's problem -- state-space
size and OD coverage changed too. The informative run is the 2-D sweep: graph
size (5x5 ... 30x30) against model size, depth and width varied separately.

## Expectation setting

Vafa et al. (2025) found a 12-layer, 768-dim transformer has weak inductive bias
toward state on a **1-D lattice with five states**. These maps have 100. A
well-sized model failing here is a live possibility and the more informative
outcome -- a cleaner result than Manhattan's, because capacity is ruled out by
construction.
