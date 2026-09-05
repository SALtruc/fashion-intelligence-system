# Task 1 — Running the Parallel Setup

Task 1 is about four hours of training on one CUDA GPU. This directory splits that into
independent jobs that run on separate machines and recombine exactly, bringing the wall
clock down to under an hour on five machines.

## What is in this directory

| File | Role |
|---|---|
| `01_task1_article_type.ipynb` | The **combine notebook**. Correct on its own on a bare machine, and also the notebook you run last to assemble a parallel run. |
| `worker_*.ipynb` | Ten **worker notebooks**, one job each. `JOB_FILTER` is already set in every one; there is nothing to configure. |

A worker is the combine notebook with the other jobs' cells removed and three cells
substituted: its header, its `JOB_FILTER` cell, and a closing summary. Every other cell is
byte-identical to the combine notebook. That is what makes all machines agree on
`RUN_FINGERPRINT`, and therefore what makes a checkpoint trained on one machine acceptable
on another.

Do not hand-edit a worker. Edit `01_task1_article_type.ipynb` and regenerate:

```
python scripts/make_task1_workers.py
```

## The short version

1. One machine runs `notebooks/00_eda_and_preprocessing.ipynb`.
2. Copy `preprocessed_datasets/` from that machine to every other machine.
3. Give each machine its worker notebooks from the plan below. Run All, top to bottom.
4. Copy every machine's `models/task1/checkpoints/` back to one machine.
5. That machine runs `01_task1_article_type.ipynb`.

## How many machines

The nine jobs total about 243 minutes of work. The longest single job, `lrsearch` at
~56 min, cannot be split, so five machines already reach the floor and a sixth buys nothing.

| Machines | Wall clock | Notes |
|---|---|---|
| 1 | ~4 h | Do not use the workers at all. Just run the combine notebook end to end. |
| 2 | ~2 h | |
| 3 | ~1 h 25 min | |
| **5** | **~56 min** | **Recommended.** Hits the lower bound set by `lrsearch`. |
| 6+ | ~56 min | No further gain. |

## The plan (5 machines)

| Machine | Notebooks, in this order | Wall clock |
|---|---|---|
| A | `worker_lrsearch` | ~56 min |
| B | `worker_resnet`, then `worker_sweep` | ~49 min |
| C | `worker_seeds_2024`, then `worker_cnn` | ~45 min |
| D | `worker_logit_adjusted`, then `worker_hog_svm` | ~39 min |
| E | `worker_seeds_1337`, then `worker_phase2` | ~54 min |

**Only one ordering in that table is a real constraint: `resnet` before `sweep` on machine
B.** The sweep trains no backbone — it attaches five stage-2 heads to
`model_resnet_stage1.pt` — so it stops on an assertion without that file. Pairing the two
onto one machine means nothing has to be copied between them mid-run. Everything else in
the table is a load-balancing choice; reorder or reshuffle it freely.

Machines finish at different times. There is nothing to synchronise until step 3.

### Variation: one machine for the whole seed study

`worker_seeds.ipynb` runs all three seeds instead of the two split notebooks. Before
starting it, copy `model_cnn.pt` and `model_resnet_decoupled.pt` onto that machine. With
them, seed 42 reuses those validation logits exactly as the combine machine does, and the
job takes ~64 min. Without them it retrains seed 42 under its own keys — about 49 minutes
of work the combine machine then discards, because it reuses the Section 6 logits for that
row. The run is correct either way, but that is what makes the unsplit seed study a
~113 min job and the critical path, which is why the split exists.

### Variation: a machine without a GPU

`ALLOW_CPU = False` by default and the notebook refuses to start on a CPU rather than
appearing to hang. `worker_hog_svm` is HOG plus liblinear and never touches the GPU, so it
is the one job worth handing to a machine without one: set `ALLOW_CPU = True` in
Section 1.1 there. `ALLOW_CPU` is absent from `RUN_FINGERPRINT`, so its checkpoints are
still accepted everywhere.

## Step 1 — the manifest, once

Run `notebooks/00_eda_and_preprocessing.ipynb` on one machine and copy the resulting
`preprocessed_datasets/` directory to every machine that will run a notebook, including the
combine machine.

**Do not regenerate it per machine.** The split and the normalisation constants come from
those files and feed `RUN_FINGERPRINT`. Regenerating risks a different split, and a machine
with a different split produces checkpoints every other machine refuses.

## Step 2 — run the workers

Open the assigned notebook and Run All. There is nothing to set. Each one ends with a
`Worker complete: <job>` cell listing the files it produced and their total size.

Every notebook prints its fingerprint near the top:

```
Run fingerprint: e6b15f5c51de -- every machine in a parallel run must agree.
```

If a machine prints anything else, stop and fix it before spending an hour there. Its
output will be refused.

## Step 3 — copy the checkpoints back

Copy the contents of each machine's `models/task1/checkpoints/` into the combine machine's
`models/task1/checkpoints/`.

- No two jobs write the same filename, so the copies merge without collisions.
- Include `.joblib` and `.npz`, not only `.pt`. `model_hog_svm_scores.npz` is what the SVM
  restore actually reads; without it the combine machine refits liblinear over 124
  one-against-rest problems, the longest CPU-bound block in the notebook.
- `epoch_*.pt` files are mid-run resume state. Copying them is harmless; skipping them
  saves transfer.

## Step 4 — the combine run

Run `01_task1_article_type.ipynb` on the gathering machine. `JOB_FILTER = None` is already
the default, which is combine mode.

Every job with a checkpoint restores in seconds. Anything missing is simply trained there
instead, so a worker that failed or never ran costs time, not correctness. It produces:

- `outputs/task1_predictions.csv` — the `articleType` prediction for every prediction row
- `models/task1/` — the selected model's weights, class index, normalisation constants, run config
- `models/task1/task1_results.csv` and `task1_seed_study.csv`
- `outputs/figures/`

## Rules that must hold

1. **Copy `preprocessed_datasets/`; never regenerate it per machine.**
2. **Do not edit Section 1.1.** Every constant there is hashed into `RUN_FINGERPRINT`.
   Change one and every checkpoint from every other machine is refused.
3. **Never set `QUICK_RUN = True` on a machine whose work you intend to keep.** It changes
   the epoch count, the seed list and the row count, so it changes the fingerprint and
   orphans that machine's output. It is for a structural smoke test only.
4. **Every machine runs at full capacity, and there is no knob for that.** The resource
   budget and the duty-cycle throttle have been removed: thread pools take every visible
   core, VRAM is uncapped, and compute is never paced. `ALLOW_CPU`, `RESUME`, `USE_COMPILE`
   and `DETERMINISTIC` remain outside the fingerprint, so set those per machine as you like;
   machines with different core counts share checkpoints freely.

## Job reference

| Job | Runtime | Needs first | Banks |
|---|---|---|---|
| `hog_svm` | ~4 min | — | `model_hog_svm.joblib`, `model_hog_svm_scores.npz` |
| `cnn` | ~7 min | — | `model_cnn.pt` |
| `resnet` | ~41 min | — | `model_resnet_stage1.pt`, `model_resnet_decoupled.pt` |
| `logit_adjusted` | ~35 min | — | `model_resnet_logit_adjusted.pt` |
| `phase2` | ~27 min | — | `model_phase2_backbone_<tag>.pt`, `model_phase2_decoupled_<tag>.pt` |
| `sweep` | ~8 min | `model_resnet_stage1.pt` (hard) | five `model_stage2_power_*.pt` |
| `lrsearch` | ~56 min | — | six `model_lrsearch_*.pt` (2 architectures x 3 rates) |
| `seeds_1337` | ~27 min | — | `model_seed_cnn_1337.pt` and the two ResNet seed files |
| `seeds_2024` | ~38 min | — | as above, for seed 2024 |
| `seeds` | ~64 min | `model_cnn.pt`, `model_resnet_decoupled.pt` (soft) | all six of the above; replaces the two split jobs |

Runtimes are the recorded run's own measured figures. That run was throttled to a 65% duty
cycle — the throttle has since been removed and every machine now runs flat out — and the same
work varied by up to 2x within it from contention. Treat them as a reliable ranking and the
right order of magnitude rather than precise numbers, and expect an unthrottled machine to
come in under them.

## Verifying the setup

Three gates, all of which should exit 0. Run them from the repository root, in the project
environment (`uv sync` first: `--strict` fails if pyflakes is missing rather than quietly
dropping the undefined-name check):

```
python scripts/check_task1_workers.py --strict    # the notebooks as they stand
python scripts/make_task1_workers.py --check      # workers match what the generator produces
python scripts/make_task1_workers.py --legacy     # the derivation rule is the real one
```

The first checks that every worker cell is byte-identical to the combine notebook, that no
worker refers to a name defined only in a cell it dropped, and that `RUN_FINGERPRINT` still
hashes to `e6b15f5c51de`. Run it after any edit to the combine notebook, before
distributing anything.

## Troubleshooting

| Symptom | Cause |
|---|---|
| A machine prints a fingerprint other than `e6b15f5c51de` | Its `preprocessed_datasets/` differs, or Section 1.1 was edited. Its output will be refused. |
| `assert source_backbone is not None` in `worker_sweep` | `model_resnet_stage1.pt` is not in `models/task1/checkpoints/` on that machine. |
| `worker_seeds` running far past ~64 min | `model_cnn.pt` and `model_resnet_decoupled.pt` were not copied in, so it is retraining seed 42. |
| `unknown job(s) [...]` on startup | `JOB_FILTER` was edited to a name not in `KNOWN_JOBS`. The run refuses to start rather than training nothing for an hour. |
| The combine run retrains something a worker already did | That checkpoint did not arrive, or arrived with a different fingerprint. Check the copy included `.joblib` and `.npz`. |
