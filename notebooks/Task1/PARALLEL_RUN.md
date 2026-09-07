# Task 1 worker and combine workflow

Checked 6 September 2026. The source of truth is
[01_task1_article_type.ipynb](01_task1_article_type.ipynb) and
[the layout module](../../scripts/task1_layout.py). There are **8 generated worker files**.
The seed-variance jobs have been retired along with the study itself.

## Run order

1. Install the locked environment with `uv sync --frozen` on every machine.
2. Run EDA once and copy the exact `preprocessed_datasets/train_manifest.csv`, supplied data
   and source files to every machine. Current workers use supplied data only; the prepared
   dataset1 export is not connected to their loader.
3. Run assigned workers top to bottom. Keep model/data settings consistent.
4. Return all files from each `models/task1/checkpoints/<arm>/`, including `.joblib` and `.npz`,
   to the combine machine. Retain per-machine result CSVs separately for provenance.
5. Run the combine notebook with `JOB_FILTER = None` and `RESUME = True`. It restores model
   banks, computes comparisons and inference, and exports results. Missing or incompatible
   checkpoints may trigger training. Completed HOG grid arms restore from their fingerprinted
   score/model checkpoints; only missing or incompatible arms are fitted again.

Do not copy `checkpoints_invalid/`. The eight files listed below are the only current
workers; anything else claiming to be one is from a superseded revision.

## Setting a machine up

Same on every machine. Nothing here is per-job; the job is chosen by which file you run.

| # | Step | Command or setting | Why |
|---|---|---|---|
| 1 | Clone or copy the repository | — | Source must be byte-identical across machines |
| 2 | Install the environment | `uv sync --frozen` | `--frozen` pins `uv.lock`; a resolved-fresh environment is a different environment |
| 3 | Place the supplied data | `datasets/` | Workers read only supplied data |
| 4 | Place the EDA manifest | `preprocessed_datasets/train_manifest.csv` | Copy the **one** file produced by a single EDA run; regenerating it per machine risks a different split |
| 5 | Set the arm | `ARM` in Section 1.0 | `"supplied"` for the ordinary run, `"enriched"` for the external-data comparison. Every machine in one pass must agree |
| 6 | Confirm the fingerprint | first cells print `Fingerprint: e6b15f5c51de` | That value is the **supplied** arm. The enriched arm prints a different one, which is correct — it trains on 697 more rows. Any other difference means step 3 or 4 diverged |

There is no capacity setting to choose. Task 1 sizes its CPU pools to every visible core and
does not pace the CUDA stream, so a worker takes the machine it is given.

Do **not** edit `JOB_FILTER`: each `worker_*.ipynb` already carries its own. Leave `RESUME`
at its default so an interrupted job restarts from its last epoch checkpoint instead of from
scratch. `ALLOW_CPU` stays `False` except on a CPU-only machine running `hog_svm` or
`hogsearch`, which need no GPU.

### Running a worker without opening Jupyter

```bash
uv run jupyter execute --timeout=-1 --output="run_{notebook_name}" notebooks/Task1/worker_cnn.ipynb
```

`--timeout=-1` disables the per-cell timeout, which a 115-minute grid cell otherwise trips.
`--output` writes the executed copy to `run_worker_cnn.ipynb` and leaves the tracked worker
file clean, so `check_task1_workers.py` still passes afterwards. Checkpoints are written as a
side effect either way; the executed copy is your provenance record of the printed scores.

## What each job needs from the machine

| Job | Device | Trains | GPU memory | Can share a machine with |
|---|---|---|---|---|
| `hog_svm` | **CPU only** | liblinear SVM | none | any GPU job |
| `hogsearch` | **CPU only** | 6 liblinear fits | none | any GPU job |
| `cnn` | GPU | PlainCNN, 40 epochs | ~0.6 GB + model | a CPU job |
| `cnnsearch` | GPU | 6 × PlainCNN, 12 epochs | ~0.6 GB + model | a CPU job |
| `resnet` | GPU | SmallResNet, 40 epochs + head | ~0.6 GB + model | a CPU job |
| `lrsearch` | GPU | 6 × SmallResNet, 12 epochs | ~0.6 GB + model | a CPU job |
| `sweep` | GPU | 5 heads on a frozen backbone | ~0.6 GB + model | a CPU job |
| `stage2grid` | GPU | 6 heads on a frozen backbone | ~0.6 GB + model | a CPU job |

The GPU memory figure is the fixed part: `BatchStream` holds the whole split on the device as
uint8, which is 436 MB for the 30,278 training rows and 109 MB for the 7,568 validation rows at
60x80x3. Model, optimiser and activations sit on top; measure the total with `nvidia-smi`
before packing a card.

**Concurrency on one GPU does not create throughput.** The total GPU work is fixed, so two
training jobs on one card finish in roughly the time they would take back to back. The real
wins are (a) separate machines and (b) overlapping the two CPU-only HOG jobs with a GPU job,
which is close to free. Two GPU jobs on one card is not an error, it just does not buy
anything: give each its own machine or run them back to back.

## Current jobs

| Worker suffix / job | Purpose | Estimate | Required checkpoint first |
|---|---|---|---|
| `hog_svm` | HOG + linear SVM baseline | ~3 min | None |
| `cnn` | PlainCNN seed 42 | ~9 min | None |
| `resnet` | Stage-1 ResNet and decoupled head, seed 42 | ~64 min | None |
| `sweep` | Five stage-2 sampler strengths | ~23 min | `model_resnet_stage1.pt` |
| `hogsearch` | Six HOG C × class-weight arms | ~16 min | None |
| `cnnsearch` | Six CNN LR × weight-decay arms, 12 epochs each | ~17 min | None |
| `lrsearch` | Six ResNet LR × weight-decay arms, 12 epochs each | ~115 min | None |
| `stage2grid` | Six stage-2 LR × sampler-strength arms | ~28 min | `model_resnet_stage1.pt` |

The estimates are `RUNTIMES` in [the layout module](../../scripts/task1_layout.py) and are
derived from per-epoch costs measured on Apple Silicon MPS at fp32: 14 s per CNN epoch, 96 s
per stage-1 ResNet epoch, 28 s per stage-2 head epoch and 159 s per liblinear fit. The run
that produced them is no longer kept in this repository, so treat them as an order of
magnitude and scale them to your own hardware before scheduling.
The eight jobs total ~275 minutes.

Each filename is `worker_<suffix>.ipynb`. `logit_adjusted` and `phase2` are historical jobs,
not current job names. Current grids and their result paths are in the
[pipeline guide](../../docs/SUGGESTED_PIPELINE.md).

Run `resnet` before `sweep` and `stage2grid`, or copy its trained stage-1 checkpoint before
starting either. They train heads only and assert that a trained backbone exists.

## Schedules by machine count

The supplied arm is eight jobs totalling ~275 minutes. Adding the enriched arm of the
Section 11 experiment brings **~351 minutes** in total, because that arm repeats only the
three jobs that produce a model: `hog_svm`, `cnn` and `resnet`.

Three floors bound any schedule, and the last two are what actually bind:

- the total divided by the machine count;
- **`lrsearch` at ~115 minutes**, which no amount of hardware splits;
- **`resnet` + `sweep` + `stage2grid` at ~115 minutes**, which stay on one machine so the
  stage-1 checkpoint dependency is satisfied locally with nothing to copy mid-run.

| Machines | Wall clock | Arms covered | Note |
|---:|---:|---|---|
| 1 | ~256 min | supplied only | Overlapping the CPU-only HOG jobs with GPU work saves 19 minutes |
| 2 | ~132 min | supplied only | Within four minutes of the floor; the two 115-minute blocks cannot be split |
| 3 | ~115 min | supplied only | Optimal for one arm |
| **4** | **~115 min** | **both** | **The fourth machine runs the entire enriched arm for free** |

**This is what changes with four machines.** Against one arm a fourth machine bought nothing,
because `lrsearch` alone was already the make-span. The enriched arm is 76 minutes of work
that depends on none of it, so it drops onto a fourth machine inside the same 115-minute
window. Four machines give you the whole experiment in the time three machines needed for
half of it.

### Two machines — 132 min (supplied arm only)

| Machine | `ARM` | Jobs, in order | Total |
|---|---|---|---:|
| A | supplied | `lrsearch`, `cnnsearch`, and `hogsearch` + `hog_svm` alongside on the CPU | 132 min |
| B | supplied | `resnet`, `sweep`, `stage2grid`, `cnn` | 124 min |

### Three machines — 115 min (supplied arm only)

| Machine | `ARM` | Jobs, in order | Total |
|---|---|---|---:|
| A | supplied | `lrsearch` | 115 min |
| B | supplied | `resnet`, `sweep`, `stage2grid` | 115 min |
| C | supplied | `cnnsearch`, `cnn`, `hogsearch`, `hog_svm` | 45 min |

### Four machines — 115 min, both arms (recommended)

| Machine | `ARM` | Jobs, in order | Total |
|---|---|---|---:|
| A | `"supplied"` | `lrsearch` | 115 min |
| B | `"supplied"` | `resnet`, `sweep`, `stage2grid` | 115 min |
| C | `"supplied"` | `cnnsearch`, `cnn`, `hogsearch`, `hog_svm` | 45 min |
| D | `"enriched"` | `resnet`, `cnn`, `hog_svm` | 76 min |

The split is deliberately clean along the arm boundary: **machine D is the entire enriched
arm and no machine changes `ARM` part-way through a pass.** Machine D writes to
`models/task1/checkpoints/enriched/` and cannot collide with the others, which write to
`.../supplied/`.

Machine D needs `preprocessed_datasets/task1_dataset1_arms/` present before it starts. Produce
it once with `python scripts/make_dataset1_arms.py` and copy it alongside the manifest at
setup step 4, or regenerate it there — it is seeded, so both give the same split.

Machine C finishes first and is the natural place to run the combine notebook from.

### Combining, with two arms

Run the combine notebook **twice**, once per arm, after the checkpoints are back:

1. `ARM = "supplied"`, `JOB_FILTER = None`. Scores the supplied arm and writes
   `models/task1/arm_results_supplied.json`.
2. `ARM = "enriched"`, `JOB_FILTER = None`. Same for the enriched arm, then prints the
   Section 11 comparison table because both files now exist.

The second pass automatically skips the four Section 7.4 grids — the notebook forces them off
for any arm but the supplied one, so the enriched arm reuses the supplied arm's tuning rather
than retuning against its own data. Set `RUN_SAMPLER_SWEEP = False` on that pass too if you do
not want the 23-minute sampler sweep repeated; it is an analysis of where Section 6's gain
comes from, not an input to the experiment.

The estimates come from a run on Apple Silicon MPS at fp32 that is no longer kept here.
Measure actual runtimes on your own hardware and re-balance if the ratios between jobs move.

## The second arm: the external-data comparison

Section 11 of the combine notebook asks whether adding 697 external crops to training helps.
It needs the same models trained twice, once per arm, and the arms differ only in `ARM`.

| Arm | `ARM` | Jobs to run | Time |
|---|---|---|---:|
| supplied | `"supplied"` | all eight, the ordinary pass | ~275 min |
| enriched | `"enriched"` | `hog_svm`, `cnn`, `resnet` only | **~76 min** |

The enriched arm re-runs only the three jobs that produce a *model*. The four tuning grids and
the sampler sweep are not repeated, and must not be: hyper-parameters are tuned once on the
supplied arm and applied to both, because retuning per arm would move the data and the recipe
together and leave the difference unattributable.

On two machines the enriched arm is ~64 minutes — `resnet` alone on one, `hog_svm` and `cnn`
together on the other.

Prerequisite: run `python scripts/make_dataset1_arms.py` once, on any machine, and copy
`preprocessed_datasets/task1_dataset1_arms/` alongside the manifest at setup step 4. It splits
dataset1 into the 697 crops the enriched arm trains on and the 461 that neither arm trains on
and both are scored against. Every machine in a pass must hold the same split; regenerating it
per machine is safe (it is seeded) but copying is one less thing to verify.

Whichever arm runs second prints the comparison table. Neither arm can overwrite the other.

## What comes back from each machine

Copy the whole of `models/task1/checkpoints/<arm>/` from every worker machine into the combine
machine's `models/task1/checkpoints/<arm>/`, preserving filenames and the arm directory. The
two arms never mix: they write to separate directories and carry different fingerprints, so a
checkpoint copied into the wrong one is refused rather than silently used. Files are keyed by
name, so
copies from different machines merge into one directory without collisions.

| Job | Files it banks | Prefix |
|---|---|---|
| `hog_svm` | 2 | `model_hog_svm.joblib`, `model_hog_svm_scores.npz` |
| `cnn` | 1 | `model_cnn.pt` |
| `resnet` | 2 | `model_resnet_stage1.pt`, `model_resnet_decoupled.pt` |
| `sweep` | 5 | `model_stage2_power_0p0.pt` … `model_stage2_power_1p0.pt` |
| `hogsearch` | 12 | `model_search_hog_c*_weight_*.joblib` and `_scores.npz` |
| `cnnsearch` | 6 | `model_search_cnnsearch_lr*_wd*.pt` |
| `lrsearch` | 6 | `model_search_lrsearch_lr*_wd*.pt` |
| `stage2grid` | 6 | `model_stage2_grid_p*_lr*.pt` |

`epoch_*.pt` files are resume state, not results. They are safe to leave behind once the
matching `model_*.pt` exists. The checkpoint the deployed model is exported from is
`model_resnet_decoupled.pt`, written by the `resnet` job; the combine run fails the export if
it is missing.

Keep each machine's executed `run_worker_*.ipynb` as the provenance record of what it printed.

## Configuration and identity

Full runs default to `ALLOW_CPU = False`. A CPU-only HOG worker must explicitly allow CPU.
`QUICK_RUN = True` enables a reduced structural run and changes checkpoint identity; its
results are not full-run results. The source currently enables CUDA AMP, whereas the saved
historical config records AMP off. Record performance settings when comparing results.

Section 1.0 sizes the CPU pools and the HOG process pool to every visible core, read from
`sched_getaffinity` where it exists so a container or `taskset` run gets the cores it was
actually given. The CUDA stream is unpaced and GPU memory is uncapped, because a hard
fractional allocator cap can OOM a valid model. None of it changes `RUN_FINGERPRINT`.

The recorded supplied-only fingerprint is `e6b15f5c51de`. It covers selected hyperparameters,
row/class counts and normalization summaries. It does not hash every pixel, class ordering
or runtime option, so also distribute identical source and data. Runtime controls such as
`ALLOW_CPU`, `RESUME`, `USE_COMPILE` and `DETERMINISTIC` are outside that fingerprint;
some affect numerical reproducibility even though checkpoints remain loadable.

## Editing and validation

Edit the combine notebook, then regenerate workers from the project root:

```bash
uv run python scripts/make_task1_workers.py
uv run python scripts/check_task1_workers.py --strict
uv run python scripts/make_task1_workers.py --check
```

Shared worker cells must match the combine source byte-for-byte. `--strict` checks names,
anchors, composition and the recorded fingerprint; it requires `pyflakes`, installed by the
default dev dependency group. `--check` verifies generator output without modifying files.
Both checks pass for the source snapshot reviewed on 6 September 2026.

`uv run python scripts/make_task1_workers.py --legacy` compares the generator against a pinned
historical Git revision, `cd44bc8f16c5`, and reproduces the four hand-derived workers that
are still jobs there. `--strict` now runs it as check [6]; run it standalone to see the diff when it
fails. It needs repository history and is skipped, not failed, in a snapshot without `.git`.

## Troubleshooting

| Symptom | Check |
|---|---|
| Different fingerprint or unexpected retraining | Compare manifest, source constants, normalization and copied bank files |
| Sweep/grid cannot find a stage-1 backbone | Supply `model_resnet_stage1.pt` from the matching ResNet job |
| Backbone has `num_batches_tracked=1` | It is the untrained shape-check probe; use the trained checkpoint |
| Unknown job at startup | Use current job names above, not historical phase2/logit-adjusted/seed names |

Final outputs are under `models/task1/` (model export, classes, config and result CSVs),
`predictions/` (`task1_predictions.csv` and `task1_test_logits.npy`) and `outputs/figures/`
(report figures). The prediction CSV still needs the other tasks' columns. See
[artifact handover](../../artifacts/README.md).

## The independent evaluation notebook

[02_independent_evaluation.ipynb](02_independent_evaluation.ipynb) is not a worker and is not
generated. It runs on the combine machine after the combine notebook has exported
`models/task1/task1_model.pt`, scoring that exact model against the external cosmetics
collections in `dataset1/` and `dataset2/`. It trains nothing, takes a few minutes, and writes
figures 10 and 11 into `outputs/figures/`. Section 8.7 of the combine notebook quotes its
result, so run it before the report is finalised.
