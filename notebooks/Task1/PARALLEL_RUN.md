# Task 1 worker and combine workflow

Checked 6 September 2026. The source of truth is
[01_task1_article_type.ipynb](01_task1_article_type.ipynb) and
[the layout module](../../scripts/task1_layout.py). There are **11 generated worker files**.
The all-seed worker is an alternative to the two seed-specific workers, so do not schedule
all three unless deliberately repeating work.

## Run order

1. Install the locked environment with `uv sync --frozen` on every machine.
2. Run EDA once and copy the exact `preprocessed_datasets/train_manifest.csv`, supplied data
   and source files to every machine. Current workers use supplied data only; the prepared
   dataset1 export is not connected to their loader.
3. Run assigned workers top to bottom. Keep model/data settings consistent.
4. Return all files from each `models/task1/checkpoints/`, including `.joblib` and `.npz`,
   to the combine machine. Retain per-machine result CSVs separately for provenance.
5. Run the combine notebook with `JOB_FILTER = None` and `RESUME = True`. It restores model
   banks, computes comparisons and inference, and exports results. Missing or incompatible
   checkpoints may trigger training. Completed HOG grid arms restore from their fingerprinted
   score/model checkpoints; only missing or incompatible arms are fitted again.

Do not copy `checkpoints_invalid/`. Do not use notebooks under `executed_run_2026-09-06/`
as current workers: they retain historical jobs and outputs.

## Setting a machine up

Same on every machine. Nothing here is per-job; the job is chosen by which file you run.

| # | Step | Command or setting | Why |
|---|---|---|---|
| 1 | Clone or copy the repository | — | Source must be byte-identical across machines |
| 2 | Install the environment | `uv sync --frozen` | `--frozen` pins `uv.lock`; a resolved-fresh environment is a different environment |
| 3 | Place the supplied data | `datasets/` | Workers read only supplied data |
| 4 | Place the EDA manifest | `preprocessed_datasets/train_manifest.csv` | Copy the **one** file produced by a single EDA run; regenerating it per machine risks a different split |
| 5 | Set the capacity knob | `hardware` in Section 1.0 of the worker | See the co-tenancy table below |
| 6 | Confirm the fingerprint | first cells print `Fingerprint: e6b15f5c51de` | A different value means step 3 or 4 diverged — stop and fix it before training |

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
| `seeds_1337` | GPU | CNN + ResNet at one seed | ~0.6 GB + model | a CPU job |
| `seeds_2024` | GPU | CNN + ResNet at one seed | ~0.6 GB + model | a CPU job |
| `sweep` | GPU | 5 heads on a frozen backbone | ~0.6 GB + model | a CPU job |
| `stage2grid` | GPU | 6 heads on a frozen backbone | ~0.6 GB + model | a CPU job |

The GPU memory figure is the fixed part: `BatchStream` holds the whole split on the device as
uint8, which is 436 MB for the 30,278 training rows and 109 MB for the 7,568 validation rows at
60x80x3. Model, optimiser and activations sit on top; measure the total with `nvidia-smi`
before packing a card.

**Concurrency on one GPU does not create throughput.** The total GPU work is fixed, so two
training jobs on one card finish in roughly the time they would take back to back. The real
wins are (a) separate machines and (b) overlapping the two CPU-only HOG jobs with a GPU job,
which is close to free. If you do co-tenant two GPU jobs deliberately, give each
`hardware = 0.5` so their duty-cycle targets sum to the card rather than fighting for it.

## Current jobs

| Worker suffix / job | Purpose | Estimate | Required checkpoint first |
|---|---|---|---|
| `hog_svm` | HOG + linear SVM baseline | ~3 min | None |
| `cnn` | PlainCNN seed 42 | ~9 min | None |
| `resnet` | Stage-1 ResNet and decoupled head, seed 42 | ~64 min | None |
| `seeds_1337` | CNN and ResNet seed 1337 | ~78 min | None |
| `seeds_2024` | CNN and ResNet seed 2024 | ~73 min | None |
| `seeds` | All three seeds; replaces the two rows above | ~152 min | Seed-42 CNN/ResNet files avoid redundant training |
| `sweep` | Five stage-2 sampler strengths | ~23 min | `model_resnet_stage1.pt` |
| `hogsearch` | Six HOG C × class-weight arms | ~16 min | None |
| `cnnsearch` | Six CNN LR × weight-decay arms, 12 epochs each | ~17 min | None |
| `lrsearch` | Six ResNet LR × weight-decay arms, 12 epochs each | ~115 min | None |
| `stage2grid` | Six stage-2 LR × sampler-strength arms | ~28 min | `model_resnet_stage1.pt` |

The estimates are `RUNTIMES` in [the layout module](../../scripts/task1_layout.py) and are
derived from the per-epoch costs the archived run under `executed_run_2026-09-06/` measured on
Apple Silicon MPS at fp32: 14 s per CNN epoch, 96 s per stage-1 ResNet epoch, 28 s per stage-2
head epoch and 159 s per liblinear fit. Scale them to your own hardware before scheduling.
Excluding `seeds`, the ten remaining jobs total ~426 minutes.

Each filename is `worker_<suffix>.ipynb`. `logit_adjusted` and `phase2` are historical jobs,
not current job names. Current grids and their result paths are in the
[pipeline guide](../../docs/SUGGESTED_PIPELINE.md).

Run `resnet` before `sweep` and `stage2grid`, or copy its trained stage-1 checkpoint before
starting either. They train heads only and assert that a trained backbone exists.
For the all-seed worker, copy `model_cnn.pt` and `model_resnet_decoupled.pt` to reuse seed 42;
without them, it performs extra training that the combine run does not need.

## Schedules by machine count

Ten jobs total ~426 minutes of work (`seeds` excluded — it is the alternative to the two
seed-specific workers). Two floors bound any schedule: the total divided by the machine count,
and `lrsearch` at ~115 minutes, which no amount of hardware splits.

| Machines | Wall clock | Against the floor | Note |
|---:|---:|---|---|
| 1 | ~407 min (6 h 47) | 426 min sequential | Overlapping the CPU-only HOG jobs with GPU work saves the 19 minutes |
| 2 | ~213 min (3 h 33) | 213 = 426/2 | Optimal |
| 3 | ~151 min (2 h 31) | 142 = 426/3 | Optimal **without** a mid-run checkpoint copy; see below |
| 4 | ~115 min (1 h 55) | 115 = `lrsearch` | Optimal |
| 5 | ~115 min (1 h 55) | 115 = `lrsearch` | **No gain over four.** `lrsearch` is the floor |

**Four machines saturates this workload.** A fifth machine cannot help, because `lrsearch`
alone is already the make-span.

Every assignment below keeps `resnet`, `sweep` and `stage2grid` on one machine, so the stage-1
checkpoint dependency is satisfied locally and nothing has to be copied mid-run.

### Two machines — 213 min

| Machine | Jobs, in order | Total |
|---|---|---:|
| A | `lrsearch`, `seeds_2024`, `hogsearch`, `cnn` | 213 min |
| B | `resnet`, `sweep`, `stage2grid`, `seeds_1337`, `cnnsearch`, `hog_svm` | 213 min |

### Three machines — 151 min

| Machine | Jobs, in order | Total |
|---|---|---:|
| A | `lrsearch`, `hogsearch`, `hog_svm` | 134 min |
| B | `resnet`, `sweep`, `stage2grid`, `cnn`, `cnnsearch` | 141 min |
| C | `seeds_1337`, `seeds_2024` | 151 min |

C is the binding machine: the two seed jobs are indivisible and sum to 151. Splitting them
across A and B pushes either past 188, so 151 is the best a copy-free three-machine plan can
do. Accepting one copy — `resnet` on B, then shipping `model_resnet_stage1.pt` to C during a
nine-minute window — reaches 143, which is rarely worth the coordination.

### Four machines — 115 min (recommended)

| Machine | Jobs, in order | Total |
|---|---|---:|
| A | `lrsearch` | 115 min |
| B | `resnet`, `sweep`, `stage2grid` | 115 min |
| C | `seeds_1337`, `cnnsearch`, `hogsearch` | 111 min |
| D | `seeds_2024`, `cnn`, `hog_svm` | 85 min |

### Five machines — 115 min

| Machine | Jobs, in order | Total |
|---|---|---:|
| A | `lrsearch` | 115 min |
| B | `resnet`, `sweep`, `stage2grid` | 115 min |
| C | `seeds_1337`, `hog_svm` | 81 min |
| D | `seeds_2024`, `cnn` | 82 min |
| E | `hogsearch`, `cnnsearch` | 33 min |

Same make-span as four machines. Use this shape only if a spare machine is free anyway, or to
buy slack against one machine failing.

The estimates come from the archived run under `executed_run_2026-09-06/`, which used Apple
Silicon MPS at fp32. Measure actual runtimes on your own hardware and re-balance if the ratios
between jobs move.

## What comes back from each machine

Copy the whole of `models/task1/checkpoints/` from every worker machine into the combine
machine's `models/task1/checkpoints/`, preserving filenames. Files are keyed by name, so
copies from different machines merge into one directory without collisions.

| Job | Files it banks | Prefix |
|---|---|---|
| `hog_svm` | 2 | `model_hog_svm.joblib`, `model_hog_svm_scores.npz` |
| `cnn` | 1 | `model_cnn.pt` |
| `resnet` | 2 | `model_resnet_stage1.pt`, `model_resnet_decoupled.pt` |
| `seeds_1337` | 3 | `model_seed_cnn_1337.pt`, `model_seed_resnet___decoupled_1337*.pt` |
| `seeds_2024` | 3 | `model_seed_cnn_2024.pt`, `model_seed_resnet___decoupled_2024*.pt` |
| `sweep` | 5 | `model_stage2_power_0p0.pt` … `model_stage2_power_1p0.pt` |
| `hogsearch` | 12 | `model_search_hog_c*_weight_*.joblib` and `_scores.npz` |
| `cnnsearch` | 6 | `model_search_cnnsearch_lr*_wd*.pt` |
| `lrsearch` | 6 | `model_search_lrsearch_lr*_wd*.pt` |
| `stage2grid` | 6 | `model_stage2_grid_p*_lr*.pt` |

`epoch_*.pt` files are resume state, not results. They are safe to leave behind once the
matching `model_*.pt` exists. The three ensemble members the final model needs are
`model_resnet_decoupled.pt`, `model_seed_resnet___decoupled_1337.pt` and
`model_seed_resnet___decoupled_2024.pt`; the combine run fails the export if any is missing.

Keep each machine's executed `run_worker_*.ipynb` as the provenance record of what it printed.

## Configuration and identity

Full runs default to `ALLOW_CPU = False`. A CPU-only HOG worker must explicitly allow CPU.
`QUICK_RUN = True` enables a reduced structural run and changes checkpoint identity; its
results are not full-run results. The source currently enables CUDA AMP, whereas the saved
historical config records AMP off. Record performance settings when comparing results.

Set `hardware` in Section 1.0 on every Task 1 worker. It accepts `0.1` through `1.0`; CPU
threads, native pools, and HOG processes use `max(1, floor(hardware * visible_cores))`. On CUDA,
training and validation batches use CUDA-event timing plus inter-batch sleep for the same compute
duty-cycle target. GPU memory stays uncapped because a hard fractional allocator cap can OOM a
valid model; this setting is runtime-only and does not change `RUN_FINGERPRINT`.

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
historical Git revision, `cd44bc8f16c5`, and reproduces the five hand-derived workers that
existed there. `--strict` now runs it as check [6]; run it standalone to see the diff when it
fails. It needs repository history and is skipped, not failed, in a snapshot without `.git`.

## Troubleshooting

| Symptom | Check |
|---|---|
| Different fingerprint or unexpected retraining | Compare manifest, source constants, normalization and copied bank files |
| Sweep/grid cannot find a stage-1 backbone | Supply `model_resnet_stage1.pt` from the matching ResNet job |
| Backbone has `num_batches_tracked=1` | It is the untrained shape-check probe; use the trained checkpoint |
| Unknown job at startup | Use current job names above, not historical phase2/logit-adjusted names |
| All-seed worker repeats seed 42 | Supply the CNN and decoupled ResNet seed-42 banks |

Final outputs are under `models/task1/` (model export, classes, config and result CSVs),
`predictions/` (`task1_predictions.csv` and `task1_test_logits.npy`) and `outputs/figures/`
(report figures). The prediction CSV still needs the other tasks' columns. See
[artifact handover](../../artifacts/README.md).

## The independent evaluation notebook

[02_independent_evaluation.ipynb](02_independent_evaluation.ipynb) is not a worker and is not
generated. It runs on the combine machine after the combine notebook has exported
`models/task1/task1_model.pt`, scoring that exact ensemble against the external cosmetics
collections in `dataset1/` and `dataset2/`. It trains nothing, takes a few minutes, and writes
figures 10 and 11 into `outputs/figures/`. Section 8.7 of the combine notebook quotes its
result, so run it before the report is finalised.
