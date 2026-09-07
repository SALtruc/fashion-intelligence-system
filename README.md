# Machine Learning Assignment 2 — Fashion Intelligence System

COSC2753 Machine Learning · Assignment 2 (2026B) · RMIT

Documentation checked against the local source and artifacts on **6 September 2026**.
Submission requirements and outstanding deliverables are in [SUBMISSION.md](SUBMISSION.md).

## Current state

| Component | Entry point | Status |
|---|---|---|
| EDA and preprocessing | [00_eda_and_preprocessing.ipynb](notebooks/00_eda_and_preprocessing.ipynb) | Audits raw data and exports the shared manifest |
| Task 1: article type | [01_task1_article_type.ipynb](notebooks/Task1/01_task1_article_type.ipynb) | Current training, comparison, tuning and prediction workflow |
| Dataset1 preparation | [00_prepare_dataset1.ipynb](notebooks/Task1/00_prepare_dataset1.ipynb) | Versioned enriched manifests prepared; training still uses supplied data only |
| External evaluation | [02_independent_evaluation.ipynb](notebooks/Task1/02_independent_evaluation.ipynb) | Evaluates the deployed supplied-only model on external imagery |
| Tasks 2–4 and final prediction | `notebooks/02_*`, `03_*`, `04_*`, `05_*` | Empty placeholder files; not runnable |

`notebooks/01_task1_article_type_classification.ipynb` is also empty. The older Colab
edition has been removed: it carried a separate upload workflow and mirrored neither the
current combine notebook nor the worker layout.

The saved Task 1 results report **0.7693 macro-F1** and **0.8774 accuracy** for the ResNet
with decoupled classifier retraining, the model Section 8.7 now recommends and which the
notebook scores again under horizontal-flip TTA. These are recorded supplied-only results,
not results from dataset1 enrichment or the newer tuning grids, and they predate the removal
of the seed-variance study. The [Task 1 report draft](docs/REPORT_TASK1.md) still describes
the superseded three-seed ensemble and has to be rewritten against a fresh run.

## Setup and data

From the project root, with `uv` installed:

```bash
uv sync --frozen
uv run jupyter notebook
```

`pyproject.toml` requires **Python 3.12.0**. The lockfile selects dependencies; the default
dev group includes `pyflakes` for worker checks. Linux and Windows use the explicit
PyTorch CUDA 12.8 index; macOS uses the default package source. The training notebook
selects CUDA, then Apple Silicon MPS, then CPU.
Full training requires CUDA or MPS under the default `ALLOW_CPU = False`.
AMP, channels-last layout and the on-device image cache are CUDA-only paths. `QUICK_RUN = True`
enables a reduced CPU smoke run, whose metrics must not be used in the report.

Place the course-provided data as follows; retain the raw files unchanged:

```text
datasets/train/styles_train.csv
datasets/train/images_train/<id>.jpg
datasets/test/styles_prediction.csv
datasets/test/images_test/<id>.jpg
```

See [datasets/README.md](datasets/README.md) for counts and path details.

## Run order

1. Run `notebooks/00_eda_and_preprocessing.ipynb` top to bottom. It writes
   `preprocessed_datasets/train_manifest.csv`; share this exact manifest across machines.
2. Run `notebooks/Task1/01_task1_article_type.ipynb` top to bottom on the training machine.
   That is ~7 hours of training on one machine. To split it, follow the
   [worker guide](notebooks/Task1/PARALLEL_RUN.md) — it carries machine setup, a per-job
   resource profile and ready-made schedules for two and three machines (three machines reaches
   the ~115-minute floor; a fourth adds nothing) — then run this notebook in combine mode
   (`JOB_FILTER = None`, `RESUME = True`). Missing compatible checkpoints cause training to
   run again.
3. Run `notebooks/Task1/02_independent_evaluation.ipynb` with `model_resnet_decoupled.pt`
   present. The checkpoint it loads must be the one being reported.
4. Complete Tasks 2–4 and the final prediction workflow before submission.

To execute EDA from a shell and retain a separate executed notebook:

```bash
uv run jupyter nbconvert --to notebook --execute notebooks/00_eda_and_preprocessing.ipynb --output 00_eda_and_preprocessing_executed.ipynb --ExecutePreprocessor.timeout=-1
```

The output is written beside the source notebook. A complete audit decodes about 38,000 images.

## Data and result contract

- Shared code is in [src/preprocessing.py](src/preprocessing.py). Images are converted to RGB,
  resized with preserved aspect ratio and white padding to **60 × 80 (width × height)**.
- EDA does not create a universal split. Each task filters its target and calls `make_split`;
  all models for that target must use the same seed, eligible rows and group-aware split.
  Task 1 uses 30,278 training and 7,568 validation rows; 110 of 124 classes are scoreable.
  `splits/` is currently a placeholder.
- Fit normalization and class weights on training rows; augment training only. Use macro-F1,
  accuracy and class-level diagnostics. Submitted models use no pretrained weights.
- Dataset1's prepared export has 31,436 training rows and the same 7,568 validation rows.
  Adoption requires loader and checkpoint-identity changes; see the
  [pipeline guide](docs/SUGGESTED_PIPELINE.md). It must then be excluded from held-out evaluation.

## Files to retain

| Location | Contents |
|---|---|
| `models/task1/` | Exported model, classes, config, results and checkpoints |
| `models/task1/runs/` | Historical worker result CSVs |
| `models/task1/checkpoints_invalid/` | Quarantined invalid sweep checkpoints; never use for inference |
| `predictions/` | Task 1 prediction CSV and test logits — the submission deliverable |
| `outputs/figures/` | Report figures |
| `preprocessed_datasets/` | Global EDA manifest and optional versioned dataset1 exports |
| `artifacts/`, `splits/` | Reserved locations; current Task 1 uses the paths above |

The combine notebook writes `predictions/task1_predictions.csv`: 5,829 rows with only
`articleType` filled. The final file must preserve the template's IDs, order and header
`id,gender,articleType,season,usage`, with all four targets completed. An older copy from a
superseded notebook revision is still present at `outputs/task1_predictions.csv`; it is not
the current output path.

The current `.gitignore` excludes raw dataset CSV/JPEG files, preprocessed CSV/JPEG files
and artifact contents (except README and `.gitkeep` files). **It does not exclude `models/`,
`outputs/`, `predictions/`, or the external image collections.** Not being excluded is not the
same as being committed: the external image collections are tracked, while `models/` and
`outputs/` are not. Run `git status` before assuming an archive contains them. Keep the course
data private and supply required files through the agreed submission route.

## Documentation and checks

- [Current models and remaining candidates](docs/SUGGESTED_MODEL.md)
- [Pipeline and reproduction contract](docs/SUGGESTED_PIPELINE.md)
- [External data audit](docs/EXTERNAL_DATA_AUDIT.md)
- [Independent evaluation scope and provenance](docs/INDEPENDENT_EVALUATION_DATA.md)
- [Artifact handover](artifacts/README.md)

```bash
uv run python scripts/check_task1_workers.py --strict
uv run python scripts/make_task1_workers.py --check
```

Both commands validate the current workers without training. `--strict` also runs the
optional `--legacy` gate, which reproduces the five hand-derived workers from the pinned
revision `cd44bc8f16c5` and so needs repository history. It is skipped, not failed, where
that history is absent.
