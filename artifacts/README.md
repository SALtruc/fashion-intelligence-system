# Artifact locations and handover

Checked 6 September 2026. This directory is reserved for task artifacts. Current Task 1
writes to **`models/task1/`, `predictions/` and `outputs/figures/`**, so copying `artifacts/`
alone is insufficient.

| Location | What to retain |
|---|---|
| `models/task1/task1_model.pt` | Exported model, ensemble member weights, class order, normalization, configuration and inference settings |
| `models/task1/task1_classes.json` | Readable class-index mapping |
| `models/task1/task1_config.json` | Recorded training and runtime configuration |
| `models/task1/task1_results.csv` | Saved comparison table; includes historical experiments |
| `models/task1/task1_seed_study.csv` | Saved seed study |
| `models/task1/task1_lr_search.csv` | Historical learning-rate search, distinct from current grid outputs |
| `models/task1/checkpoints/` | Per-job model banks and resumable epoch files |
| `models/task1/runs/` | Historical per-machine result tables |
| `predictions/task1_predictions.csv` | Partial submission: articleType only |
| `predictions/task1_test_logits.npy` | Stored test scores |
| `outputs/figures/` | Exported report figures |
| `preprocessed_datasets/train_manifest.csv` | Shared EDA input to target-specific splits |

The combine notebook exports every ensemble member. Its representative `state_dict` alone
is not the ensemble: inference averages member/view softmax probabilities, including the
horizontal mirror when enabled, before selecting a class. Keep the recorded class order,
normalization, image size and inference settings with the weights.

The independent evaluation notebook loads three files from `models/task1/checkpoints/`:
`model_resnet_decoupled.pt`, `model_seed_resnet___decoupled_1337.pt`, and
`model_seed_resnet___decoupled_2024.pt`. It does not load the export as a generic selected model.
Do not copy files from `checkpoints_invalid/` into the active directory.

## Reproducible handover

Include the source notebooks/scripts, `pyproject.toml`, `uv.lock`, the exact audited manifest,
raw-data access, model files and their companion metadata. Record the source revision,
hardware/runtime settings, seed, split parameters, class mapping and measured scores. The
revision is recoverable: this is a Git working tree, so record the commit the run was made
from rather than an invented identifier, and note whether the tree was dirty.

Task 1's supplied-only fingerprint is `e6b15f5c51de` for the recorded data and recipe.
It hashes selected configuration/data summaries, not every image byte or runtime option.
Matching fingerprints are necessary for reuse but do not prove full data identity or valid
training; the quarantined sweep files demonstrate that limitation.

Prepared dataset1 exports are a different data contract. Retain their versioned manifests,
metadata and external images, and establish a separate enriched-run identity before training.
See [the pipeline guide](../docs/SUGGESTED_PIPELINE.md).

## Ignore rules

`.gitignore` excludes `artifacts/**` except directories, `.gitkeep` and README files. It also
excludes raw/preprocessed dataset CSV/JPEG files. It does **not** currently exclude `models/`
or `outputs/`, and does not exclude `predictions/`. Check actual inclusion before sharing an
archive; a path's presence here does
not establish its Git tracking status. Keep large files in the agreed private handover and
include the final models required by the assignment submission.
