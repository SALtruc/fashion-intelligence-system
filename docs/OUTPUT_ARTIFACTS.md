# Task 1 output artifacts and worker handover

Paths below are relative to the project root. The Task1 notebook is the source of truth;
these paths are created when the relevant cell runs.

Each worker writes finished and resumable checkpoints under `models/task1/checkpoints/`:

- `model_*.pt` are finished neural-model banks; `epoch_*.pt` are resumable epoch states.
- `model_hog_svm.joblib` plus `model_hog_svm_scores.npz` are the HOG + SVM baseline.
- `model_search_hog_*.joblib` plus matching `_scores.npz` files are the six HOG grid arms.
- CNN, ResNet, seed-study, sampler-sweep, and stage-2-grid arms use the same `model_*.pt`
  and `epoch_*.pt` naming convention.
- Checkpoints contain the run fingerprint. Incompatible files are ignored rather than mixed
  into a run. HOG grid score files also record the arm hyperparameters and restore safely.

The worker stop cell prints every `.pt`, `.joblib`, and `.npz` file found in the checkpoint
folder. Copy that folder to the combine machine and preserve filenames. A checkpoint's
presence is not evidence that its training completed; use the matching result CSV and the
fingerprint-aware loader when assembling a run.

The combine notebook additionally writes:

| Path | Purpose |
|---|---|
| `models/task1/task1_model.pt` | Selected Task1 model export, including the inference settings that reproduce its predictions |
| `models/task1/task1_ood_gate.pt` | Section 8.6 out-of-distribution gate: class centroids, shared precision and the rejection threshold |
| `models/task1/task1_classes.json` | Class order used by the model |
| `models/task1/task1_config.json` | Run configuration and normalization metadata |
| `models/task1/task1_results.csv` | Current validation comparison table |
| `models/task1/task1_hog_search.csv` | HOG C × class-weight search results |
| `models/task1/task1_cnnsearch.csv` | CNN learning-rate × weight-decay search results |
| `models/task1/task1_lrsearch.csv` | ResNet learning-rate × weight-decay search results |
| `models/task1/task1_stage2_grid.csv` | Stage-2 learning-rate × sampler-power search results |
| `predictions/task1_predictions.csv` | Task1 article-type predictions in template column order |
| `predictions/task1_test_logits.npy` | Task1 test log probabilities |
| `outputs/figures/` | Figures produced by the executed combine notebook |

The combine notebook does not claim or create full-budget confirmation files for the reduced
12-epoch backbone searches. Those grids are reported as search evidence and are not silently
promoted into the 40-epoch main-model comparison.

This document intentionally covers Task1 only. Other assignment tasks are outside the
scope of this work.
