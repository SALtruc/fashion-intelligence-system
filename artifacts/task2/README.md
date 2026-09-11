# Task 2 model outputs

The season classifier's weights and generated run output. `task2_model.pt` is **tracked in git** — the submitted DenseNet-121, so a clone can predict
without a Drive link. The three candidate checkpoints under `checkpoints/` are not: 94 MB of
models the report quotes rather than re-runs. Get those from the team Drive.

- `task2_model.pt` — the submitted model, DenseNet-121. Tracked.
- `checkpoints/` — the three trained candidates the ultimate judgement compares:
  `model_densenet121.pt`, `model_efficientnet_b0.pt`, `model_random_forest.joblib`, plus the
  Random Forest's aligned validation scores and feature importances.
- `task2_test_scores.npy` — per-class scores for the 5,829 test images, from the final run.
- `figures/`, and one directory per model holding its metrics, history and validation
  predictions, written by `src/task2_utils.py` as the notebooks run.

`src/task2_utils.py` resolves every path here; nothing hard-codes the location. The one thing
that is **not** here is the prediction CSV: it is a submission deliverable, so it stays tracked
at `predictions/task2/task2_predictions.csv`, which is where
`notebooks/01_final_prediction.ipynb` reads it from.
