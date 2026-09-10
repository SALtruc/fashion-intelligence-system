# Task 1 model outputs

This folder is the canonical organized copy of the verified Kaggle run `task1_full_16ay9832`. It is populated by `scripts/organize_kaggle_results.py`; the raw Kaggle download was removed after this canonical tree was verified.

- `final/` — the five refitted model artifacts, including the selected `resnet_resample_final.pt`.
- `predictions/` — reporting predictions and the Task 1 template-order submission.
- `tables/` — search, confirmation, refit, reporting, uncertainty and split tables.
- `figures/` — run figures used by the report.
- `sota/` — pretrained-reference embeddings and provenance.
- `telemetry/` — GPU samples, phase durations and the distributed log.
- `selection.json`, `deployment.json`, `run.json` — selection decision, deployable metadata and SHA-256 evidence.

For inference, use `python -m src.task1_inference`; it defaults to this folder. For a new raw Kaggle download, run `python scripts/organize_kaggle_results.py --source <download>/task1_full_*`. The organizer copies and verifies files; it does not delete the source.

