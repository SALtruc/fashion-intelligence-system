# Task 1 model outputs

This folder is the canonical organized copy of the verified Kaggle run `task1_full_16ay9832`. It is
populated by `src/task1/organize_kaggle_results.py`; the raw Kaggle download was removed after this
canonical tree was verified.

Like every other task's weights, it lives under `artifacts/` and is **gitignored** — only this README
is tracked. Get the folder from the team Drive alongside the Task 3 and Task 4 checkpoints. The
seven run figures are the exception: they are small, the report quotes them, so they are tracked in
`notebooks/task1/figures/` instead.

- `final/` — the five refitted model artifacts, including the selected `resnet_resample_final.pt`.
- `predictions/` — reporting predictions, and the template-order submission copy whose SHA-256
  `run.json` records. The submitted file is `predictions/task1/task1_predictions.csv` at the
  repository root, and the two must stay identical.
- `tables/` — search, confirmation, refit, reporting, uncertainty and split tables.
- `sota/` — pretrained-reference embeddings and provenance.
- `telemetry/` — GPU samples, phase durations and the distributed log.
- `selection.json`, `deployment.json`, `run.json` — selection decision, deployable metadata and
  SHA-256 evidence for all 67 artifacts of the run.

For inference, use `python -m src.task1.task1_inference`; it defaults to this folder. For a new raw
Kaggle download, run `python src/task1/organize_kaggle_results.py --source <download>/task1_full_*`.
The organizer copies and verifies files; it does not delete the source.

Verifying the manifest on Windows needs newline normalisation: git checks these text files out with
CRLF, while the hashes were taken on Linux. Compare `sha256` of the file with `\r\n` replaced by
`\n`, or all 58 present files will look like mismatches.
