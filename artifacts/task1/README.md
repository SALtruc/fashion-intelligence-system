# Task 1 model outputs

This folder is the canonical organized copy of the verified Kaggle run `task1_full_16ay9832`. It is
populated by `src/task1/organize_kaggle_results.py`; the raw Kaggle download was removed after this
canonical tree was verified.

`submitted/` holds the four files a prediction needs, and is **tracked in git**, with `deployment.json`, `selection.json`
and `run.json`, so `python -m src.task1.task1_inference` works from a clone alone. The other four
candidates and the run evidence are gitignored and live on the team Drive. The seven run figures
are tracked too, under `notebooks/task1/figures/`.

The submitted checkpoint is in `submitted/`; the four losing candidates stay in `models/`.
`run.json` keys them all under `final/` because that is where the Kaggle run wrote them, so the
predictor searches `final/`, then `models/`, then this folder -- and the SHA-256 in `run.json` is
what actually decides a file is the right one. All five verify.

- `submitted/` — the submitted model with deployment.json, selection.json and run.json. Tracked.
- `models/` — the four candidates it was selected over. Local only.
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
