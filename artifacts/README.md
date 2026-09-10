# Artifact locations and handover

Checked 9 September 2026.

A Task 1 run writes to **`models/task1/`**, not here, so copying `artifacts/` alone is not a
handover. This directory holds preserved records of past runs: a run's evidence is kept by
copying it aside, because the notebook overwrites `models/task1/` in place.

## What is here

- [`kaggle-full-run-20260909/`](kaggle-full-run-20260909/) — the recorded full-budget run:
  the executed notebook with its outputs, its figures, the environment probe, and a zip of
  every artifact the run produced. This is the evidence the report's numbers are quoted from.

## What a completed run leaves behind

Everything lands under `models/task1/`. Retain all of it:

| Location | What to retain |
|---|---|
| `models/task1/final/` | The three refitted models: `hog_svm_final.joblib`, `cnn_final.pt`, `resnet_final.pt` |
| `models/task1/predictions/task1_predictions.csv` | Partial submission: `articleType` filled, the other three targets blank |
| `models/task1/predictions/reporting_all_models.csv` | Per-row reporting predictions for all three families |
| `models/task1/tables/` | Comparison, all 18 search arms, confirmed contenders, bootstrap intervals, per-epoch histories, split membership |
| `models/task1/figures/` | The five report figures |
| `models/task1/selection.json` | Which family won, under which rule, on what evidence |
| `models/task1/deployment.json` | What to ship: artifact, class order, normalisation, HOG parameters |
| `models/task1/run.json` | Protocol, input digest, and a SHA-256 of every file above |
| `preprocessed_datasets/train_manifest.csv` | Shared EDA input to the target-specific splits |

A `state_dict` alone does not reproduce a prediction. Each saved model carries its class
order, normalisation statistics, image size and recipe alongside the weights; keep them
together.

## Reproducible handover

Include the source notebooks, `pyproject.toml`, `uv.lock`, the exact audited manifest,
raw-data access, and the model files with their companion metadata. Record the source
revision, hardware and runtime settings, seed, split parameters, class mapping and measured
scores. The revision is recoverable: this is a Git working tree, so record the commit the run
was made from rather than an invented identifier, and note whether the tree was dirty.
`run.json` already carries the protocol, the library versions and the input digest, which
covers most of this — but it cannot know the commit, so record that separately.

The input digest hashes the manifest file and the sorted ids of the three splits, not every
image byte. A matching digest is strong evidence that two runs saw the same rows; it is not
proof that the underlying image files were byte-identical.

## Ignore rules

`.gitignore` excludes `artifacts/**` except directories, `.gitkeep` and README files. It also
excludes raw and preprocessed dataset CSV/JPEG files. It does **not** exclude `models/` or
`outputs/`. Not being excluded is not the same as being tracked — check actual inclusion with
`git status` before assuming an archive contains a file. Keep large files in the agreed
private handover and include the final models the assignment submission requires.
