> **Historical document — not the current Task 1 result or run instructions.**
> Use [the current report](REPORT_TASK1.md), [README](../README.md), and [patch record](TASK1_PATCH_NOTES.md). This file is retained as development history.

# Task 1 output artifacts

Paths below are relative to the project root. `notebooks/Task1/02_task1_full_run.ipynb` is
the source of truth; each path appears when the cell that writes it runs.

## One run, one directory

Task 1 is a single notebook executed top to bottom, so there is one output location and no
handover between machines. Everything lands under `models/task1/`, grouped by what the file
is rather than by which stage produced it:

| Path under `models/task1/` | Contents |
|---|---|
| `final/hog_svm_final.joblib` | The refitted HOG + linear SVM, with its class order, HOG parameters and the `C` it was selected at |
| `final/cnn_final.pt` | The refitted PlainCNN: weights, class order, normalisation statistics and the recipe that produced it |
| `final/resnet_final.pt` | The refitted SmallResNet, same contents |
| `predictions/task1_predictions.csv` | The submission — one `articleType` per test row, in template order |
| `predictions/reporting_all_models.csv` | Per-row reporting-split predictions for all three families, so any metric in the report can be recomputed without retraining |
| `tables/task1_results.csv` | The reporting comparison: macro-F1, accuracy and the support-bucket breakdown per family |
| `tables/search_all_models.csv` | All 18 search arms — six per family — with their tuning scores |
| `tables/selection_candidates.csv` | The three confirmed contenders the winner was chosen from |
| `tables/paired_holdout_intervals.csv` | Paired stratified bootstrap intervals for each pairwise macro-F1 difference |
| `tables/history_{cnn,resnet}_search_<0-5>.csv` | Per-epoch history for each neural search arm |
| `tables/history_{cnn,resnet}_confirm.csv` | Per-epoch history for the confirmation runs |
| `tables/refit_{cnn,resnet}.csv` | Per-epoch history for the final refits |
| `tables/split_{fit,tuning,reporting}.csv` | Exact membership of the three splits |
| `figures/fig01_class_distribution.png` … `fig05_top_confusions.png` | The report figures, numbered in the order the notebook produces them |
| `selection.json` | Which family won, under which rule, on what evidence |
| `deployment.json` | What to ship: artifact path, class order, normalisation, HOG parameters |
| `run.json` | The protocol, an input digest, and a SHA-256 of every file above |

Figures are numbered by production order, not named after plot titles, so a filename does
not change when a title is reworded.

## Provenance

`run.json` carries three things, and they answer different questions:

- **`protocol`** — what was done: seed, image size, split shares, search and confirmation
  epoch budgets, patience, label smoothing, augmentation strengths, both search grids, and
  the Python, PyTorch, NumPy, pandas and scikit-learn versions in force.
- **`inputs`** — what it was done to: a SHA-256 of the manifest file, the row counts of the
  three splits, the class count, and a SHA-256 over the sorted ids in each split. Hashing
  the split membership rather than every image byte gives the same guarantee far more
  cheaply: the manifest hash fixes which files exist and the split hash fixes which rows
  went where.
- **`files`** — what came out: a SHA-256 of every artifact listed above, so "the run produced
  its outputs" is a checkable claim rather than an assertion.

## Reruns

A rerun overwrites the previous one in place. Nothing is timestamped or versioned, because
the notebook is deterministic given the same manifest and seed and there is no partial-run
state to protect. To keep a run's evidence, copy `models/task1/` aside before rerunning, the
way [`artifacts/kaggle-full-run-20260909/`](../artifacts/kaggle-full-run-20260909/) preserves
the recorded full run.

`QUICK_RUN = True` writes the same tree from a handful of rows per class. Its numbers are
not reportable; use it to check that the path completes, then rerun with `QUICK_RUN = False`.

This document covers Task 1 only. The other assignment tasks are outside its scope.
