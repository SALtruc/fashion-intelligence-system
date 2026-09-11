# Artifacts

Every task's trained weights and generated run output. One folder per task, and each has its
own README saying what is inside and how to run it.

## What is tracked

Each task keeps everything a prediction needs in one **`submitted/`** folder, and that folder
is the only part of this tree in git — the brief asks for a ZIP that runs on its own, and a
marker should not need a Drive link to make a prediction. Everything else is gitignored: the
losing candidates, the run evidence, the training history and the figures all travel on Drive.

| file | size | run it with |
|---|---:|---|
| `task1/submitted/resnet_resample_final.pt` | 45 MB | `python -m src.task1.task1_inference` |
| `task2/submitted/task2_model.pt` | 28 MB | `src.task2_utils.predict_test_set()` |
| `task3/submitted/task3_gender_usage_C_weighted.pt` | 1 MB | `python src/task3/predict_test.py` |
| `task4/submitted/` (encoder + gallery) | 115 MB | `python src/task4/retrieve_topk.py` |

Task 4 needs two files rather than one: retrieval ranks a query against a catalogue, so the
encoder without its gallery answers nothing.

Task 1 also tracks `deployment.json`, `selection.json` and `run.json` — the selection
decision, what to ship, and a SHA-256 for every artifact the run produced. Task 4 tracks
`image_preprocessing.json`, the letterbox size and channel statistics inference must
reproduce exactly.

## A state_dict alone is not a model

Each checkpoint here carries its class order, normalisation statistics, image size and
recipe alongside the weights, and the inference scripts read all of it rather than
re-deriving anything. A statistic recomputed on the data being predicted is not a statistic
any more. Keep the metadata with the weights.

## Restoring the rest from Drive

The Drive folder mirrors this layout. Two things to watch when restoring:

- **Check names against contents.** A previous copy of `artifacts/task1/` arrived with every
  file at the top level misnamed — `task1_predictions.csv` was a 45 MB checkpoint,
  `hog_svm_final.joblib` was a PNG. `run.json` records a SHA-256 for all 67 files; verify
  against it rather than trusting a filename.
- **Task 4 has two galleries.** The one beside the training checkpoint is an earlier
  development population of 30,389 items. The tracked `task4/submitted/gallery_embeddings.npy` is the
  33,968 the hold-out benchmark scored, and it is the one every script reads.

## Ignore rules

`.gitignore` excludes `artifacts/**` except directories, README files, and each task's
`submitted/` folder. Not being excluded is not the same as being tracked — check
with `git status` before assuming an archive contains a file.
