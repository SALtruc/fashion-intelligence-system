> **Historical document — not the current Task 1 result or run instructions.**
> Use [the current report](REPORT_TASK1.md), [README](../README.md), and [patch record](TASK1_PATCH_NOTES.md). This file is retained as development history.

# Current pipeline and reproduction contract

Checked 9 September 2026 against source and available artifacts. Task 1 is implemented;
Tasks 2–4 and final prediction remain empty placeholders. Start with
[the root README](../README.md) for environment and data setup.

## 1. Audit once, then split per target

Run `notebooks/00_eda_and_preprocessing.ipynb`. It parses the supplied metadata, checks
images, audits duplicates and labels, and writes `preprocessed_datasets/train_manifest.csv`.
Keep raw images and metadata unchanged, and share that exact manifest across machines.

`src/preprocessing.py` is the shared data-access module for notebook 00 and Tasks 2–4:

| Function | Contract |
|---|---|
| `load_manifest(target=None)` | Read the audited manifest and resolve `filename` against supplied images; optionally retain rows labelled for one target |
| `standardize_image(image)` | EXIF orientation, RGB conversion, aspect-preserving white padding to 60×80 (width × height) |
| `load_image_array(path)` | Apply the shared transform, optionally scale to [0, 1] |
| `make_split(frame, target, validation_share=0.2, random_state=42)` | Group-aware stratified split; classes with fewer than two groups stay in training |
| `describe_split(...)` | Summarize target support and partition sizes |
| `compute_normalisation(...)` | Fit image statistics on the supplied training frame |

Task 1's notebook inlines the same transform constants so that it stands alone; notebook 00
remains the source of truth for them.

A universal split under `splits/` is not implemented. Missing labels are filtered per target,
so one task's missing values do not discard usable rows from another. Within each task, keep
eligible rows, seed, grouping and split shares fixed for every comparison.

## 2. Run the Task 1 notebook

Run `notebooks/Task1/02_task1_full_run.ipynb` top to bottom. It is self-contained: it defines
every transform, model, training loop and metric it uses, imports no project module, and
reads no checkpoint written elsewhere. There is no worker, launcher or generator step.

Task 1 has **37,846 labelled rows** and **124 classes**, split three ways and group-aware:
**24,223 fit / 6,055 tuning / 7,568 reporting**. All 124 classes appear in fitting, 106 in
tuning and 110 in reporting — a class with a single group cannot be split, so its rows go
wholly to training and its quality goes unmeasured on the held-out splits. Macro-F1 is
computed over the classes present in the split being scored. Support buckets use training
counts: head ≥1,000, body 100–999, tail 10–99, rare <10. Exact-image grouping cannot rule
out alternate views of the same product.

The notebook caches standardized uint8 images on the device, computes normalisation from
training rows only, and augments training only: horizontal flip, ±10° rotation, 8%
translation and brightness/contrast jitter, applied on the GPU. Tuning and reporting
transforms are deterministic. Training uses batch size 128, AdamW with three warmup epochs
then cosine decay, label smoothing 0.05, class-balanced cross-entropy and early stopping on
tuning macro-F1 with patience 8.

`QUICK_RUN = True` reduces the data and the budget; it is a structural check only and its
numbers must not be reported. Full training selects CUDA, then Apple Silicon MPS; CPU is
refused unless `ALLOW_CPU = True`. Mixed precision, channels-last layout and the device-side
image cache are CUDA-only paths, so record the runtime settings with every result.

## 3. Compare models under the equal-budget protocol

Three families — HOG/SVM, PlainCNN and SmallResNet — each get **six configurations on an
equal budget**, and the winner is chosen on **tuning macro-F1 alone**. No family gets a
bespoke extra stage the others do not; the retired layout gave the ResNet a sampler sweep
and a stage-2 grid the other two never received, so its margin was partly a budget artefact.

| Family | Grid |
|---|---|
| HOG/SVM | `C` ∈ {0.003, 0.01, 0.03, 0.1, 0.3, 1.0} |
| CNN | learning rate ∈ {3e-4, 1e-3, 3e-3} × weight decay ∈ {1e-4, 1e-3} |
| ResNet | the same learning-rate × weight-decay grid |

The search runs at 12 epochs. The best recipe in each family is then confirmed at the full
40-epoch budget, the confirmed contenders are compared, all three families are refit on
fit + tuning, and only then is the reporting split touched. Ties break on accuracy, then on
the candidate identifier.

Report macro-F1, accuracy and the support-bucket breakdown, then quantify the differences
with paired stratified bootstrap intervals and inspect the winner's largest confusions.
`models/task1/tables/` holds `search_all_models.csv`, `selection_candidates.csv`,
`task1_results.csv` and `paired_holdout_intervals.csv`; see
[OUTPUT_ARTIFACTS.md](OUTPUT_ARTIFACTS.md) for the full layout.

## 4. No external-data arm

An earlier revision added 697 external crops to training and evaluated the model on 1,857
more. The enrichment did not improve the deployed model on the reporting split, so both the
data and the machinery that carried it — the preparation notebook, the arm split, the
independent-evaluation notebook and their documentation — have been removed. Nothing in the
current tree reads external imagery, and the out-of-domain failure that evaluation recorded
is not reproducible here.

## 5. Export and finish the assignment

The notebook writes everything under `models/task1/`: the three refitted models in `final/`,
the tables and figures, the predictions, and `selection.json`, `deployment.json` and
`run.json`. `run.json` records the protocol, an input digest and a SHA-256 of every file,
so a result can be traced back to the settings and the rows that produced it. A saved model
carries its own class order and normalisation, so it can be loaded without the notebook.

Current predictions contain 5,829 `articleType` values; gender, season and usage are blank.
Implement and evaluate the remaining tasks, then combine predictions using the supplied
`id,gender,articleType,season,usage` template without changing ID order or adding an index.
The final prediction notebook is currently empty and cannot perform this merge.

Package the trained models and reproduction code, then complete the full report and
[submission checklist](../SUBMISSION.md). Saved Task 1 results alone do not complete the
four-task assignment.
