# Current pipeline and reproduction contract

Checked 6 September 2026 against source and available artifacts. Task 1 is implemented;
Tasks 2–4 and final prediction remain empty placeholders. This replaces the earlier generic
proposal. Start with [the root README](../README.md) for environment and data setup.

## 1. Audit once, then split per target

Run `notebooks/00_eda_and_preprocessing.ipynb`. It parses the supplied metadata, checks images,
audits duplicates and labels, and writes `preprocessed_datasets/train_manifest.csv`.
Keep raw images and metadata unchanged. Share the same manifest across worker machines.

`src/preprocessing.py` provides:

| Function | Contract |
|---|---|
| `load_manifest(target=None)` | Read the audited manifest and resolve `filename` against supplied images; optionally retain rows labelled for one target |
| `standardize_image(image)` | EXIF orientation, RGB conversion, aspect-preserving white padding to 60×80 (width × height) |
| `load_image_array(path)` | Apply the shared transform, optionally scale to [0, 1] |
| `make_split(frame, target, validation_share=0.2, random_state=42)` | Group-aware stratified split; classes with fewer than two groups stay in training |
| `describe_split(...)` | Summarize target support and partition sizes |
| `compute_normalisation(...)` | Fit image statistics on the supplied training frame |

A universal split under `splits/` is not implemented. Missing labels are filtered per target,
so one task's missing values do not discard usable rows from another. Within each task,
keep eligible rows, seed, grouping and validation share fixed for every comparison.

Task 1 has **37,846 supplied rows**, **30,278 train / 7,568 validation**, and **124 classes**.
Only 110 classes appear in validation. Macro-F1 excludes the 14 absent classes; their quality
is unmeasured. Support buckets use training counts: head ≥1,000, body 100–999, tail 10–99,
rare <10. Exact-image grouping cannot rule out alternate views of the same product.

## 2. Train the supplied-only Task 1 workflow

Run `notebooks/Task1/01_task1_article_type.ipynb`, or use its generated workers and then
combine. See [PARALLEL_RUN.md](../notebooks/Task1/PARALLEL_RUN.md) for prerequisites.

The notebook caches standardized uint8 images, computes training-only normalization and
applies per-sample training augmentation: horizontal flip, ±10° rotation, 8% translation
and color jitter. Validation/test transforms are deterministic. Default training uses
batch size 128, up to 40 epochs, learning rate 1e-3, weight decay 1e-4, three warmup epochs,
label smoothing 0.05 and early stopping patience 8 on macro-F1. Stage 2 freezes the backbone
and retrains the classifier for 10 epochs at learning rate 1e-2 with balanced sampling.

`RESUME = True` restores compatible model banks. `QUICK_RUN = True` changes the data/budget
and is only a structural check. Full training selects CUDA or Apple Silicon MPS, with CPU fallback refused by default.
Mixed precision is currently enabled on CUDA, while the saved historical config records
AMP disabled. Therefore current defaults should not be described as bit-identical reproduction
of the historical run. Keep runtime settings with every result.

## 3. Compare models and tuning grids

Current jobs cover HOG/SVM, PlainCNN, SmallResNet with decoupling, the fixed-split seed study,
a five-value sampler sweep, and four grids:

| Grid | Values | Output under `models/task1/` |
|---|---|---|
| HOG | C = 0.01, 0.1, 1.0 × class weight balanced/none | `task1_hog_search.csv` |
| CNN | LR = 3e-4, 1e-3, 3e-3 × weight decay 1e-4, 1e-3; 12 epochs per arm | `task1_cnnsearch.csv` |
| ResNet | Same LR/weight-decay grid and 12-epoch budget | `task1_lrsearch.csv` |
| Stage 2 | LR = 3e-3, 1e-2, 3e-2 × sampler power 0.5, 1.0 | `task1_stage2_grid.csv` |

These output CSVs are not present in the checked snapshot. `task1_lr_search.csv` (with an
extra underscore) is an older experiment and cannot stand in for `task1_lrsearch.csv`.
A partial epoch checkpoint does not establish a completed grid result.

Report macro-F1, accuracy, balanced accuracy, weighted F1, top-5 and support buckets, then
inspect errors, calibration and inference cost. The current final-selection candidates are
the decoupled ResNet, the ensemble and any qualifying sampler-sweep winner. Historical
logit-adjusted and multi-task results remain in saved CSVs but their jobs were removed.

## 4. Optional dataset1 enrichment: preparation is complete, adoption is pending

`notebooks/Task1/00_prepare_dataset1.ipynb` checks the retained dataset1 snapshot, freezes the
supplied split and exports version **`c447dd49cbcb349c`** under
`preprocessed_datasets/task1_dataset1/`. The folder contains:

- `train_manifest.csv`: 31,436 rows = 30,278 supplied + 1,158 dataset1 rows.
- `validation_manifest.csv`: the unchanged 7,568 supplied validation rows.
- `class_support.csv`: supplied, external and combined support.
- `integration_metadata.json`: source/output hashes, split settings, classes and version.

External rows add 397 Eyeshadow, 384 Lipstick and 377 Nail Polish examples. Only articleType
is retained as external supervision; unverified auxiliary labels are missing. `relative_path`
is resolved against the repository root on each machine. Do not pass the combined training
manifest back through `make_split`, replace the global EDA manifest, or use its external rows
for independent evaluation after training on them.

The current combine notebook/workers still call the supplied-only loader and split.
To adopt enrichment, use the preparation notebook's loading example, fit normalization and
class weights on combined training, retain supplied-only support buckets for comparable
reporting, add the data version to cache/checkpoint identity, choose a separate output
location and regenerate workers. Merely running preparation changes no trained model.
The current fingerprint does not include the enriched metadata's complete data identity.

Compare supplied-only and enriched runs on the exact same validation rows. Only eight
validation images cover the three enriched classes, so per-class conclusions are fragile.
Source/version/licence gaps remain documented in [the audit](EXTERNAL_DATA_AUDIT.md).

## 5. External evaluation

`02_independent_evaluation.ipynb` loads the three fixed supplied-only ResNet checkpoints,
checks their fingerprints and averages softmax over members and horizontal mirrors.
It checks external file hashes against the eligible supplied manifest, but this alone cannot
prove independence from a future enriched training set or exclude related product views.
Verify the complete training population before making a held-out claim.

The recorded evaluation includes dataset1 (1,158 images) and dataset2 (699 images).
Use [INDEPENDENT_EVALUATION_DATA.md](INDEPENDENT_EVALUATION_DATA.md) for provenance,
label-quality limitations and the model-specific interpretation of those results.

## 6. Export and finish the assignment

The combine notebook writes `models/task1/task1_model.pt`, class/config JSON files, result
CSVs, `predictions/task1_predictions.csv`, `predictions/task1_test_logits.npy` and the figures
under `outputs/figures/`. The export
includes all selected ensemble members and inference settings. The independent evaluation
notebook instead requires the individual checkpoint files.

Current predictions contain 5,829 articleType values; gender, season and usage are blank.
Implement and evaluate the remaining tasks, then combine predictions using the supplied
`id,gender,articleType,season,usage` template without changing ID order or adding an index.
The final prediction notebook is currently empty and cannot perform this merge.

Package the trained models and reproduction code, then complete the full report and
[submission checklist](../SUBMISSION.md). Saved Task 1 results alone do not complete the
four-task assignment.
