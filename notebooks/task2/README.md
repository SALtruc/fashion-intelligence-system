# Task 2: Season Classification Workflow

Task 2 predicts one of four season labels—**Fall, Spring, Summer, or Winter**—from a catalogue product image. Season is partly a merchandising label rather than a directly observable object property, so the models must use indirect evidence such as garment type, colour, material appearance, and body coverage.

The training data is strongly imbalanced. The frozen split currently contains:

| Season | Training | Validation |
|---|---:|---:|
| Fall | 8,208 | 2,052 |
| Spring | 1,247 | 312 |
| Summer | 14,928 | 3,732 |
| Winter | 5,877 | 1,470 |

Summer represents approximately 49.3% of the data, whereas Spring represents approximately 4.1%. For this reason, model selection uses macro-F1 and balanced accuracy alongside overall accuracy. The majority baseline illustrates the problem: predicting Summer for every image gives 49.3% accuracy but only 0.165 macro-F1.

## Notebook execution order

Run the notebooks from the repository root and in numerical order. Notebooks 02–04 consume the exact prepared data exported by Notebook 01. Notebook 05 compares their validation evidence and packages one selected model. Notebook 06 performs a separate external evaluation, while Notebook 07 applies only the selected model to the unlabelled test set.

| Notebook | Purpose | Principal inputs | Principal outputs |
|---|---|---|---|
| `01_task2_setup.ipynb` | Defines the evaluation contract, creates the frozen 80:20 leakage-safe split, standardises images to 60×80 with aspect-preserving white padding, calculates training-only normalisation, extracts the 2,007 Random Forest features, and evaluates simple baselines. | Raw training CSV and images | Split CSV, metadata and fingerprint, image arrays, feature arrays, baseline validation scores |
| `02_task2_random_forest.ipynb` | Trains the classical image-feature baseline, evaluates its learning behaviour, and reports feature-family evidence. | Notebook 01 outputs | Random Forest model, aligned validation scores, feature-importance CSV |
| `03_task2_efficientnet_b0.ipynb` | Trains the small-input EfficientNet-B0 under the shared neural pipeline and explains correctness for each season. | Notebook 01 outputs and `src/task2_utils.py` | EfficientNet checkpoint containing weights, configuration, history, and aligned validation logits |
| `04_task2_densenet121.ipynb` | Trains the small-input DenseNet-121 under the same pipeline and explains correctness for each season. | Notebook 01 outputs and `src/task2_utils.py` | DenseNet checkpoint containing weights, configuration, history, and aligned validation logits |
| `05_task2_analysis_ultimate_judgement.ipynb` | Validates comparable checkpoints, ranks all models, analyses the selected model, states limitations, and exports the durable inference package. | Baseline evidence and checkpoints from Notebooks 02–04 | Comparison tables, diagnostic figures, and exactly one selected-model package |
| `06_task2_independent_evaluation.ipynb` | Compares the Notebook 05 winner with the external Fashion-Product-Season SigLIP checkpoint on labelled external images. It reports common metrics, per-class behaviour, prediction distributions, paired correctness, agreement, and confusion matrices. | Selected-model package, Task 2 metadata, ExtraSeasonData, and the Hugging Face checkpoint | External comparison tables, paired predictions, score arrays, and confusion matrices |
| `07_task2_prediction.ipynb` | Loads the selected package, applies its recorded preprocessing to the official test images, and produces final season predictions. | Selected-model package and raw test data | Final prediction CSV and class-score array |

Notebook 06 is independent of the frozen validation comparison used to choose the winner. It is intended to test generalisation under dataset shift, not to reselect the model after observing test-like external results.

## Main implementation decisions

- **Frozen split:** `splits/task2_season_split.csv` is reused by every candidate, preventing performance differences from being caused by different validation samples.
- **Leakage control:** identical image IDs remain entirely in either training or validation.
- **Image representation:** images are resized to 60×80 while preserving aspect ratio and padding unused space with white. This avoids geometric distortion and reduces training cost, with a recognised loss of fine texture detail.
- **Exported NumPy arrays:** preprocessing is performed once and saved as `.npy` arrays. Memory mapping lets later notebooks reuse exactly the same pixels and features without repeatedly decoding thousands of image files.
- **Classical features:** the 2,007-feature vector combines RGB/HSV summary statistics, colour histograms, HOG shape descriptors, and foreground occupancy. It provides an interpretable classical comparison without claiming to preserve all visual information.
- **Comparable neural training:** EfficientNet-B0 and DenseNet-121 share augmentation, batching, normalisation, optimisation, early stopping, checkpoint recovery, and metric code from `src/task2_utils.py`.
- **Selection evidence:** Notebook 05 verifies the data fingerprint, validation-ID ordering, class order, score dimensions, and finite values before comparing checkpoints.
- **Inference durability:** the selected package stores the model configuration, weights or estimator, class order, normalisation values, target image size, and preprocessing fingerprint needed by Notebooks 06 and 07.

## Current validation result

The current full-run evidence selects **DenseNet-121**.

| Model | Accuracy | Macro-F1 | Balanced accuracy | Top-2 accuracy | Selection score |
|---|---:|---:|---:|---:|---:|
| DenseNet-121 | 0.7561 | 0.7515 | 0.7211 | 0.9344 | 0.7835 |
| EfficientNet-B0 | 0.7511 | 0.7454 | 0.7102 | 0.9421 | 0.7795 |
| Random Forest | 0.6677 | 0.6840 | 0.6832 | 0.9296 | 0.7266 |
| Majority baseline | 0.4933 | 0.1652 | 0.2500 | 0.6875 | 0.3844 |

DenseNet-121's per-season F1 scores are 0.698 for Fall, 0.783 for Spring, 0.791 for Summer, and 0.734 for Winter. These figures come from the current exported results and will change if the models are retrained.

## Related files and directories

```text
Machine-Learning-Assignment-2/
├── datasets/
│   ├── train/styles_train.csv
│   ├── train/images_train/
│   ├── test/styles_test.csv
│   └── test/images_test/
├── notebooks/task2/
│   ├── 01_task2_setup.ipynb
│   ├── 02_task2_random_forest.ipynb
│   ├── 03_task2_efficientnet_b0.ipynb
│   ├── 04_task2_densenet121.ipynb
│   ├── 05_task2_analysis_ultimate_judgement.ipynb
│   ├── 06_task2_independent_evaluation.ipynb
│   ├── 07_task2_prediction.ipynb
│   └── README.md
├── src/task2_utils.py
├── splits/task2_season_split.csv
├── preprocessed_datasets/task2/
├── models/task2/
│   └── checkpoints/
├── outputs/task2/
│   ├── figures/
│   └── independent_evaluation/
└── ExtraSeasonData/
    ├── build_extra_season_data.py
    └── ExtraSeasonData/
        ├── extraSeasonData.csv
        └── extraSeasonImages/
```

### `src/task2_utils.py`

This is the shared implementation contract for Task 2. It centralises paths, directory creation, data loading, fingerprint validation, common metrics, per-class tables, image loading, Random Forest feature extraction, model construction, augmentation, neural training, checkpoint recovery, validation-prediction export, and checkpoint loading. Centralising these operations reduces accidental differences between model notebooks.

### `splits/task2_season_split.csv`

This records each labelled image ID, its frozen `train` or `validation` assignment, and its original position. It is the authoritative split and should not be regenerated between model runs unless every candidate is deliberately rerun.

### `preprocessed_datasets/task2/`

- `task2_metadata.json`: class order, labels, split IDs, preprocessing settings, normalisation statistics, and fingerprint.
- `task2_deep_learning_train_images.npy` and `task2_deep_learning_validation_images.npy`: standardised uint8 image tensors.
- `task2_random_forest_train_features.npy` and `task2_random_forest_validation_features.npy`: aligned 2,007-column feature matrices.
- `task2_baseline_validation_scores.npz`: validation evidence for the non-learned baselines.

### `models/task2/`

- `checkpoints/model_random_forest.joblib` and `model_random_forest_scores.npz`: fitted forest and aligned validation evidence.
- `checkpoints/model_efficientnet_b0.pt`: completed EfficientNet-B0 checkpoint.
- `checkpoints/model_densenet121.pt`: completed DenseNet-121 checkpoint.
- `checkpoints/random_forest_feature_importance.csv`: feature-level Random Forest evidence.
- `task2_results.csv`: common overall metrics and ranking score.
- `task2_per_class_results.csv`: selected-model precision, recall, and F1 by season.
- `task2_training_summary.csv`: neural best epochs and training durations.
- `task2_model.pt` or `task2_model.joblib`: the single selected package consumed by downstream notebooks. The current winner produces `task2_model.pt`.

### `outputs/task2/`

- `figures/confusion_matrix.png`, `calibration.png`, and `training_curves.png`: selected-model diagnostics.
- `task2_predictions.csv`: final test metadata with the predicted `season` column.
- `task2_test_scores.npy`: four-class test score matrix aligned with the prediction rows.
- `independent_evaluation/`: Notebook 06 exports external model comparison, per-class results, prediction distributions, paired predictions, score arrays, and confusion matrices here when run.

### `ExtraSeasonData/`

`build_extra_season_data.py` samples genuine source annotations from `fnauman/fashion-second-hand-front-only-rgb`. It supports all four Task 2 labels, maps the source label `Autumn` to `Fall`, defaults to 200 images per class, and enforces a range of 100–200 images per class. During generation it hashes decoded RGB content, skips exact duplicates, and fills the quota from unused source rows. It stages the images, CSV, and ZIP before replacing the previous valid package. The generated CSV contains `id`, `season`, and `articleType`, while the matching JPEG files are stored in `extraSeasonImages/`.

The current package was regenerated with seed 42. Four duplicate source
candidates—three Spring and one Winter—were detected while their class quotas
were being filled. All four were skipped and replaced before final IDs were
assigned. The completed package contains:

| Season | Images | Represented article types |
|---|---:|---:|
| Fall | 200 | 23 |
| Spring | 200 | 24 |
| Summer | 200 | 24 |
| Winter | 200 | 22 |
| **Total** | **800** | — |

An independent integrity check confirmed 800 readable JPEGs, 800 unique file
hashes, 800 unique decoded-pixel hashes, 800 matching CSV rows, and 800 matching
ZIP members. Perceptual similarity is not used as an automatic deletion rule:
different garments can have very similar low-resolution hashes because the
source uses consistent white catalogue backgrounds.

To regenerate the canonical package on Windows, run:

```powershell
uv pip install datasets
Set-Location ExtraSeasonData
..\.venv\Scripts\python.exe build_extra_season_data.py `
  --output-dir ExtraSeasonData `
  --overwrite
```

The builder first validates source metadata, then performs type-aware candidate
ordering, content-hash duplicate rejection, replacement sampling, image staging,
CSV creation, and ZIP creation. Only after every stage succeeds does it replace
the earlier output package.

> **Current compatibility note:** ExtraSeasonData has now been regenerated with all four seasons, but Notebook 06 still contains the earlier Spring/Winter-only label validation. Notebook 06 must be updated to accept Fall and Summer before the current external package can be evaluated.

## Environment and execution

Install the locked project dependencies and launch Jupyter from the repository root:

```powershell
uv sync
uv run jupyter notebook
```

Use `QUICK_RUN = True` in the neural notebooks for a pipeline check. Use the full configuration for reported model comparisons. Notebook 06 additionally downloads the pinned external SigLIP checkpoint through Hugging Face, so it requires network access and sufficient checkpoint storage.

## Reproducibility rules

1. Keep the raw dataset unchanged.
2. Run Notebook 01 before all model notebooks.
3. Do not mix arrays, checkpoints, or scores with different fingerprints.
4. Run or complete Notebooks 02–04 before Notebook 05.
5. Treat Notebook 05 validation results as the basis for model selection.
6. Treat Notebook 06 as a separate domain-shift evaluation.
7. Run Notebook 07 only after Notebook 05 has exported exactly one selected package.
8. Do not compare metrics from quick runs with metrics from full runs as if they were equivalent experiments.
