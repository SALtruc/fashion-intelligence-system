# Suggested End-to-End Pipeline

## COSC2753 Machine Learning Assignment 2 — Fashion Intelligence System

This document proposes a reproducible pipeline for:

1. Fashion item type classification (`articleType`)
2. Fashion season classification (`season`)
3. Fashion audience and occasion classification (`gender` and `usage`)
4. Visual search for similar fashion items

Support each processing choice with data, validation results, or literature.

> Important assignment constraint:
>
> - Submitted final models must be trained using the assignment data.
> - Pre-trained models may be used for comparison only.
> - Do not use pre-trained weights in submitted final models.

## 1. What the system should include

Task 3 has two targets, so use four classification models:

| System | Input | Target/output | Recommended primary metric |
|---|---|---|---|
| Model A: item type | Fashion image | `articleType` | Macro-F1 |
| Model B: season | Fashion image | `season` | Macro-F1 |
| Model C: audience | Fashion image | `gender` | Macro-F1 |
| Model D: occasion | Fashion image | `usage` | Macro-F1 |
| System E: visual search | Query image + gallery | Top-K similar image IDs | Recall@K / mAP@K |

Keep `gender` and `usage` separate because they describe different concepts. Use images as model inputs and metadata for EDA/leakage checks unless a multimodal system is explicitly defined.

## 2. Pipeline overview

```text
        Raw CSV + image folders
                        |
                        v
        Data audit and exploratory analysis
                        |
                        v
        Data cleaning and quality checks
                        |
                        v
        Create one fixed, reproducible train/validation split
                        |
        +-------------------------------+
        |                               |
        v                               v
Training-only preprocessing      Validation preprocessing
  (fit on train only)              (transform only)
        |                               |
        +---------------+---------------+
                        v
        +--> Simple CNN baseline --------+
        |                                |
        +--> Advanced DL model 1 -------+--> Tune and compare
        |                                |    using the same
        +--> Optional DL model 2 --------+    validation protocol
                        |
                        v
        Select final model using multiple criteria
                        |
                        v
        Use the selected trained model for final predictions
                        |
                        v
        Predict test labels / retrieve gallery images / save files
```

Use this pipeline for every target, with task-specific settings supported by EDA.

## 3. Stage 0 — Make the experiments reproducible

Before modelling, record:

- random seeds, software versions, hardware, dataset version, and CSV/image counts;
- image size, colour mode, normalization, augmentation, batch size, split file, and class mappings;
- model configuration, checkpoint rule, metrics, training time, and prediction code.

Use one configuration file or clearly defined constants for paths and seeds. Do not commit the educational dataset or large model files when repository policy excludes them.

## 4. Stage 1 — Exploratory data analysis (EDA)

Understand the data before cleaning, splitting, or modelling. Do not modify raw files during EDA.

### 4.1 Tabular and label analysis

For `styles_train.csv`, inspect:

1. Row count, unique IDs, missing values, invalid labels, and data types.
2. Class counts and rare classes for `articleType`, `season`, `gender`, and `usage`.
3. Relationships between targets and possible leakage in other metadata.

Use class counts to decide whether accuracy is sufficient. Report macro-F1, per-class recall, and confusion matrices when imbalance is present.

### 4.2 Image analysis

Measure or visualize:

- dimensions, aspect ratios, colour mode, and corrupted/unreadable files;
- brightness, contrast, background, and object scale;
- common, rare, and visually ambiguous examples.

Use contact sheets to check label visibility, background usefulness, and whether resizing/cropping could remove details.

### 4.3 EDA conclusions

Summarize:

- class imbalance;
- missing/unreadable files;
- suspicious labels;
- image properties that guide preprocessing or augmentation.

Perform cleaning and splitting in Stage 2.

## 5. Stage 2 — Data cleaning and stratified splitting

Clean the data using documented rules, then create one fixed split for all experiments.

Do not modify raw files in `datasets/`. Create cleaned CSV files derived from the given metadata CSV.

### 5.1 Data cleaning before splitting

Perform these checks before creating the train/validation split:

- confirm required CSV columns and data types;
- match CSV IDs to images and remove or record missing, unreadable, or corrupted files;
- identify duplicate CSV rows, duplicate image files, near-duplicate images, or repeated image variants;
- handle duplicate groups in data cleaning only, keeping one version from each group;
- remove rows with missing or invalid targets for the relevant task;
- record kept IDs, removed IDs, reasons, and cleaning rules.

Fixed preprocessing may happen before splitting when it uses no labels or dataset statistics: decode images, convert to RGB, resize/pad, and convert pixels to a fixed range.

Do not calculate normalization statistics, class weights, label mappings, thresholds, or other data-dependent values before splitting.

Save data-cleaning outputs to disk and share them through Drive. Outputs must be CSV files derived from the given metadata CSV, plus image files only when cleaned copies are needed. Never replace raw files.

### 5.2 Stratified split

Using the cleaned CSV, create one fixed train/validation split, such as 80/20, with similar class proportions in both parts.

Use the same split indices for all models. Use the shared split in `splits/` when applicable. Check that every target class is represented and document any rare class that cannot be split safely.

Never tune on unlabeled test images or repeatedly change the validation split. If product group IDs are available, keep each group in one partition and preserve class proportions as far as possible.

## 6. Stage 3 — Prepare images for each task

### 6.1 Common image preprocessing

Start with the same pipeline for every task:

1. Load the image using its CSV ID.
2. Convert it to RGB.
3. Resize to a documented resolution, such as 128×128 or 224×224.
4. Apply a documented padding or centre-crop policy.
5. Convert pixels to floating point and normalize.

### 6.2 Data-dependent preprocessing

Fit these steps using the training partition only, then apply the same result to validation and test data:

- label encoders and class-index mappings;
- class weights from training-label frequencies;
- learned normalization statistics and other learned parameters;
- decision thresholds and task-specific settings selected from experiments.

Resizing and converting pixels to `[0, 1]` are fixed steps. Calculate custom normalization values from training images only.

Apply preprocessing on-the-fly in the data loader. Save only the pipeline and learned settings as artifacts, such as `.pkl` or `.joblib`, plus its config.

### 6.3 Neural-network input format

All classifiers receive image tensors:

1. Load using the CSV ID.
2. Convert to RGB and resize, pad, or crop.
3. Convert to a floating-point tensor.
4. Normalize using fixed values or training-image statistics.
5. Apply deterministic processing to validation and test images.

Use one tensor format for fair architecture comparisons.

### 6.4 General and task-specific preprocessing

Use common preprocessing for every task. Add task-specific steps only when EDA supports them, and record fixed steps, learned values, and training-only steps.

Data augmentation belongs only to task-specific preprocessing. Apply it only to training images; keep validation and test processing deterministic.

Possible augmentations include horizontal flips, small rotations/translations, small scale changes, and mild brightness/contrast changes. Avoid transformations that change the target meaning.

- **`articleType`:** use training-only class weights or balanced sampling when needed; inspect minority recall and common confusions.
- **`season`:** check rare classes and weak visual cues; compare against a majority-class predictor.
- **`gender`:** check label meanings, fit the mapping on training data, and do not merge classes without a clear reason.
- **`usage`:** clean missing/inconsistent values before splitting; use training-only class weights and macro-F1.

## 7. Stage 4 — Compare models for each task

Repeat this process for each classification target and visual search:

1. Perform task-specific preprocessing from Stage 3.
2. Choose one baseline model.
3. Investigate one or two models that may perform better.
4. Tune and compare them using the same validation protocol.
5. Finalize the best model and compare it with a pre-trained model or public Kaggle/notebook result when available.

Keep models as separate alternatives, keep the set small, and record the reason for each candidate.

### 7.1 Non-learning reference

For every target, calculate a majority-class predictor. It is a reference, not the neural baseline. For imbalanced targets, consider a stratified random predictor as a second weak reference.

### 7.2 Choose one baseline model

Train one small CNN from scratch for each target using the common image-tensor pipeline. It can contain 2–4 convolution blocks, activation/pooling, optional normalization/dropout, global average pooling, and a task-specific head.

Save validation predictions, training curves, and resource use.

### 7.3 Investigate other candidate models

Train one or two advanced models using only assignment images:

1. **Deeper CNN:** increase baseline depth or channel capacity.
2. **Optional ResNet-style CNN or metric-learning network:** use residual blocks for classification or learn an embedding for visual search.

If time or compute is limited, train only the deeper CNN. Give every extra model a clear purpose.

Use the same resolution and fixed split for all targets. Use cross-entropy with training-only class weights, or another suitable balanced loss. Select checkpoints using a declared validation objective, such as macro-F1, not simply the final epoch.

### 7.4 Finalize the model and compare externally

Finalize one model for each required task using Section 9. Treat pre-trained models and public scores as external benchmarks only.

Compare results only when dataset, split, target, metric, and preprocessing are similar. Check for extra data, pre-trained weights, metadata, or leakage, and label protocol differences.

### 7.5 Hyperparameter tuning protocol

Tune a small, pre-declared set of parameters, such as resolution, learning rate, optimizer, batch size, depth, dropout, weight decay, augmentation strength, loss strategy, and patience.

Use an inner split or cross-validation when practical; otherwise use the fixed validation split consistently and note possible optimism. Keep a dated experiment table and freeze the selected configuration before test evaluation.

## 8. How to evaluate the models

### 8.1 Classification metrics

Report for each task and model:

- macro-F1 as the primary metric;
- weighted-F1, accuracy, and per-class precision/recall/F1;
- a normalized confusion matrix;
- training/prediction time and model size where relevant.

If possible, report variation across seeds. Compare models on the same validation examples; treat small differences without uncertainty as inconclusive.

### 8.2 Understand classification errors

For each target:

1. Inspect high-confidence correct and incorrect examples.
2. Inspect minority-class errors separately.
3. Link systematic confusions to image quality, pose, background, or labels.
4. Check whether candidate models correct baseline errors.
5. Check calibration when users will see confidence scores.

### 8.3 Evaluate visual search

Build a gallery from labelled training images. For each validation image, retrieve Top-K neighbours and exclude the query if it is also in the gallery.

Define relevance before evaluation: same `articleType`, same `articleType` plus `subCategory`, or a manually checked subset.

Report Recall@K, Precision@K, mAP@K/NDCG@K when appropriate, qualitative retrieval grids, latency, and gallery size. Use the same neighbour method and embedding normalization for all search models.

Compare a baseline CNN embedding, a deeper CNN embedding, and optionally a metric-learning embedding trained on assignment data. Also inspect colour, silhouette, texture, and overall appearance.

## 9. Choose the final models

Decide what “best” means before looking at final test predictions:

1. Meet a minimum quality threshold against the majority baseline.
2. Prefer higher validation macro-F1 and consider per-class recall.
3. Reject leakage, unstable seeds, and severe overfitting.
4. Consider time, size, memory, complexity, and deployment.
5. For visual search, consider retrieval quality, latency, and gallery size.
6. Select one final model for each required output.

A lower-scoring model may be better if it is smaller, faster, more stable, or better on minority classes. Choose extra complexity only when its improvement is repeatable and meaningful. Do not use the unlabeled test set for model selection.

## 10. Final model use and separate evaluation

After selecting the models:

1. Freeze preprocessing, architecture, hyperparameters, mappings, and decision rules.
2. Use the selected classifier for each target.
3. Build the final visual-search gallery from valid labelled training images.
4. Generate predictions for every test ID in `images_test`.
5. Preserve the exact format and column order:

   ```text
   id,gender,articleType,season,usage
   ```

6. Check every test ID appears once, required values are present, and no extra columns exist.
7. Save final weights and a script that reproduces the prediction file.

For outside evaluation data, document its source, keep it separate from training/validation, and discuss differences in backgrounds, poses, camera quality, and fashion trends.

## 11. Targeted improvement experiments

Run a small number of experiments, with one hypothesis per experiment:

| Experiment | Hypothesis | Evidence to collect |
|---|---|---|
| No augmentation vs augmentation | Augmentation improves robustness | Macro-F1, recall, curves |
| Unweighted vs class-weighted loss | Weighting helps rare labels | Minority F1, confusion matrix |
| 128×128 vs 224×224 | Higher resolution preserves detail | Score versus cost |
| Random vs grouped split | Random splitting may overestimate generalization | Metric change and examples |
| Simple vs deeper CNN | More capacity captures useful features | Same validation metrics |
| Crop/padding alternatives | Framing affects prediction | Metrics and examples |

Record every controlled comparison. A controlled negative result is still useful.

## 12. Suggested report structure within the five-page limit

Focus the report on decisions, evidence, and critical analysis:

1. **Problem and system design:** inputs, outputs, constraints, and model choices.
2. **Data and preprocessing:** EDA, quality, split, leakage controls, and image representation.
3. **Models investigated:** majority reference, baseline, advanced models, and external comparison.
4. **Results and error analysis:** comparison table, figures, controlled tests, and limitations.
5. **Final decision:** model per task, evidence, deployment trade-offs, and independent evaluation.

Use the appendix for supporting figures and detailed tables.

## 13. Experiment tracking and final checklist

Keep one row per experiment with the task, split, preprocessing, model, hyperparameters, seed, metrics, cost, and decision.

- [ ] EDA covers imbalance, missing/corrupt images, image properties, and suspicious labels.
- [ ] A fixed shared split is used and is stratified or group-aware where appropriate.
- [ ] Data-cleaning outputs are saved as metadata-derived CSV files, with image copies only when needed.
- [ ] No validation/test information fits preprocessing, thresholds, or model selection.
- [ ] General and task-specific preprocessing runs on-the-fly; only the pipeline/settings are saved as artifacts.
- [ ] The majority predictor is reported as a non-learning reference.
- [ ] One baseline and one or two advanced architectures are trained from scratch per target.
- [ ] Final trained models are saved for `articleType`, `season`, `gender`, and `usage`.
- [ ] Pre-trained/public models are labelled as comparison only.
- [ ] Macro-F1, weighted-F1, accuracy, per-class metrics, and confusion matrices are reported.
- [ ] Visual search reports Recall@K/Precision@K or mAP@K and defines relevance.
- [ ] Error analysis and controlled tests support the final decision.
- [ ] Final models are selected only after the configuration is frozen.
- [ ] Prediction CSV has exactly `id,gender,articleType,season,usage` and one row per test image.
- [ ] Code, model-loading instructions, environment information, and required files are included.
