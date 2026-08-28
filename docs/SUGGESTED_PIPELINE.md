# Suggested End-to-End Pipeline

## COSC2753 Machine Learning Assignment 2 — Fashion Intelligence System

This document proposes a reproducible pipeline for the four parts of the assignment:

1. Fashion item type classification (`articleType`)
2. Fashion season classification (`season`)
3. Fashion audience and occasion classification (`gender` and `usage`)
4. Visual search: retrieving the most visually similar fashion items

Test the system with new images, just as it would be used in practice.

For each processing step, give a reason based on the data, validation results, or relevant literature. Do not choose a method only because it is familiar.

> Important assignment constraint:
>
> - The group must train the submitted final models.
> - Pre-trained models, such as ImageNet models, may be used for comparison only.
> - Do not use pre-trained weights in the submitted final models.

## 1. What the system should include

Task 3 has two separate targets, so use four classification models:

| System | Input | Target/output | Recommended primary metric |
|---|---|---|---|
| Model A: item type | Fashion image | `articleType` | Macro-F1 |
| Model B: season | Fashion image | `season` | Macro-F1 |
| Model C: audience | Fashion image | `gender` | Macro-F1 |
| Model D: occasion | Fashion image | `usage` | Macro-F1 |
| System E: visual search | Query image + gallery | Top-K similar image IDs | Recall@K / mAP@K, using a clearly stated rule for what counts as similar |

This satisfies the minimum requirement and also covers visual search.

Keep `gender` and `usage` as separate models.

They describe different concepts and have different class distributions. Combining them would create many rare label combinations and make errors harder to understand.

Use the image as the main model input.

The target columns (`articleType`, `season`, `gender`, and `usage`) are labels, not input features.

Use other metadata for EDA and leakage checks only. Do not use it in the final image-only classifier unless a multimodal system is explicitly defined.

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

Use the same overall pipeline for every target, but allow task-specific settings.

For example, class weighting may help `articleType`, while images may contain little information about `season`. Measure and discuss these differences.

## 3. Stage 0 — Make the experiments reproducible

Before modelling, record:

- random seed(s), Python version, package versions, and hardware;
- the exact dataset version and number of CSV rows/images;
- the image size, colour mode, normalization, augmentation policy, and batch size;
- the split file used by every experiment;
- the class-to-index mapping for each target;
- the model configuration, training epoch, rule for choosing the saved model, and metric values.

Use one configuration file, or clearly defined constants, for paths and random seeds.

Save the label mappings, preprocessing settings, model weights, metrics, and prediction code. This allows another group member to repeat the predictions.

Do not commit the educational dataset or large model files if repository policy excludes them. Explain where an evaluator can obtain them.

## 4. Stage 1 — Exploratory data analysis (EDA)

Use EDA to understand the dataset before cleaning, splitting, or modelling.

Report findings that may affect modelling decisions. Do not modify the raw files during EDA.

### 4.1 Tabular and label analysis

For `styles_train.csv`, inspect:

1. The number of rows, unique IDs, missing values, invalid label values, and data types.
2. The number of classes and sample count for `articleType`, `season`, `gender`, and `usage`.
3. The percentage of samples in the largest and smallest classes and the number of rare classes.
4. Relationships between targets, such as `articleType` versus `gender`, `season`, and `usage`.
5. `year`, `baseColour`, `masterCategory`, and `subCategory` for useful patterns or possible leakage.

Class counts show whether accuracy is enough.

If a few common item types dominate the data, accuracy may be high even when rare-class performance is poor.

Use macro-F1, per-class recall, and a confusion matrix as the main evidence.

### 4.2 Image analysis

Measure or visualize:

- image dimensions and aspect ratios;
- colour mode and the number of corrupted or unreadable files;
- brightness, contrast, background, and object scale;
- examples from common and rare classes;
- examples of visually ambiguous labels.

Create contact sheets for random samples and important classes.

Use them to check whether:

- the labels can be learned from the images;
- the background is distracting;
- resizing or cropping could remove important details.

### 4.3 EDA conclusions

Summarize findings that affect the next stages:

- class imbalance;
- missing or unreadable files;
- suspicious labels;
- image properties that may guide preprocessing or augmentation.

Perform cleaning and splitting in Stage 2.

## 5. Stage 2 — Data cleaning and stratified splitting

Clean the dataset using documented rules. Then create one fixed split for all experiments.

Do not modify the raw files in `datasets/`. Create cleaned metadata CSV files derived from the given metadata CSV.

### 5.1 Data cleaning before splitting

Perform these checks before creating the train/validation split:

- confirm that the required CSV columns are present and have the expected data types;
- match each CSV ID to its image file and remove or record missing, unreadable, or corrupted images;
- identify duplicate CSV rows, duplicate image files, near-duplicate images, or repeated image variants;
- handle duplicate groups during data cleaning only;
- keep one selected version from each duplicate group in the cleaned CSV file and remove the other versions;
- remove rows with a missing or invalid target for the relevant task;
- record the kept ID, removed IDs, reason, and cleaning rule for each duplicate group.

Some simple, fixed preprocessing can be done before splitting because it does not learn from the dataset:

- decode the images;
- convert them to RGB;
- resize or pad them to a fixed size;
- convert pixels to a fixed range.

Apply the same rule to every image. These steps do not use labels or calculate statistics, so they do not cause validation leakage.

Do not calculate these values before splitting:

- normalization statistics;
- class weights;
- label mappings;
- decision thresholds;
- other data-dependent values.

Data cleaning outputs should be saved into disk since these techniques are consistent and applied for the whole dataset. Share them through Drive so every group member uses the same data and split. Never replace the raw files with cleaned or processed versions.

### 5.2 Stratified split

Using the cleaned CSV file, create a fixed train/validation split, such as 80/20. Keep class proportions as similar as possible in both parts.

Use the same split indices for all models. This makes model comparisons fair. The repository already provides a shared split in `splits/`; all group members should use it.

For each target, verify that all classes appear in the split.

If a rare class is missing from validation or cannot be split safely, document the issue. Use a task-specific split only when necessary.

Never tune on the unlabeled test images. Do not repeatedly change the validation split to get a better score.

If same-product group IDs are available, keep each complete product group in only one partition. Preserve class proportions as far as possible.

The goal is to prevent related images from appearing on both sides of the evaluation split.

## 6. Stage 3 — Prepare images for each task

### 6.1 Common image preprocessing

Start with a consistent image pipeline:

1. Load the image using its CSV ID.
2. Convert to RGB.
3. Resize to a documented resolution such as 128×128 or 224×224.
4. Use a consistent policy for aspect ratio, such as resize-with-padding or resize-and-centre-crop.
5. Convert pixel values to floating point and normalize.

### 6.2 Data-dependent preprocessing

Some preprocessing methods depend on statistics or categories learned from many samples.

Fit them using the training partition only. Apply the same result to validation and test data:

- label encoders and class-index mappings;
- class weights calculated from training-label frequencies;
- learned image-normalization statistics, if custom statistics are used;
- any learned preprocessing parameters or decision thresholds;
- task-specific augmentation settings selected from experiments;
- any other preprocessing value calculated from the data.

Fixed steps such as resizing and converting pixels to `[0, 1]` do not learn from the data. Calculate dataset-specific normalization values from training images only.

Do not store a separate copy of the fully processed dataset.

Apply preprocessing on-the-fly in the data loader during training, validation, and testing.

Save only the pipeline and its learned settings as artifacts, such as `.pkl` or `.joblib` files. Also save the configuration needed to reproduce them.

### 6.3 Neural-network input format

All classifiers receive image tensors:

1. Load the image using its CSV ID.
2. Convert it to RGB and resize, pad, or crop it to the selected resolution.
3. Convert the image to a floating-point tensor.
4. Normalize pixels using fixed values or statistics calculated from the training images only.
5. Apply deterministic tensor preprocessing to validation and test images.

Use the same tensor format for all neural models. This makes architecture and setting comparisons fairer.

### 6.4 General and task-specific preprocessing

Use the common image preprocessing for every task.

Add task-specific steps only when EDA supports them. For each task, record:

- fixed preprocessing steps;
- values learned from the training split;
- steps used only during training.

Data augmentation is a task-specific step. Choose it from the EDA findings and apply it only to training images. Do not augment validation or test images.

Possible augmentations include:

- horizontal flips;
- small rotations and translations;
- small scale changes;
- brightness and contrast changes.

Avoid transformations that change the target's meaning. For example, avoid strong colour changes when colour may help predict clothing type or season.

- **`articleType`:** expect many classes and strong imbalance.
  - Calculate class weights or choose a balanced sampler from training labels only.
  - Inspect minority-class recall and confusions such as shirt/T-shirt or trousers/jeans.
- **`season`:** check for rare classes and weak visual cues.
  - Avoid strong colour changes if colour may help predict the season.
  - Check whether the model is better than a majority-class predictor.
- **`gender`:** check the meaning of each label and whether an “Unisex” class exists.
  - Fit the label mapping on the training split.
  - Do not merge classes without a clear reason.
- **`usage`:** clean missing or inconsistent values before splitting.
  - Calculate class weights from training labels only.
  - Use macro-F1 and per-class results because formal, casual, and sports labels may overlap visually.

## 7. Stage 4 — Compare models for each task

Repeat the following process independently for each classification target and for visual search:

1. Perform the task-specific preprocessing described in Stage 3.
2. Choose one baseline model using the common image-tensor pipeline.
3. Investigate one or two other models that may perform better, such as a deeper CNN or a custom ResNet-style model.
4. Tune and compare the candidate models using the same validation protocol, then finalize the best model for the task.
5. Compare the finalized model with a pre-trained model or a comparable public Kaggle/notebook result, when available.

Treat the models as separate alternatives. Do not pass the baseline output into another model.

The baseline is a simple reference. The other models test whether a different or larger model performs better.

Keep the model set small so the comparison is easy to understand and record.

### 7.1 Non-learning reference

For every target, calculate a majority-class predictor.

This is a reference point, not the baseline model. It shows the score possible without image understanding.

For imbalanced targets, also consider a stratified random predictor as a second weak reference.

### 7.2 Choose one baseline model

Train one small CNN from scratch for each target using the common image-tensor pipeline.

A suitable baseline can contain:

- 2–4 convolution blocks;
- activation functions and pooling;
- batch normalization where useful;
- global average pooling;
- dropout and a task-specific classification head.

This baseline is simple to train and understand. It still learns visual features from the images.

Save its validation predictions, training curves, and resource use for comparison.

### 7.3 Investigate other candidate models

Train one or two advanced neural models using only the assignment images. Recommended options are:

1. **DL model 1 — deeper CNN:** increase the depth and channel capacity of the baseline while retaining global average pooling and regularization.
2. **Optional DL model 2 — custom ResNet-style CNN or metric-learning network:** use residual blocks for classification, or learn an embedding specifically for visual search.

If compute or time is limited, train only the deeper CNN.

If you use a second model, give it a clear purpose. For example, test whether residual connections improve item recognition or whether metric learning improves retrieval.

Do not add a model only to increase the model count.

Use the same input resolution and fixed train/validation split for all targets.

Use cross-entropy loss with training-only class weights, or another suitable balanced loss for severe imbalance.

Select the saved model using validation macro-F1 or another declared validation objective. Do not select it simply because it is from the final epoch.

Choose the early-stopping patience before inspecting the results.

Advanced models may learn more detailed shape, texture, and colour patterns.

They may also overfit, especially on rare classes. Check this with training/validation curves, augmentation tests, and saved-model results.

Compare performance and resource use. A larger model is not automatically better.

### 7.4 Finalize the model and compare externally

After comparing the baseline and candidate models, finalize one model for each required task using Section 9.

Then compare it with a pre-trained feature extractor or a published/Kaggle result, if available. Treat this only as an external benchmark.

Label the external result clearly. Do not use pre-trained weights in the submitted final models.

Compare results only when the dataset, split, target, metric, and preprocessing are similar. Otherwise, describe the comparison as a rough indication.

Possible comparison conditions:

- CNNs trained from scratch: eligible submitted models;
- pre-trained ImageNet CNN fine-tuned or used as a frozen feature extractor: external comparison only;
- public Kaggle/notebook score: literature/benchmark context only, after checking its protocol.

Before comparing with a public score, check whether it used:

- a different split;
- extra data;
- pretrained weights;
- metadata;
- data leakage.

### 7.5 Hyperparameter tuning protocol

Tune a small, pre-declared set of important parameters for the baseline and each selected DL model.

Use an inner training split or cross-validation if there is enough computing power.

Otherwise, use the fixed validation split consistently. Note that repeated use may make the result optimistic.

Possible parameters include:

- image resolution;
- learning rate, optimizer, and batch size;
- number of convolution blocks and channel width;
- dropout and weight decay;
- augmentation strength;
- class-weighting or loss-function strategy;
- early-stopping patience.

Do not tune every parameter at once. Start with a small search, keep the configuration with the best validation macro-F1, and then test it with several random seeds if possible.

Keep a dated experiment table with the configuration, seed, training time, validation metrics, and observations.

Do not select a model using only one favourable metric. Freeze the selected configuration before evaluating it on the test data.

## 8. How to evaluate the models

### 8.1 Classification metrics

Report, for each task and model:

- macro-F1 as the primary metric;
- weighted-F1 to show performance under the observed class distribution;
- accuracy as a familiar secondary measure;
- per-class precision, recall, and F1;
- confusion matrix, preferably normalized by true class;
- training and prediction time, and model size where relevant.

Macro-F1 gives every class equal importance. It is useful when rare classes matter.

Weighted-F1 and accuracy add context but can hide poor performance on rare labels.

If possible, report confidence intervals or variation across random seeds. At minimum, repeat the best configurations with several seeds and report the mean ± standard deviation.

Compare models on the same validation examples.

Use paired bootstrap confidence intervals or a paired test on per-example correctness when claiming that one model is better.

Treat a small difference without uncertainty as inconclusive.

### 8.2 Understand classification errors

For each target:

1. Inspect the highest-confidence correct and incorrect examples.
2. Inspect errors for minority classes separately.
3. Identify systematic confusions and relate them to image quality, class ambiguity, background, pose, or label noise.
4. Compare the baseline and each DL model: do they fail on the same examples, or does a DL model correct the baseline's systematic errors?
5. Check calibration or confidence reliability if the system will expose confidence to users.

This analysis shows where the model works and where it fails. It is more useful than a single leaderboard score.

### 8.3 Evaluate visual search

Construct a gallery from the labelled training images.

For each validation image, retrieve its Top-K nearest gallery images using cosine similarity or Euclidean distance. Exclude the query image if it is also in the gallery.

The dataset has no human-annotated visual-similarity score. Define a relevance rule before evaluation.

A reasonable primary rule is the same `articleType`. A stricter rule can require the same `articleType` and `subCategory`, or use a manually checked subset.

Report:

- Recall@K: whether at least one relevant item appears in the first K results;
- Precision@K: proportion of the first K results that are relevant;
- mAP@K or NDCG@K when ranked relevance is available;
- qualitative retrieval grids for diverse query types;
- retrieval latency and gallery size.

Use the same nearest-neighbour method for all competing search models. Apply the same all-neural design:

- neural baseline embedding from the simple CNN's global-average-pooling layer;
- DL model 1 embedding from the deeper CNN;
- optionally DL model 2 embedding from a metric-learning network trained only on the assignment data.

Normalize embeddings consistently. Compare them using cosine similarity or Euclidean distance.

The final visual-search system should use neural embeddings only.

Same-category relevance is only an approximation of human-perceived similarity.

Also check retrieval results for colour, silhouette, texture, and overall appearance.

## 9. Choose the final models

Decide what “best” means before looking at final test predictions.

Use this rule:

1. Meet a minimum quality threshold against the majority baseline.
2. Prefer higher validation macro-F1, with per-class recall considered explicitly.
3. Reject models with evidence of leakage, unstable seed performance, or severe overfitting.
4. Consider prediction time, model size, memory use, implementation complexity, and ease of deployment.
5. For visual search, balance retrieval quality against query latency and gallery scalability.
6. Select one final model for each required output, even if different tasks favour different architectures.

A model with slightly lower macro-F1 may be the better practical choice if it is:

- smaller;
- faster;
- more stable;
- better on minority classes.

Choose a more complex model only when its improvement is repeatable and meaningful. Show this trade-off in a results table.

Do not choose the final model using the unlabeled test set. The test set is reserved for producing the final submission after the configuration is frozen.

## 10. Final model use and separate evaluation

After selecting the trained models:

1. Freeze preprocessing, architecture, hyperparameters, class mappings, and decision criteria.
2. Use the selected classification model for each target.
3. For the visual-search system, build the final gallery from all valid labelled training images.
4. Generate predictions for every test ID in `images_test`.
5. Preserve the exact sample-prediction format and column order:

   ```text
   id,gender,articleType,season,usage
   ```

6. Verify that every test ID appears exactly once, no required value is missing, and no extra columns are present.
7. Save the final weights and a script that loads them and reproduces the prediction file.

The independent evaluation can compare your results with external work that uses similar data and goals.

It can also use a small manually selected set or outside data if permitted.

For any outside data:

- document its source;
- make it available to the evaluator when required;
- keep it separate from training and validation;
- discuss differences in backgrounds, poses, camera quality, and fashion trends.

## 11. Targeted improvement experiments

For a stronger investigation, run a small number of targeted experiments. Each experiment should test one hypothesis:

| Experiment | Hypothesis | Evidence to collect |
|---|---|---|
| No augmentation vs augmentation | Augmentation improves robustness to pose/scale changes | Macro-F1, per-class recall, curves |
| Unweighted vs class-weighted loss | Weighting helps rare labels | Minority-class F1 and confusion matrix |
| 128×128 vs 224×224 | Higher resolution preserves useful detail | Score versus training/inference cost |
| Random split vs grouped split | Random splitting may overestimate generalization | Metric change and examples of related images |
| Simple CNN vs deeper CNN | Greater capacity captures more useful visual information | Same validation queries and metrics |
| Background-preserving vs simple crop/padding | Background or object framing affects prediction | Target metrics and qualitative examples |

Record every controlled comparison. A negative result is still useful when it is controlled and explained.

## 12. Suggested report structure within the five-page limit

The report should focus on decisions, evidence, and critical analysis:

1. **Problem and system design:** inputs, outputs, constraints, and why the four classifiers plus search system are appropriate.
2. **Data and preprocessing:** EDA findings, data quality, split strategy, leakage controls, and image representation.
3. **Models investigated:** majority reference, one neural-network baseline, one or two advanced DL models trained from scratch, and external comparison where used.
4. **Results and error analysis:** compact comparison table, confusion matrices or selected figures, controlled comparisons, and limitations.
5. **Final decision:** final model per task, evidence for the choice, deployment trade-offs, and independent evaluation.

Use the appendix for supporting figures, detailed class tables, and additional confusion matrices. Do not put the final decision only in the appendix.

## 13. Experiment tracking and final checklist

Keep one row per experiment.

Record the task, split version, preprocessing, model, hyperparameters, seed, metrics, training cost, and decision. This prevents accidental cherry-picking of results.

- [ ] EDA includes class imbalance, missing/corrupt images, image properties, and suspicious labels.
- [ ] A fixed shared split is used consistently and is stratified or group-aware where appropriate.
- [ ] Data-cleaning outputs are saved as CSV files, and cleaned image copies are saved only when needed.
- [ ] No validation or test information is used to fit preprocessing, tune thresholds, or choose the model.
- [ ] General and task-specific preprocessing is applied on-the-fly; only the pipeline and its settings are saved as artifacts.
- [ ] The majority-class predictor is reported only as a non-learning reference.
- [ ] One neural-network baseline and one or two advanced DL architectures are trained from scratch and tuned per target.
- [ ] At least four different final trained models are saved: `articleType`, `season`, `gender`, and `usage`.
- [ ] The submitted DL model(s) are trained from scratch.
- [ ] Pre-trained/public models are labelled as comparison only and their protocols are checked.
- [ ] All final classifiers and search representations use neural image models and tensor inputs.
- [ ] Macro-F1, weighted-F1, accuracy, per-class metrics, and confusion matrices are reported.
- [ ] Visual search reports Recall@K/Precision@K or mAP@K and explains its similarity rule.
- [ ] Error analysis and controlled improvement tests support the final decision.
- [ ] Final models are selected only after the configuration is frozen.
- [ ] Prediction CSV has exactly `id,gender,articleType,season,usage` and one row per test image.
- [ ] Code, model-loading instructions, environment information, and required files are included.
