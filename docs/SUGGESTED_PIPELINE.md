# Suggested End-to-End Pipeline

## COSC2753 Machine Learning Assignment 2 — Fashion Intelligence System

This document proposes a reproducible pipeline for the four parts of the assignment:

1. Fashion item type classification (`articleType`)
2. Fashion season classification (`season`)
3. Fashion audience and occasion classification (`gender` and `usage`)
4. Visual search: retrieving the most visually similar fashion items

The system should be tested in the same way it will be used: give it a new image, then check its predicted labels or similar images. Explain each processing step using evidence from the data, validation results, or relevant literature. Do not choose a method only because it is familiar.

> Important assignment constraint: the final submitted models must be trained by the group. Pre-trained models such as ImageNet models may be used for comparison or an independent benchmark, but their weights must not be used as the submitted final models.

## 1. What the system should include

Although the brief describes three classification tasks, Task 3 has two separate targets. The recommended implementation is:

| System | Input | Target/output | Recommended primary metric |
|---|---|---|---|
| Model A: item type | Fashion image | `articleType` | Macro-F1 |
| Model B: season | Fashion image | `season` | Macro-F1 |
| Model C: audience | Fashion image | `gender` | Macro-F1 |
| Model D: occasion | Fashion image | `usage` | Macro-F1 |
| System E: visual search | Query image + gallery | Top-K similar image IDs | Recall@K / mAP@K, using a clearly stated rule for what counts as similar |

This produces four distinct classification models, satisfying the minimum requirement, while also covering the advanced visual-search task. Gender and usage should be separate models because they represent different concepts and have different label distributions. A combined `gender_usage` label would create many sparse combinations and make a prediction error difficult to interpret.

The model should use the image as its main input. Metadata such as `articleType`, `season`, `gender`, and `usage` is what the model predicts, not an input feature. Use `productDisplayName` and other metadata for data exploration, data-quality checks, and leakage checks. Do not use them in the final image-only classifier unless the project explicitly defines a system that uses both images and metadata.

## 2. Pipeline overview

```text
Raw CSV + image folders
        |
        v
Data audit and exploratory analysis
        |
        v
Leakage/duplicate checks and invalid-image filtering
        |
        v
Create one fixed, reproducible train/validation split
        |
        +----------------------------+
        |                            |
        v                            v
Training-only preprocessing      Validation preprocessing
  (fit on train only)              (transform only)
        |                            |
        +-------------+--------------+
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
       Retrain selected configuration on all labelled data
                      |
                      v
       Predict test labels / retrieve gallery images / save files
```

Use the same overall pipeline for every target, but allow the preprocessing and model settings to differ. For example, class weighting may be important for `articleType`, while images may contain little direct information about `season`. Measure and discuss these differences. The only non-neural reference is the majority-class predictor.

## 3. Stage 0 — Make the experiments reproducible

Before modelling, record:

- random seed(s), Python version, package versions, and hardware;
- the exact dataset version and number of CSV rows/images;
- the image size, colour mode, normalization, augmentation policy, and batch size;
- the split file used by every experiment;
- the class-to-index mapping for each target;
- the model configuration, training epoch, rule for choosing the saved model, and metric values.

Use one configuration file, or clearly defined constants, for paths and random seeds. Save label mappings, preprocessing settings, model weights, metrics, and prediction code so another group member can repeat the predictions. Do not commit the educational dataset or large model files if repository policy excludes them; explain where an evaluator can obtain them.

## 4. Stage 1 — Explore and check the data

Explore the data before choosing the split, preprocessing, or model. Focus on questions that can change your modelling decisions.

### 4.1 Tabular and label audit

For `styles_train.csv`:

1. Check the number of rows, unique IDs, duplicate IDs, missing values, invalid label values, and unexpected data types.
2. Verify that each labelled ID has exactly one corresponding image and identify missing or unreadable image files.
3. Count every class for `articleType`, `season`, `gender`, and `usage`.
4. Report the percentage of samples in the largest and smallest classes and the number of rare classes.
5. Inspect relationships between targets, such as `articleType` versus `gender`, `season`, and `usage`.
6. Inspect `year`, `baseColour`, `masterCategory`, and `subCategory` for useful descriptive patterns and suspicious leakage.

The class counts determine whether accuracy is enough. If a few common item types dominate the data, a model can have high accuracy but perform poorly on rare classes. This is why macro-F1, recall for each class, and a confusion matrix should be the main evidence.

### 4.2 Image audit

Measure or visualize:

- image dimensions and aspect ratios;
- colour mode and corrupted files;
- brightness, contrast, background, and object scale;
- near-duplicate or repeated images;
- examples from common and rare classes;
- examples of visually ambiguous labels.

Create contact sheets for random samples and important classes. They can show whether the labels can be learned from the images, whether the background is distracting, and whether resizing or cropping might remove important details.

### 4.3 Dataset quality and leakage audit

Check for exact duplicates and, if possible, near duplicates using image hashes or simple image fingerprints. Also check whether the same product, image variant, or very similar photograph appears in both partitions. If nearly identical images appear in both train and validation, the score may be unrealistically high.

If duplicates are found, group them together and place the complete group in only one partition. Record the rule and the number of affected samples. If no reliable product-group identifier exists, state the limitation and use the strongest available duplicate check.

## 5. Stage 2 — Prepare and split the data safely

Separate steps that can be done before splitting from steps that must learn values from the training data.

### 5.1 Operations safe before splitting

These do not learn a value from the labels or from the validation distribution:

- matching CSV IDs to image filenames;
- removing rows whose image is missing or unreadable;
- converting images to a consistent colour mode;
- checking file integrity;
- removing exact duplicate records according to a documented rule;
- defining the target columns and excluding rows with a missing target for that task.

Removing invalid records before the split is safe because it checks data quality rather than fitting a model. Record the rule and the number of removed records.

### 5.2 Operations that must be training-only

Fit these using the training partition only. Then apply the same result to validation and test data:

- label encoders and class-index mappings;
- class weights calculated from target frequencies;
- learned image normalization statistics, if custom statistics are used;
- any learned preprocessing parameters or decision thresholds;
- augmentation policy parameters selected from experiments.

For neural networks, fixed steps such as resizing and converting pixels to `[0, 1]` do not learn from the data. If you use dataset-specific normalization values, calculate them from training images only.

### 5.3 Stratified split

Create a fixed train/validation split, such as 80/20, while keeping class proportions as similar as possible in both parts. Use the same split indices for all models so that differences come from the methods, not from different validation images. The repository already provides a shared split in `splits/`; all group members should use it.

For separate target models, verify the class coverage of the shared split for all four targets. If a rare class is absent from validation or cannot be stratified safely, document the issue and use a task-specific stratified split only when necessary. Never tune on the unlabeled test images or repeatedly change the validation split to obtain a better result.

If duplicate or product groups are available, keep each group in only one partition and preserve class proportions as far as possible. The main goal is to prevent related images from appearing on both sides of the evaluation split.

## 6. Stage 3 — Prepare images for each task

### 6.1 Common image preprocessing

Start with a consistent image pipeline:

1. Load the image using its CSV ID.
2. Convert to RGB.
3. Resize to a documented resolution such as 128×128 or 224×224.
4. Use a consistent policy for aspect ratio, such as resize-with-padding or resize-and-centre-crop.
5. Convert pixel values to floating point and normalize.
6. Apply random augmentation only to training images.

Reasonable training augmentations are small horizontal flips, rotations, translations, scale changes, and brightness/contrast changes. Avoid transformations that change the target's meaning, such as aggressive colour changes if colour is informative for clothing type or season. Validation and test images must receive deterministic preprocessing only.

### 6.2 Neural-network input format

All classifiers receive image tensors:

1. Load the image using its CSV ID.
2. Convert it to RGB and resize, pad, or crop it to the selected resolution.
3. Convert the image to a floating-point tensor.
4. Normalize pixels using fixed values or statistics calculated from the training images only.
5. Apply augmentation only in the training data loader.
6. Apply deterministic tensor preprocessing to validation and test images.

Use the same tensor format when comparing neural models. This makes it clearer whether differences come from the model architecture or its settings.

### 6.3 Target-specific considerations

- **`articleType`:** expect many classes and strong imbalance. Use class-weighted training, inspect minority-class recall, and analyse confusions between visually similar types such as shirt/T-shirt or trousers/jeans.
- **`season`:** inspect whether some classes are rare and whether the labels are inferred from weak visual cues. Report whether the model is genuinely better than a majority-class predictor.
- **`gender`:** inspect what each label means and whether an “Unisex” or similar class exists. Do not silently merge classes; justify any mapping.
- **`usage`:** inspect missing or inconsistent values and class imbalance. Formal, casual, sports, and similar labels may be visually overlapping, so macro-F1 and per-class results matter more than accuracy alone.

## 7. Stage 4 — Compare models for each task

For each classification target, use a focused model set:

1. **One neural-network baseline** using the common image-tensor pipeline.
2. **One or two advanced deep-learning models** trained from scratch, for example a deeper CNN or a custom ResNet-style model.
3. **Hyperparameter tuning** for each candidate using the same validation protocol.

The models are separate alternatives; the baseline's output is not passed into the other models. The baseline is a simple reference. The advanced models test whether a larger model improves the result enough to justify its extra cost. Keeping the model set small makes the comparison easier to understand and record.

### 7.1 Baseline 0: non-learning reference

For every target, calculate a majority-class predictor. This establishes the score obtainable without image understanding. For imbalanced targets, also consider a stratified random predictor as a second weak reference.

### 7.2 Neural-network baseline

Train a small CNN from scratch for each target using the common image-tensor pipeline. A suitable baseline contains 2–4 convolution blocks, activation functions, pooling, batch normalization where useful, global average pooling, dropout, and a task-specific classification head.

This is the main baseline because it is simple to train and understand, while still learning visual features from the images. Save its validation predictions, training curves, and resource use so you can compare it fairly with every advanced model.

### 7.3 Advanced deep-learning models trained from scratch

Train one or two advanced neural models using only the assignment images. Recommended options are:

1. **DL model 1 — deeper CNN:** increase the depth and channel capacity of the baseline while retaining global average pooling and regularization.
2. **Optional DL model 2 — custom ResNet-style CNN or metric-learning network:** use residual blocks for classification, or learn an embedding specifically for visual search.

If compute or time is limited, train only the deeper CNN. If the second model is used, it should test a clear hypothesis, such as whether residual connections improve fine-grained item recognition or whether metric learning improves retrieval quality. Do not add a second architecture only to increase the number of models.

Use the same input resolution and fixed train/validation split for all targets. Use cross-entropy loss with training-only class weights or a suitable balanced-loss variant for severe imbalance. Select the saved model by validation macro-F1 or another clearly declared validation objective, not simply by the final training epoch. Use early stopping based on a patience value chosen before inspecting the results.

The advanced models may improve on the simple CNN because they can learn more detailed patterns of shape, texture, and colour. They may also overfit, especially on rare classes. Use training/validation curves, augmentation tests, and checkpoint results to check this. Compare both performance and resource use; a larger model is not automatically better.

### 7.4 Optional research comparison

For an independent comparison, evaluate a pre-trained feature extractor or a published/Kaggle result only as a separate benchmark. Label it clearly as external and do not use its weights in the submitted final models. Compare results only when the dataset, split, target, metric, and preprocessing are similar enough. Otherwise, say that the comparison gives only a rough indication.

Possible comparison conditions:

- CNNs trained from scratch: eligible submitted models;
- pre-trained ImageNet CNN fine-tuned or used as a frozen feature extractor: external comparison only;
- public Kaggle/notebook score: literature/benchmark context only, after checking its protocol.

Do not claim that a public score is better or worse without checking whether it used a different split, extra data, pretrained weights, metadata, or data leakage.

### 7.5 Hyperparameter tuning protocol

Tune a small, pre-declared set of high-impact parameters for the one baseline and each selected DL model. Use an inner training split/cross-validation for tuning if the team has enough compute; otherwise use the fixed validation split consistently and acknowledge that repeated validation use can make the estimate optimistic. Candidate parameters include:

- image resolution;
- learning rate, optimizer, and batch size;
- number of convolution blocks and channel width;
- dropout and weight decay;
- augmentation strength;
- class-weighting or loss-function strategy;
- early-stopping patience.

Do not tune every parameter at once. Start with a small search, keep the configuration with the best validation macro-F1, and then test it with several random seeds if possible.

Keep a dated experiment table with the configuration, seed, training time, validation metrics, and observations. Do not select a model after looking only at one favourable metric. The final configuration should be frozen before the last training run.

## 8. How to evaluate the models

### 8.1 Classification metrics

Report, for each task and model:

- macro-F1 as the primary metric;
- weighted-F1 to show performance under the observed class distribution;
- accuracy as a familiar secondary measure;
- per-class precision, recall, and F1;
- confusion matrix, preferably normalized by true class;
- training and prediction time, and model size where relevant.

Macro-F1 gives every class equal importance, so it is useful when rare classes matter. Weighted-F1 and accuracy add context but can hide poor performance on rare labels. If possible, report confidence intervals or variation across random seeds. At minimum, repeat the best configurations with several seeds and report the mean ± standard deviation.

Compare models on identical validation examples. Use paired bootstrap confidence intervals or a paired test on per-example correctness when making a strong claim that one model is better. A small metric difference without uncertainty should be described as inconclusive.

### 8.2 Understand classification errors

For each target:

1. Inspect the highest-confidence correct and incorrect examples.
2. Inspect errors for minority classes separately.
3. Identify systematic confusions and relate them to image quality, class ambiguity, background, pose, or label noise.
4. Compare the baseline and each DL model: do they fail on the same examples, or does a DL model correct the baseline's systematic errors?
5. Check calibration or confidence reliability if the system will expose confidence to users.

This analysis is more useful than a single leaderboard score because it shows where the model works and where it fails.

### 8.3 Evaluate visual search

Construct a gallery from the labelled training images. For each validation image, retrieve its Top-K nearest gallery images using cosine similarity or Euclidean distance in the selected embedding space. Exclude the query image itself if it is present in the gallery.

Because the dataset has no human-annotated visual-similarity score, define a practical relevance rule before evaluation. A reasonable primary rule is that two items have the same `articleType`. A stricter rule can require the same `articleType` and `subCategory`, or use a manually checked subset. Report:

- Recall@K: whether at least one relevant item appears in the first K results;
- Precision@K: proportion of the first K results that are relevant;
- mAP@K or NDCG@K when ranked relevance is available;
- qualitative retrieval grids for diverse query types;
- retrieval latency and gallery size.

Use the same nearest-neighbour method for all competing search models. Apply the same all-neural design:

- neural baseline embedding from the simple CNN's global-average-pooling layer;
- DL model 1 embedding from the deeper CNN;
- optionally DL model 2 embedding from a metric-learning network trained only on the assignment data.

Normalize embeddings consistently and compare them using cosine similarity or Euclidean distance. The final visual-search system should use neural embeddings only.

State clearly that same-category relevance is a proxy and not identical to human-perceived similarity. A visually similar search result should also be checked qualitatively for colour, silhouette, texture, and overall appearance.

## 9. Choose the final models

Decide what “best” means before looking at final test predictions. A recommended rule is:

1. Meet a minimum quality threshold against the majority baseline.
2. Prefer higher validation macro-F1, with per-class recall considered explicitly.
3. Reject models with evidence of leakage, unstable seed performance, or severe overfitting.
4. Consider prediction time, model size, memory use, implementation complexity, and ease of deployment.
5. For visual search, balance retrieval quality against query latency and gallery scalability.
6. Select one final model for each required output, even if different tasks favour different architectures.

A model with slightly lower macro-F1 may be the better practical choice if it is much smaller, faster, more stable, and makes fewer serious errors on minority classes. Choose a more complex model only when its improvement is repeatable and meaningful. Show this trade-off in a decision table supported by your results.

Do not choose the final model using the unlabeled test set. The test set is reserved for producing the final submission after the configuration is frozen.

## 10. Final training and separate evaluation

After selecting the configurations:

1. Freeze preprocessing, architecture, hyperparameters, class mappings, and decision criteria.
2. Retrain each selected classification model on all labelled, valid training data.
3. For the visual-search system, build the final gallery from all valid labelled training images.
4. Generate predictions for every test ID in `images_test`.
5. Preserve the exact sample-prediction format and column order:

   ```text
   id,gender,articleType,season,usage
   ```

6. Verify that every test ID appears exactly once, no required value is missing, and no extra columns are present.
7. Save the final weights and a script that loads them and reproduces the prediction file.

The independent evaluation can compare your results with external work that uses similar data and goals. It can also use a small manually selected set or outside data if permitted. Document any outside data, make it available to the evaluator when required, and keep it separate from training and validation. Discuss possible differences in backgrounds, poses, camera quality, and fashion trends.

## 11. Targeted improvement experiments

For a stronger investigation, implement a small number of targeted experiments. Each should test a hypothesis:

| Experiment | Hypothesis | Evidence to collect |
|---|---|---|
| No augmentation vs augmentation | Augmentation improves robustness to pose/scale changes | Macro-F1, per-class recall, curves |
| Unweighted vs class-weighted loss | Weighting helps rare labels | Minority-class F1 and confusion matrix |
| 128×128 vs 224×224 | Higher resolution preserves useful detail | Score versus training/inference cost |
| Random split vs duplicate/group-aware split | Random splitting may overestimate generalization | Duplicate rate and metric change |
| Simple CNN vs deeper CNN | Greater capacity captures more useful visual information | Same validation queries and metrics |
| Background-preserving vs simple crop/padding | Background or object framing affects prediction | Target metrics and qualitative examples |

Do not run controlled comparisons without recording them. A negative result is valuable when it is correctly controlled and explained.

## 12. Suggested report structure within the five-page limit

The report should emphasise decisions, evidence, and critical analysis:

1. **Problem and system design:** inputs, outputs, constraints, and why the four classifiers plus search system are appropriate.
2. **Data and preprocessing:** EDA findings, data quality, split strategy, leakage controls, and image representation.
3. **Models investigated:** majority reference, one neural-network baseline, one or two advanced DL models trained from scratch, and external comparison where used.
4. **Results and error analysis:** compact comparison table, confusion matrices or selected figures, controlled comparisons, and limitations.
5. **Final decision:** final model per task, evidence for the choice, deployment trade-offs, and independent evaluation.

Use the appendix for supporting figures, detailed class tables, and additional confusion matrices. Do not put the final decision only in the appendix.

## 13. Experiment tracking and final checklist

Keep one row per experiment with the task, split version, preprocessing, model, hyperparameters, seed, macro-F1, weighted-F1, accuracy, training cost, and decision. This provides evidence for the report and prevents accidental cherry-picking of results.

- [ ] EDA includes class imbalance, missing/corrupt images, image properties, and duplicate checks.
- [ ] A fixed shared split is used consistently and is stratified or group-aware where appropriate.
- [ ] No validation or test information is used to fit preprocessing, tune thresholds, or choose the model.
- [ ] The majority-class predictor is reported only as a non-learning reference.
- [ ] One neural-network baseline and one or two advanced DL architectures are trained from scratch and tuned per target.
- [ ] At least four different final trained models are saved: `articleType`, `season`, `gender`, and `usage`.
- [ ] The submitted DL model(s) are trained from scratch.
- [ ] Pre-trained/public models are labelled as comparison only and their protocols are checked.
- [ ] All final classifiers and search representations use neural image models and tensor inputs.
- [ ] Macro-F1, weighted-F1, accuracy, per-class metrics, and confusion matrices are reported.
- [ ] Visual search reports Recall@K/Precision@K or mAP@K and explains its similarity rule.
- [ ] Error analysis and controlled improvement tests support the final decision.
- [ ] Final models are retrained only after the configuration is frozen.
- [ ] Prediction CSV has exactly `id,gender,articleType,season,usage` and one row per test image.
- [ ] Code, model-loading instructions, environment information, and required files are included.
