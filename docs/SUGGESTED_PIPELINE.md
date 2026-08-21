# Suggested End-to-End Pipeline

## COSC2753 Machine Learning Assignment 2 — Fashion Intelligence System

This document proposes a reproducible pipeline for the four parts of the assignment:

1. Fashion item type classification (`articleType`)
2. Fashion season classification (`season`)
3. Fashion audience and occasion classification (`gender` and `usage`)
4. Visual search: retrieving the most visually similar fashion items

The central design principle is that the system must be evaluated as it would be used in practice: a new image is provided as input, and the system predicts its labels or retrieves similar images. The report should therefore justify each processing step with evidence from the data, validation results, or relevant literature rather than choosing a method only because it is familiar.

> Important assignment constraint: the final submitted models must be trained by the group. Pre-trained models such as ImageNet models may be used for comparison or an independent benchmark, but their weights must not be used as the submitted final models.

## 1. Proposed system scope

Although the brief describes three classification tasks, Task 3 contains two meaningful targets. The recommended implementation is therefore:

| System | Input | Target/output | Recommended primary metric |
|---|---|---|---|
| Model A: item type | Fashion image | `articleType` | Macro-F1 |
| Model B: season | Fashion image | `season` | Macro-F1 |
| Model C: audience | Fashion image | `gender` | Macro-F1 |
| Model D: occasion | Fashion image | `usage` | Macro-F1 |
| System E: visual search | Query image + gallery | Top-K similar image IDs | Recall@K / mAP@K, with a clearly stated relevance proxy |

This produces four distinct classification models, satisfying the minimum requirement, while also covering the advanced visual-search task. Gender and usage should be separate models because they represent different concepts and have different label distributions. A combined `gender_usage` label would create many sparse combinations and make a prediction error difficult to interpret.

The model should use the image as its main input. Metadata such as `articleType`, `season`, `gender`, and `usage` is the prediction target, not an input feature. `productDisplayName` and other metadata can be used for exploratory analysis, data-quality checking, and analysis of possible leakage, but they should not be used by the final image-only classifier unless the project explicitly defines a multimodal system.

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
       Predict test labels / retrieve gallery images / save artefacts
```

The same conceptual pipeline is used for every target, but the preprocessing and model configuration may differ. For example, class weighting may be essential for `articleType`, while image appearance may provide little direct information about `season`; that difference should be measured and discussed rather than hidden. The only non-neural reference is the majority-class predictor.

## 3. Stage 0 — Establish reproducibility and data contracts

Before modelling, record:

- random seed(s), Python version, package versions, and hardware;
- the exact dataset version and number of CSV rows/images;
- the image size, colour mode, normalization, augmentation policy, and batch size;
- the split file used by every experiment;
- the class-to-index mapping for each target;
- the model configuration, training epoch, checkpoint-selection rule, and metric values.

Use a single configuration file or clearly defined constants for paths and seeds. Save label mappings, preprocessing settings, model weights, metrics, and prediction code so another group member can reproduce inference. Do not commit the educational dataset or large model files if the repository policy excludes them; document where an evaluator can obtain them.

## 4. Stage 1 — General exploratory data analysis

EDA should be performed before choosing the split, preprocessing, or model. It should answer questions that affect modelling decisions.

### 4.1 Tabular and label audit

For `styles_train.csv`:

1. Check the number of rows, unique IDs, duplicate IDs, missing values, invalid label values, and unexpected data types.
2. Verify that each labelled ID has exactly one corresponding image and identify missing or unreadable image files.
3. Count every class for `articleType`, `season`, `gender`, and `usage`.
4. Report the percentage of samples in the largest and smallest classes and the number of rare classes.
5. Inspect relationships between targets, such as `articleType` versus `gender`, `season`, and `usage`.
6. Inspect `year`, `baseColour`, `masterCategory`, and `subCategory` for useful descriptive patterns and suspicious leakage.

The class distributions determine whether accuracy is sufficient. If a few common item types dominate the data, a model can obtain high accuracy while performing poorly on minority classes. This is why macro-F1, per-class recall, and a confusion matrix should be primary evidence.

### 4.2 Image audit

Measure or visualize:

- image dimensions and aspect ratios;
- colour mode and corrupted files;
- brightness, contrast, background, and object scale;
- near-duplicate or repeated images;
- examples from common and rare classes;
- examples of visually ambiguous labels.

Create contact sheets for random samples and for each important class. This can reveal whether the labels are visually learnable, whether the background dominates the image, and whether resizing or cropping could remove important features.

### 4.3 Dataset quality and leakage audit

Check for exact duplicates and, where feasible, near duplicates using image hashes or low-dimensional image fingerprints. Also investigate whether the same product, image variant, or highly similar photographs occur in both partitions. A random split that places near-identical images in both train and validation can produce an unrealistically high score.

If duplicates are found, group them together and place the complete group in only one partition. Record the rule and the number of affected samples. If no reliable product-group identifier exists, state the limitation and use the strongest available duplicate check.

## 5. Stage 2 — Safe data preparation and splitting

This stage should be divided into operations that are safe globally and operations that must be learned from training data.

### 5.1 Operations safe before splitting

These do not learn a value from the labels or from the validation distribution:

- matching CSV IDs to image filenames;
- removing rows whose image is missing or unreadable;
- converting images to a consistent colour mode;
- checking file integrity;
- removing exact duplicate records according to a documented rule;
- defining the target columns and excluding rows with a missing target for that task.

Filtering out invalid records before the split is reasonable because it is a data-integrity operation, not a model-fitting operation. The decision and resulting counts must be logged.

### 5.2 Operations that must be training-only

Fit these using the training partition only, then apply the learned transformation unchanged to validation and test data:

- label encoders and class-index mappings;
- class weights calculated from target frequencies;
- learned image normalization statistics, if custom statistics are used;
- any learned preprocessing parameters or decision thresholds;
- augmentation policy parameters selected from experiments.

For neural networks, fixed operations such as resizing and converting pixels to `[0, 1]` are not fitted to the data. Dataset-specific normalization values, if used, must be calculated from training images only.

### 5.3 Stratified split

Create a fixed train/validation split, for example 80/20, using a stratified strategy. Use the same split indices for all competing models so that differences are attributable to the methods rather than to different validation samples. The repository already specifies a shared split in `splits/`; all group members should use that file.

For separate target models, verify the class coverage of the shared split for all four targets. If a rare class is absent from validation or cannot be stratified safely, document the issue and use a task-specific stratified split only when necessary. Never tune on the unlabeled test images or repeatedly change the validation split to obtain a better result.

If duplicate/product groups are available, use a group-aware split first and stratification as far as the data permits. The priority is that related images do not cross the evaluation boundary.

## 6. Stage 3 — Task-specific preprocessing

### 6.1 Common image preprocessing

Start with a controlled, reproducible image pipeline:

1. Load the image using its CSV ID.
2. Convert to RGB.
3. Resize to a documented resolution such as 128×128 or 224×224.
4. Use a consistent policy for aspect ratio, such as resize-with-padding or resize-and-centre-crop.
5. Convert pixel values to floating point and normalize.
6. Apply random augmentation only to training images.

Reasonable training augmentations are small horizontal flips, rotations, translations, scale changes, and brightness/contrast changes. Avoid transformations that change the target semantics, such as aggressive colour changes if colour is informative for clothing type or season. Validation and test images must receive deterministic preprocessing only.

### 6.2 Neural-network input pipeline

All classifiers receive image tensors:

1. Load the image using its CSV ID.
2. Convert it to RGB and resize, pad, or crop it to the selected resolution.
3. Convert the image to a floating-point tensor.
4. Normalize pixels using fixed values or statistics calculated from the training images only.
5. Apply augmentation only in the training data loader.
6. Apply deterministic tensor preprocessing to validation and test images.

The same tensor representation should be used when comparing the neural models. This isolates the effect of architecture and hyperparameters.

### 6.3 Target-specific considerations

- **`articleType`:** expect many classes and strong imbalance. Use class-weighted training, inspect minority-class recall, and analyse confusions between visually similar types such as shirt/T-shirt or trousers/jeans.
- **`season`:** inspect whether some classes are rare and whether the labels are inferred from weak visual cues. Report whether the model is genuinely better than a majority-class predictor.
- **`gender`:** inspect label semantics and whether an “Unisex” or similar class exists. Do not silently merge classes; justify any mapping.
- **`usage`:** inspect missing or inconsistent values and class imbalance. Formal, casual, sports, and similar labels may be visually overlapping, so macro-F1 and per-class results matter more than accuracy alone.

## 7. Stage 4 — Model investigation for each task

For each classification target, use a focused model set:

1. **One neural-network baseline** using the common image-tensor pipeline.
2. **One or two advanced deep-learning models** trained from scratch, for example a deeper CNN or a custom ResNet-style model.
3. **Hyperparameter tuning** for each candidate using the same validation protocol.

The models are independent competitors, not a sequence in which the output of the baseline is passed into another model. The baseline provides a simple neural reference, while the advanced models test whether greater representational capacity improves the result enough to justify additional cost. This focused design gives a meaningful comparison without producing many unrecorded experiments.

### 7.1 Baseline 0: non-learning reference

For every target, calculate a majority-class predictor. This establishes the score obtainable without image understanding. For imbalanced targets, also consider a stratified random predictor as a second weak reference.

### 7.2 Neural-network baseline

Train a small CNN from scratch for each target using the common image-tensor pipeline. A suitable baseline contains 2–4 convolution blocks, activation functions, pooling, batch normalization where useful, global average pooling, dropout, and a task-specific classification head.

This is the primary baseline because it is simple enough to train and interpret while still learning visual features directly from the images. Save its validation predictions, training curves, and resource requirements so that the same analysis can be applied to every advanced model.

### 7.3 Advanced deep-learning models trained from scratch

Train one or two advanced neural models using only the assignment images. Recommended options are:

1. **DL model 1 — deeper CNN:** increase the depth and channel capacity of the baseline while retaining global average pooling and regularization.
2. **Optional DL model 2 — custom ResNet-style CNN or metric-learning network:** use residual blocks for classification, or learn an embedding specifically for visual search.

If compute or time is limited, train only the deeper CNN. If the second model is used, it should test a clear hypothesis, such as whether residual connections improve fine-grained item recognition or whether metric learning improves retrieval quality. Do not add a second architecture only to increase the number of models.

Use the same input resolution and fixed train/validation split for all targets. Use cross-entropy loss with training-only class weights or a suitable balanced-loss variant for severe imbalance. Select the checkpoint by validation macro-F1 or a clearly declared validation objective, not by the final epoch. Use early stopping based on a patience value chosen before inspecting the results.

The advanced models may improve over the simple CNN because they learn richer hierarchical shape, texture, and colour representations. They may nevertheless overfit, especially for rare classes; training/validation curves, augmentation ablations, and checkpoint selection provide evidence for that claim. Compare them on both performance and resource cost rather than assuming that a larger model is automatically better.

### 7.4 Optional research comparison

For the independent comparison, evaluate a pre-trained feature extractor or publicly reported/Kaggle result only as a separate benchmark. Clearly label it as external and do not use its weights in the submitted final models. Comparisons are valid only if the dataset, split, target definition, metric, and preprocessing are sufficiently comparable; otherwise state why the comparison is directional rather than exact.

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

Do not tune all parameters at once. Start with a small search, retain the best configuration according to validation macro-F1, and then verify it over multiple random seeds where feasible.

Keep a dated experiment table with the configuration, seed, training time, validation metrics, and observations. Do not select a model after looking only at one favourable metric. The final configuration should be frozen before the last training run.

## 8. Evaluation framework

### 8.1 Classification metrics

Report, for each task and model:

- macro-F1 as the primary metric;
- weighted-F1 to show performance under the observed class distribution;
- accuracy as a familiar secondary measure;
- per-class precision, recall, and F1;
- confusion matrix, preferably normalized by true class;
- training/inference time and model size where relevant.

Macro-F1 gives every class equal importance and is appropriate when minority classes matter. Weighted-F1 and accuracy provide context but can hide failures on rare labels. Report confidence intervals or variation across seeds where feasible. At minimum, repeat the best configurations over several seeds and report mean ± standard deviation.

Compare models on identical validation examples. Use paired bootstrap confidence intervals or a paired test on per-example correctness when making a strong claim that one model is better. A small metric difference without uncertainty should be described as inconclusive.

### 8.2 Classification error analysis

For each target:

1. Inspect the highest-confidence correct and incorrect examples.
2. Inspect errors for minority classes separately.
3. Identify systematic confusions and relate them to image quality, class ambiguity, background, pose, or label noise.
4. Compare the baseline and each DL model: do they fail on the same examples, or does a DL model correct the baseline's systematic errors?
5. Check calibration or confidence reliability if the system will expose confidence to users.

This analysis supports the ultimate judgement better than a single leaderboard number because it explains where the model is safe or unsafe to use.

### 8.3 Visual-search evaluation

Construct a gallery from the labelled training images. For each validation image, retrieve its Top-K nearest gallery images using cosine similarity or Euclidean distance in the selected embedding space. Exclude the query image itself if it is present in the gallery.

Because the dataset does not provide a human-annotated “visual similarity” score, define the relevance proxy before evaluation. A defensible primary proxy is same `articleType`; stricter secondary proxies can require the same `articleType` and `subCategory`, or compare with a manually inspected subset. Report:

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

## 9. Ultimate judgement: selecting the final models

Define “best” before examining the final test predictions. A recommended decision rule is:

1. Meet a minimum quality threshold against the majority baseline.
2. Prefer higher validation macro-F1, with per-class recall considered explicitly.
3. Reject models with evidence of leakage, unstable seed performance, or severe overfitting.
4. Consider inference latency, model size, memory, implementation complexity, and ease of deployment.
5. For visual search, balance retrieval quality against query latency and gallery scalability.
6. Select one final model for each required output, even if different tasks favour different architectures.

A model with slightly lower macro-F1 may be the better real-world choice if it is much smaller, faster, more stable, and has fewer catastrophic minority-class failures. Conversely, a more complex model should be selected only when its improvement is repeatable and meaningful. The report should make this trade-off explicit in an evidence-based decision table.

Do not choose the final model using the unlabeled test set. The test set is reserved for producing the final submission after the configuration is frozen.

## 10. Final training and independent evaluation

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

The assignment’s independent evaluation can include a comparison with external work using similar data and goals. It can also include a small manually curated set or externally collected data if permitted. Any external data must be documented, available to the evaluator where required, and kept separate from training and validation. Discuss domain shift, such as different backgrounds, poses, camera quality, or fashion trends.

## 11. Ablation and improvement experiments

For a stronger investigation, implement a small number of targeted experiments. Each should test a hypothesis:

| Experiment | Hypothesis | Evidence to collect |
|---|---|---|
| No augmentation vs augmentation | Augmentation improves robustness to pose/scale changes | Macro-F1, per-class recall, curves |
| Unweighted vs class-weighted loss | Weighting helps rare labels | Minority-class F1 and confusion matrix |
| 128×128 vs 224×224 | Higher resolution preserves useful detail | Score versus training/inference cost |
| Random split vs duplicate/group-aware split | Random splitting may overestimate generalization | Duplicate rate and metric change |
| Simple CNN vs deeper CNN | Greater capacity captures more useful visual information | Same validation queries and metrics |
| Background-preserving vs simple crop/padding | Background or object framing affects prediction | Target metrics and qualitative examples |

Do not run ablations without recording them. A negative result is valuable when it is correctly controlled and explained.

## 12. Recommended report structure within the five-page limit

The report should emphasise decisions, evidence, and critical analysis:

1. **Problem and system design:** inputs, outputs, constraints, and why the four classifiers plus search system are appropriate.
2. **Data and preprocessing:** EDA findings, data quality, split strategy, leakage controls, and feature representation.
3. **Models investigated:** majority reference, one neural-network baseline, one or two advanced DL models trained from scratch, and external comparison where used.
4. **Results and error analysis:** compact comparison table, confusion matrices or selected figures, ablations, and limitations.
5. **Ultimate judgement:** final model per task, evidence for the choice, deployment trade-offs, and independent evaluation.

Use the appendix for supporting figures, detailed class tables, and additional confusion matrices. Do not put the actual judgement only in the appendix.

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
- [ ] Visual search reports Recall@K/Precision@K or mAP@K and explains its relevance proxy.
- [ ] Error analysis and ablation results support the ultimate judgement.
- [ ] Final models are retrained only after the configuration is frozen.
- [ ] Prediction CSV has exactly `id,gender,articleType,season,usage` and one row per test image.
- [ ] Code, model-loading instructions, environment information, and required artefacts are included.
