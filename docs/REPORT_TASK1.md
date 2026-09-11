# Task 1: Fashion article-type classification

Task 1 report material, updated from completed run `task1_full_16ay9832` on 10 September 2026. This replaces the superseded draft. It must be condensed with Tasks 2–4 into the assignment's five-page, 11-point, single-column report, with up to two appendix pages. Add the group's actual names and IDs in that combined report.

## Problem and evaluation design

We predict `articleType` from a product image. After preprocessing, the Task 1 population contains 37,846 labelled images across 124 classes. Large class-support differences motivate macro-F1 as the selection metric: accuracy alone can conceal poor rare-class recognition. The majority baseline obtains 17.40% accuracy but only 0.0027 macro-F1. Accuracy and weighted F1 remain useful secondary summaries; macro-F1 does not measure business error costs by itself.

The fitting, tuning and reporting partitions contain 24,223, 6,055 and 7,568 images. No IDs or exact-duplicate group IDs overlap. Fitting contains all 124 classes, tuning 106 and reporting 110. Classes absent from evaluation cannot be claimed as validated. Normalization is fitted on fitting data during development and recomputed on fitting plus tuning for final refitting. Reporting images do not fit these statistics.

Earlier pilots informed this protocol. Partition disjointness is verified, but the saved record does not establish that the same reporting population was never inspected during earlier development. We therefore qualify internal holdout claims and distinguish them from external evaluation.

## Data treatment and model choices

The EDA notebook supplies the audited manifest and collapses exact-duplicate image groups. Task-specific label filtering keeps rows whose article type is available even if another target is missing. Images undergo EXIF orientation handling, RGB conversion and white padding to 60 x 80, preserving aspect ratio. Mild geometric and colour augmentation is applied only in training. This is intended to preserve product shape while broadening appearance variation; a no-augmentation ablation has not isolated its effect.

HOG plus linear SVM provides an engineered-shape baseline (Dalal and Triggs, 2005). PlainCNN tests learned convolutional features. SmallResNet uses residual blocks (He et al., 2016) and a stride-1 3 x 3 stem to retain more thumbnail detail than an aggressively downsampling stem. These choices are motivated by the input, but separate stem/depth ablations were not run.

Each neural architecture is investigated with inverse-frequency loss reweighting and class-balanced resampling. Both have six learning-rate/weight-decay configurations and 24 search epochs. Each combination's winner is confirmed for up to 40 epochs, with patience eight. AdamW, warmup, cosine decay and label smoothing are shared. HOG/SVM searches six C values; an optional decision offset is selected on tuning data. Equal neural trial counts do not mean equal computation across families.

## Results and interpretation

| Model | Reporting macro-F1 | Accuracy | Rare-bucket F1 |
|---|---:|---:|---:|
| HOG/SVM | 0.6345 | 79.92% | 0.4620 |
| CNN, reweight | 0.5453 | 65.76% | 0.4357 |
| CNN, resample | 0.7447 | 85.56% | 0.5476 |
| ResNet, reweight | 0.5525 | 66.17% | 0.4386 |
| **ResNet, resample** | **0.7654** | **87.35%** | **0.5857** |

The SVM offset selected tau=0, so it adds no gain. Resampling outperforms the tested reweighting recipes in both architectures. This is evidence about these configurations, not proof that all weighted losses are inferior. No untreated-imbalance control was trained. The ResNet winner peaks at the 40-epoch confirmation limit, leaving convergence uncertain. Training uses one seed.

The selected ResNet's macro-F1 exceeds CNN/resample by 0.0206. A paired within-class bootstrap gives a 95% interval of [-0.0020, 0.0393], including zero. Its advantage over HOG/SVM is 0.1309 with interval [0.1000, 0.1507]. These intervals describe observed evaluation examples, not training-seed variation or unseen rare-class diversity.

There are 957 reporting errors. Tshirts to Tops (101), Casual Shoes to Sports Shoes (78), and Sports Shoes to Casual Shoes (40) are frequent directed confusions. Their labels suggest visual similarity, but counts alone cannot establish the cause. Head/body/tail/rare F1 is 0.8823/0.8386/0.7742/0.5857: the headline masks materially weaker rare-class performance.

## Ultimate judgement and practical use

The frozen tuning rule selects **ResNet/resample**, with learning rate 0.001, weight decay 0.0001 and 40 refit epochs. We retain it for assisted catalogue tagging where batch processing can tolerate a larger model. Its checkpoint is 45.01 MB compared with CNN's 4.85 MB. Recorded GPU reporting-batch prediction times are 3.15 s and 0.48 s respectively; these exclude some pipeline overhead and are not a customer-facing CPU latency guarantee. CNN is a credible compact alternative, and the uncertain accuracy margin prevents a universal superiority claim.

## External context and limitations

The [independent literature comparison](INDEPENDENT_EVALUATION_TASK1.md) contrasts this result with Condition-CNN and a ResNet-BERT fashion classifier. Differences in taxonomy, modalities, resolution, pretraining and evaluation protocols prevent a common leaderboard interpretation. This provides the literature-comparison route of independent evaluation; no fresh external-image test has been performed.

Locally executed pretrained references are a separate comparison: SigLIP2 probe achieves 0.8219 macro-F1, SigLIP2 fine-tuning 0.8074, ConvNeXt probe 0.8000 and ConvNeXt fine-tuning 0.7916. These models use external pretraining and are excluded from submission selection. Their gaps cannot be attributed to architecture alone. Upscaling the shared thumbnails cannot establish a guaranteed floor or ceiling on higher-resolution performance.

The final claim is limited to this supplied catalogue distribution and the 110 observed reporting classes. Fourteen classes are unmeasured; near duplicates may remain after exact deduplication. A stronger study would independently assess new catalogue sources, repeat training seeds, and evaluate calibrated review thresholds on data reserved for that purpose. These are outstanding experiments, not results of this update.

## Evidence and references

Current tables, figures, selection, deployment metadata and run records are under `models/task1/`. Use figure 03 for confirmation, 04 for reporting comparison, 05 for confusions and 07 for pretrained references. The original run records and predictions remain unchanged.

- Dalal, N. and Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
- Tschannen, M. et al. (2025). [SigLIP 2](https://arxiv.org/abs/2502.14786).
- Liu, Z. et al. (2022). [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR.
- Full external-study citations and source-access limits: [independent comparison](INDEPENDENT_EVALUATION_TASK1.md).

