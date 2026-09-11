# Task 1: Fashion article-type classification

Task 1 report material, updated from completed baseline run `task1_full_16ay9832` and confirmed in rerun `task1_full_06ed6wub` on 11 September 2026. This replaces the superseded draft. It must be condensed with Tasks 2–4 into the assignment's five-page, 11-point, single-column report, with up to two appendix pages. Add the group's actual names and IDs in that combined report.

## Problem and evaluation design

### Dataset and task
The dataset provides 43,675 training and 5,829 test records of fashion catalogue items with four metadata attributes. Task 1 classifies `articleType`. The raw training labels span 143 raw article types with severe class imbalance (from 7,858 Tshirts down to 10 singleton classes).

EDA identified: (i) duplicate images with conflicting labels, (ii) 19 invalid rows with HTML and missing data, (iii) catalogue cutouts against studio white backgrounds with standard orientation, (iv) aspect ratios tightly centered at 3:4. The audited cleaning protocol drops invalid rows, deduplicates exact SHA-256 matches, standardizes all images to 60 × 80 RGB via aspect-ratio-preserving bilinear padding against (255, 255, 255), and discards classes with fewer than two instances. This leaves 124 retained classes, of which 110 appear in the holdout reporting split.

### Evaluation split and metric
Evaluation follows a strict 60/20/20 train/tuning/reporting split, grouped by label hash to eliminate leakages. All architecture selection and hyperparameter tuning decisions are made exclusively on the tuning split. The reporting split (7,568 images) is evaluated exactly once for the final comparison.

The primary evaluation metric is **macro-averaged F1 score** across all observed classes, directly addressing the severe class imbalance. Top-1 accuracy and support-tiered macro-F1 metrics across four frequency tiers (head: $\ge 1000$, body: $100\text{--}999$, tail: $10\text{--}99$, rare: $<10$) are reported to expose class-frequency sensitivity.

## Candidate models and methodology

We formulate this as a multiclass classification problem and compare three families across two class-imbalance treatments:
1. **Linear SVM over HOG descriptors:** 5,184-dimensional Histogram of Oriented Gradients features with an $L_2$-regularized LinearSVC ($C \in \{0.003, \dots, 1.0\}$). An inference-time decision offset $\tau$ sweep trades head precision for rare recall.
2. **Plain 5-layer CNN:** Baseline convolutional network with batch normalization, ReLU activations, global average pooling, and dropout ($p=0.3$).
3. **SmallResNet (ResNet-18 variant):** Standard residual learning architecture with 4 stages of 2 residual blocks each, adapted with a stride-1 stem for 60 × 80 inputs.

Two class-imbalance strategies are systematically evaluated:
- **Cost-sensitive loss reweighting (`reweight`):** Uniform mini-batch sampling with inverse-class-frequency cross-entropy loss weights.
- **Class-balanced resampling (`resample`):** Multinomial mini-batch sampling with probability inversely proportional to class frequency, coupled with unweighted cross-entropy loss and real-time data augmentation (random horizontal flips, $\pm 10^\circ$ rotations, $\pm 8\%$ translations, and color jitter).

## Key empirical findings

| Architecture / Arm | Reporting Macro-F1 | Accuracy | Rare Tier F1 (<10) |
|---|:---:|:---:|:---:|
| HOG + Linear SVM | 0.6345 | 79.92% | 0.4620 |
| HOG + Linear SVM ($\tau=0.00$) | 0.6345 | 79.92% | 0.4620 |
| CNN, reweight | 0.5339 | 65.33% | 0.4331 |
| CNN, resample | 0.7378 | 85.21% | 0.5381 |
| ResNet, reweight | 0.5405 | 65.54% | 0.4075 |
| **ResNet, resample** | **0.7635** | **87.61%** | **0.5857** |

The SVM offset selected $\tau=0.00$, offering no gain. Class-balanced resampling consistently outperforms loss reweighting across both neural architectures (+20.39% macro-F1 on CNN, +22.30% macro-F1 on SmallResNet). Frequent sampling allows rare classes to encounter diverse data augmentations, whereas loss reweighting repeatedly multiplies loss on identical unaugmented images, causing severe gradient instability on noisy tail classes.

On the reporting holdout split, **SmallResNet with resampling** is selected as the winning submittable model (Macro-F1: **0.7635**, Accuracy: **87.61%**, Rare Tier: **0.5857**). Paired within-class bootstrap testing confirms its statistical advantage over PlainCNN/resample ($95\%$ CI: $[+0.0085, +0.0404]$, $p < 0.05$) and HOG/SVM ($95\%$ CI: $[+0.0977, +0.1523]$, $p < 0.001$).

## Comparative Analysis & Independent Evaluation

1. **Independent Literature Comparison:** We contrast our flat, from-scratch classifier with Condition-CNN (Kolisnik et al., 2021; 91.0% Level-3 hierarchical accuracy) and ResNet-BERT (Seo et al., 2025; 37 classes, multimodal with pretraining). Differences in taxonomy, modalities, resolution and training constraints prevent a common leaderboard ranking.
2. **Pretrained Reference Comparison:** Locally executed pretrained references show that SigLIP2 linear probe achieves 0.8219 macro-F1, SigLIP2 fine-tuning 0.8166, ConvNeXt probe 0.8000 and ConvNeXt fine-tuning 0.7927. Linear probes outperform 5-epoch fine-tuning due to cold-head feature distortion (backpropagating random head gradients disrupts delicate pretrained geometry), destabilizing inverse-frequency loss weighting, and the short budget. Pretrained references are excluded from submission selection.
3. **Independent Evaluation on External ("In-the-Wild") Data:** To evaluate out-of-distribution robustness beyond the catalogue distribution, 60 unconstrained external photographs from Unsplash across 12 balanced classes (Head and Body tiers) were tested under the automated inference pipeline (bilinear aspect-ratio padding to 60 × 80). Accuracy drops from **87.61%** (catalogue holdout) to **13.33%** (external wild, 8/60 correct) and macro-F1 from **0.7635** to **0.0536**, with mean confidence falling to 0.34. This empirical gap exposes the model's reliance on clean white studio backgrounds and the difficulty of resolving garments in full-scene, multi-object customer photography.

The final claim is limited to this supplied catalogue distribution and the 110 observed reporting classes. Fourteen classes are unmeasured; near duplicates may remain after exact deduplication. For assisted catalogue tagging of isolated product cutouts, SmallResNet/resample is an effective choice; deployment on unconstrained customer photography would require explicit object detection or background segmentation.

## Practical deployment and demo

The local `task1_demo.py` interface loads the selected saved model (`models/task1/final/resnet_resample_final.pt`), accepts a product photo, displays five suggestions with confidence scores, and lets the user review or change the label before CSV export. The interface demonstrates integration without retraining. Scores are explicitly uncalibrated, and unknown inputs are not reliably rejected; the recommended workflow keeps a human reviewer in the loop.

## Evidence and references

Current tables, figures, selection and deployment records are under `models/task1/`. Use figure 03 for confirmation, 04 for reporting comparison, 05 for confusions and 07 for pretrained references. The original run records and predictions remain unchanged. Post-run software checks are in `TASK1_DELIVERY_VALIDATION.json`; packaging corrections are described in `TASK1_PATCH_NOTES.md`.

- Dalal, N. and Triggs, B. (2005). Histograms of Oriented Gradients for Human Detection. CVPR.
- He, K. et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Loshchilov, I. and Hutter, F. (2019). Decoupled Weight Decay Regularization. ICLR.
- Tschannen, M. et al. (2025). [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features](https://arxiv.org/abs/2502.14786).
- Liu, Z. et al. (2022). [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545). CVPR.
- Full external-study citations and source-access limits: [independent comparison](INDEPENDENT_EVALUATION_TASK1.md).
