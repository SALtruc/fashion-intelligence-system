# Possible Deep-Learning Models

## COSC2753 Machine Learning Assignment 2 — Fashion Intelligence System
This document lists candidate models for investigation; it does not select a
final model. The dataset contains approximately 38,600 labelled images, so
small and medium-sized models are practical candidates for training from scratch.
## Assignment constraint and common setup
Submitted models must be trained using the assignment data. ImageNet or other
external pretrained weights may be used only for a clearly labelled comparison.
All candidates should use the same fixed split, preprocessing, and evaluation
protocol. Augmentation, class weighting, focal loss, and balanced sampling may
be investigated for imbalanced labels.
## Task 1 — Fashion item type classification
**Target:** `articleType`
- **Small custom CNN:** A fast, interpretable baseline that tests whether image
  content is sufficient and provides a reference for more complex models.
- **ResNet-18/34:** Residual connections stabilise deeper training and can learn
  garment shape, texture, colour, and structure.
- **DenseNet-121:** Dense feature reuse may preserve details needed to separate
  visually similar types, but it can require more memory.
- **EfficientNet-B0/B1:** Compound scaling gives an accuracy-versus-compute
  comparison and allows input-resolution effects to be investigated.
- **MobileNetV3:** Lightweight depthwise convolutions support mobile or web
  deployment, although fine-grained accuracy must be measured.

Investigate class imbalance, image resolution, background reliance, and
confusions such as shirts versus T-shirts or jeans versus trousers.

## Task 2 — Fashion season classification
**Target:** `season`
- **Small custom CNN:** A low-capacity reference for a target that may contain
  weak visual signals or noisy metadata labels.
- **ResNet-18/34:** Can combine local details with global structure such as
  sleeve length, layering, material, and garment coverage.
- **EfficientNet-B0/B1:** Tests whether efficient scaling and higher resolution
  capture subtle season-related details.
- **Multi-task CNN:** A shared backbone with season and auxiliary heads such as
  `articleType` or `usage`; loss weights and negative transfer must be tested.

Compare against a majority-class reference to determine whether season is
visually predictable rather than mainly a metadata convention.

## Task 3 — Gender and occasion/usage classification
**Targets:** `gender` and `usage`
- **Two independent CNNs:** Separate models allow target-specific preprocessing,
  losses, class weights, and error analysis.
- **Shared CNN with two heads:** Learns common fashion features while retaining
  separate gender and usage outputs; it is cheaper but may cause negative transfer.
- **Multi-task ResNet-18/34:** Tests whether residual shared features improve
  both targets while preserving independent classification heads.
- **EfficientNet or DenseNet with two heads:** Tests compound scaling or dense
  feature reuse for the two related targets.
- **Combined `gender_usage` classifier:** Models joint labels directly, but may
  create sparse classes and makes errors harder to interpret.

Separate targets are easier to evaluate because gender and usage describe
different concepts. Report each target independently, even with a shared model.

## Task 4 — Advanced fashion visual search
**Output:** Top-K visually similar gallery images.
- **Classification CNN embedding:** Use a global-average-pooling vector with
  cosine similarity; simple, but it may learn category rather than exact visual
  similarity.
- **Siamese network:** Learns image-pair similarity with shared weights; it
  requires defensible positive and negative pair construction.
- **Triplet network:** Learns to rank an anchor closer to a positive than a
  negative; hard-negative selection can improve retrieval but may destabilise training.
- **Supervised contrastive CNN:** Uses multiple positive and negative examples to
  shape the embedding space; it requires careful batch construction.
- **Convolutional autoencoder:** Learns embeddings through reconstruction without
  explicit pairs, but may prioritise background or lighting.
- **Joint classification/metric-learning model:** Combines semantic labels and
  retrieval objectives, but introduces loss-weight choices.

Possible relevance definitions are the same `articleType`, the same
`articleType` plus `subCategory`, or manual judgements on a query subset.
Report Recall@K, Precision@K, mAP@K, and qualitative retrieval examples.

## Models requiring caution
Large Vision Transformers and very deep CNNs may overfit or require excessive
compute when trained from scratch. GANs do not directly solve these objectives,
while unsupervised embeddings may learn pixel or background similarity instead
of meaningful fashion similarity.
## Experiment records
Record architecture, random/pretrained initialisation, input size, augmentation,
loss, optimiser, learning rate, batch size, epochs, checkpoint rule, metrics,
training time, model size, inference latency, and seed variation. Architecture
selection should be based on these measurements, class-level errors, robustness,
and deployment constraints.
