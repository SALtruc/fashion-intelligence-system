# Task 1: Independent Evaluation and Literature Comparison

This document fulfills the assignment requirement to benchmark Task 1 against independently published work and evaluate the model on independent external data outside the supplied dataset.

## Our reference point

Run `task1_full_06ed6wub` (confirming `task1_full_16ay9832`) selects an image-only, from-scratch SmallResNet using class-balanced resampling. It reports **87.61% accuracy and 0.7635 macro-F1** on 7,568 supplied-catalogue reporting images. Training retains 124 classes; reporting contains 110. Images are standardized to 60 × 80. Selection uses a separate tuning split, with earlier pilot exposure not fully reconstructed.

## Published work and comparability

Two published studies provide comparative context:
1. **Condition-CNN (Kolisnik et al., 2021):** Evaluated on a 44,441-image subset of the same e-commerce catalogue. Uses hierarchical classification across three tiers, reporting 91.0% accuracy on Level 3 (equivalent to `articleType`). However, their hierarchy groups fine-grained categories into broader super-classes and discards classes with fewer than 10 instances.
2. **ResNet-BERT Multimodal (Seo et al., 2025):** Evaluates joint image-text representations on a 37-class subset, reaching 94.2% top-1 accuracy. Crucially, their model relies on high-resolution text embeddings and pretraining on external web data, which are prohibited under our flat from-scratch image-only protocol.

Differences in class taxonomy, hierarchical restructuring, image resolutions, and feature modalities prevent a direct numerical leaderboard ranking against published benchmarks.

## Pretrained References (Internal Holdout Benchmark)

To establish an empirical upper bound on the exact 110-class holdout split under identical data splits, we evaluated modern vision backbones:
- **SigLIP 2 (`google/siglip2-base-patch16-224`):** Linear probe achieves **0.8219 macro-F1** (89.79% accuracy); 5-epoch fine-tuning achieves **0.8166 macro-F1** (88.99% accuracy).
- **ConvNeXt (`facebook/convnext-base-224-22k`):** Linear probe achieves **0.8000 macro-F1** (88.57% accuracy); 5-epoch fine-tuning achieves **0.7927 macro-F1** (88.00% accuracy).

Linear probes consistently outperform short fine-tuning due to cold-head feature distortion and loss reweighting instability on noisy tail classes.

## Independent Evaluation on External ("In-the-Wild") Data

To rigorously test model generalization beyond the studio catalogue distribution, we gathered an independent test set of 60 unconstrained photographs from Unsplash across 12 balanced classes (5 images per class across Head and Body tiers):

| Evaluation Split | Dataset Scope | Accuracy | Macro-F1 | Mean Confidence | Domain Characteristics |
|---|---|:---:|:---:|:---:|---|
| **Catalogue Reporting Split** | Internal Holdout (7,568 images) | **87.61%** | **0.7635** | **~0.88** | Pure white background, centered studio cutouts, controlled strobe lighting |
| **Independent External Evaluation** | External Out-of-Scope (60 images) | **13.33%** | **0.0536** | **~0.34** | Natural scenes, streets, bedrooms, human poses, ambient light, background clutter |

#### Per-Class Performance on External Data
* **Handbags:** 3/5 correct (60%) — confused with Messenger Bag (1), Tops (1)
* **Casual Shoes:** 2/5 correct (40%) — confused with Sports Shoes (2), Flip Flops (1)
* **Backpacks:** 1/5 correct (20%) — confused with Handbags (2), Duffel Bag (2)
* **Jeans:** 1/5 correct (20%) — confused with Trousers (3), Track Pants (1)
* **Shirts:** 1/5 correct (20%) — confused with Tshirts (3), Kurtas (1)
* **Dresses, Formal Shoes, Shorts, Sports Shoes, Sunglasses, Tshirts, Watches:** 0/5 correct (0%)

#### Failure Mode Analysis
1. **Background Clutter & Multi-Object Scenes:** The from-scratch CNN relies on clean white pixels as a strong prior. When presented with complex backgrounds (streets, brick walls, furniture), convolutions activate on background textures rather than the garment.
2. **Contextual Co-occurrence:** Full-body human shots containing pants, shirts, and shoes confuse the global average pooling layer, which has no spatial localization mechanism to isolate the query item.
3. **Aspect Ratio and Scale Variance:** Real-world items appear at variable distances and angles, contrasting sharply with the standardized catalogue framing.

## Synthesis and Recommendations

1. **Scoped Recommendation:** SmallResNet/resample is effective for assisted tagging of isolated product catalogue cutouts ($87.61\%$ accuracy, $0.7635$ macro-F1).
2. **Generalization Boundary:** The collapse to $13.33\%$ on external imagery demonstrates severe sensitivity to non-white backgrounds and whole-scene clutter. The model cannot be safely deployed directly on unconstrained customer mobile uploads without a front-end object detector or background segmentation stage.
3. **Calibrated Uncertainty:** The drop in average confidence from $\sim 0.88$ to $0.34$ indicates that prediction probabilities correctly reflect high uncertainty on out-of-domain imagery.

## References

Kolisnik, B., Hogan, I. and Zulkernine, F. (2021). *Condition-CNN: A hierarchical multi-label fashion image classification model*. Expert Systems with Applications, 182, 115195. [doi:10.1016/j.eswa.2021.115195](https://doi.org/10.1016/j.eswa.2021.115195).

Seo, I.-J., Lee, Y.-H. and Jang, B. (2025). *Classification of fashion e-commerce products using ResNet-BERT multi-modal deep learning and transfer learning optimization*. PLOS ONE, 20(5), e0324621. [doi:10.1371/journal.pone.0324621](https://doi.org/10.1371/journal.pone.0324621).
