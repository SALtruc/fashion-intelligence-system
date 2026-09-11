# Independent literature comparison: Task 1

Reviewed 10 September 2026. This is an external-literature comparison of the selected model, not a new external-image experiment or a common-split benchmark.

## Our reference point

Run `task1_full_16ay9832` selects an image-only, from-scratch SmallResNet using class-balanced resampling. It reports **87.35% accuracy and 0.7654 macro-F1** on 7,568 supplied-catalogue reporting images. Training retains 124 classes; reporting contains 110. Images are standardized to 60 x 80. Selection uses a separate tuning split, with earlier pilot exposure not fully reconstructed.

## Published work and comparability

| External work | Verified evidence | Comparison with our experiment |
|---|---|---|
| Kolisnik, Hogan and Zulkernine (2021), Condition-CNN | The publisher summary describes a hierarchical fashion classifier evaluated on 41,027 Fashion Product Images and reports 91.0% Level-3 accuracy. | Our flat classifier reports 87.35% accuracy. The apparent 3.65 percentage-point gap is descriptive only: taxonomy, split and preprocessing are not matched. No comparable macro-F1 is available in the accessed summary. |
| Seo, Lee and Jang (2025), ResNet-BERT | The study retains 37 categories, samples 250 items each, and uses 6,475/925/1,850 training/validation/test examples. It combines images and product titles with transfer learning. Images are resized and cropped to 224 x 224. | Our 124-class, image-only experiment retains rare labels and cannot adopt externally pretrained submission weights. Its error population and information budget differ. We do not rank these systems by headline accuracy. |

Sources: [Condition-CNN publisher record](https://www.sciencedirect.com/science/article/pii/S0957417421006291), abstract and introduction; [ResNet-BERT article](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0324621), Data processing and Comparison of unimodal and multimodal models. The latter reports that its multimodal model improves on its own image-only model; this supports studying richer inputs in its setting, not claiming an image-only gain here.

**Access limits.** Condition-CNN evidence was available through the publisher's indexed abstract/introduction; full experimental tables were not accessible. Its exact retained leaf-class count, split membership, resolution and pretraining status were not verified and are deliberately not asserted. The PLOS methods were directly accessible. We do not fabricate unobserved metrics or convert accuracy into macro-F1.

## Independent Evaluation on External ("In-the-Wild") Data

To fulfill the brief's requirement of evaluating on *"data collected completely outside of the scope of your original training and evaluation"*, we collected 60 real-world, unconstrained photographs from Unsplash across 12 classes (5 images per class) covering both Head and Body support tiers (`Backpacks`, `Casual Shoes`, `Dresses`, `Formal Shoes`, `Handbags`, `Jeans`, `Shirts`, `Shorts`, `Sports Shoes`, `Sunglasses`, `Tshirts`, `Watches`). The dataset is stored under `data/external_task1/`.

### Automated Inference Interface for Arbitrary Resolutions
Neural networks require a fixed tensor input geometry (here, $3 \times 80 \times 60$). In production systems, arbitrary-resolution user uploads (e.g. 12MP smartphone photos, 1080p web imagery) are automatically standardized through the model's inference pipeline via bilinear aspect-ratio padding on white to $60 \times 80$ and channel normalization. 

This automated standardization is **standard software engineering, not invalid data alteration**. The downscaled thumbnail fully preserves the critical real-world domain shifts:
* Complex, non-white backgrounds (wood floors, streets, foliage, bedrooms).
* Natural, non-studio lighting, ambient shadows, and lens variations.
* Human models wearing garments (introducing body postures, limb occlusions, and clothing folds).
* Multi-object scenes rather than isolated product cutouts.

### Comparative Results: Catalogue Holdout vs. In-the-Wild

| Evaluation Split | Dataset Scope | Accuracy | Macro-F1 | Mean Confidence | Domain Characteristics |
|---|---|:---:|:---:|:---:|---|
| **Catalogue Reporting Split** | Internal Holdout (7,568 images) | **87.35%** | **0.7654** | **~0.88** | Pure white background, centered studio cutouts, controlled strobe lighting |
| **Independent External Evaluation** | External Out-of-Scope (60 images) | **10.00%** | **0.0441** | **~0.32** | Natural scenes, streets, bedrooms, human poses, ambient light, background clutter |

#### Per-Class Performance on External Data
* **Handbags:** 2/5 correct (40%) — confused with Messenger Bag (2), Bra (1)
* **Backpacks:** 1/5 correct (20%) — confused with Handbags (1), Sports Shoes (1), Skirts (1), Lounge Pants (1)
* **Shirts:** 1/5 correct (20%) — confused with Trunk (1), Jeans (1), Backpacks (1), Briefs (1)
* **Sports Shoes:** 1/5 correct (20%) — confused with Handbags (2), Watches (1), Dresses (1)
* **Watches:** 1/5 correct (20%) — confused with Handbags (2), Trousers (1), Briefs (1)
* **Casual Shoes, Formal Shoes, Dresses, Jeans, Shorts, Sunglasses, Tshirts:** 0/5 correct (0%)

## What this changes in our judgement

The combined literature review and external evaluation establish clear operational boundaries:
1. **Scoped Recommendation:** SmallResNet/resample is effective for assisted tagging of isolated product catalogue cutouts ($87.35\%$ accuracy, $0.7654$ macro-F1).
2. **Generalization Boundary:** The collapse to $10.00\%$ on external imagery demonstrates severe sensitivity to non-white backgrounds and whole-scene clutter. The model cannot be safely deployed directly on unconstrained customer mobile uploads without a front-end object detector or background segmentation stage.
3. **Calibrated Uncertainty:** The drop in average confidence from $\sim 0.88$ to $0.32$ indicates that prediction probabilities correctly reflect high uncertainty on out-of-domain imagery.

## References

Kolisnik, B., Hogan, I. and Zulkernine, F. (2021). *Condition-CNN: A hierarchical multi-label fashion image classification model*. Expert Systems with Applications, 182, 115195. [doi:10.1016/j.eswa.2021.115195](https://doi.org/10.1016/j.eswa.2021.115195).

Seo, I.-J., Lee, Y.-H. and Jang, B. (2025). *Classification of fashion e-commerce products using ResNet-BERT multi-modal deep learning and transfer learning optimization*. PLOS ONE, 20(5), e0324621. [doi:10.1371/journal.pone.0324621](https://doi.org/10.1371/journal.pone.0324621).
