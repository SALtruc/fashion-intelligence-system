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

## What this changes in our judgement

The comparison supports a scoped catalogue-assistance recommendation rather than a state-of-the-art claim. Our rare-class score (0.5857) and 14 unobserved reporting classes are central limitations that a reduced-class comparison would conceal. The literature motivates a future hierarchical approach using supplied category labels, but this is an unimplemented research direction, not an improvement delivered by the current model.

The selected ResNet remains fixed. The literature was added after training and is used to contextualize it, not to retune on reporting data. A stronger empirical comparison would require a common taxonomy, identical data partitions, matched input modalities and allowed pretraining, and macro-F1 plus support-stratified errors. Fresh external images would test domain generalization; none were collected or scored in this update.

This addresses the specification's literature-comparison route. It does not establish external-image robustness or remove uncertainty about prior inspection of the internal holdout. The final report should state both distinctions explicitly.

## References

Kolisnik, B., Hogan, I. and Zulkernine, F. (2021). *Condition-CNN: A hierarchical multi-label fashion image classification model*. Expert Systems with Applications, 182, 115195. [doi:10.1016/j.eswa.2021.115195](https://doi.org/10.1016/j.eswa.2021.115195).

Seo, I.-J., Lee, Y.-H. and Jang, B. (2025). *Classification of fashion e-commerce products using ResNet-BERT multi-modal deep learning and transfer learning optimization*. PLOS ONE, 20(5), e0324621. [doi:10.1371/journal.pone.0324621](https://doi.org/10.1371/journal.pone.0324621).
