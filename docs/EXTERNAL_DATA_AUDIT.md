# External cosmetics data audit

Audit date: 2026-09-05. This report covers the actual files in
`notebooks/Task1/dataset1/` and records the earlier dataset2 visual screening.
It does not report a training experiment or measured improvement in accuracy.

## Current cleanup status

On 2026-09-05, all 42 flagged dataset1 images and their CSV rows were removed
with user approval. There are now 1,158 matching image/label pairs: 397 Eyeshadow,
384 Lipstick, and 377 Nail Polish. Retained image bytes and label values/order
were verified unchanged. See the
[removal manifest](../notebooks/Task1/dataset1/audit/removal_manifest.json).
The redundant local backup and one-off audit script were subsequently removed
at the user's request; the user maintains the source elsewhere. Audit results
and the removal manifest are retained. Dataset2 and training notebooks were
not changed by this cleanup.

## Readiness

| Collection | Scope of visual screening | Images flagged | Assessment |
|---|---|---:|---|
| dataset1 | All 1,200 images; candidates rechecked against source and 60x80 input | 42 / 1,200 (3.5%) | 42 flags removed; 1,158 retained candidates pending provenance and integration checks |
| dataset2 | All 699 images; representative flags rechecked individually | 148 / 699 (21.2%) | Needs substantial crop and label review before integration |

These percentages are provisional AI-assisted visual screening results, not
ground-truth annotation-error rates. The two screening registers use different
categories and were not calibrated against independent human reviewers.
Do not interpret the difference as a statistically validated quality ranking.

## Dataset1: original full visual screening (before cleanup)

All 1,200 images were examined in 12 ordered contact sheets, 100 images per sheet.
Sources were displayed with preserved aspect ratio. Candidate images were
rechecked in larger source views beside the model's RGB, white-padded 60x80 input.
Strong rotation, different sizes, dark backgrounds, and multiple objects were
not automatically counted as failures.

| Assigned class | Reviewed | Wrong target visible | Poor crop / limited view | Ambiguous or obscured target | No obvious issue |
|---|---:|---:|---:|---:|---:|
| Eyeshadow | 400 | 0 | 3 | 0 | 397 |
| Lipstick | 400 | 0 | 0 | 16 | 384 |
| Nail Polish | 400 | 6 | 3 | 14 | 377 |
| Total | 1,200 | 6 | 6 | 30 | 1,158 |

The exact decisions are in
[dataset1's visual register](../notebooks/Task1/dataset1/audit/visual_review.csv).
Every row has its assigned class, status, proposed action, reason, and review method.
[The visual summary](../notebooks/Task1/dataset1/audit/visual_summary.json) binds
the review to hashes of the original CSV and image collection. These visual
records remain historical; proposed actions are not pending removals. The refreshed
technical table covers retained images. The local pre-cleanup backup, including
the original technical table, was removed at the user's request.

Removed examples (no longer stored in this workspace):

- `900206.jpg`, Eyeshadow:
  a 12x39 dark blurred fragment.
- `901052.jpg`, Nail Polish:
  a 25x23 blurred fragment with almost no product detail.
- `900802.jpg`, Nail Polish:
  an eyeshadow-palette crop without a clearly identifiable nail-polish bottle.
- `901126.jpg`, Nail Polish:
  lipstick and packaging dominate the crop.
- `900455.jpg`, Lipstick:
  an eyeshadow palette obscures most of the intended lipstick.

"Wrong target visible" describes a mismatch between the assigned class and
usable crop content. It does not distinguish an incorrect original label from
an incorrect bounding box or heavy occlusion, and it does not authorize a
replacement label. "Ambiguous target" is deliberately separate: the intended
product may still be present behind another product.

The original flags proposed `hold_for_review`. The approved cleanup has now
removed all 42 images and their CSV rows, without recropping or relabelling.
The 1,158 retained candidates are not independently certified correct.

## Dataset1: technical audit

The completed audit produced the technical reports in dataset1/audit using
the methods below. The one-off script was removed after completion at the
user's request; the results are retained.

The refreshed audit found 1,158 metadata rows and 1,158 readable JPEGs, with unique
metadata IDs, no missing image references, and no unlabelled JPEGs. Class counts
are 397 Eyeshadow, 384 Lipstick, and 377 Nail Polish. All retained files can be
converted to the common 60x80 RGB input.

Contrary to the old README, none of dataset1's images is natively 60x80.
After cleanup, width ranges from 21 to 296 pixels and height from 30 to 294.
The original CSV includes season, usage, and label_source.

The comparisons use:

1. SHA-256 of file bytes.
2. SHA-256 of RGB pixel bytes prefixed by the source image dimensions.
3. SHA-256 of bilinearly resized 16x16 RGB thumbnail bytes.
4. SHA-256 of aspect-preserving white-padded 60x80 RGB input bytes.
5. A 64-bit horizontal dHash, using a 9x8 Lanczos-resized grayscale image,
   with Hamming distance at most two.

| Comparison | Reference images | Exact matches under each representation | dHash candidate images |
|---|---:|---:|---:|
| Supplied training images | 38,612 | 0 | 0 |
| Supplied test images | 5,829 | 0 | 0 |
| dataset2 images (excluding quarantine) | 699 | 0 | 0 |

There are also zero internal exact duplicate groups under each representation,
and zero internal dHash pairs at this threshold. Details are in
[technical_summary.json](../notebooks/Task1/dataset1/audit/technical_summary.json),
[image_audit.csv](../notebooks/Task1/dataset1/audit/image_audit.csv),
[reference_matches.csv](../notebooks/Task1/dataset1/audit/reference_matches.csv),
and [near_duplicate_candidates.csv](../notebooks/Task1/dataset1/audit/near_duplicate_candidates.csv).

These checks do not prove absence of every possible related image. Repeated
views of recurring products are apparent in the visual review. Pose, background,
occlusion, and crop changes can evade a small hash threshold. No unique-product
count can be established without source-photo and product identifiers.

Mean native-image border brightness is 101.75/255, averaging each image's
top/bottom rows and side columns without double-counting the corners, then
averaging over images. This describes the external set only. The old 99.6%
source-separation claim was not reproduced and is not a finding of this audit.

## Dataset2: previously completed visual screening

The [dataset2 register](../notebooks/Task1/dataset2/audit/visual_review.csv)
records the earlier screen of all 699 files. It is preserved here so that the
reported 148 flags have explicit IDs rather than only a headline count.

| Assigned class | Label-review candidates | Crop-review candidates |
|---|---:|---:|
| Compact | 0 | 0 |
| Concealer | 3 | 6 |
| Foundation and Primer | 1 | 1 |
| Highlighter and Blush | 2 | 19 |
| Kajal and Eyeliner | 34 | 7 |
| Lip Gloss | 14 | 5 |
| Lip Liner | 51 | 5 |
| Total | 105 | 43 |

The candidate sets do not overlap. Examples include a Nivea cream container
labelled Concealer (910117), a moisturizer labelled Foundation and Primer
(910271), nail-polish bottles labelled Lip Gloss (910512, 910549), and exposed
lipstick labelled Lip Liner (910616). Crop examples include patterned fabric
labelled Concealer (910132), pale fabric labelled Foundation and Primer
(910280), and blue fabric labelled Highlighter and Blush (910306).

These 105 label flags are not confirmed corrected annotations; preserve source
labels until checked against the source material. The crop flags include
background-dominated, severely truncated, or unrecognizable product views.
A clean duplicate check does not address these failures.

Dataset2's inherited documentation reports Roboflow provenance and CC BY 4.0.
The exact upstream project/version and original annotations still need to be
established. Claims that all unsuitable classes were skipped are not supported
by the visual findings. The historical leakage manifests have not been changed.

## Integration implications

The current Task 1 supplied-only split has 30,278 training rows and 7,568
validation rows, with 124 training classes but only 110 classes in validation.
After removing the 42 flags, these dataset1 candidate additions remain:

| Class | Supplied training | Validation | External candidates | Potential combined training |
|---|---:|---:|---:|---:|
| Eyeshadow | 4 | 1 | 397 | 401 |
| Lipstick | 12 | 3 | 384 | 396 |
| Nail Polish | 15 | 4 | 377 | 392 |

This would give 31,436 training rows. No integration or training has been
performed. Only eight validation examples cover these three classes, limiting
the strength of any per-class improvement claim.

Before integration:

- Preserve the supplied dataset and audited manifest. Append accepted external
  rows only after the original Task 1 split is frozen.
- Use source-aware relative image paths and the existing common transform.
  Fit data-dependent preprocessing using the declared training population only.
- Keep evaluation support buckets based on original training support so that
  adding rows does not redefine the reported rare-class group.
- Treat propagated gender/season/usage values separately from verified labels.
  Do not infer general Task 2/3 suitability from those populated columns.
- Recover the source archive, exact version, original image and annotation IDs,
  bounding boxes, mapping rules, and provenance/redistribution information.
- Compare supplied-only and enriched models on unchanged validation rows. Keep
  independent evaluation images entirely separate from training.
- Version accepted rows, labels, image hashes, ordered split membership, and
  preprocessing settings across machines. Regenerate workers from the combine
  notebook; separate new checkpoints from the supplied-only baseline.

The existing arrangement of independent jobs on different machines can remain.
The data contract and checkpoint identity need updating before those machines
participate in the same enriched experiment.

## Historical documentation corrections

The updated dataset1 README replaces inaccurate size/schema claims and totals
from another split, removes commands for absent scripts, and distinguishes old
leakage PASS records from current visual quality. Its source/licence claims are
identified as inherited and unverified. No independent ExternalEval directory
or successful integration is claimed.

The original provenance notes and leakage files do not demonstrate 1,200 unique
products, fully reproducible extraction, current model gains, or suitable labels
for every task. None of those claims should be carried into the assignment report
without further evidence.
