# Dataset1: external cosmetics training candidates

Audited on 2026-09-05. After approved cleanup, this folder contains 1,158 candidate training crops
for Task 1: 397 Eyeshadow, 384 Lipstick, and 377 Nail Polish.

**Status checked 6 September 2026: all 42 flags removed; versioned training manifests
are prepared, but the current Task 1 training loader still uses supplied data only.
Upstream provenance remains unresolved.**
All 1,200 images received visual contact-sheet screening; flagged images were
rechecked at larger source size and alongside the actual 60x80 padded input.
The user-approved cleanup subsequently removed all 42 flagged images and their
CSV rows on 2026-09-05. Retained images and label values/order are unchanged.
See the [removal manifest](audit/removal_manifest.json). The redundant local
backup and one-off audit script were removed at the user's request; the user
maintains the source elsewhere. Visual records below preserve the original 1,200-image
review; their proposed actions are historical, not pending removals.

## Original visual audit outcome (before cleanup)

| Assigned class | Images reviewed | Wrong target visible | Poor crop / limited view | Ambiguous or obscured target | No obvious issue |
|---|---:|---:|---:|---:|---:|
| Eyeshadow | 400 | 0 | 3 | 0 | 397 |
| Lipstick | 400 | 0 | 0 | 16 | 384 |
| Nail Polish | 400 | 6 | 3 | 14 | 377 |
| Total | 1,200 | 6 | 6 | 30 | 1,158 |

The 42 flags are **3.5% of the images** and are review judgements, not 42 confirmed
incorrect source annotations. Categories are exclusive, so rows are counted once.

- **Wrong target visible:** another product dominates and the labelled product
  cannot be clearly identified. All six are labelled Nail Polish:
  900802, 900883, 900968, 901004, 901067, and 901126.
  For example, `900802.jpg` shows an eyeshadow palette, while
  `901126.jpg` is dominated by lipstick and packaging.
  Check the original bounding box and annotation before deciding to recrop,
  relabel, or exclude. No replacement label has been assigned.
- **Poor crop:** 900206, 901051, 901052, and 901061 have severe blur or too little
  recognizable product content. `900206.jpg` is only 12x39 pixels;
  `901052.jpg` is 25x23.
- **Limited view:** 900181 and 900297 show very little class-specific detail,
  particularly at the model's input resolution. This is a conservative review
  flag rather than a claim that an edge-on product is always unusable.
- **Ambiguous target:** 30 crops contain overlapping cosmetics that obscure the
  intended object. For example, `900455.jpg` is labelled Lipstick,
  but an eyeshadow palette dominates. A multi-object image is not automatically
  bad; the concern here is whether the assigned target can be identified.
- **No obvious issue:** the other 1,158 images passed this visual screen.
  This does not authenticate the product category, prove unique products, or
  guarantee benefit to a model.

The removed examples above are no longer stored in this workspace.
The historical full row-level decisions and reasons are in
[audit/visual_review.csv](audit/visual_review.csv).
The audit's scope, snapshot hashes, limitations, and integration guidance are in
[the shared audit report](../../../docs/EXTERNAL_DATA_AUDIT.md).

## Files and actual schema

| File | Meaning |
|---|---|
| images/ | 1,158 retained JPEG crops; 42 approved IDs removed |
| external_cosmetics.csv | 1,158 retained rows; label values and ordering preserved |
| audit/visual_review.csv | All 1,200 visual decisions, reasons, and proposed actions |
| audit/visual_summary.json | Visual-review totals and the exact source snapshot reviewed |
| audit/image_audit.csv | Per-image integrity, dimensions, hashes, and border brightness |
| audit/technical_summary.json | Recorded technical checks for the cleaned dataset |
| audit/reference_matches.csv | Candidate matches to supplied train/test images and dataset2 |
| audit/near_duplicate_candidates.csv | Internal dHash candidates at Hamming distance at most 2 |
| images_gate_manifest.json | Historical leakage report supplied with the folder |
| images_leakage_report.csv | Historical per-image leakage report, with paths from another machine |

The CSV has nine columns:

`id,gender,masterCategory,subCategory,articleType,season,usage,source,label_source`

All are populated. The source tag is `external_cosmetics_coco_v1`.
Season and usage are explicitly described as propagated per articleType, not
independently observed source labels. The file assigns Women / Spring / Casual
to every row. Gender provenance is not established by the shipped annotations.
Do not treat these values as verified extra supervision for Tasks 2 or 3.

## Geometry and preprocessing

The previous README said all images were 60x80. That was incorrect for these files:

- Retained widths range from 21 to 296 pixels; heights from 30 to 294 pixels.
- None of the 1,158 retained images is natively 60x80.
- All images decode and can be transformed to RGB at 60x80.
- Use the existing `src.preprocessing.standardize_image()` at load time:
  preserve aspect ratio and pad with white. Keep the source JPEGs unchanged.

Different sizes are not themselves a quality failure. The very small and
obscured crops above are flagged because of their visible content, not merely
because resizing is needed.

## Integrity and duplicate screening

The completed audit found:

- 1,158 metadata rows, unique IDs, and 1,158 readable matching JPEGs after cleanup.
- No missing image references or unlabelled JPEGs.
- No internal exact duplicates by file bytes, dimension-qualified RGB pixels,
  16x16 RGB thumbnails, or standardized 60x80 RGB pixels.
- No internal 64-bit dHash pairs within Hamming distance 2.
- No matches under those checks to 38,612 supplied training images, 5,829
  supplied test images, or the 699 images in dataset2.

These checks are **not a certification of correct labels or independent
products**. The contact sheets show many repeated views of recurring palettes,
lipsticks, and bottles. Different poses, crops, lighting, and occlusion can evade
a small dHash threshold. Original source-photo and product identifiers are absent,
so the number of independent products cannot be established.

The historical PASS manifests refer to `D:\g2\Dataset\ExternalCosmetics`.
Their `clean_and_usable=1200` field describes the old leakage gate, not this
visual review. They have been retained as historical records, not overwritten.

Technical results remain under `audit/`; the one-off audit script was removed
after completion. The shared audit report documents the checks performed.
Visual decisions are recorded separately from the technical checks.

## Provenance still to establish

The original supplied README attributes the crops to "Cosmetic Images",
Lanz Vincent Vencer (2022), and reports CC0 1.0, 15,000 images, and COCO
annotations. These are inherited provenance claims, not independently verified
facts about this exact crop collection.

The source archive, original image IDs, bounding boxes, category mapping,
preparation script, and exact upstream dataset/version URL are not included.
The previously referenced preparation and verification commands do not exist
in this repository. Obtain these materials and confirm attribution/redistribution
details before describing the extraction as reproducible.

## Task 1 use

For an enriched model, keep the existing supplied-only split fixed and append accepted
external rows to training only. Those rows must then be excluded from held-out evaluation.
The recorded supplied-only model has not trained on dataset1, so the existing external
evaluation can use it as a robustness probe for that model; see
[the evaluation scope](../../../docs/INDEPENDENT_EVALUATION_DATA.md).
Do not append these rows to the global manifest before calling `make_split`.

| Class | Current supplied train | Current validation | Retained external candidates | Potential combined train |
|---|---:|---:|---:|---:|
| Eyeshadow | 4 | 1 | 397 | 401 |
| Lipstick | 12 | 3 | 384 | 396 |
| Nail Polish | 15 | 4 | 377 | 392 |

Counts use the current audited manifest and Task 1's seed-42, 80/20 split.
Removing all 42 flags leaves 1,158 candidates: training would increase from
30,278 to 31,436 rows, with 7,568 validation rows unchanged. The preparation notebook has exported this split to
`preprocessed_datasets/task1_dataset1/c447dd49cbcb349c/`, with ordered train/validation
manifests, class support and hash metadata. Training adoption and measured enrichment
results remain pending. Run [00_prepare_dataset1.ipynb](../00_prepare_dataset1.ipynb)
for validation/export and its source-aware loading example; do not split its training
export again or replace the global EDA manifest.

The three classes have only eight validation examples combined. Report their
support and individual scores; a change on one image can substantially move
a class's F1. Compare supplied-only and enriched runs on the same validation
rows, keeping evaluation support buckets based on original training counts.

Native external border brightness averages 101.75 on the 0-255 scale under the
documented perimeter calculation. Real backgrounds and repeated products differ
from the catalogue domain. The old README's 99.6% source-separation claim and
test-region enrichment figures were not reproduced and should not be cited as
current findings. Improvement must be measured, not inferred from row counts.

Before distributing a combined run, version the accepted rows, labels, image
hashes, split ordering, and preprocessing configuration; update the shared loader
and regenerate workers. Preserve supplied-only checkpoints in a separate run
directory. The existing independent-job machine layout can remain.
