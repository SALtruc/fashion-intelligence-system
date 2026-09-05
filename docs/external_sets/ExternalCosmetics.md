# ExternalCosmetics — extra TRAINING images for three starved classes

> Copy of the `README.md` that ships inside this set's folder on the team Drive
> (`A2_ExternalData/ExternalCosmetics/`). Kept in the repo because the images are not
> committed, so without this the provenance, licence and leakage evidence would
> live only in a Drive folder. `../../src/` holds the scripts that built it.
>
> Paths written `../ExternalX/` below are relative to the **data** folder, not
> to this file; the sibling documents here are `ExternalCosmetics.md`,
> `ExternalCosmetics2.md` and `ExternalEval.md`.

COSC2753 Assignment 2, group SG_G3. This folder is referenced by the report under
"Preprocessing data, including extra-collecting activities". The brief requires any
extra-collected data to be accessible to the evaluator; this README exists so the
folder is self-explanatory rather than 1,200 unlabelled JPEGs.

**This is training data. It is NOT an evaluation set.** See `../ExternalEval/` for
the independent evaluation set — the two exist for different reasons and must not be
confused.

## What is in here

| file | what it is |
|---|---|
| `images/` | 1,200 JPEGs at their **native crop size** (not 60×80 — see "Image size" below), named by id (900000–901199) |
| `external_cosmetics.csv` | one row per image: `id, gender, masterCategory, subCategory, articleType, source` |
| `images_gate_manifest.json` | the leakage-check result (verdict PASS) |
| `images_leakage_report.csv` | per-image leakage verdict, all 1,200 rows |

400 images each of **Eyeshadow**, **Lipstick**, **Nail Polish**.

## Source and licence

Derived from the **"Cosmetic Images"** dataset (COCO format), contributor
**Lanz Vincent Vencer**, 2022 — 15,000 images, 29,990 instance annotations, four
categories.

**Licence: CC0 1.0 Universal (Public Domain Dedication)**
<https://creativecommons.org/publicdomain/zero/1.0/>

CC0 imposes no attribution requirement; the contributor is credited here anyway.

The source's fourth category, `Makeup Brush` (7,372 annotations), was deliberately
**not** used: it is not one of the 124 `articleType` values in the provided dataset,
so adding it would have introduced a label the submission format cannot express.

## How the images were produced

`src/prepare_external_cosmetics.py --source <archive> --per-class 400`

1. Read the COCO instance annotations.
2. Crop each annotated object by its bounding box.
3. Assign ids from **900000** upward, chosen to sit far outside the provided id range
   (1163–60000) so an external row can never be mistaken for a provided one.

## Image size

These 1,200 crops are stored at their **native bounding-box size** — 1,165 distinct
sizes across 1,200 files, from 12×39 to 296×294. None is 60×80.

Batch 2 and the evaluation set *are* stored at exactly 60×80, so this set is the odd
one out. That inconsistency was found late, by an assertion that fired on the channel
statistics, and it is recorded here rather than quietly corrected because the report
should be able to describe the data as it actually is.

**It is not a defect in the pipeline.** Every consumer resizes on read through the
same transform (`preprocessing.standardize_image`, and `load_images` in the Task 3
notebook): scale to fit inside 60×80 preserving aspect ratio, pad with white, centred.
Storing at native size and standardising once on read is in fact marginally *better*
than shipping pre-resized files, which would be resampled twice.

Two consequences that do belong in the report:

* the padding is white, so it lifts these crops' measured border brightness a long way
  towards the catalogue's — see the domain-gap section below, where it changes the
  headline number from 101.5 to **201.2**;
* a handful of source crops are tiny (the smallest is 12×39), so they are upscaled and
  are genuinely blurry. They were kept because a blurred lipstick is still a lipstick
  at 60×80, but a class whose images are mostly upscaled is a fair thing to flag.

Re-check at any time with:

```bash
python -c "from PIL import Image; from pathlib import Path; from collections import Counter; print(Counter(Image.open(f).size for f in Path('Dataset/ExternalCosmetics/images').glob('*.jpg')).most_common(3))"
```

## Why these three classes specifically

Not "the classes with fewest images" — the classes with fewest images **that are also
concentrated in the region the graded test set is drawn from.**

The provided test set is the 5,829 highest ids, and warehouse composition drifts along
the id axis, so a class's overall rarity is not the same as its rarity where it will be
graded.

| articleType | provided train images | how many sit in the test-like id region | enrichment | after adding 400 |
|---|---|---|---|---|
| Eyeshadow | 5 | 5 (100%) | 6.6× | 405 |
| Lipstick | 15 | 15 (100%) | 6.6× | 415 |
| Nail Polish | 19 | 19 (100%) | 6.6× | 419 |

6.6× is the maximum possible value: every single instance of these classes lives in
the test-like region. Training total goes from 27,596 to 28,796 (+4.3%).

## Leakage: what was checked and how to re-check it

The tutor's condition on extra data was explicit: *"if you get more data for train,
make sure there is no data leakage to the test set that you are provided."*

Result — see `images_gate_manifest.json`:

```
images_found                : 1200
matches_provided_TEST_exact : 0
matches_provided_TEST_near  : 0
matches_provided_train      : 0
clean_and_usable            : 1200
verdict                     : PASS
```

To reproduce:

```bash
python src/verify_external_data.py --build-reference          # once
python src/verify_external_data.py --check Dataset/ExternalCosmetics/images
```

The check resizes every candidate to 60×80 and compares it against all 38,612 train
and 5,829 test images two ways: exact 16×16 thumbnail match, and dHash within Hamming
radius 2. Exit code is 1 on FAIL.

**A zero is only meaningful if the detector can produce a non-zero.** The script has a
self-test that feeds it the real provided test folder; it flags 5,829 of 5,829. It has
also been validated against realistic tampering — provided test images upscaled to
480×640 and re-encoded at JPEG quality 92 were caught 4 out of 4, all by the
near-duplicate branch.

```bash
python src/verify_external_data.py --self-test
```

## Known limitation, stated deliberately

These crops come out of in-the-wild photographs, so they carry whatever background was
behind the product. The provided catalogue images are cut out on white.

Measured with `src/verify_external_data.py --domain-gap` (mean RGB value of the
outermost 3-pixel frame; "near-white" is that mean above 240; catalogue sampled at
3,000 images):

| | mean border brightness | near-white border | separable from catalogue |
|---|---|---|---|
| these crops, **as stored** | 101.5 | 0.0% | 98.9% |
| these crops, **as loaded** (60×80) | **201.2** | **3.7%** | **90.4%** |
| provided catalogue | **247.2** | **75.7%** | — |

**Read the "as loaded" row, not the "as stored" one.** These crops are shipped at
their native size, so the standard 60×80 transform pads them with white — which is
most of the difference between the two rows, and the padded version is the only one a
model ever sees. An earlier version of this file reported 104.7 / 0.0% / 99.6% for
this set, which was the "as stored" figure measured with a definition that no longer
exists in the repo; it described a set of pixels nothing is trained on. The command
above is now the definition.

"separable" is the best single-threshold balanced accuracy at telling this set apart
from the catalogue by border brightness alone; 50% would mean indistinguishable. At
90% a model could in principle learn "darker border → cosmetic" instead of learning
what a cosmetic looks like.

**This does not invalidate the reported Task 1 score.** These rows are appended to
`train` only — see `src/train_task1_condition_resnet.py`, where the external frame is
concatenated into `train_df` and nowhere else, behind the opt-in
`--external-cosmetics` flag. Validation stays 100% provided catalogue imagery, so a
model that only learned the background shortcut would score no better on validation.

Untested idea worth an hour: normalise the crop backgrounds to white and re-measure.
If the score holds, the shortcut was not being used; if it improves, the shortcut was
costing accuracy.

## What this data does NOT help

`external_cosmetics.csv` has no `season` and no `usage` column, because the source
provides neither and neither can be read reliably off a cosmetics photograph. So this
data contributes to **Task 1 (articleType)** and **Task 3 gender** only. It does
nothing for Task 2 or Task 3 usage.
