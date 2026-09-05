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
| `images/` | 1,200 JPEGs, 60×80, named by id (900000–901199) |
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
3. Resize to exactly **60×80** — the provided dataset's image size. Matching this
   matters: a different size would mean the model saw external and provided images
   through different amounts of resampling.
4. Assign ids from **900000** upward, chosen to sit far outside the provided id range
   (1163–60000) so an external row can never be mistaken for a provided one.

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
behind the product. The provided catalogue images are cut out on white. Measured on a
balanced 400 vs 400 sample:

| | mean border brightness | share with a near-white border |
|---|---|---|
| these crops | **104.7** | **0.0%** |
| provided catalogue | **248.9** | **97.8%** |

A single brightness threshold separates the two sources **99.6%** of the time (50%
would mean indistinguishable). So a model could in principle learn "dark background →
cosmetic" instead of learning what a cosmetic looks like.

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
