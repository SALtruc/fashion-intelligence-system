# ExternalCosmetics2 — batch 2 of extra TRAINING images

> Copy of the `README.md` that ships inside this set's folder on the team Drive
> (`A2_ExternalData/ExternalCosmetics2/`). Kept in the repo because the images are not
> committed, so without this the provenance, licence and leakage evidence would
> live only in a Drive folder. `../../src/` holds the scripts that built it.
>
> Paths written `../ExternalX/` below are relative to the **data** folder, not
> to this file; the sibling documents here are `ExternalCosmetics.md`,
> `ExternalCosmetics2.md` and `ExternalEval.md`.

COSC2753 Assignment 2, group SG_G3. Second batch of extra-collected training data.
Batch 1 is `../ExternalCosmetics/`; the independent **evaluation** set is
`../ExternalEval/` and must never be confused with either of these.

**This is training data.** 699 crops covering seven cosmetics classes that batch 1
did not reach.

## What is in here

| file | what it is |
|---|---|
| `images/` | 699 JPEGs, 60×80, ids 910000–910699 |
| `external_cosmetics2.csv` | `id, gender, masterCategory, subCategory, articleType, season, usage, source, label_source` |
| `images_gate_manifest.json` | leakage-check result (verdict PASS) |
| `images_leakage_report.csv` | per-image leakage verdict |
| `quarantine/` | 1 flagged crop + README explaining why it was investigated and excluded |

| articleType | provided train | + batch 2 | after |
|---|---|---|---|
| Compact | 4 | 100 | 104 |
| Concealer | 2 | 100 | 102 |
| Foundation and Primer | 12 | 100 | 112 |
| Highlighter and Blush | 4 | 100 | 104 |
| Kajal and Eyeliner | 13 | 99 | 112 |
| Lip Gloss | 6 | 100 | 106 |
| Lip Liner | 12 | 100 | 112 |

## Source and licence

**Roboflow Universe — "makeup products detection"**, an object-detection dataset of
2,076 photographs with 4,754 instance annotations across 31 classes.

**Licence: CC BY 4.0.**

The export contains three splits (train 1,456 / valid 413 / test 207), each with its
own `_annotations.coco.json`. All three are merged before cropping — that split is
Roboflow's, irrelevant to us, and reading only one folder would have discarded about
a third of the available annotations. For classes with four examples that is not an
acceptable loss.

Deliberately **not** Openverse. `../ExternalEval/` is built from Openverse, and
training on the same source we evaluate against would undercut the point of having an
independent evaluation set at all.

## Why these seven classes

Same rule as batch 1: not "fewest images" but **fewest images and concentrated in the
region the graded test set is drawn from.** The provided test set is the 5,829 highest
ids, and all seven of these classes have **100% of their instances** in that region —
an enrichment of 6.6×, the maximum possible.

Two of the nine starved cosmetics classes are still unfixed because this source has
no annotations for them:

- **Lip Plumper** (4 images, needs 16)
- **Body Wash and Scrub** (1 image, needs 19)

## How the crops were produced

```bash
python prepare_external_cosmetics2.py --source D:/makeup_rb --list-classes
python prepare_external_cosmetics2.py --source D:/makeup_rb --per-class 100
```

Crop each annotated object by its bounding box, resize to exactly **60×80** (the
provided dataset's size), assign ids from **910000** — clear of batch 1's
900000–901199 and far outside the provided range 1163–60000.

`--per-class 100` was chosen because it is the largest round number every target
class could supply: `powder` → Compact has only 110 annotations available.

Source class names were mapped explicitly; anything unmapped is **skipped rather than
guessed**. Notably `mascara`, `lip balm`, `bb cream` and `bronzer` are not among the
124 `articleType` values, and `nail polish remover` is a different product from
`Nail Polish`, so all were left out.

## Leakage

```
images_found                : 699
matches_provided_TEST_exact : 0
matches_provided_TEST_near  : 0
matches_provided_train      : 0
clean_and_usable            : 699
verdict                     : PASS
```

Reproduce:

```bash
python src/verify_external_data.py --check Dataset/ExternalCosmetics2/images \
    --labels Dataset/ExternalCosmetics2/external_cosmetics2.csv
```

The first run flagged **one** crop (910405) as matching the provided train set. It was
investigated, not waved through: mean absolute pixel difference to its nearest
"match" was **124.87/255** where a true duplicate sits near 0, and visually the crop is
a dark rectangular cosmetic on a grey surface while its four nearest matches are women
wearing dresses and tops. A hash collision, not a leak. It was excluded anyway so the
shipped set carries zero flags, and it is kept in `quarantine/` with the reasoning.

## Labels — propagated, not observed

`gender`, `season` and `usage` are **not** in the source. They are propagated per
articleType from the provided training data, and the `label_source` column records
this so nobody mistakes them for source data.

This is legitimate because these are catalogue conventions rather than visual
properties, and they are deterministic in the provided data: every one of the 12
cosmetics `articleType` values is **100% `season=Spring`**, and `Personal Care` overall
is 99.6% Spring and 99.8% `usage=Casual`. All seven classes here are
`Women / Spring / Casual`.

### The side effect, which is large and must be reported honestly

`Spring` is the rarest season — 1,093 of 27,596 train rows (4.0%), and the known
bottleneck for Task 2. Both external batches are entirely Spring:

**Spring: 1,093 → 2,992, a factor of 2.74.**

That is a real gain and a real risk in one move. Spring was already 65% Personal Care;
after both batches it is far higher. The model is being taught, hard, that a cosmetic
implies Spring.

It is defensible because that *is* the rule generating the label. But any Task 2
improvement must be described accurately: it comes from cosmetics, not from the model
learning seasonality. The Spring rows that are **not** Personal Care — Casual Shoes 77,
Sports Shoes 60, Tshirts 45 and others — get no help at all.

**The check worth running:** measure Spring recall separately on Personal Care and on
non-Personal-Care validation rows. If only the former improves, the gain is a
categorical shortcut and the report should say so.

## Domain gap, measured

Like batch 1, these crops come out of in-the-wild photographs — here, phone photos of
products on fabric and carpet — so they carry real backgrounds while the provided
catalogue images are cut out on white.

Measured with `src/verify_external_data.py --domain-gap` (mean RGB value of the
outermost 3-pixel frame; "near-white" is that mean above 240):

| | mean border brightness | near-white border | separable from catalogue |
|---|---|---|---|
| batch 2 (this folder) | **120.6** | **0.0%** | **99.0%** |
| batch 1, as loaded | 201.2 | 3.7% | 90.4% |
| `ExternalEval` | 122.4 | 2.7% | 96.3% |
| provided catalogue | **247.2** | **75.7%** | — |

Batch 2 is stored at exactly 60×80, so "as stored" and "as loaded" are the same
measurement here. Batch 1 is not, and its row above is the as-loaded figure — see
`ExternalCosmetics.md`.

A single brightness threshold tells batch 2 from the catalogue **99.0%** of the time
(50% would mean indistinguishable). So a model could learn "darker border → cosmetic"
instead of learning what a cosmetic looks like.

**This does not invalidate the validation scores.** These rows go into `train` only;
validation stays 100% provided catalogue imagery, so a model relying purely on the
background shortcut would score no better there.

External data is now **6.4%** of the training set (1,899 of 29,495 rows). Worth
stating in the report alongside the gap measurement.

Untested and cheap: normalise the crop backgrounds to white and re-measure. Every
outcome is informative — unchanged means the shortcut was not being used, better means
it was costing accuracy, worse means the background carried real signal.

## Effect on the macro-F1 ceiling

The reachable ceiling if every class below a threshold scores F1 = 0, over 124 classes:

| classes under | before both batches | after both batches | ceiling |
|---|---|---|---|
| 3 images | 14 | 13 | 0.887 → **0.895** |
| 5 images | 21 | 18 | 0.831 → **0.855** |
| 10 images | 32 | 27 | 0.742 → **0.782** |
| 20 images | 48 | 38 | 0.613 → **0.694** |

Read this as **permission, not promise** — a class having 100 images makes a non-zero
F1 possible, it does not deliver one.

**38 classes remain under 20 images, needing about 514 more.** The cosmetics vein is
now nearly exhausted; what is left is mostly apparel and accessories
(`Lehenga Choli`, `Nehru Jackets`, `Rain Trousers`, `Baby Dolls`, `Shapewear`, …),
which are harder to source as clean product shots.
