# ExternalEval — the INDEPENDENT EVALUATION set

> Copy of the `README.md` that ships inside this set's folder on the team Drive
> (`A2_ExternalData/ExternalEval/`). Kept in the repo because the images are not
> committed, so without this the provenance, licence and leakage evidence would
> live only in a Drive folder. `../../src/` holds the scripts that built it.
>
> Paths written `../ExternalX/` below are relative to the **data** folder, not
> to this file; the sibling documents here are `ExternalCosmetics.md`,
> `ExternalCosmetics2.md` and `ExternalEval.md`.

COSC2753 Assignment 2, group SG_G3. This folder exists to satisfy the brief's §3.3,
*"Independent Evaluation of your Ultimate Judgement"*, which offers two routes:

> * Using data collected completely outside of the scope of your original training and
>   evaluation
> * Comparing your performance to other works in literature

This is the first route.

## Read this first: do not train on these images

**These 261 images must never appear in any training run.** Their entire value is that
no model in this project has ever seen them. Train on them once and that value is gone
permanently, and the independent-evaluation claim in the report becomes false.

`../ExternalCosmetics/` is the opposite — that folder *is* training data. The two are
not interchangeable and the report should never cite one for the other's purpose.

## Why a set like this is needed at all

Every number the project reports comes from Myntra catalogue photography: 60×80,
product centred, cut out on white, studio lighting. That is a narrow and very
consistent visual convention.

A model can score well on that convention and still be relying on it. Held-out
validation cannot detect this, because validation follows the same convention. The
only way to find out is to test on images that break it.

So the images here are deliberately *not* catalogue shots. They are ordinary
photographs — objects on tables, garments on people, things outdoors. The performance
drop between the provided validation set and this set is the measurement: it is how
much of the model's skill was about fashion items, and how much was about Myntra's
photography.

That is also what the brief's HD requirement asks for — *"explore how the current
status of the data will affect to the result of the models"*.

## What is in here

| file | what it is | size |
|---|---|---|
| `images/` | **261** final images, 60×80, ids 800000–800555 (not contiguous) | 1.4 MB |
| `external_eval.csv` | the labels: `id, articleType, gender, season, usage, note` plus licence, creator, landing URL | 56 KB |
| `ATTRIBUTION.txt` | per-image credit — creator, licence, source page | 28 KB |
| `labels.csv` | **all 553** candidates with keep=1/0 and a reason for every rejection | 28 KB |
| `provenance.csv` | full provenance for all 553, including the search phrase used | 176 KB |
| `images_gate_manifest.json` | leakage-check result (verdict PASS) | 1 KB |
| `images_leakage_report.csv` | per-image leakage verdict | 16 KB |
| `quarantine/` | 3 images the gate flagged, plus a README explaining why they were investigated and excluded | 276 KB |
| `rejected/` | the 292 rejected candidates, kept for audit — local only, not uploaded | 54 MB |
| `images_raw/` | the original full-resolution downloads — local only | 43 MB |
| `search_cache/` | cached API pages so collection can resume — local only | 5.1 MB |

## Source and licence

Collected via the **Openverse API** (<https://api.openverse.org>), which indexes
openly-licensed media and returns per-image licence and attribution metadata.

Licence filter was `cc0,pdm,by`. **`by-sa` was deliberately excluded** so that no
share-alike obligation attaches to this derived dataset.

Actual licences in the final 261: CC BY 2.0 dominates, with some CC0, Public Domain
Mark, CC BY 2.5 / 3.0 / 4.0. Sources are mostly Flickr, plus Wikimedia, rawpixel, NASA
and the Cleveland Museum of Art.

CC BY **requires attribution on redistribution**, which is why `ATTRIBUTION.txt`
exists and must travel with the images.

## How it was built

```bash
python collect_external_eval.py --collect --per-class 10   # 553 candidates, 56 classes
python collect_external_eval.py --sheet                    # review sheet
#   ... every candidate reviewed by hand ...
python collect_external_eval.py --finalize                  # apply labels, write credits
python verify_external_data.py --check images --labels external_eval.csv
```

Images were centre-cropped to 3:4 and resized to exactly **60×80**, matching the
provided data. This is not cosmetic: if the eval images differed in geometry, any
performance drop would be confounded by resampling differences and would prove nothing.

**Every one of the 553 candidates was reviewed by a human and 292 were rejected — a
47.2% keep rate.** Keyword search is noisy in ways that are not obvious in advance, and
the rejection reasons are recorded per image in `labels.csv`.

## Leakage

```
images_found                : 261
matches_provided_TEST_exact : 0
matches_provided_TEST_near  : 0
matches_provided_train      : 0
clean_and_usable            : 261
verdict                     : PASS
```

Near-zero was expected by construction — Flickr photographs are not Myntra catalogue
shots — but "by construction" is an argument, not evidence, so the check was run
anyway. Reproduce with:

```bash
python src/verify_external_data.py --check Dataset/ExternalEval/images \
    --labels Dataset/ExternalEval/external_eval.csv
```

**The gate initially reported FAIL**, flagging 3 of the original 556 images, two of
them as near-matching the provided test set. All three were investigated and found to
be hash collisions rather than leaks: no exact match existed (closest mean absolute
pixel difference 15.4/255, where a true duplicate sits near 0), each flagged image tied
*six* provided images at the same distance where a real leak matches one, and visual
comparison showed a brown-strapped watch matched against a pink child's watch, a silver
watch and a black-faced steel watch — different watches sharing a silhouette on white.

They were excluded regardless, and kept in `quarantine/` with the reasoning, so the
judgement can be audited. Measured false-positive rate on genuine photographs:
**3/556 = 0.54%**, and in the safe direction — the gate can raise a false alarm but
cannot issue a false all-clear.

## Composition, and what it can and cannot measure

**52 of 56 attempted classes** survived review. Per-class counts run from 9 (Socks,
Compact) down to 1 (Lehenga Choli), so this supports a solid overall figure and only
indicative per-class observations.

| target | coverage | detail |
|---|---|---|
| `gender` | 5 / 5 | Women 126, Unisex 85, Men 44, Boys 4, Girls 2 |
| `usage` | 7 / 8 | Casual 166, Formal 25, Sports 25, Party 24, Ethnic 12, Travel 5, **Home 4**. No Smart Casual. |
| `season` | **3 / 4** | Summer 193, Fall 50, Winter 18. **No Spring — that season cannot be measured here.** |
| `articleType` | 52 | see `external_eval.csv` |

**Four classes yielded nothing usable**, which is itself a result worth reporting:
`Basketballs` 0/10 (the query returns the sport, never the ball as an object),
`Concealer` 0/10 (concealed storage, a concealed-handgun auction, a bed),
`Jumpsuit` 0/10 (one political protest series, near-identical frames), `Nehru Jackets`
0/6 (crowd and western-suit photographs; the whole open-licence supply is ~12 images).

## Three caveats that belong in the report

**1. The `season` labels are the weakest thing here, and the skew is the annotator's.**
Summer is 74% of this set against 49.6% in the provided data. That gap is not a property
of the images — season-agnostic accessories (wallets, watches, sunglasses, keychains)
were defaulted to Summer because that is the provided data's modal value. A model scored
on `season` against this set is being scored partly against that prior. Report `season`
results with this caveat attached, not as a headline number.

**2. `Spring` is absent**, so the set covers 3 of 4 season classes.

**3. But `usage = Home` has 4 images here, against 1 in the entire provided training
set.** This is the most useful fact in this section: the external set can *measure*
classes the provided data cannot meaningfully *teach*. Same pattern for Travel (5 here
vs 25 provided) and Party (24 vs 13). Where the training tail is thinnest, this set is
comparatively rich — which is exactly what an independent evaluation should contribute.

## The one gap a second person should close

All labels come from a **single annotator**, so there is no inter-annotator agreement
figure. The fix is cheap and worth real marks: have a second person independently label
a random **50** of the 261 without seeing these labels, then report per-target
agreement. That number is evidence of rigour on its own, and it converts caveat 1 above
from an apology into a measured confidence interval. Start with `season` and `usage`;
`articleType` and `gender` will agree closely.

## Running the models over this set

`src/ui_inference.py::Predictor` in the team repo is the entry point. It discovers
checkpoints from `models/` dynamically, so no per-task wiring is needed. Report each
task's metric on this set beside the same metric on the provided validation split — the
gap between them is the finding, not either number alone.
