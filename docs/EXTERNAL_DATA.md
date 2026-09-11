# Externally collected data — SG_G3

Three sets were collected. **Two are training data, one must never be trained on.**
The images live on the team Drive at `A2_ExternalData/`; they are not committed. This
file is the provenance, licence and leakage record the brief asks for, and the only
document you need to read before using any of it.

| set | rows | ids | stored size | purpose | source | licence |
|---|---|---|---|---|---|---|
| `ExternalCosmetics` | 1,200 | 900000–901199 | **native crop** | TRAIN | "Cosmetic Images" COCO set, Lanz Vincent Vencer 2022 | **CC0 1.0** |
| `ExternalCosmetics2` | 699 | 910000–910699 | 60×80 | TRAIN | Roboflow Universe, *makeup products detection* | **CC BY 4.0** |
| `ExternalEval` | 261 | 800000–800555 | 60×80 | **EVALUATION ONLY** | Openverse API (`cc0,pdm,by`) | mostly CC BY 2.0 |

Ids start at 800000/900000/910000, far outside the provided range (1163–60000), so an
external row can never be mistaken for a provided one.

## `ExternalEval` — do not train on these 261 images

Their entire value is that no model in this project has seen them. Train on them once
and the independent-evaluation claim in the report becomes false. They are deliberately
*not* catalogue shots — ordinary photographs, objects on tables, garments on people —
so the drop between provided validation and this set measures how much of a model's
skill was about fashion items and how much was about Myntra's photography.

## Attribution obligation

`ExternalEval` is largely **CC BY**, which requires attribution on redistribution, so
`ATTRIBUTION.txt` (per-image creator, licence, source page) **must travel with the
images**. `by-sa` was excluded at collection time on purpose, so no share-alike
obligation attaches to anything here. `ExternalCosmetics` is CC0 and needs no
attribution; the contributor is credited above anyway.

## Leakage

Every set was gated against the provided data before use. Verdict **PASS** for all
three, recorded per set in `images_gate_manifest.json` and per image in
`images_leakage_report.csv`. Flagged candidates were investigated and excluded, not
silently dropped: 1 crop in batch 2, 3 images in `ExternalEval`, each with a
`quarantine/README`. Re-check at any time:

```bash
python src/verify_external_data.py --all
```

The tutor's condition was explicit — *"if you get more data for train, make sure there
is no data leakage to the test set that you are provided"* — so this is the one section
worth re-running rather than trusting.

## Four caveats that belong in the report

**1. Batch 1 is stored at native crop size, not 60×80.** 1,165 distinct sizes across
1,200 files, from 12×39 to 296×294. This is not a pipeline defect: every consumer
standardises on read (scale to fit 60×80, pad white, centred), and standardising once
on read beats resampling twice. Two consequences are real, though — white padding lifts
these crops' border brightness a long way towards the catalogue's, and a few source
crops are tiny (smallest 12×39) so they are upscaled and genuinely blurry.

**2. Batch 2's `gender`, `season` and `usage` are propagated, not observed.** They are
not in the source; they are filled per `articleType` from the provided training data,
and the `label_source` column records that. It is defensible because these are
catalogue conventions rather than visual properties, and they are deterministic in the
provided data — all 12 cosmetics `articleType` values are 100% `season=Spring`. All
seven batch-2 classes are `Women / Spring / Casual`.

The side effect is large. `Spring` is the rarest season and Task 2's bottleneck:
**1,093 → 2,992 train rows, a factor of 2.74**, all of it cosmetics. So any Task 2
improvement must be described as coming from cosmetics, not from the model learning
seasonality — the Spring rows that are *not* Personal Care (Casual Shoes 77, Sports
Shoes 60, Tshirts 45) get no help at all. **The check worth running:** Spring recall
separately on Personal Care and non-Personal-Care validation rows. If only the former
improves, the gain is a categorical shortcut and the report should say so.

**3. The domain gap is measurable and large.** Mean RGB of the outermost 3-pixel frame,
via `src/verify_external_data.py --domain-gap`:

| | border brightness | near-white border | separable from catalogue |
|---|---|---|---|
| provided catalogue | **247.2** | 75.7% | — |
| batch 1, as loaded | 201.2 | 3.7% | 90.4% |
| batch 2 | 120.6 | 0.0% | **99.0%** |
| `ExternalEval` | 122.4 | 2.7% | 96.3% |

A single brightness threshold separates batch 2 from the catalogue 99% of the time, so
a model could learn "darker border → cosmetic" instead of what a cosmetic looks like.
This does **not** invalidate any validation score: external rows join `train` only, and
validation stays 100% provided imagery.

**4. Two starved classes are still unfixed** because neither source annotates them:
`Lip Plumper` (4 images, needs 16) and `Body Wash and Scrub` (1, needs 19).

## Measured effect, so far

On **Task 3** the extra training rows had **no measurable effect on either target**.
Four runs across two platforms; the sign flipped on both `gender` and `usage`. That is
what §7 of the Task 3 notebook predicted before measuring — every external row carries
the majority class of both targets, so there was no reason to expect a gain. Collected,
tested, no effect is a reportable result.

Tasks 1 and 2 have more to gain, because the rows were chosen to fill *their* starved
`articleType` classes: all seven batch-2 classes have **100% of their instances in the
region the graded test set is drawn from** (the 5,829 highest ids), an enrichment of
6.6× — the maximum possible. **Please run the ablation** with and without, one variable
at a time. A marker will ask.

## Using it

```bash
# training rows for Tasks 1, 2, 3 -- adds to TRAIN only, never to validation
from src.external_data import load_external_training_rows
extra = load_external_training_rows()          # 1,899 rows

# the independent evaluation set -- predict, never fit
from src.external_data import load_external_eval
ev = load_external_eval()                      # 261 rows
```

Set `A2_EXTERNAL_DATA` to the folder if it is not auto-detected. The loader searches
Colab and local paths; `src/external_data.py` holds the list.

## Where the detail lives

The long-form record is machine-readable rather than prose, so it can be checked
instead of read:

| question | file |
|---|---|
| licence and creator of one image | `ExternalEval/ATTRIBUTION.txt` |
| why a candidate was rejected | `ExternalEval/labels.csv` (all 553, keep=1/0 + reason) |
| full provenance incl. search phrase | `ExternalEval/provenance.csv` |
| leakage verdict per image | `<set>/images_leakage_report.csv` |
| how a set was built | `src/prepare_external_cosmetics*.py`, `src/collect_external_eval.py` |
