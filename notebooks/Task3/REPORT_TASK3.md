# Task 3 — report contribution

For Trực. **Section A is the main text, trimmed to fit ~1 page**; Section B is appendix
material to use only if the two appendix pages have room. Every number is on the team's
frozen split `splits/train_val_grouped_sha256.csv` (37,745 rows, 15% validation), so it
is directly comparable with the other tasks.

---

## Section A — main text

> ≈1.15 pages as written. **If you need to cut:** the 261-photograph paragraph
> compresses to one sentence, and "What did not work" can lose its last two sentences.
> **Do not cut "The methodological finding"** — it is the strongest content here for the
> judgement mark, and no other task in the report has an equivalent.

### Evaluation framework

`gender` has 5 classes, the smallest holding 483 images; `usage` has 8, the smallest
holding **1**. Accuracy is therefore unusable as the headline: predicting `Casual`
everywhere scores **76.1% accuracy** and **0.108 macro-F1**. All results are macro-F1
over all 8 `usage` classes, against two floors — the majority class (0.141 / 0.108) and
1-NN on 15×20 grayscale pixels with no training at all (0.534 / 0.327). The 1-NN floor
is the informative one: most of the apparent skill on the head classes needs no learning.

### Investigation and ultimate judgement

Three designs were compared with everything else held constant: two independent models
(A), one model on the joint `gender × usage` label (B), and one shared convolutional body
with two heads (C). A led on macro-F1 in all four runs. **The ultimate judgement is
nevertheless C with a class-weighted loss** (289k parameters, plus mirror test-time
augmentation). A's advantage on `gender` is 0.013–0.025 and consistent in sign across
all four runs, so we treat it as real rather than as noise — three fresh seeds put
`gender`'s own spread at only 0.007. But class weighting's gain on `usage` is **+0.06 to
+0.10**, two to four times larger, so C trades a ~0.02 loss on one target for a ~0.08
gain on the other and improves the mean of the two by about +0.03. That, and not an
appeal to measurement error, is why C ships; A would also cost 2× the parameters.

The submitted model scores **gender 0.7202, usage 0.4676**, and is the **median of three
runs rather than the best**; choosing the best on validation would inflate the figure we
then report. Across those runs the design gives gender 0.720–0.750 and usage
0.436–0.472, and those ranges are what it is worth.

### Why the scores are not higher — measured, not assumed

`usage` accuracy is at the ceiling its labels allow. An oracle *given the true
`articleType`* and nothing else reaches **0.8945** accuracy; all six metadata fields
move it to 0.8928, i.e. nowhere; our CNN reaches **0.8972** from pixels alone. The
residual is products of one type carrying different labels — `Tshirts` are 86% `Casual`,
14% `Sports` — a marketing decision that is not photographed.

`gender` is the opposite, and is where the image earns its keep: the best oracle reaches
0.8002 against the CNN's **0.9057**, so the picture is worth **+10.6 accuracy points**
over the whole catalogue description.

That explains the combined headline too. **`both correct` is the product of the two
accuracies to within 0.0023** across all three designs, so it has no slack of its own —
it moves only when a factor moves, and one factor is pinned. Macro-F1 on `usage` has a
second, arithmetic limit: dividing by 8 means every permanently unlearnable class costs
exactly **0.125**, and `Home` (1 training image) and `Party` (10) scored 0.000 in every
run, putting the ceiling at **0.875**. The 0.500 we predicted from class counts before
training was met (four runs: 0.466–0.502).

### Generalisation, honestly

A random validation split flatters the model. The **forward split** — train on low ids,
validate on the highest, which is how the graded test set is drawn — costs 0.138 on
`gender` (0.717 → 0.579) and is our honest estimate of the graded score. It was also the
steadiest number we measured: 0.573–0.582 across four runs on two GPUs.

We also evaluated on **261 independently collected photographs** no model here had seen.
`gender` macro-F1 falls to 0.131. That collapse is real but partly an artefact of the
set's construction: rare classes were deliberately over-collected, making it 32.6%
`Unisex` against the catalogue's 5.4%. The clean figure is the class whose share barely
moved — `usage=Casual` falls **0.933 → 0.746**, and that 0.19 is what we attribute to
the photographs.

### What did not work, and why that is still a result

**The extra collected training data had no measurable effect on this task** — four
measurements across two platforms, sign flipped on both targets. We predicted that
before measuring, because all 1,899 extra rows carry the majority class of both targets,
and we report it rather than dropping it. Three further levers (logit adjustment, mirror
TTA, `articleType` pretraining) each gave a reproducible but tiny **+0.003 to +0.008** on
`usage` and nothing surviving a sign test on `gender`. Two of them correct the class
prior at inference — which is what class weighting does in the loss: **+0.08 in the loss
against +0.008 at inference**, the same idea tenfold apart.

### The methodological finding

Running the identical notebook four times, **two runs of the same code differ by up to
0.036 on a trained model while the untrained baselines reproduce to four decimal
places** — so data, split and metric are identical and the difference is training alone.
Note what that means: with the seed fixed, initialisation and batch order are fixed too,
so the residual comes from nondeterministic GPU kernels rather than from sampling. Three
runs at genuinely different seeds separate the two targets sharply — `gender` spreads
only **0.007**, `usage` **0.047** — and the asymmetry is not mysterious. All of `usage`'s
instability sits in its tiny classes: across those three seeds `Travel` F1 swings
0.000/0.571/0.222 on **three** validation images while `Casual` moves 0.003 on 4,306.
`gender`'s rarest validation class has 66. **So the same headline metric is trustworthy
to three decimals on one target and to barely one on the other**, and any `usage` effect
below ~0.05 has to be argued per class, not from the macro. This retired two conclusions
we had drawn from single runs — that the external data hurt `gender`, and a strict
three-way ranking of the designs — and later retired a third: a 215-image external
`Party` set raised the `usage` macro by +0.007, but `Party` F1 was 0.0000 in all nine
models trained, and the entire apparent gain decomposed onto `Travel`, a class the new
data never touched.

---

## Section B — appendix tables

**B1. Main results, team split, all 8 `usage` classes**

| model | gender | usage | gender acc | usage acc |
|---|---|---|---|---|
| majority class | 0.141 | 0.108 | 0.546 | 0.761 |
| 1-NN, 15×20 grayscale, untrained | 0.534 | 0.327 | 0.771 | 0.777 |
| B — joint label | 0.727 | 0.408 | 0.897 | 0.897 |
| C — shared body | 0.717 | 0.407 | 0.897 | 0.894 |
| A — two models | **0.749** | 0.408 | 0.906 | 0.890 |
| **C + class weighting (submitted)** | 0.720 | **0.468** | 0.867 | 0.869 |

**B2. Metadata oracles — the ceiling a pixel-only model is asked to beat**

| the oracle is given | usage acc | gender acc |
|---|---|---|
| the true `articleType` | **0.8945** | 0.794 |
| all six metadata fields | 0.8928 | **0.8002** |
| the other target | 0.761 | 0.608 |
| *our CNN, pixels only* | *0.8972* | *0.9057* |

**B3. One model, four evaluation regimes**

| | gender | usage |
|---|---|---|
| team split, 15% validation | 0.717 | 0.407 |
| per-target split, 20% validation | 0.687 | 0.406 |
| forward split (highest ids, like the test set) | 0.579 | 0.359 |
| 261 independent photographs | 0.131 | 0.116 |

**B4. Reproducibility — what survived**

| effect | estimate | held on other hardware? |
|---|---|---|
| class weighting, `usage` | +0.081 | yes, +0.061 |
| A − B, `gender` | +0.026 | yes, +0.022 |
| A − C, `gender` | +0.022 | sign only; size varies 0.006–0.037 |
| external data, either target | ~0 | sign flips |
| C versus B | unresolved | sign flips |

**B5. `usage` per class, class weighting off and on** — design C, run 4 of the
notebook (not the submitted seed; the submitted model's own scores are in A)

| class | train n | val n | unweighted | weighted |
|---|---|---|---|---|
| Casual | 24,662 | 4,306 | 0.933 | 0.918 |
| Ethnic | 2,182 | 383 | 0.867 | 0.861 |
| Formal | 1,861 | 343 | 0.783 | 0.770 |
| Sports | 3,300 | 614 | 0.669 | 0.654 |
| Smart Casual | 46 | 9 | 0.000 | 0.250 |
| Travel | 22 | 3 | 0.000 | 0.286 |
| Party | 10 | 3 | 0.000 | 0.000 |
| Home | 1 | **0** | — | — |

---

## Notes for Trực — not for the report

* If Section A is too long, cut from the bottom. The order is deliberate: judgement and
  the ceiling argument first, methodology last.
* **A data-quality item you should know about.** The de-duplicated `ColabDataset`
  changes **two `gender` labels** against the raw file — ids 36762 (`Men` → `Unisex`)
  and 39107 (`Boys` → `Unisex`). Two rows of 37,745, so no metric moves, but the
  de-duplicated labels are therefore *not* a subset of the provided ones, and anyone who
  filters the raw CSV themselves rather than using the zip will get different labels.
* The submitted weights are **not in git** — `artifacts/**` is gitignored by our own
  convention. `task3_gender_usage_C_weighted.pt` (1.2 MB) is on the team Drive and needs
  to go into the Canvas submission.
* `predictions/task3_gender_usage_nguyen.csv` has `gender` and `usage` filled for all
  5,829 test rows in the original order, with `articleType` and `season` left empty for
  whoever merges the four tasks.
