# Task 3 - report contribution

For Trực. **Section A is the main text, trimmed to fit ~1 page**; Section B is appendix
material to use only if the two appendix pages have room. Every number is on the team's
frozen split `splits/train_val_grouped_sha256.csv` (37,745 rows, 15% validation), so it
is directly comparable with the other tasks.

---

## Section A - main text

> ~1.15 pages as written. **If you need to cut:** the 261-photograph paragraph
> compresses to one sentence, and "What did not work" can lose its last two sentences.
> **Do not cut "The methodological finding"** - it is the strongest content here for the
> judgement mark, and no other task in the report has an equivalent.

### Evaluation framework

`gender` has 5 classes, the smallest holding 483 images; `usage` has 8, the smallest
holding **1**. Accuracy is therefore unusable as the headline: predicting `Casual`
everywhere scores **76.1% accuracy** and **0.108 macro-F1**. All results are macro-F1
over all 8 `usage` classes, against two floors - the majority class (0.141 / 0.108) and
1-NN on 15x20 grayscale pixels with no training at all (0.534 / 0.327). The 1-NN floor
is the informative one: most of the apparent skill on the head classes needs no learning.

### Investigation and ultimate judgement

Three designs were compared with everything else held constant: two independent models
(A), one model on the joint `gender x usage` label (B), and one shared convolutional body
with two heads (C). A led on macro-F1 in all four runs. **The ultimate judgement is
nevertheless C with a class-weighted loss** (289k parameters, plus mirror test-time
augmentation). A's advantage on `gender` is 0.006-0.037 over five runs, mean **+0.023**
and consistent in sign, so we treat it as real rather than as noise - six runs of one
configuration put `gender`'s own spread at 0.011. Class weighting's gain on `usage` is
larger but less stable than we first reported: **+0.081, +0.061 and +0.028** on three
platforms, mean **+0.057**. C therefore trades about 0.02 on one target for about 0.06
on the other.

A hyper-parameter sweep looked for a cheaper way to recover that `gender` gap and
appeared to find one - 30 epochs instead of 20 gained **+0.019**, positive on both
seeds - but the gain did not survive being tested on a split that had not selected it.
See "Selection bias, caught in our own work" below; the shipped configuration is
unchanged.

The submitted model scores **gender 0.7202, usage 0.4676**, and is the **median of three
runs rather than the best**; choosing the best on validation would inflate the figure we
then report. Across those runs the design gives gender 0.720-0.750 and usage
0.436-0.472, and those ranges are what it is worth.

### Why the scores are not higher - measured, not assumed

`usage` accuracy is at the ceiling its labels allow. An oracle *given the true
`articleType`* and nothing else reaches **0.8945** accuracy; all six metadata fields
move it to 0.8928, i.e. nowhere; our CNN reaches **0.8972** from pixels alone. The
residual is products of one type carrying different labels - `Tshirts` are 86% `Casual`,
14% `Sports` - a marketing decision that is not photographed.

`gender` is the opposite, and is where the image earns its keep: the best oracle reaches
0.8002 against the CNN's **0.9057**, so the picture is worth **+10.6 accuracy points**
over the whole catalogue description.

That explains the combined headline too. **`both correct` is the product of the two
accuracies to within 0.0023** across all three designs, so it has no slack of its own -
it moves only when a factor moves, and one factor is pinned. Macro-F1 on `usage` has a
second, arithmetic limit: dividing by 8 means every permanently unlearnable class costs
exactly **0.125**, and `Home` (1 training image) and `Party` (10) scored 0.000 in every
run, putting the ceiling at **0.875**. The 0.500 we predicted from class counts before
training was met (four runs: 0.466-0.502).

### Generalisation, honestly

A random validation split flatters the model. The **forward split** - train on low ids,
validate on the highest, which is how the graded test set is drawn - costs 0.138 on
`gender` (0.717 -> 0.579) and is our honest estimate of the graded score. It was also the
steadiest number we measured: 0.573-0.582 across four runs on two GPUs.

We also evaluated on **261 independently collected photographs** no model here had seen.
`gender` macro-F1 falls to 0.131. That collapse is real but partly an artefact of the
set's construction: rare classes were deliberately over-collected, making it 32.6%
`Unisex` against the catalogue's 5.4%. The clean figure is the class whose share barely
moved - `usage=Casual` falls **0.933 -> 0.746**, and that 0.19 is what we attribute to
the photographs.

### What did not work, and why that is still a result

**The extra collected training data had no measurable effect on this task** - four
measurements across two platforms, sign flipped on both targets. We predicted that
before measuring, because all 1,899 extra rows carry the majority class of both targets,
and we report it rather than dropping it. Three further levers (logit adjustment, mirror
TTA, `articleType` pretraining) each gave a reproducible but tiny **+0.003 to +0.008** on
`usage` and nothing surviving a sign test on `gender`. Two of them correct the class
prior at inference - which is what class weighting does in the loss: **+0.08 in the loss
against +0.008 at inference**, the same idea tenfold apart.

### Selection bias, caught in our own work

Section 10.1 warns that tau is chosen by argmax on the validation set it is then scored on, so its
gain cannot come out negative. The hyper-parameter sweep then walked into the same trap
with a different knob, and we can put a number on it. `epochs=30` won the sweep on the
random validation split by **+0.019** on `gender`. Re-run on the **forward split** - the
high-id holdout that mimics the graded test set and had selected nothing - the same change
measured **-0.002**, with the sign flipping across three fresh seeds. We had recorded
"adopt" as the prediction beforehand. It was wrong, and reporting that is the point: the
sweep's winner was **the sweep**, not the model, and the only reason we know is that the
confirmation used a split the selection had never touched. Every "improvement" in section 10
should be read with that in mind, and it is why the sweep is reported as sensitivity and
the shipped configuration is unchanged.

### The methodological finding

Running the identical notebook four times, **two runs of the same code differ by up to
0.036 on a trained model while the untrained baselines reproduce to four decimal
places** - so data, split and metric are identical and the difference is training alone.
Note what that means: with the seed fixed, initialisation and batch order are fixed too,
so the residual comes from nondeterministic GPU kernels rather than from sampling. Three
runs at genuinely different seeds separate the two targets sharply, and six runs of the
same configuration settle it at `gender` **0.011** against `usage` **0.064** - the
asymmetry is not mysterious. All of `usage`'s
instability sits in its tiny classes: across those three seeds `Travel` F1 swings
0.000/0.571/0.222 on **three** validation images while `Casual` moves 0.003 on 4,306.
`gender`'s rarest validation class has 66. **So the same headline metric is trustworthy
to three decimals on one target and to barely one on the other**, and any `usage` effect
below ~0.05 has to be argued per class, not from the macro. This retired two conclusions
we had drawn from single runs - that the external data hurt `gender`, and a strict
three-way ranking of the designs - and later retired a third: a 215-image external
`Party` set raised the `usage` macro by +0.007, but `Party` F1 was 0.0000 in all nine
models trained, and the entire apparent gain decomposed onto `Travel`, a class the new
data never touched.

---

## Section B - appendix tables

**B1. Main results, team split, all 8 `usage` classes**

| model | gender | usage | gender acc | usage acc |
|---|---|---|---|---|
| majority class | 0.141 | 0.108 | 0.546 | 0.761 |
| 1-NN, 15x20 grayscale, untrained | 0.534 | 0.327 | 0.771 | 0.777 |
| B - joint label | 0.727 | 0.408 | 0.897 | 0.897 |
| C - shared body | 0.717 | 0.407 | 0.897 | 0.894 |
| A - two models | **0.749** | 0.408 | 0.906 | 0.890 |
| **C + class weighting (submitted)** | 0.720 | **0.468** | 0.867 | 0.869 |

**B2. Metadata oracles - the ceiling a pixel-only model is asked to beat**

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

**B4. Reproducibility - what survived**

| effect | estimate | held on other hardware? |
|---|---|---|
| class weighting, `usage` | +0.081 | yes, +0.061 and +0.028 |
| A - B, `gender` | +0.026 | yes, +0.022 |
| A - C, `gender` | +0.022 | sign only; size varies 0.006-0.037 |
| external data, either target | ~0 | sign flips |
| C versus B | unresolved | sign flips |

**B5. `usage` per class, class weighting off and on** - design C, run 4 of the
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
| Home | 1 | **0** | - | - |

---

**B6. Hyper-parameter sensitivity** - one knob moved at a time around the shipped
configuration, two seeds each, against the six runs that already measured the default.
Reported as sensitivity: no configuration is adopted on the strength of a validation
score measured on the split this report quotes.

| configuration | `gender` | delta | beyond band? | `usage` | delta | beyond band? |
|---|---|---|---|---|---|---|
| **default** (lr 1e-3, dropout 0.2, bs 256, 20 epochs) | 0.7317 | - | - | 0.4494 | - | - |
| lr 3e-4 | 0.6964 | **-0.035** | yes | 0.4310 | -0.018 | no |
| lr 3e-3 | 0.7489 | **+0.017** | yes | 0.4505 | +0.001 | no |
| dropout 0.0 | 0.7413 | +0.010 | no | 0.4439 | -0.006 | no |
| dropout 0.4 | 0.7273 | -0.004 | no | 0.4423 | -0.007 | no |
| 30 epochs | 0.7510 | **+0.019** | yes | 0.4551 | +0.006 | no |

Noise bands: `gender` 0.011, `usage` 0.064. **Nothing moves `usage` beyond its band and
three configurations move `gender` beyond its** - the same asymmetry section 9.2 found, and for
the same reason: `usage`'s instability lives in classes holding 3 to 9 validation images,
so it cannot resolve an effect this size, while `gender`'s smallest validation class has
66. Two knobs, `lr 3e-3` and `30 epochs`, beat the shipped default on `gender`. Neither
is adopted here, and the reason is measured rather than asserted. `epochs=30` was the
stronger candidate: both its seeds landed above all six baseline runs, an exact rank-sum
`p = 0.0357`. But it was the best of five knobs, and correcting for having looked five
times leaves `1 - (1 - 0.0357)^5 = 0.166`. So it was re-tested on the **forward split**,
which had never selected anything, with three fresh seeds and an adoption rule and a
prediction both fixed in advance:

| split | selected `epochs=30`? | `gender` delta (30 - 20) | per seed |
|---|---|---|---|
| random validation | **yes** | **+0.019** | +0.024, +0.015 |
| forward split | no | **-0.002** | +0.005, **-0.016**, +0.006 |

**+0.019 on the split that chose it, -0.002 on the split that did not.** The prediction
recorded before the run was to adopt; it was wrong, and the rule refused. `epochs=20`
stands and the submitted model is unchanged. `usage` did rise on the forward split in all
three seeds, but by +0.007 against that arm's own 0.007 spread - at the noise floor, and
not what the rule was about.

---

**B7. The two best ingredients, combined** - `D articleType-pretrained` is the
strongest `gender` model the notebook measures and `C weighted` the strongest `usage`
one, and they had only ever been measured apart. This measures them together over
3 paired repeats, **on the forward split** rather than on the validation split this
report quotes. That choice is the whole design: `epochs=30` won the sweep by +0.019 on
random validation and then measured -0.002 on the forward split, so a candidate tested
only where it was chosen cannot be told apart from its own selection. The rule and a
prediction were both fixed before the run.

| repeat | `gender` C | `gender` D | delta | `usage` C | `usage` D | delta |
|---|---|---|---|---|---|---|
| 51 | 0.6096 | 0.6117 | +0.0021 | 0.3502 | 0.3521 | +0.0019 |
| 52 | 0.6086 | 0.6349 | +0.0263 | 0.3530 | 0.3369 | -0.0161 |
| 53 | 0.5936 | 0.5848 | -0.0088 | 0.3523 | 0.3459 | -0.0064 |
| **mean** | **0.6039** | **0.6105** | **+0.0065** | **0.3518** | **0.3450** | **-0.0069** |

Verdict: **KEEP `C weighted`**. Of the three pre-committed conditions - `gender` up in every repeat (no), the mean above the C arm's own spread of 0.0160 (no), `usage` down by no more than its spread of 0.0028 (no) - **0 of 3** held.

The interesting part is not the verdict but where the effect went, and the mechanism
check answers it. The diagnostic localised `gender`'s loss to one class: Unisex, F1
0.529 on random validation and 0.235 on the forward
split, and the sole class where an `articleType -> modal gender` lookup beats the CNN.
Design D carries exactly that signal, so if the story is right Unisex should rise.
It does: `Unisex` F1 moves +0.0152 on average (+0.0039, +0.0245, +0.0173) from a C-arm level of 0.2349. The direction is right and the size is not.

| `gender` class | C | D | delta |
|---|---|---|---|
| Boys | 0.5942 | 0.5789 | -0.0153 |
| Girls | 0.4366 | 0.4638 | +0.0272 |
| Men | 0.8626 | 0.8657 | +0.0030 |
| Unisex | 0.2349 | 0.2502 | +0.0152 |
| Women | 0.8913 | 0.8938 | +0.0025 |

`Unisex` improves in **every** repeat, yet the macro average they feed does not: `gender` is up in 2 of 3 and its mean of +0.0065 sits below the arm's own spread. The gains are small and `Boys` loses -0.0153, so five class-level movements average out to almost nothing. The hypothesis is confirmed where it was made - at the class - and still does not produce a model worth submitting.

There is also a reason to prefer C beyond the averages. Across the same repeats the D arm's own `gender` spread is **0.0501** against C's **0.0160**, 3.1x as wide - its worst repeat (0.5848) falls below every C run. What gets submitted is one training run, not a mean over three, so an arm that swings that far is the worse deliverable even where its average is level. The likely cause is visible upstream: each D repeat inherits whatever its own pre-training produced, and that pre-training is the weak, high-variance task described below.

Two measurements explain the weakness, and neither is visible on a random split.
First, the forward split's training rows cover only **101 of 121** article types against the random split's 121 of 121, so 20 classes have no training example at all and the pre-training task is capped at 0.835 before the model makes a single error; the absent classes are the ones appearing only among the high ids, which
is precisely what the model has to generalise to. Second, the pre-training reached articleType macro-F1 0.2064 here against 0.3104 on the random split - the same code, a harder task. **So design D's
advantage, as measured in section 10.3, is partly an artefact of a random split showing the
pre-training every article type in the catalogue.** The way the graded test set is
actually cut does not. That is a qualification on our own reported result, not on
someone else's.

---

## Notes for Trực - not for the report

* If Section A is too long, cut from the bottom. The order is deliberate: judgement and
  the ceiling argument first, methodology last.
* **A data-quality item you should know about.** The de-duplicated `ColabDataset`
  changes **two `gender` labels** against the raw file - ids 36762 (`Men` -> `Unisex`)
  and 39107 (`Boys` -> `Unisex`). Two rows of 37,745, so no metric moves, but the
  de-duplicated labels are therefore *not* a subset of the provided ones, and anyone who
  filters the raw CSV themselves rather than using the zip will get different labels.
* The submitted weights are **not in git**: `artifacts/**` is gitignored by team
  convention, so `task3_gender_usage_C_weighted.pt` (1.2 MB) is on the team Drive under
  `artifacts/task3`. What is in git is `results/task3/task3_final_metadata.json`, which
  records its sha256, the commit that produced it, the split, the framework version and
  the validation scores - enough to identify the file without shipping it. It needs to
  go into the Canvas submission, since deliverable 2 asks for the models themselves.
* `predictions/task3_gender_usage_nguyen.csv` has `gender` and `usage` filled for all
  5,829 test rows in the original order, with `articleType` and `season` left empty for
  whoever merges the four tasks.
