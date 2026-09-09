# Review: `Task3UsageExternal.zip`

Reviewer: Hoang Nguyen (Task 3 owner) - 08/09/2026 - 508 images, 29 MB

**Verdict - the collection process is sound and leakage-free. Trained and measured
(section 5, nine models, paired three-seed A/B): it does not move `usage` macro-F1, and
`Party` F1 stays at 0.0000 in every single run. The model learns the external Party
dresses almost perfectly (held-out recall 0.00 -> 0.88) and transfers none of it to
the provided catalogue. The `Home` half additionally encodes a label meaning this
dataset does not use. Recommend: keep it in the report as a measured negative
result, do not train the submitted model on it.**

---

## 1. Integrity and provenance - passes, with one file to fix

| Check | Result |
|---|---|
| Manifest rows vs image files | 508 vs 508 - match |
| `sha256` in manifest vs actual file bytes | **508 / 508 verified** |
| Duplicate images inside the set | 0 (the 4 pairs in `within_external_duplicates.csv` were correctly removed) |
| Licence recorded | CC BY-NC 4.0, with source URL and the original competition repo |
| Human precision audit | present, per group, with an explicit 0.85 threshold |

The audit is the strongest part of this work. All 8 `fashion4events` groups scored
**precision 0.000** and were excluded, and `Robes` was excluded at 0.833 for
missing the threshold. Rejecting 9 of 14 candidate groups on measured precision is
exactly the discipline the brief asks for.

**One defect: `provenance.json` is stale.** It reports the pre-audit manifest:

| Field | `provenance.json` | Actual manifest |
|---|---|---|
| `rows` | 583 | **508** |
| `usage_counts` | Party 292, Home 291 | **Party 256, Home 252** |
| `split_counts` | train 495, holdout 88 | **train 429, holdout 79** |

The 75-row gap is the excluded `Robes` group. It also carries a local absolute
path (`D:\Dowloads\...`). The tutor reads this file - please regenerate it.

## 2. Data leakage - clean, at both levels tested

Checked all 508 external images against **all 44,441 provided images** (38,612
train + 5,829 test):

- **Exact:** 0 byte-identical files.
- **Near-duplicate:** 16x16 normalised-thumbnail cosine and 64-bit dHash. Highest
  cosine to any test image was 0.944 - a black cocktail dress against a *deodorant
  spray can*. The closest "matches" are unrelated items that happen to be a dark
  object on a light ground at thumbnail scale; the two metrics never agree on the
  same pair, which is the signature of no true duplicate.
- Identifier spaces cannot collide (`imaterialist:NNNNNN` vs integer ids
  1163-60000), and provided test ids 52003-60000 are disjoint from train.

**No leakage. This requirement is satisfied.**

## 3. Why it cannot raise `usage` macro-F1

Three independent reasons, in increasing order of severity.

**(a) Half the set targets a class with nothing to score against.** In the frozen
split `Home` has **0 validation images** and **1 training image** in the entire
provided set. In the forward split (highest ids, our honest test proxy) `Home` has
**0 instances**. Its F1 is 0 by construction whatever we train.

**(b) That one `Home` instance is a cushion cover.** The provided dataset's
`usage=Home` means *homeware* (`masterCategory=Home`), not clothing worn at home.
`Nightgowns` and `Pajamas` **do not exist as article types anywhere in the provided
data**. Training 252 sleepwear photos as `Home` teaches the model a concept this
label does not denote - it can only produce false positives on test items that are
really `Casual`.

**(c) The rare `usage` labels are annotation noise, not a visual concept.** Within
the provided data:

| articleType | n | how it is actually labelled |
|---|---|---|
| Backpacks | 710 | Casual 616, Sports 83, **Travel 11** |
| Handbags | 1439 | Casual 1426, **Travel 5**, Sports 6, other 2 |
| Dresses | 345 | Casual 337, **Party 7**, Formal 1 |
| Watches | 2252 | Casual 2163, **Smart Casual 21**, Formal 18, **Party 1** |
| Wallets | 829 | Casual 788, Formal 30, **Smart Casual 3** |

The 11 `Travel` backpacks are photographed identically to the 616 `Casual` ones.
No pixel feature separates them, so no volume of additional images can either.
Of 122 article types, **not one has a rare class as its modal `usage`** except
`Cushion Covers` (n=1).

**The ceiling this implies.** Fit a plain `articleType -> modal usage` lookup on
train and score it on validation - strictly more information than a 60x80 image
model can recover:

```
              precision  recall      F1   support
Casual           0.8857  0.9893  0.9346      4306
Ethnic           0.9652  0.8695  0.9148       383
Sports           0.9457  0.5391  0.6867       614
Formal           0.8974  0.4082  0.5611       343
Home             0.0000  0.0000  0.0000         0
Party            0.0000  0.0000  0.0000         3
Smart Casual     0.0000  0.0000  0.0000         9
Travel           0.0000  0.0000  0.0000         3
                                  macro-F1 = 0.3872
```

**All four rare classes score exactly 0.0000 even with perfect product identity.**
The submitted CNN's 0.4676 already *exceeds* this ceiling, because class-weighted
loss makes it gamble on rare classes and land some. `usage` macro-F1 near 0.47 is
not an under-trained model; it is four structurally unlearnable classes each
costing a fixed 0.125 of an 8-class macro average.

**(d) Domain gap, for completeness.** Adversarial validation after the exact
pipeline preprocessing (resize to 60x80 RGB): a plain logistic regression separates
external from provided images with **AUC 0.978 / accuracy 0.929**. The external
photos are full-body model shots at 146 distinct resolutions (mostly 600x600);
every provided image is a 60x80 product shot. A network can shortcut on source
rather than concept - consistent with our independent-source evaluation, where
`usage` macro-F1 fell from 0.4007 on validation to 0.1274 on outside photos.

## 4. Measurability

Even the semantically-valid `Party` half can only be scored on **3 validation
images** (7 in the test proxy). Our measured seed-to-seed spread on `usage`
macro-F1 is **0.036**; flipping one of those 3 images moves the 8-class macro by
roughly 0.025-0.040. Any single-run result is therefore inside the noise floor by
construction, so a paired multi-seed A/B is the minimum that could say anything -
which is what section 5 then runs.

## 5. Measured, not predicted - we ran it

Section 4 argued the effect would be unmeasurable. That is a prediction, so it was tested:
a **paired A/B, three seeds, nine trained models** (~10 min each on an RTX 4070),
`Home` dropped, the `gender` head masked on external rows via `ignore_index` so the
manifest's "do not train gender" instruction is honoured while the shared body still
sees the images. Script: `task3/experiment_party_external.py`, results
`task3/experiment_party_external.csv`.

Three arms, because adding the data changes **two** things at once - the model gains
215 Party images *and* the inverse-sqrt class weight on Party falls from **1.450 to
0.357**, removing the pressure to chase the class in the same step. `party` is the
honest as-shipped comparison; `party_fixedw` freezes the weight vector at the base
arm's values to separate data from reweighting.

| arm | usage macro-F1 | gender macro-F1 | **Party F1** | k (Party predicted) | c (of 3 correct) | external holdout recall |
|---|---|---|---|---|---|---|
| base | 0.4584 | 0.7303 | **0.0000** | 3.3 | **0/3** | **0.00** |
| party | 0.4653 | 0.7287 | **0.0000** | 2.7 | **0/3** | **0.88** |
| party_fixedw | 0.4380 | 0.7293 | **0.0000** | 7.7 | **0/3** | **0.94** |

Paired deltas against base, per seed:

| arm | s42 | s43 | s44 | mean | 95% CI |
|---|---|---|---|---|---|
| party | +0.0315 | -0.0166 | +0.0058 | **+0.0069** | [-0.032, +0.046] |
| party_fixedw | +0.0193 | -0.0732 | -0.0075 | **-0.0205** | [-0.097, +0.056] |

**Both arms flip sign across seeds and both intervals contain zero.** The base arm's
own seed-to-seed spread is **0.0465**, so a mean of +0.0069 is roughly one seventh of
the noise it would have to clear.

**The decisive number is not the macro at all - it is that `Party` F1 is 0.0000 in
all nine runs, and `c = 0/3` in all nine.** Not one of the three validation Party
images was ever classified correctly, with or without 215 extra Party dresses. So
decomposing the delta by class:

| class | val n | contribution to the `party` delta (mean) |
|---|---|---|
| **Party** | **3** | **+0.0000** |
| Travel | 3 | **+0.0179** |
| Smart Casual | 9 | -0.0075 |
| Formal | 343 | -0.0037 |
| Casual / Ethnic / Sports / Home | 4306 / 383 / 614 / 0 | +0.0001 combined |

Every unit of the apparent gain comes from `Travel` and `Smart Casual` - classes the
external data never touched. In the base arm alone, `Travel` F1 swings **0.0000 /
0.5714 / 0.2222** across three seeds (spread 0.5714 on three images) while `Casual`
moves 0.0032 across 4,306. **The "+0.007 improvement" is Travel jitter with a
different label on it.**

**And the external data was learned almost perfectly.** Recall on Trực's own 41-image
held-out Party set goes from **0.00 -> 0.88** (0.94 with the weight held). The model
did not fail to learn party dresses; it learned them nearly perfectly and transferred
**none** of it to the provided catalogue's Party images. That is precisely the
domain gap section 3(d) measured at AUC 0.978 - now confirmed by outcome rather than by
proxy.

Masking worked: `gender` moves -0.0016 and -0.0010, i.e. nothing.

## 6. Recommendations

1. **Fix `provenance.json`** to describe the shipped 508-row manifest.
2. **Drop the `Home` rows** from any training use - wrong label semantics, 0 val
   and 0 test-proxy instances. Keep the files and the audit record.
3. **Do not train the submitted model on this set.** Not on the section 3 ceiling argument
   alone - on the section 5 measurement: nine trained models, `Party` F1 0.0000 in every
   one, `c = 0/3` in every one, and a paired mean of +0.007 against a 0.047 noise
   band that came entirely from a class the data never touched.
4. **Do put this in the report.** A rejected external source with measured
   per-group precision (0.000 across all `fashion4events` groups), a verified
   leakage check against all 44,441 provided images, and a ceiling argument
   explaining *why* more data cannot help is worth more than a +0.01 F1 that would
   sit inside the noise band anyway.
5. **If more `usage` data is still wanted**, the only classes worth the effort are
   `Travel` (100% bags: backpacks, rucksacks, duffels) and `Smart Casual`
   (watches, shirts, formal shoes, wallets) - as **isolated product shots on white**,
   to match the provided domain. But note section 3(c): the provided data already contains
   710 backpacks and 2,252 watches, and labels them `Casual`. The bottleneck is
   label consistency, not sample count.
