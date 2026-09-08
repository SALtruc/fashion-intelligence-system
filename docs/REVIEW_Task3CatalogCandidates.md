# Review: `Task3CatalogCandidates_review_ready.zip` + the Task 3 work on `feat/task4-retrieval-v2`

Reviewer: Hoang Nguyen (Task 3 owner) · 08/09/2026

**Verdict — the collection machinery is careful and leakage-free, and the targeting is a
real improvement on the last set. But every one of its six mapping groups teaches a
label the provided dataset contradicts, by ratios from 1:1 to 1554:1. The predictable
effect is damage to `Formal` (343 validation images, F1 0.755) in exchange for a hoped-for
gain on three classes holding 15 validation images between them. Recommend: one paired
run at most, judged per class, and keep the collection work in the report as a measured
negative result.**

---

## Part 1 — the dataset

### What is in it

| | |
|---|---|
| Manifest rows | 840 |
| Images on disk | **753** |
| Sources | `amazon_berkeley_objects` 750 · `myntra_catalog_v2` 90 |
| Target classes | Party 340 · Smart Casual 250 · Travel 250 — **no `Home`** |
| Recorded size | 126–256 px, mean 237 × 224 |
| Licence | ABO: *"metadata is inconsistent across official pages; retain attribution and follow the most restrictive CC BY-NC 4.0 terms"* · Myntra: CC0 |

Dropping `Home` and aiming at `Travel`/`Smart Casual`/`Party` is exactly right — those are
the three rare classes that actually have validation instances, and `Travel|backpack`
matches the provided semantics (provided `Travel` is 100% bags). This is a much better
targeted set than `Task3UsageExternal`.

### Two defects in the manifest

**87 of the 90 Myntra rows never downloaded.** The CSV records it itself:

| source | `image_exists=False` | `True` |
|---|---|---|
| amazon_berkeley_objects | 0 | 750 |
| myntra_catalog_v2 | **87** | **3** |

That matters more than the raw count, because `supervise_usage=True` is set on *exactly*
the 90 Myntra rows and `False` on all 750 ABO rows. So the only rows the pipeline would
trust for supervision are the ones that are missing, and everything present requires
individual review.

**The audit is not done.** All 119 rows in `task3_catalog_audit.csv` carry
`review_status=pending`. The file is ready *for* review, not reviewed — so no accepted
manifest can be produced from it yet.

### Data leakage — clean

Checked all 753 images against all **44,441** provided images (38,612 train + 5,829 test):

- **Exact:** 0 byte-identical files.
- **Near-duplicate** (16×16 normalised-thumbnail cosine + 64-bit dHash): highest cosine to
  any test image 0.976, and inspecting the top 16 pairs by either metric shows no
  duplicate — the closest pair is a **teal duffel bag against a silver wristwatch**, then a
  backpack against a computer mouse. The two metrics never agree on the same pair, which is
  the signature of no true match; a tall dark object centred on white looks like any other
  at thumbnail scale.
- Identifier spaces cannot collide: Myntra `source_id` values are 7–8 digits
  (14025812, 10759170) against the provided range 1163–60000.

**No leakage. The requirement is satisfied.** Worth stating plainly in the report, because
`myntra_catalog_v2` shares a source with the provided dataset and that deserved the check.

### Domain gap — no better than the set that already failed

Adversarial validation after the pipeline's own preprocessing (`ImageOps.pad` to 60×80):

| set | AUC | accuracy |
|---|---|---|
| `Task3CatalogCandidates` (this one) | **0.9844** | 0.964 |
| `Task3UsageExternal` (imaterialist) | 0.9775 | 0.929 |

I expected ABO product shots to sit *closer* to the provided catalogue than imaterialist's
full-body model photos, and tested whether the gap was an artefact of padding square 256×256
images into a 3:4 frame. It is not — centre-cropping instead of padding makes it **worse**
(AUC 0.9946). The gap is intrinsic.

This matters because we already know what a gap this size does. Trained on the imaterialist
set, the model reached **0.88 recall on that set's own held-out images and transferred 0%**
to the provided catalogue: `Party` F1 stayed 0.0000 across all nine models trained.

### The decisive problem — the labels contradict the target

Each mapping group teaches a rule. Here is what the provided training data says about the
same article types:

| taught | n | provided labelling of the same articleType | verdict |
|---|---|---|---|
| `Tops` → **Party** | 81 | Casual **1554**, Ethnic 45, Sports 12, Formal 2, Party **1** | **1554 : 1 against** |
| `Formal Shoes` → **Smart Casual** | 250 | **Formal 585**, Casual 16, Smart Casual **9** | **65 : 1 against** |
| `Backpacks` → **Travel** | 224 | Casual **616**, Sports 83, Travel **11** | **56 : 1 against** |
| `Dresses` → **Party** | 259 | Casual **337**, Party **7**, Formal 1 | **48 : 1 against** |
| `Duffel Bag` → **Travel** | 7 | Casual **80**, Travel 3, Sports 3 | 27 : 1 against |
| `Rucksacks` → **Travel** | 19 | Casual 6, Travel 5 | 1 : 1 |

This is not a collection error. It is that the provided `usage` labels are internally
inconsistent — the same product category carries different labels — so *any* externally
sourced mapping that is correct in the world is wrong against the graded ground truth.
The consequence is directional and predictable:

- **250 formal shoes labelled `Smart Casual` fight 585 provided rows labelled `Formal`.**
  `Formal` has **343 validation images** and currently scores F1 **0.755**. That is a class
  with real weight and real headroom, and this data pulls on it.
- The hoped-for gain sits on `Party` (3 val images), `Smart Casual` (9) and `Travel` (3) —
  **15 images deciding 3/8 of the macro average.**

Measurable downside, unmeasurable upside. That asymmetry is the whole argument.

---

## Part 2 — the Task 3 work on the branch

The methodology is sound and in several places stricter than mine: the immutable
`train_val_grouped_sha256.csv` split, external rows appended only after splitting, paired
bootstrap confidence intervals, and cRT reported honestly as *"crosses zero, not a real
win"*. The `allow_individual_human_labels` change makes the audit gate **stricter**, not
looser — individually reviewed rows only, no group extrapolation. Two issues.

### 1. The metric convention silently favours conservative models

`evaluate_task3_ensemble.py:90-91,109-110` and `refine_task3_decoupled.py:158` call

```python
f1_score(y_true, y_pred, average="macro", zero_division=0)     # no labels=
```

With no `labels=`, sklearn averages over the classes present in `y_true ∪ y_pred` only.
`usage=Home` has **0 validation instances**, so:

- a model that never predicts `Home` is scored over **7** classes;
- a model that predicts `Home` even once is scored over **8**, and eats a hard 0.125.

That is a property of the model's caution, not its quality — and it runs *against* the
long-tail methods on the branch, which exist precisely to make rare-class predictions.
Concretely, on my own shipped model the convention is worth **+0.067**:

| | 8 classes | sklearn default |
|---|---|---|
| my `usage` macro-F1 | **0.4676** | **0.5344** |

**Fix: pass `labels=` with all eight classes, for everyone, before any numbers are
compared.** I raise it because it currently flatters my model, not his.

### 2. The only committed prediction file is a failed arm

`artifacts/task3/task3_usage_independent_weighted_external_usage_focal2_samplersqrt_val_predictions.csv`
covers exactly my 5,661 validation rows, so it scores directly:

| class | val n | that arm | mine |
|---|---|---|---|
| Casual | 4306 | **0.4161** | **0.9157** |
| Ethnic | 383 | 0.4049 | 0.8531 |
| Formal | 343 | 0.4769 | 0.7549 |
| Sports | 614 | 0.4168 | 0.6658 |
| Smart Casual | 9 | 0.0291 | 0.2136 |
| Travel | 3 | 0.0108 | 0.2645 |
| Party / Home | 3 / 0 | 0.0000 | 0.0000 |
| **usage macro-F1** | | **0.2193** | **0.4676** |

`Casual` F1 of 0.416 on 4,306 images is below a majority-class baseline (~0.86): the
sqrt sampler plus focal loss has over-balanced so hard it destroyed the common classes,
and it predicts `Home` 49 times where there are none. This is clearly an ablation, not the
headline model — the doc quotes a 0.6456 combined reference — but it is the only
predictions file in the repo, so **please commit the good arm's `val_predictions.csv`
too**. Three columns (`id,usage_true,usage_pred`) is enough; I can score any arm on the
shared split under a fixed convention in about a minute.

---

## Recommendations

1. **Fix the metric convention first** (`labels=` over all 8 classes). Until then no two
   people's numbers on this task are comparable.
2. **Finish or drop the audit.** All 119 rows are `pending`, and 87 of the 90
   supervision-eligible rows have no image. Neither an accepted manifest nor a training
   arm can be built from the zip as it stands.
3. **If it is run, run it once, paired, and judge it per class** — `Formal` (343 val) and
   `Sports` (614 val) are the only classes stable enough to measure. Watch `Formal` for
   the predicted drop. Do not read the macro: on our measurements a single run's `usage`
   macro moves 0.047 from training noise alone, which is larger than any effect 15
   validation images can demonstrate.
4. **Do not run nine arms on it.** Nine arms against a 0.047 noise band on 15 decisive
   images will produce a best-of-nine number that is selection on validation, not a result.
5. **Keep the collection work in the report as a measured negative result.** The audit
   design, the licence handling, the `image_exists` bookkeeping and a leakage check against
   all 44,441 provided images are worth more marks than a macro-F1 movement that sits
   inside the noise — especially against the rubric line *"explore how the current status
   of the data affects the results"*, which the label-contradiction table above answers
   directly.
6. **Offer:** my GPU is free and everything above was measured on it, so any arm run here
   is directly comparable to the numbers in this document. Colab exhaustion need not block
   the experiment.
