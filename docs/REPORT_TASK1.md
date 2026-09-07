# Fashion Intelligence System — Task 1: Article Type Classification

**COSC2753 Machine Learning · Assignment 2 (2026B)**

Group *<number>* — *<Name (sID)>, <Name (sID)>, <Name (sID)>, <Name (sID)>*

> **Formatting note before submission.** The spec allows 5 pages of text at 11 pt, single
> column, plus 2 pages of appendices; the cover page and reference list are excluded. Content
> past page 5 is not marked. This draft covers **Task 1 only** — Tasks 2–4 must be folded in and
> the whole cut to length. Figures referenced here live in `outputs/figures/`.

> **Evidence status, 6 September 2026.** This is a draft of the recorded supplied-only
> experiment, not a report of the newly added tuning grids or dataset1 enrichment.
> `task1_results.csv` includes historical logit-adjusted and multi-task runs that are no
> longer current jobs. Dataset1 manifests are prepared but unused by current training.
> Tasks 2–4 and final prediction are empty placeholders. Recheck quantitative diagnostics
> against retained executed outputs before incorporating them into the final report.

---

## 1. The problem, and the metric that defines it

Given a 60×80 product image, predict its `articleType`: one of 124 classes over 37,846 labelled
rows. The distribution is the defining feature of the problem, not a detail of it. `Tshirts`
holds 6,584 rows; the rarest class holds one. That is a **6,584:1 imbalance**, and it dictates
every subsequent choice.

Accuracy is unusable as a primary metric here, and we can show it rather than assert it: a
classifier that always predicts `Tshirts` reaches **17.4% accuracy at 0.0027 macro-F1**. Any
metric that rewards that model is measuring the wrong thing. We therefore report **macro-F1 over
the classes present in validation** as the primary metric, with accuracy, balanced accuracy,
weighted F1 and top-5 alongside.

Because two models with identical macro-F1 can have opposite head/tail profiles, every result is
additionally broken down by **training-support bucket** — head (≥1000 rows), body (100–999),
tail (10–99), rare (<10). Buckets are assigned from *training* support so the grouping is a
property of the learning problem rather than of the particular validation draw.

**The split** is a fixed, group-aware, stratified 80/20 partition: 30,278 training and 7,568
validation rows. Whole `group_id` values stay on one side so byte-identical images cannot
straddle the split. Classes with fewer than two groups go entirely to training, since a class
with a single example cannot also be evaluated; 14 of the 124 classes consequently have no
validation example and are excluded from every macro average. They are unmeasured, not verified.

## 2. Data and preprocessing

An audit notebook (`00_eda_and_preprocessing.ipynb`) validates every image, detects exact
duplicates by SHA-256, collapses duplicate groups, and emits a manifest. Two findings drove
downstream decisions:

- **Article types separate largely by silhouette.** This motivated a HOG + linear SVM baseline
  that encodes shape directly, rather than a trivial baseline that would have been easy to beat
  and uninformative.
- **The portrait frame carries class cues at its top and bottom.** Random cropping was therefore
  excluded from the augmentation policy; it would destroy the cue the EDA identified.

Images are decoded once through a shared transform into a `uint8` cache. **Normalisation
constants are fitted on training rows only** and persisted with the model, because inference
must reproduce them exactly. Augmentation — horizontal flip, ±10° rotation, 8% translation,
colour jitter — is applied per batch on-device with independent parameters per sample. Label
smoothing of 0.05 is justified by the label noise the audit documented.

## 3. Models investigated

The saved comparison table contains the following supplied-only approaches, including historical ablations.

| Model | Macro-F1 | Top-1 | Rare-bucket F1 |
|---|---|---|---|
| Baseline: majority class | 0.0027 | 0.1740 | 0.000 |
| 1. HOG + linear SVM | 0.6343 | 0.8056 | 0.403 |
| 2. CNN from scratch | 0.6967 | 0.8787 | 0.508 |
| 3a. ResNet, plain cross-entropy | 0.7536 | 0.8844 | 0.562 |
| 3. ResNet + decoupled classifier | 0.7693 | 0.8774 | 0.621 |
| 3b. ResNet, logit-adjusted (τ=1) | 0.6873 | 0.5525 | 0.488 |
| 5. Multi-task ResNet + softened decoupling | 0.7811 | 0.8835 | 0.651 |
| **4. Ensemble ×3 + flip TTA** | **0.8010** | **0.8936** | **0.679** |

**HOG + linear SVM** establishes what silhouette alone buys: 0.634 macro-F1. That is the bar a
learned representation has to clear to justify itself.

**A plain CNN** is kept deliberately simple so that the ResNet's gain is attributable to two
named changes rather than to accumulated tricks.

**A small ResNet** uses a stride-1 3×3 stem rather than the ImageNet 7×7 stride-2 stem plus max
pool, which would reduce an 80-pixel input to 20 pixels before the first residual block. No
pretrained weights are used anywhere, per the assignment constraint.

**Decoupled classifier retraining** (Kang et al., 2020) is the key long-tail treatment: train the
representation instance-balanced, then retrain only the classifier under class-balanced sampling
with the backbone frozen. It costs one forward pass per epoch and buys +0.016 macro-F1, almost
all of it in the tail and rare buckets.

**Logit-adjusted cross-entropy** (Menon et al., 2021) is reported as an *ablation, not a
candidate*: it trades 33 points of top-1 accuracy for a tail gain that decoupling achieves more
cheaply. Including it as a rejected option is the point — it isolates which change paid.

**A multi-task ResNet** tests whether supervising `subCategory` alongside `articleType` helps.
It does, modestly (0.781), and is the closest single-model rival to the ensemble.

## 4. Hyper-parameter setting and tuning

The saved run and the current source have different search coverage. The historical
searches below explain the saved report; current grid outputs must be reported separately.

**Historical learning-rate search**: 3 log-spaced rates × 2 architectures, at a reduced 15-epoch budget with
early stopping disabled so every arm trains equally. The result is deliberately *not* acted on
for the ResNet: 3e-3 leads the deployed 1e-3 by 0.0037 macro-F1, against a measured seed spread
of 0.0073 — half a standard deviation of the variation the same configuration produces on its
own. Changing the rate on that evidence would be fitting the search's own noise. For the CNN the
gap is real (0.0806) and is recorded as a limitation: the CNN is the plain reference for the
attribution claim, and tuning the reference but not the finalist would bias the comparison.

**Sampler-strength sweep, not yet validly executed**: five values of `power` interpolating
between a uniform sampler and full class balancing. This separates *retraining the head* from
*rebalancing it* — the two are bundled in Section 6 and in most reports of Kang et al. The
figures previously quoted here (0.7424 uniform against 0.7693 rebalanced, so +0.027 attributable
to the rebalancing) come from a run whose checkpoints were quarantined under
`models/task1/checkpoints_invalid/`: every arm scored at chance. **The decomposition is the
claim the sweep is designed to settle, not a result it has settled**, and the report must not
quote those numbers until the `sweep` worker has been rerun.

**Historical post-hoc logit adjustment**: a 21-point τ grid, **fitted out-of-fold** on disjoint
halves of validation with the selection bias measured explicitly. It came out at +0.0000, and τ
was left at 0. This search result alone does not establish calibration.

**Current grids, results pending.** The source now implements HOG C × class weighting,
CNN and ResNet learning rate × weight decay (12 epochs per arm), and stage-2 learning rate
× sampler power. The expected `task1_hog_search.csv`, `task1_cnnsearch.csv`,
`task1_lrsearch.csv` and `task1_stage2_grid.csv` are absent from this snapshot. The old
`task1_lr_search.csv` is a different experiment. No completed joint-grid interaction or
new winner is claimed. See [the pipeline](SUGGESTED_PIPELINE.md) for exact grids.

## 5. Ultimate judgement

**We recommend the three-seed ensemble under horizontal-flip test-time augmentation.**

1. **Performance.** 0.8010 macro-F1 and 0.8936 top-1, against 0.6967 and 0.6343 macro-F1 for the
   CNN and HOG baselines.
2. **Where the gain sits.** Against plain cross-entropy the improvement is monotonic in scarcity —
   head +0.011, body +0.012, tail +0.047, **rare +0.117**. The head bucket is effectively
   unchanged, so the macro-F1 gain is bought almost entirely in the tail.
3. **What it costs.** Not accuracy — the ensemble improves both metrics. The cost is compute: six
   forward passes per image against one. For a catalogue tagged once at ingest, that is
   defensible; for per-request inference it would not be.
4. **Error severity.** 805 errors against the CNN's 918 (10.6% vs 12.1%). The share staying inside
   the correct `subCategory` is 71.3% against 72.3% — **a wash, and we do not claim the errors
   became systematically cheaper**. What improves is their number and reach: cross-family error
   rate falls from 3.36% to 3.05%.
5. **Reliability.** ECE 0.0305 supports an operating policy: 82.6% of the catalogue auto-tagged
   at 95% accuracy, or 59.1% at 98%, with the rest routed to review.
6. **Is the margin real?** Two different noise sources, two different answers. Against *seed*
   variation the ranking is safe: the margin over the seed-42 decoupled ResNet is 0.0316 against a ResNet seed
   standard deviation of 0.0073. The margin over the best saved single model (multi-task)
   is smaller, at 0.0199; the ResNet seed study does not measure multi-task variability. Against *evaluation-set sampling*
   noise it is not: bootstrapping the 7,568 validation rows 1,000 times gives a 95% CI of
   **[0.7676, 0.8280]** on the headline figure, and the ensemble beats the multi-task rival in
   only 80% of resamples. **We report the ensemble as the better model but not as significantly
   better than the runner-up**, which the single-number comparison would have hidden.
7. **Is the choice inflated by selecting on the same split we report?** Measured, not assumed.
   Choosing the winner on a random half of validation and reporting it on the held-out half over
   200 halvings gives a mean optimism of **+0.0048** — small — and the ensemble is selected in
   **178 of 200** halvings. The choice is robust even though its individual margin is not
   significant.

## 6. Independent evaluation

**Published comparison remains to be verified.** Earlier notes cite Condition-CNN and a
public multi-task ResNet50 result. Confirm each original source, label space, split,
initialization and metric before quoting its score. Different protocols do not support a
numerical claim about the cost of dropping pretrained weights or a head-to-head ranking.

**Against data collected outside the project.** We evaluated the deployed ensemble on 1,857
COCO-derived cosmetics crops of article types the model knows, with no recorded byte-identical match to
the eligible supplied manifest. **Top-1 is 0.0000. So is top-5.** Mean probability on the true class is
0.0024 — *below* the 1/124 uniform prior — while the model stays 35% confident in its wrong
answer, and 18.7% of those images would clear a 50% confidence gate and be filed silently.

The same ensemble scores **0.9744** on supplied imagery of the same three classes, which indicates
poor transfer to the external collections, subject to their label and provenance limitations. The supplied catalogue is white-background
studio photography (78% near-white pixels); the external crops are objects in context (26%).
Errors concentrate on `Handbags`, `Bra`, `Briefs` — large, centrally-framed silhouettes — which
is consistent with sensitivity to framing and silhouette. These observations do not isolate
a cause: backgrounds, cropping, label mapping and source uncertainty may all contribute.

[Provenance and evaluation scope](INDEPENDENT_EVALUATION_DATA.md) limit the claim:
exact-hash separation does not prove independent products, dataset1 screening does not certify
labels, and upstream provenance is unresolved. Dataset1 may be a held-out probe for this
supplied-only ensemble, but must be excluded from evaluation if used for enriched training.
The evaluation notebook loads fixed ensemble checkpoints and must be revisited if selection changes.

## 7. Limitations

- **Split variance is unestimated.** The seed study varies optimisation with the split fixed. The
  bootstrap CI quantifies evaluation-set sampling noise but not the variation from a different
  validation draw, which would require repeated group-aware resplitting and retraining.
- **Selection and reporting share a split.** Early stopping, the learning-rate choice and the
  model choice all read the validation set the headline is quoted from. Section 5.7 bounds the
  resulting optimism at +0.0048 but does not eliminate it.
- **14 of 124 classes are unmeasured**, having no validation example, and 5 of the 110 scoreable
  classes score zero F1. `Suits` cannot be predicted at all — its only record has no image.
- **The CNN baseline is untuned by choice**, so the ResNet's margin over it is an upper bound.
- **Exact-duplicate detection only.** Alternate views and colourways of the same product are not
  detected by SHA-256.

## 8. What we would do next

Repeated resplitting to separate split variance from seed variance; a nested selection protocol
so the reported figure is never the one selection maximised; and — the highest-value item by
some distance — **domain augmentation**. The independent evaluation shows the system is bounded
to catalogue-style photography. Background randomisation and scale jitter during training are
the obvious first attempt at making the deployed claim wider than the one we can currently
defend.

## References

Kang, B., Xie, S., Rohrbach, M., Yan, Z., Gordo, A., Feng, J., & Kalantidis, Y. (2020).
*Decoupling representation and classifier for long-tailed recognition.* ICLR.

Kolisnik, B., Hogan, I., & Zulkernine, F. (2021). *Condition-CNN: A hierarchical multi-label
fashion image classification model.* Expert Systems with Applications, 182.

Menon, A. K., Jayasumana, S., Rawat, A. S., Jain, H., Veit, A., & Kumar, S. (2021).
*Long-tail learning via logit adjustment.* ICLR.

---

## Appendix A — figures

`outputs/figures/`, regenerated by the current notebooks: augmentation preview (01), training
histories (02), head/tail trade-off (03), sampler sweep (04), confusion matrix (05), per-class F1
against support (06), calibration (07) from `01_task1_article_type.ipynb`; domain shift (10) and
confidence under shift (11) from `02_independent_evaluation.ipynb`.

Saliency (08) and learning-rate search (09) are still on disk but **no current notebook produces
them** — the saliency section and the old single-axis learning-rate figure were both removed. Do
not cite them in the report unless the code that draws them is restored.

## Appendix B — reproduction

**No result table is present in this repository, and no recorded output survives.** `models/`
does not exist, so `task1_results.csv` and every grid CSV including `task1_stage2_grid.csv`
are absent and must be produced by a run. The executed workers that once held the cell output
behind these numbers have been deleted, along with the stale figures under `outputs/figures/`.
**Every figure quoted in this report is therefore unsourced within this repository** and must
be re-derived from a fresh run before submission; the numbers below stand only as a record of
what a superseded revision reported.

The supplied-only `RUN_FINGERPRINT` is `e6b15f5c51de`. It checks selected recipe/data summaries,
not complete image identity or every runtime setting. The current code enables CUDA AMP;
the saved config records AMP disabled. Matching fingerprints therefore do not establish
bit-identical execution or guarantee a valid trained backbone. Preserve the invalid-sweep
quarantine and follow [the worker guide](../notebooks/Task1/PARALLEL_RUN.md).
