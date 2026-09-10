> **Historical document — not the current Task 1 result or run instructions.**
> Use [the current report](REPORT_TASK1.md), [README](../README.md), and [patch record](TASK1_PATCH_NOTES.md). This file is retained as development history.

# Model inventory and remaining experiments

Checked 9 September 2026. This file separates implemented Task 1 models from proposals for
unfinished tasks. The [report draft](REPORT_TASK1.md) contains saved results; the
[pipeline guide](SUGGESTED_PIPELINE.md) describes how to reproduce the current code.

## Task 1: articleType

The source is `notebooks/Task1/02_task1_full_run.ipynb`. All learned models use random
initialization and the same three-way Task 1 split. Images use the shared RGB,
aspect-preserving, white-padded 60×80 transform.

| Model | Role | Reporting macro-F1 |
|---|---|---|
| HOG + linear SVM | Shape-feature baseline, `C` grid of six | **0.6345 — selected** |
| PlainCNN | Neural baseline, learning-rate × weight-decay grid of six | 0.5352 |
| SmallResNet | Deeper neural contender, same grid | 0.4548 |
| Paired stratified bootstrap | The noise band every comparison above is read against | all three pairs exclude zero |

Selection is on tuning macro-F1 alone, on an equal budget: six configurations per family at
12 epochs, the best in each confirmed at 40, then all three refit and scored once on the
reporting split. There is no bespoke extra stage for any one family.

Retired with the unequal-budget layout: the decoupled stage-2 classifier, its sampler sweep
and learning-rate × sampler grid, horizontal-flip TTA, the logit-adjusted and multi-task
ResNet variants, and the majority/stratified-random reference rows. Their recorded numbers
came from a protocol that no longer exists and are not comparable to the table above.

Worth trying next, in rough order of expected value: a pretrained backbone as a clearly
labelled comparison (the assignment forbids one as a submitted model but permits it as a
comparison), HOG features fed to a non-linear classifier, and a HOG/CNN ensemble — the two
families fail on different classes, so their errors are not fully correlated.

## Remaining tasks: proposals, not implementations

The Task 2, Task 3 and Task 4 notebook files are empty. No final models or metrics for them
are supplied in this snapshot.

| Task | Candidate comparison | Evaluation |
|---|---|---|
| Season | Majority reference, scratch CNN, scratch residual CNN | Macro-F1, accuracy, per-class recall; test whether the metadata label is visually predictable |
| Gender and usage | Independent CNN heads versus a shared backbone with separate heads | Score each target separately and check whether sharing harms either target |
| Visual search | Classification embeddings versus a metric-learning model | Define relevance, exclude query/self matches, report Precision@K, Recall@K and mAP@K |

Use the existing preprocessing and target-specific split contract. Add architectures only
with a stated hypothesis and comparable evaluation budget. Record initialization, augmentation,
loss, optimizer, learning rate, batch size, epochs, checkpoint rule, training time, model size,
inference latency and seeds. Pretrained models may be clearly labelled comparisons only;
submitted final models must comply with the assignment's no-pretrained-weights constraint.
