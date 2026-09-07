# Model inventory and remaining experiments

Checked 6 September 2026. This file separates implemented Task 1 models from proposals for
unfinished tasks. The [report draft](REPORT_TASK1.md) contains saved results; the
[pipeline guide](SUGGESTED_PIPELINE.md) describes how to reproduce the current code.

## Task 1: articleType

The current source is `notebooks/Task1/01_task1_article_type.ipynb`. All learned models use
random initialization and the supplied-only fixed Task 1 split. Images use the shared RGB,
aspect-preserving, white-padded 60×80 transform.

| Model or experiment | Current role |
|---|---|
| Majority and stratified-random predictors | Non-learning references |
| HOG + linear SVM | Shape-feature baseline; C × class-weight grid available |
| PlainCNN | Neural baseline; learning-rate × weight-decay grid available |
| SmallResNet, plain cross-entropy | Stage-1 learned representation |
| SmallResNet with decoupled classifier | Frozen-backbone, class-balanced stage-2 finalist |
| Decoupled ResNet + horizontal-flip TTA | The selected model; the TTA row is scored in Section 7.1 |
| Paired bootstrap over validation rows | Section 7.2's noise band; the bar every later comparison is read against |
| Stage-2 sampler sweep and learning-rate × sampler grid | Investigate retraining strength; a qualifying sweep model can enter final selection |
| Logit-adjusted and multi-task ResNet | Historical results retained; removed from the current worker job set |

The current `FINAL_CHOICE = "auto"` compares the available decoupled ResNet, its flip-TTA
variant and any qualifying sweep candidate by validation macro-F1. It does not automatically
promote every tuning-grid winner. Review class-level errors, the Section 7.2 bootstrap band and
cost before finalizing. Flip TTA uses two forward passes per image.

The saved result table predates the current four tuning grids. Existing historical rows are
not evidence that the current grids completed. Dataset1 preparation has produced manifests,
but no enriched training result is established by those exports.

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
