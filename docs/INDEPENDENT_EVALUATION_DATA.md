# External evaluation: data, provenance and scope

Checked 6 September 2026. [02_independent_evaluation.ipynb](../notebooks/Task1/02_independent_evaluation.ipynb)
evaluates the fixed **supplied-only** decoupled ResNet with horizontal-flip TTA.
It loads individual checkpoint banks, not the selected export generically. Its findings
apply to that model and data snapshot; changing the training population or selected model
requires revisiting the evaluation.

## Available collections

| Collection | Local folder | Retained images | Article types |
|---|---|---:|---:|
| dataset1 | [dataset1](../notebooks/Task1/dataset1/README.md) | 1,158 | 3 |
| dataset2 | [dataset2](../notebooks/Task1/dataset2/README.md) | 699 | 7 |

Images, label CSVs and screening records are present locally and are tracked in Git: 1,169
files under `dataset1` and 706 under `dataset2`, which is every file the two folders hold.
Include the required data in the evaluator handover anyway; presence in the repository does
not ensure access in a separately packaged submission.

## Held-out status is specific to the model

The current training notebook uses `load_manifest("articleType")` and the supplied-only
split. The preparation notebook separately exports dataset1-enriched training manifests
under `preprocessed_datasets/task1_dataset1/c447dd49cbcb349c/`, but no current worker loads
them. Preparing those files does not itself expose a model to the images.

Once an enriched model trains on those 1,158 rows, **dataset1 is training data for that
model and cannot be reported as its independent evaluation set**. Dataset2 remains outside
the current training population, subject to the quality and provenance limits below.
Repeatedly using either collection to guide development also weakens a final-test claim.

The evaluation notebook checks SHA-256 intersections against the eligible supplied manifest
and verifies the checkpoint fingerprint. The recorded intersection is zero. This rules out exact
file matches to that manifest, not all related products, all raw excluded files or future
external training rows. The dataset1 technical audit additionally checked supplied raw
train/test and dataset2 with several image representations and a dHash radius of two.
Neither check establishes independent source photographs or products.

The notebook filters missing files and labels outside the model's class space before scoring.
Check its displayed counts against the expected 1,158 and 699; do not silently interpret a
smaller scored subset as the complete collection.

## Label and source limitations

- Dataset1's 1,200 original images received AI-assisted visual screening; 42 flagged images
  and their CSV rows were removed, leaving 1,158. Passing that screen is not independent
  ground-truth confirmation.
- Dataset2 has 148 provisional review flags among 699 images: 105 label-review and 43
  crop-review candidates. Labels have not been corrected. The unflagged remainder is not
  a certified clean test set.
- Dataset1's source tag is `external_cosmetics_coco_v1`; its README records inherited
  attribution and licence claims. The exact upstream archive/version, original image and
  annotation IDs, bounding boxes and mapping procedure remain unresolved.
- Dataset2's Roboflow/CC BY 4.0 attribution is likewise inherited; exact project/version and
  original annotations are absent. Do not present these claims as independently verified.
- Gender, season and usage fields were propagated or have unestablished provenance; they
  are not verified extra supervision for Tasks 2–3.

The full methods, counts and limitations are in [EXTERNAL_DATA_AUDIT.md](EXTERNAL_DATA_AUDIT.md).
No new provenance verification is claimed by this documentation refresh.

## Recorded result and interpretation

The retained evaluation reports **0.0000 top-1 and top-5** across 1,857 external images.
Dataset1 alone also scores zero across its 1,158 images. The recorded comparison on supplied
imagery of dataset1's three article types is **0.9744 top-1**. These observations indicate
poor transfer to these external crops under the supplied-only model.

They do not isolate a single cause: backgrounds, framing, crop quality, label mapping and
source uncertainty can all contribute. The flag rate is not a measured annotation-error
rate, so it cannot prove that label noise has no effect. Nor does the contrast establish a
causal explanation that background alone caused the failure. Describe the collections as
local robustness probes with unresolved provenance, not reproducible public benchmarks.

Before reporting a new evaluation, freeze the model, verify the complete training/evaluation
separation, confirm file/label counts and preserve the source snapshot. Recover the upstream
provenance if possible; otherwise state the unresolved gaps explicitly and ensure evaluator
access to the exact files used.
