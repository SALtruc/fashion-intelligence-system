# Artifacts

This folder holds **generated, reproducible-by-reference files** that are needed
to run, evaluate, or hand over a model, but are normally too large or too
frequently replaced to store in Git. The folder contents are ignored by Git;
share the actual files through the team's agreed Drive location instead.

Keep artifacts separated by task:

```text
artifacts/
├── task1/
├── task2/
├── task3/
└── task4/
```

## What belongs here

Examples include:

- preprocessing/tokeniser/label-encoder configs and fitted transformers
- exported inference pipelines (`.json`, `.pkl`, `.joblib`)
- trained model checkpoints and weights (`.pt`, `.pth`, `.keras`, `.h5`)
- NumPy arrays such as embeddings, feature matrices, or cached predictions (`.npy`, `.npz`)
- experiment metadata, training history, and evaluation results (`.json`, `.csv`, `.yaml`)

Do **not** put source code, raw datasets, notebooks, or final prediction CSVs
here; those have their own repository locations.

## Naming and versioning

Use descriptive, lowercase names. Include the task, model/preprocessing name,
split or dataset version, and a version or timestamp when it helps distinguish
experiments.

```text
task2/resnet18_v3_best.pth
task2/resnet18_v3_preprocessing.json
task2/resnet18_v3_label_encoder.joblib
task2/resnet18_v3_metrics.json
task3/tfidf_word_1-2gram_v1.joblib
```

Use `best` only for the checkpoint selected by the agreed validation metric.
Avoid vague names such as `model_final.pth`, `new.pkl`, or `test.npy`.

## Required handover information

For every model or pipeline shared in Drive, add a small matching metadata file
(for example, `resnet18_v3_metadata.json`). At minimum record:

```json
{
  "artifact": "resnet18_v3_best.pth",
  "task": "task2",
  "created_at": "2026-09-02",
  "git_commit": "<commit SHA>",
  "dataset_version": "<dataset/checksum or description>",
  "split": "splits/<shared split file>",
  "preprocessing": "resnet18_v3_preprocessing.json",
  "labels": "resnet18_v3_label_encoder.joblib",
  "framework": "PyTorch <version>",
  "validation_metrics": {"macro_f1": 0.0, "accuracy": 0.0},
  "notes": "Training command, seed, and any important assumptions."
}
```

The `git_commit`, split, preprocessing configuration, and label mapping are
essential: weights alone are not a reproducible model. Use the shared
stratified split in `splits/` when reporting validation results.

## Sharing checklist

Before asking another teammate to use an artifact:

1. Upload the artifact and its metadata to the matching Drive/task folder.
2. Confirm the metadata names every required companion file.
3. Test loading it in a clean session using the recorded preprocessing and
   labels.
4. Record macro-F1 and accuracy, plus the experiment details, in the shared
   experiments sheet.
5. Tell the team which file is the current recommended checkpoint; do not
   delete older files until the replacement has been verified.

## Safety and size

- Never commit artifact binaries to this repository; `.gitignore` deliberately
  excludes them.
- Do not open `.pkl` or `.joblib` files from untrusted sources: they can execute
  code during loading. Only load files created by the team or a trusted source.
- Prefer portable formats where practical: JSON for configuration and label
  mappings, and framework-native state dictionaries/weights over a pickled
  whole model.
- Keep secrets (API keys, tokens, personal paths, credentials) out of configs,
  notebooks, metadata, and artifacts.
