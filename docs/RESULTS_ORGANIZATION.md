# Task 1 results organization

The verified Kaggle Task 1 run is now organized into the repository's canonical folders. The raw Kaggle download was removed after the organized copies and all 67 run-manifest hashes were verified.

| Repository path | Contents |
|---|---|
| `models/task1/` | Final checkpoints, metadata, tables, figures, pretrained-reference embeddings, telemetry, and reporting predictions |
| `predictions/task1/task1_predictions.csv` | Task 1 submission copy with the assignment template order |
| `splits/task1/{fit,tuning,reporting}.csv` | Exact split membership tables |
| `outputs/figures/task1/` | Report figure copies |
| `artifacts/task1/kaggle-full-16ay9832/` | Runtime, log, phase metadata, telemetry, and organization manifest |

The organizer script remains available for a future fresh Kaggle download:

```console
python scripts/organize_kaggle_results.py --source <download>/task1_full_*
```

It skips identical files, stops on differing destinations, verifies SHA-256 after each copy, and writes `ORGANIZATION_MANIFEST.json`. It copies files and does not delete the source download.
