# Task 1 patch status

The source patch addresses the Task1 runtime, reproducibility, worker-handover, and export
checks. This continuation used static source inspection and worker validation only; no
notebook was executed and no model was trained here.

Implemented:

- The default combine mode remains self-contained: `JOB_FILTER = None` makes every Task1 job
  wanted, so missing checkpoints are trained locally rather than required from workers.
- HOG baseline image arrays remain available until the HOG grid, avoiding the original combine
  failure at the first HOG search arm.
- All six HOG grid arms write fingerprinted score checkpoints and optional joblib model files
  atomically, restore compatible completed arms, and update the partial search CSV after each arm.
- Stage-2 resume state includes the FP16 gradient scaler.
- Neural and HOG checkpoints are listed by the worker-stop cell for parallel handover.
- Prediction export uses the explicit Task1 path `predictions/task1_predictions.csv`, preserves
  the exact template columns `id,gender,articleType,season,usage`, and validates unique IDs.
- Generated worker composition, fingerprint and legacy-derivation checks pass.

Still requiring an actual full run for numerical evidence:

- The tuning CSVs and final prediction values must be produced on the supplied data and target
  hardware.
- The reduced backbone grids are search evidence only; this source does not invent full-budget
  confirmation results.
- Dataset1 enrichment remains prepared but is not the default training source.
- External-data provenance, labels, licenses, and deployment representativeness require review.

Static validation completed:

```text
uv run python scripts/check_task1_workers.py --strict
uv run python scripts/make_task1_workers.py --check
```

These checks parse and compare source cells only. They establish that the workers are
faithfully derived from the combine notebook and that the fingerprint is unchanged; they do
not execute any cell, so every entry under "Implemented" above describes reviewed source and
not observed behaviour.
