# Task 1 post-run patch record

Updated 10 September 2026 against run `task1_full_16ay9832`.

- Notebook changes are editorial: original code cells, execution counts and outputs are preserved. Added observations, an actual conclusion, source references, literature context and demo instructions. Static numerical prose identifies the run and must be refreshed on rerun.
- The raw Kaggle `results/` download was preserved during organization, then removed after all 67 recorded artifact hashes were verified against the canonical copies.
- `deployment.json` retains the historical `source_notebook` value `01_task1_article_type_classification.ipynb`. The reviewed executed deliverable is `notebooks/task1-sota.ipynb`. This correction is recorded here instead of changing signed-by-hash run evidence. Its artifact basename resolves under `final/`; the new loader explicitly implements that convention.
- New inference code imports only architecture definitions extracted from the executed runtime. It verifies the checkpoint hash, selected arm, class order, recipe and normalization before loading. The GUI runs CPU FP32 inference, with no training or remote upload.
- The report and README now describe current results. Historical pipeline/model notes are marked archival to prevent contradictory recommendations.
- The literature comparison provides external research context, not a new external-image experiment or a matched benchmark. No additional model training or protocol tuning was done.
- The new ZIP is a Task 1 handoff package, not the complete multi-task Canvas submission. It excludes raw course images, training caches and duplicate rank outputs. Every original manifest-listed artifact, including reference embeddings, is included. Its own package manifest lists included files; the original run manifest remains unchanged.

See `TASK1_DELIVERY_VALIDATION.json` for checks actually executed. GUI interaction is integration-tested programmatically; a human visual/usability review is still useful before presenting the demonstration.

## Kaggle download organization

scripts/organize_kaggle_results.py copied the raw download into the codebase on 10 September 2026. Canonical runtime paths are models/task1/, predictions/task1/, splits/task1/, outputs/figures/task1/, and artifacts/task1/kaggle-full-16ay9832/. The organization manifest records every copied file and hash. One stale prediction under models/task1/predictions/ differed from the Kaggle run and was replaced after comparison. The raw source was then removed at the user request.


