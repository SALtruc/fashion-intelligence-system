# Fashion Intelligence System — COSC2753 Assignment 2

Task 1 delivery updated 10 September 2026. The completed executed notebook is
`notebooks/task1-sota.ipynb`; its organized Task 1 outputs are under `models/task1/`.
The selected from-scratch ResNet/resample achieves **0.7654 reporting macro-F1** and
**87.35% accuracy** on 7,568 images across 110 observed classes. See the notebook's
conclusion for uncertainty, rare-class coverage and pilot-history limitations.

## Try the saved model (no retraining)

Use Python 3.12. From this folder:

```console
python -m pip install -r requirements-inference.txt
python task1_demo.py
```

On Windows, `launch_task1_demo.bat` uses the existing project environment if present.
The native desktop interface lets you choose a JPG/PNG, inspect suggestions, review the
label and export it. It runs locally on CPU and needs no web service or pretrained downloads.
Python's Windows installer includes Tk; on Linux install your distribution's `python3-tk`
package and use a graphical desktop. If your environment already has PyTorch, NumPy and
Pillow, you can run directly. The tested local PyTorch version is recorded in the validation JSON;
the minimal requirements use the original run's PyTorch 2.10 release.

Single-image prediction and template-based batch inference:

```console
python -m src.task1_inference --image datasets/test/images_test/IMAGE_ID.jpg
python -m src.task1_inference --template datasets/test/styles_prediction.csv --image-dir datasets/test/images_test --output outputs/task1_cpu_predictions.csv
```

Replace IMAGE_ID with a real file ID. `--artifacts PATH` accepts another compatible run's
`models/task1` directory. CPU inference uses FP32; the Kaggle run used mixed precision.
Small numerical/runtime differences are possible. Do not overwrite the original evaluated
prediction file merely because a new CPU export exists. Other template columns and ID order
are preserved. GUI review exports have a different, human-review schema and are not the
assignment submission CSV.

## Reproduce training

The original project environment is specified by `pyproject.toml` and `uv.lock`:

```console
uv sync --frozen
uv run jupyter notebook
```

The existing lock requires Python 3.12.0 exactly; the recorded Kaggle run used Python 3.12.13.
Keep these environment differences visible when comparing reruns. For the saved-model demo,
the smaller inference requirements suffice.

Place the course-provided data here:

```text
datasets/train/styles_train.csv
datasets/train/images_train/<id>.jpg
datasets/test/styles_prediction.csv
datasets/test/images_test/<id>.jpg
```

1. Run `notebooks/00_eda_and_preprocessing.ipynb` to produce the audited manifest.
2. Open `notebooks/task1-sota.ipynb`. Set `KAGGLE=False` in the first code cell locally;
   the preserved executed copy has `KAGGLE=True` because the recorded run was on Kaggle.
3. Leave `RUN_QUICK=False` for performance evaluation. Execute Run All. GPU training is
   recommended; CPU training requires the notebook's explicit `ALLOW_CPU` option. References
   can be disabled with `RUN_PRETRAINED=False`; enabling them needs internet and more compute.
4. Outputs are written to a fresh `outputs/task1_full_*` locally, or `/kaggle/working/` on Kaggle.
   Section 8 contains run-specific post-run prose: update its numbers after any rerun.

The older `01_task1_article_type_classification.ipynb` is retained as an earlier source copy;
it is not the reviewed executed deliverable. The old Kaggle bundle generator was retired after the reviewed run; use `task1-sota.ipynb` for any deliberate rerun.

## Evidence and packaging

- `models/task1/`: canonical organized models, tables, figures, predictions and metadata.
- `predictions/task1/task1_predictions.csv`: canonical Task 1 prediction copy.
- `splits/task1/`: canonical fit, tuning and reporting membership tables.
- `outputs/figures/task1/`: report figure copies.
- `artifacts/task1/kaggle-full-16ay9832/`: runtime, telemetry and organization manifest.
- The raw Kaggle download was removed after canonical hashes were verified.
- `docs/REPORT_TASK1.md`: current Task 1 report material; combine and format with the other tasks.
- `docs/INDEPENDENT_EVALUATION_TASK1.md`: published-work comparison with comparability limits.
- `docs/TASK1_PATCH_NOTES.md`: distinction between original evidence and post-run additions.
- `scripts/validate_task1_delivery.py`: artifact, sampled inference and GUI integration checks.
- `scripts/organize_kaggle_results.py`: copies a raw Kaggle download into the canonical layout with SHA-256 checks.
- `scripts/build_task1_submission.py`: builds a Task 1 handoff ZIP containing code, models and evidence.

```console
python scripts/validate_task1_delivery.py
python scripts/build_task1_submission.py
```

The validator needs the supplied image directories and a graphical Tk installation; it does
not retrain. The builder includes the existing audited manifest and test template but not raw
course images; use the course data layout above for retraining. The app can classify a chosen
local image from the archive without those datasets. The handoff ZIP includes every original manifest-listed artifact, including reference embeddings,
so the original integrity checks remain reproducible.

## Remaining assignment work

Task 1 is not the complete four-task assignment. Tasks 2–4, the combined prediction CSV,
all group details, and the final length-checked PDF report remain separate work. See
`SUBMISSION.md`. No external-image robustness claim or guaranteed rubric grade is made.


