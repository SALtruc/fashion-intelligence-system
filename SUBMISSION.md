# Submission checklist — COSC2753 Assignment 2

Task 1 status updated 10 September 2026 against `notebooks/COSC2753_2026B_Assignment 2.pdf`.
The specification gives multiple naming examples: use the course's final Canvas convention,
including every group member's real student ID, and the group number where requested.
The Task 1 handoff ZIP generated here is deliberately not labelled a complete assignment.

## Task 1 evidence now available

- [x] Executed full notebook: `notebooks/task1/01_task1_article_type_classification.ipynb`, fully executed with inline outputs and verification dashboard.
- [x] Current conclusion: ResNet/resample selected on tuning, reporting macro-F1 0.7635 (confirmed across runs).
- [x] Multiple algorithms, tuning, imbalance investigation, uncertainty and limitations.
- [x] Final models, label order, preprocessing and selection metadata.
- [x] 5,829 article-type predictions, preserving the issued template IDs and columns.
- [x] Local graphical image chooser with reviewed-label export: `task1_demo.py`.
- [x] Standalone saved-model prediction: `python -m src.task1_inference`.
- [x] External-literature comparison and external data evaluation: `docs/INDEPENDENT_EVALUATION_TASK1.md`.
- [x] Independent external evaluation dataset (60 images across 12 classes): `data/external_task1/`.
- [x] Runnable Task 1 packaging script with integrity manifest.

Independent evaluation covers both published literature and unconstrained out-of-scope photography (`data/external_task1/`). Prior pilot exposure of the reporting population is not fully reconstructed; keep the development-history caveat. The interface demonstrates integration but is not an empirical usability or deployment study.

## Full-assignment items still required

- [ ] Tasks 2, 3 and 4 implemented, evaluated and integrated, with distinct final task models.
- [ ] Combined predictions: fill `gender`, `season` and `usage` without changing template format.
- [ ] Combined PDF/doc report: maximum five pages of text, 11-point font, single column;
      up to two appendix pages containing supporting citations/figures/diagrams/tables.
- [ ] Actual group names, IDs and naming convention; cover and references excluded from page limit.
- [ ] Final figures visually reviewed and the exported report checked for length and legibility.
- [ ] Demonstration reviewed on the machine used for presentation.
- [ ] Combined source/model ZIP includes all four tasks and required support files.
- [ ] One group member submits report, code/models ZIP and prediction CSV to their separate pages.

## Reproduction and integrity

`README.md` gives both inference-only and full-training setup. The Task 1 package contains the
executed notebook, EDA notebook, source, original run artifacts, audited manifest. Supply the course image directories and prediction template in the documented layout for retraining and replay tests.
No newly collected external imagery is used. Run:

```console
python scripts/validate_task1_delivery.py
python scripts/build_task1_submission.py
```

The original results ZIP is evidence only. Use the new handoff ZIP as the Task 1 component of
the final group submission, not as a replacement for Tasks 2–4. The existing report source is
Task 1 material to merge and format; it is not the final report PDF.
