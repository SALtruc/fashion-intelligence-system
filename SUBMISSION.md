# Submission checklist — COSC2753 Assignment 2

Task 1 material updated 10 September 2026 against the executed training notebook.
The specification gives multiple naming examples: use the course's final Canvas convention,
including every group member's real student ID, and the group number where requested.
The Task 1 training notebook and recorded model outputs are preserved here; they are not the complete assignment.

## Task 1 evidence now available

- [x] Executed full notebook: `notebooks/task1/01_task1_article_type_classification.ipynb`, 54 executed code cells, no saved errors.
- [x] Current conclusion: ResNet/resample selected on tuning, reporting macro-F1 0.7654.
- [x] Multiple algorithms, tuning, imbalance investigation, uncertainty and limitations.
- [x] Final models, label order, preprocessing and selection metadata.
- [x] 5,829 article-type predictions, preserving the issued template IDs and columns.
- [x] External-literature comparison: `docs/INDEPENDENT_EVALUATION_TASK1.md`.

The literature route does not test our model on newly collected images. Prior pilot exposure
of the reporting population is not fully reconstructed; keep the development-history caveat.

## Full-assignment items still required

- [ ] Tasks 2, 3 and 4 implemented, evaluated and integrated, with distinct final task models.
- [ ] Combined predictions: fill `gender`, `season` and `usage` without changing template format.
- [ ] Combined PDF/doc report: maximum five pages of text, 11-point font, single column;
      up to two appendix pages containing supporting citations/figures/diagrams/tables.
- [ ] Actual group names, IDs and naming convention; cover and references excluded from page limit.
- [ ] Final figures visually reviewed and the exported report checked for length and legibility.
- [ ] Combined source/model ZIP includes all four tasks and required support files.
- [ ] One group member submits report, code/models ZIP and prediction CSV to their separate pages.

## Reproduction and integrity

`README.md` gives the project setup. The Task 1 training notebook and its recorded model/output
tree are preserved under `notebooks/task1/` and `models/task1/`. Supply the course image
directories and prediction template in the documented layout for any deliberate retraining.
No newly collected external imagery is used.

The existing report source is Task 1 material to merge and format; it is not the final report PDF.
