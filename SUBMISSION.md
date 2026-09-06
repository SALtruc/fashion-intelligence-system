# Submission checklist — COSC2753 Assignment 2

Repository status checked **6 September 2026**. The requirements below are transcribed
from the local assignment PDF; check Canvas for subsequent announcements before upload.

Requirements transcribed from Section 5 of `notebooks/Task1/COSC2753_2026B_Assignment 2.pdf`.
**Only one group member submits.** Do not spread the submission across members.

Due **Sat 12 Sep 2026, 23:59**. Late submissions lose 20% per day until 16 Sep; after that
Canvas closes and all marks for the assignment are lost.

## Naming convention

Both the report and the code archive use:

```
COSC2753_A2_<studentID1_studentID2_...>
```

The spec also shows a report variant carrying the group number:
`COSC2753_A2_<Group number>_<studentID1_studentID2_...>`. Use the group-number form for the
report and the plain form for the ZIP unless an announcement says otherwise.

**A submission that does not follow the convention incurs a mark deduction.** Fill in the real
IDs before submitting; do not leave a placeholder.

## Three separate Canvas pages

The portal has three sub-pages and they are not interchangeable.

| # | Page | What goes there | Format |
|---|---|---|---|
| 1 | Report | The written report only | PDF or DOC — nothing else |
| 2 | Models & code | Everything needed to reproduce | ZIP |
| 3 | Predictions | Predictions from the ultimate judgement | as issued by `styles_prediction.csv` |

## 1. Report

- [ ] **No more than 5 pages of text**, plus up to 2 pages of appendices
- [ ] Single-column layout, **11 pt** font
- [ ] Cover page and reference list do **not** count toward the limit
- [ ] Appendices may contain **only** citations, figures, diagrams or data tables — not the
      judgement itself
- [ ] Names and student IDs of every group member
- [ ] Analyses all four required elements: preprocessing (including any extra collection),
      algorithms considered, why they were selected, evaluation of trained models, and the
      ultimate judgement with supporting evidence
- [ ] Compares **multiple models**, not one

> Over-length content is **not marked**. If the report runs to six pages, only the first five
> are read. Check the page count in the exported PDF, not in the editor.

Draft: [docs/REPORT_TASK1.md](docs/REPORT_TASK1.md) covers recorded supplied-only Task 1
results, including historical experiments absent from current jobs. Current tuning-grid
results are not yet exported. It still needs the other tasks, the cover page, names and IDs.
The Task 2–4 and final-prediction notebooks are empty placeholders, so implementing and
evaluating those tasks remains outstanding.

## 2. Models & code (ZIP)

- [ ] All notebooks and scripts used to build the system
- [ ] **At least four final models** plus the scripts that run them for prediction
- [ ] A README telling a marker how to set the environment up — where data files go, which
      packages to install
- [ ] Everything in **one folder**, then zipped
- [ ] Opens and runs on a standard machine

The repository README covers `uv sync --frozen` and the dataset layout. Before zipping, confirm
a marker can follow it from a clean checkout.

**The current `.gitignore` excludes raw/preprocessed dataset CSV/JPEG files and most
`artifacts/` contents, but does not exclude `models/`, `outputs/` or `predictions/`.** Include the required
trained models in the ZIP. If the upload limit prevents this, obtain the course's accepted
alternative delivery method; this repository does not establish that a Drive link satisfies
the models requirement. Check actual archive contents independently of ignore rules.

**Any extra-collected data must be accessible to the evaluator.** The external cosmetics
collections used in the supplied-only independent evaluation are present inside
`notebooks/Task1/dataset1/` and `dataset2/`, and both folders are fully tracked in Git
(1,169 and 706 files). Their provenance and model-specific held-out status
are documented in `docs/INDEPENDENT_EVALUATION_DATA.md`,
which should be read before the report describes them.

## 3. Predictions

- [ ] Produced by the **ultimate judgement**, not by whichever model scored highest in isolation
- [ ] Keeps the exact `styles_prediction.csv` format: `id,gender,articleType,season,usage`
- [ ] IDs are the selected test-set IDs, unchanged and in order

Current state: the combine notebook writes `predictions/task1_predictions.csv` with all 5,829
rows and `articleType` filled; `gender`, `season` and `usage` are empty and must be filled by
Tasks 2 and 3 before submission. A superseded copy remains at `outputs/task1_predictions.csv`
and is not the current output path.
The current final-prediction notebook is empty and does not perform the merge. Preserve the
original template's IDs and row order when completing it.

## Integrity

- [ ] Every source, dataset and borrowed idea is cited, including web material
- [ ] Reference list included (does not count toward the page limit)
- [ ] Code is the group's own — it goes through plagiarism software and Turnitin

## Final pass before uploading

- [ ] Report exports to PDF at the right length and font size
- [ ] ZIP opens and the folder structure is intact
- [ ] Prediction CSV has 5,829 rows, the original header, and no index column
- [ ] File names match the convention exactly
- [ ] All three Canvas pages submitted — a missing page is a missing deliverable
