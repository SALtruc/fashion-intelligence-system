# Task 3 model outputs

`submitted/task3_gender_usage_C_weighted.pt` is **tracked in git** — it is the submitted model, and a
marker who clones this repository can run it without a Drive link:

```console
python src/task3/predict_test.py      # writes predictions/task3/task3_gender_usage.csv
```

Design C: one shared convolutional body, one linear head per target, class-weighted loss,
289k parameters. Validation macro-F1 **gender 0.7202 / usage 0.4676**, with mirror TTA. The
checkpoint carries its own class order per target, image size, and the channel statistics
fitted on the training rows, so inference re-derives nothing.

Everything else here is gitignored and lives on the team Drive: the two independently
trained `usage` variants, the external-usage checkpoint, and the validation-prediction
CSVs the report's appendix quotes. `src/task3/finalise_task3.py` rebuilds the shipped
model from scratch; `predict_test.py` only runs it.
