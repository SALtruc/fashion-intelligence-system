# Task 3 - `gender` + `usage`

`03_task3_gender_usage_nguyen.ipynb` is the notebook. `task3_build.py` is the same
code as a plain script (`# %%` cell markers) - edit whichever you prefer; regenerate
the notebook from the script with the converter at the bottom of this file.

## What it answers

The brief says predict **both** `gender` and `usage` and does not say whether that is
one model or two. The notebook decides that with evidence:

| | design | backbones | outputs |
|---|---|---|---|
| **A** | two independent models | 2 | 5 - 8 |
| **B** | one model, joint label | 1 | 24 `gender x usage` pairs that occur |
| **C** | one shared backbone, two heads | 1 | 5 + 8 |

Same convolutional body, same seed, same schedule, same rows - so the difference
between them is the design choice and nothing else.

Then three ablations, one variable each: class-weighted loss, external data, and a
split that mimics the real test set.

## Running it locally on a GPU - the faster path

Colab is not required. This repo's owner has an RTX 4070 Laptop, which beats the T4
the first three runs used, and the catalogue is already staged at `D:/ColabDataset`
(both that path and `D:/g2/Dataset` for the external images are already in the search
lists in section 0.2, so nothing needs editing).

One-time setup:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python -m pip install nbconvert nbclient
python -m ipykernel install --user --name a2torch --display-name "A2 (torch cu126)"
```

Then:

Open `03_task3_gender_usage_nguyen.ipynb` and run all cells: about 90 minutes on an
RTX 4070, and the notebook is self-contained. Setting `A2_QUICK=1` in the environment
cuts it to roughly two minutes on a 4,000-row sample, which is enough to prove the
pipeline runs before committing an hour of GPU.

The same code also exists as `src/task3/task3_build.py`, cell-for-cell identical and
marked with `# %%`. That is the copy the experiment scripts execute, so they share the
notebook's preprocessing rather than duplicating it:

```bash
python src/task3/finalise_task3.py               # the submitted model and predictions
python src/task3/experiment_hyperparams.py       # the sweep behind appendix B6
python src/task3/experiment_epochs_confirm.py    # the confirmation that refuted it
python src/task3/experiment_transfer_weighted.py # appendix B7
python src/task3/diagnose_gender_ceiling.py      # per-class ceilings, trains nothing
python src/task3/experiment_party_external.py    # external data, round 1
python src/task3/experiment_catalog_external.py  # external data, round 2
```

Each writes into `predictions/task3/`, and all ten result CSVs are also folded into
`predictions/task3/task3_all_results.csv` in long form, one row per measurement.

Local staging, if `D:/ColabDataset` is ever missing: it needs
`preprocessed_datasets/train/styles_train.csv` and `images_train/`, holding exactly
the 37,745 rows the frozen split covers. Take the CSV from `ColabDataset.zip` rather
than filtering the provided `styles_train.csv` - **two `gender` labels differ** between
them (ids 36762 and 39107, `Unisex` in the de-duplicated file against `Men` and `Boys`
in the raw one), so the de-duplicated file is not a subset of the raw one and only the
zip matches what the earlier runs measured.

## Running it on Colab

Setup is built into the notebook - sections 0.1 and 0.2 mount Drive, copy the zip,
unpack it to local disk and check the unpack finished. Nothing to add by hand.

**Before you start**, both of these must be reachable from *My Drive*. A shortcut
pointing into a Shared Drive is fine - Colab mounts My Drive only, so the shortcut is
the bridge. Anything under *Shared with me* is **not** mounted; right-click it in
Drive -> **Add shortcut to Drive** first.

| | what | why |
|---|---|---|
| `ColabDataset.zip` | the provided catalogue, zipped | 43,577 separate Drive reads take ~30 min *per session*; one zip takes ~1 min |
| `A2_ExternalData/` | the collected images (folder, 20 MB) | small enough to read straight from Drive |
| `splits/task3/train_val_grouped_sha256.csv` | **the team's frozen split** (430 KB) | keep this file with the repository, or copy the same file into `A2_ExternalData/` for a notebook-only Colab upload |

The split file is not optional if the numbers are going to sit beside a teammate's.
A split generated here from seed 42 overlapped the team's frozen file by **15.6%** -
same seed, same function, different scikit-learn version. Section 3.0 loads the file
when it can find one and says loudly when it cannot.

Then:

1. **Runtime -> Change runtime type -> T4 GPU.** It runs on CPU, but many times slower.
2. Upload the notebook (`File -> Upload notebook`) or open it from Drive.
3. If your zip lives somewhere else, add the path to `COLAB_ZIP_CANDIDATES` in
   Section 0.1. Both places it has lived so far are already in the list.
4. **Set `QUICK = True`** in section 0, then **Runtime -> Run all**.

`QUICK` runs the whole notebook on 4,000 rows and 6 epochs in a few minutes. Its
*numbers are meaningless* - the point is to surface a typo before you spend an hour.
You are looking for one line at the very bottom:

```
results will be written to: /content/drive/MyDrive/A2_ExternalData/task3_results.csv
saved -> /content/drive/MyDrive/A2_ExternalData/task3_results.csv
```

It must say **Drive**, not `/content`. `/content` dies with the VM, and a run that
finishes while the laptop is asleep then leaves nothing behind - which has happened.

5. Got that? Set `QUICK = False`, then **Runtime -> Restart session and run all**.

### Expected cost

Decoding 37,745 JPEGs takes ~10 minutes and is cached to `/content/train_images.npy`,
so a re-run inside the same session skips it. Ten training runs of 20 epochs on a T4
(the section 10 additions include an `articleType` pretraining pass) put the measured total
at **70-75 minutes** end to end, twice. Budget 90.

Section 10.5 needs no GPU at all - it is pandas over the metadata - but it reads
results the trained models produced, so it cannot be run on its own.

### If something goes wrong

| symptom | cause |
|---|---|
| `FileNotFoundError` listing your My Drive contents | `COLAB_ZIP` name is wrong - pick from the list it printed |
| `images missing -- the unzip was incomplete` | delete `/content/preprocessed_datasets` and re-run section 0.1 |
| external data "not found", section 7 and section 8.2 skipped | `A2_ExternalData` has no shortcut in My Drive |
| a cell hangs with no output | almost always an `unzip` overwrite prompt; section 0.1 passes `-o` to avoid it |

## Reading the output

Everything is **macro-F1**. Accuracy appears only next to it as evidence of why it
cannot be the metric: predicting `Casual` for every row scores **76.1%** accuracy on
`usage` and **0.108** macro-F1.

The three numbers worth carrying into the report:

* `usage` macro-F1 has a **ceiling of 0.500** while the four classes under 100 images
  stay unlearnable (`Home` has **1** training image). Section 1.1 predicted it; the best
  model measured **0.5020**. Two of those four classes cost 0.125 each and no model
  recovers them.
* Section 10.5 measures the other ceiling: an oracle handed the true `articleType` reaches
  **0.8945** accuracy on `usage`, and the CNN reaches 0.8917-0.8979 from pixels alone.
  `usage` accuracy is finished. `gender` is not - the CNN beats the best metadata
  oracle by **+10.6 points**, so that is where the remaining headroom is.
* the **forward split** (section 8.1) is a better estimate of the graded score than the
  random one, because the test set is the highest ids and the label distribution
  drifts along that axis - `Women` goes 33% -> 53%.
* the **independent evaluation** (section 8.2) is the brief's section 3.3 requirement, and the gap
  between it and validation is the finding, not a disappointment.

## Notebook and output provenance

`src/task3/task3_build.py` is the plain-script mirror of the notebook. Keep narrative,
paths and model code aligned between the two files when making future changes.

Earlier versions of this README referred to `build_notebook.py` and
`check_notebook.py`; those helpers are not present in the current branch, so they must
not be presented as an active reproducibility gate. The executed model and figure
outputs currently embedded in the notebook are preserved from the completed GPU run.
This refinement changed prose, figure references, and the split resolver path only; it
did not rerun training or replace any stored metric.

The frozen split itself is unchanged after being organised under `splits/task3/`:

```
rows: 37,745 (32,084 train; 5,661 validation)
SHA-256: 905bb7e1dfbfcf5a9dbf5f2bcd89444f866588d394fe483e0b20b7703c9647ed
```

The saved split-loading output may still display its previous machine-specific path.
That line is execution provenance, not the current repository location. Run the setup
and split cells again when a fresh environment is available; no model retraining is
required for that check.
