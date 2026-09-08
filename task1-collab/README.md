# Task 1 on Colab and Kaggle

The hosted-notebook edition of the Task 1 parallel run. Ten notebooks: the combine notebook,
the eight training workers, and the independent evaluation. They are the same notebooks as
`notebooks/Task1/`, with the repository layout flattened so everything sits under one working
root, and with the housekeeping — dependency check, data staging, checkpoint hand-over, result
export — added at the top and bottom.

**The same ten files run on both platforms.** The setup cells detect where they are and set
`PROJECT_ROOT` themselves: `/content` on Colab, `/kaggle/working` on Kaggle. You do not edit
anything to switch. The one thing that genuinely differs is how the data arrives, because the
platforms disagree about it — see [Running on Kaggle](#running-on-kaggle).

**No Google Drive anywhere.** You bring the data into the session and take the results out of
it; nothing is mounted, and nothing is read from or written to Drive. On Colab that has a
consequence worth reading before you start a long job — see
[Things that will bite](#things-that-will-bite).

Checked against the source notebooks on **7 September 2026**.

> **These files are generated.** Do not edit them by hand. Edit the source under
> `notebooks/Task1/` and run `python scripts/make_task1_colab.py`; the changes land here.
> `python scripts/make_task1_colab.py --check` verifies this folder is current without
> writing anything.

## Why a separate edition is needed at all

The source notebooks find the repository root by walking *up* from the notebook's own
directory looking for `src/preprocessing.py`. That assumes the notebook sits inside a
checkout. A hosted notebook does not: the working directory is `/content` on Colab or
`/kaggle/working` on Kaggle, with the data flat beneath it. That is the one substantive
change. Everything else here is staging.

**Nothing that affects a result is changed.** Every hyper-parameter, the split, the manifests
and the augmentation policy are untouched, so the Colab edition prints the same
`RUN_FINGERPRINT` — `e6b15f5c51de` on the supplied arm — as a local run. Checkpoints trained
here and checkpoints trained from a checkout are interchangeable. `ARM`, `ALLOW_CPU`, `RESUME`
and every path are outside that fingerprint by design.

---

## Prerequisites

| # | What | Notes |
|---|---|---|
| 1 | The course data in a local checkout | `datasets/train/images_train/`, `datasets/test/` — not in git, see `datasets/README.md` |
| 2 | Python locally, once, to build the bundle | `python scripts/make_colab_bundle.py` |
| 3 | **~1.5 GB free locally**, and the upload bandwidth for it | On Colab the 590 MB bundle is re-uploaded into every session; on Kaggle it is uploaded once as a Dataset and reused |
| 4 | One account **per concurrent session** | See [Sessions and quota](#sessions-and-quota) — this is the real limit on the parallel run |

No Drive quota is needed and nothing is left in your Google account between sessions. On Colab
the cost moves to the network instead: 590 MB up per session, over a connection that then has
to stay up for as long as the job runs. **Kaggle does not have that problem** — the bundle
lives on as a Dataset and every session mounts it read-only — which is the main practical
reason to prefer Kaggle for the long jobs.

Nothing is installed on the runtime in the ordinary case: Colab and Kaggle both ship torch,
numpy, pandas, scikit-learn, scikit-image, matplotlib, seaborn and joblib. The dependency cell
checks and reports versions rather than installing — which matters on Kaggle, where internet is
off by default and an install would fail.

---

## Files needed to run

Everything is addressed relative to `PROJECT_ROOT` — `/content` on Colab, `/kaggle/working`
on Kaggle, resolved automatically. `scripts/make_colab_bundle.py` produces exactly this layout,
and Kaggle mounts the same tree read-only under `/kaggle/input/<slug>/`.

### The bundle — `task1_colab_data.zip`, 590 MB, 46,307 files

| Path in the bundle | Files | Size | Needed by |
|---|---:|---:|---|
| `src/preprocessing.py` | 1 | 10 KB | **every notebook** — the shared image transform and split |
| `preprocessed_datasets/train_manifest.csv` | 1 | 6.6 MB | **every notebook** — defines the split; do **not** regenerate |
| `datasets/train/images_train/*.jpg` | 38,612 | 481.9 MB | **every notebook** |
| `preprocessed_datasets/task1_dataset1_arms/<version>/` | 3 | 0.2 MB | the enriched arm; combine (Section 11) |
| `dataset1/external_cosmetics.csv` + `dataset1/images/*.jpg` | 1,159 | 12.5 MB | the enriched arm; combine; evaluation |
| `dataset2/external_cosmetics2.csv` + `dataset2/images/*.jpg` | 700 | 2.5 MB | evaluation only |
| `datasets/test/styles_prediction.csv` + `datasets/test/images_test/*.jpg` | 5,830 | 86.0 MB | **combine only** — the prediction run |
| `01_task1_article_type.ipynb` | 1 | 0.3 MB | **evaluation only** — it reads `SmallResNet` out of the notebook source |

`datasets/train/styles_train.csv` is deliberately **not** in the bundle. No Task 1 notebook
opens it; only `notebooks/00_eda_and_preprocessing.ipynb` does, and the manifest that notebook
writes is tracked, so the EDA run is never repeated per session.

`python scripts/make_colab_bundle.py --no-test` drops the test set for a 504 MB worker-only
bundle, which is roughly 90 seconds off each worker upload. The combine session still needs
the full one.

### Per notebook

| Notebook | Reads | Also needs |
|---|---|---|
| `worker_*.ipynb`, `COLAB_ARM = "supplied"` | `src/`, the manifest, the training images | — |
| `worker_{hog_svm,cnn,resnet}.ipynb`, `COLAB_ARM = "enriched"` | the above, plus the arms split and `dataset1/images/` | — |
| `worker_sweep.ipynb`, `worker_stage2grid.ipynb` | the supplied set | **`model_resnet_stage1.pt`** from the `resnet` session, in `models/task1/checkpoints/<arm>/` |
| `01_task1_article_type.ipynb` (combine) | everything, including `datasets/test/` | **every worker session's checkpoints**, unpacked into `models/task1/checkpoints/<arm>/` |
| `02_independent_evaluation.ipynb` | `dataset1/`, `dataset2/`, the manifest, the training images | `model_resnet_decoupled.pt`. It also reads `01_task1_article_type.ipynb`, which the bundle already carries — so only the checkpoint zip has to be dragged in |

### Written by the run, on the runtime's own disk

```
/content/
|- task1_colab_data.zip                      the bundle you dragged in
|- checkpoints_<job>_<arm>.zip               dragged in from an earlier session, if this job needs one
|- models/task1/checkpoints/supplied/        this session's training output
|- models/task1/checkpoints/enriched/        the second arm, never mixed with the first
|- models/task1/                             the exported model, classes, config, result CSVs
|- outputs/figures/                          report figures
|- predictions/                              task1_predictions.csv, task1_test_logits.npy
|- checkpoints_<job>_<arm>.zip               written by the last cell, downloaded to your machine
|- task1_outputs_<arm>.zip                   the combine session's deliverables
```

**On Colab, every one of those paths is deleted when the runtime disconnects.** The last cell
of each notebook zips what matters and pushes it through the browser, and that download is the
only copy that survives the session — so run it before closing the tab. Anything the last cell
does not collect, download yourself from the sidebar file browser while the session is alive.
On Kaggle the same tree is `/kaggle/working`, which is kept as the version output when you Save
Version, so the pressure is lower — but run the last cell there too, because the zip it writes
is the unit the next session expects.

Checkpoints move between sessions by hand: a worker session's last cell writes
`checkpoints_<job>_<arm>.zip`, you bring that file into the next session — dragged onto
`/content` on Colab, added as a Dataset on Kaggle — and its staging cell unpacks it into
`models/task1/checkpoints/<arm>/` and prints what it found. Files
are keyed by name, so several sessions' zips merge without collisions, and a zip naming the
*other* arm is listed and skipped rather than mixed in.

---

## Running one notebook on Colab

1. **Build the bundle**, once, from a local checkout that has the data:

   ```bash
   python scripts/make_colab_bundle.py
   ```

2. **Open the notebook**: colab.research.google.com → File → Upload notebook → pick the one
   you want out of this folder.

3. **Drag the data in.** Connect the runtime, open the file browser in the left sidebar, and
   drag `task1_colab_data.zip` onto `/content`. Wait for the upload to finish — the progress
   ring at the bottom of the file browser — before running anything; a partial upload fails
   the unpack. If this job needs another session's checkpoints, drag those zips in now too.

4. **Pick the runtime**: Runtime → Change runtime type → **T4 GPU**, except for
   `worker_hog_svm` and `worker_hogsearch`, which have no GPU path at all and should be given
   a **CPU** runtime so the GPU goes to a job that can use it. Each notebook's control panel
   says which it wants and warns if it got the other.

5. **Set `COLAB_ARM`** in the control panel — the first code cell, and the only cell you
   edit. `"supplied"` or `"enriched"`.

6. **Runtime → Run all.** No authorisation prompt appears, because nothing is mounted. The
   staging cell unpacks the bundle (about a minute), imports any checkpoint zips you dragged
   in, and verifies every required path before any training starts.

7. **When it finishes**, the last cell writes `checkpoints_<job>_<arm>.zip` and hands it to
   the browser as a download. **Run that cell and let the download complete before closing the
   tab** — the runtime's disk goes with the session. It also prints the fingerprint; check it
   against every other session in the pass.

The steps above are the Colab ones. The notebooks are identical on Kaggle; only steps 2–3 and
7 change, because the platforms disagree about how data gets in and out.

---

## Running on Kaggle

The same ten notebooks, unmodified. `PROJECT_ROOT = "auto"` resolves to `/kaggle/working`, and
the staging cell finds the data under `/kaggle/input` instead of unpacking a zip.

**Why the data path differs.** Kaggle does not let a notebook write next to its inputs. You add
the bundle as a *Dataset*, Kaggle extracts the zip itself and mounts the result at
`/kaggle/input/<slug>/` **read-only**, while the writable root `/kaggle/working` starts empty.
So there is no archive to unpack. The staging cell detects this, finds the extracted tree, and
**symlinks** its top-level entries into `/kaggle/working` — a copy would spend minutes and most
of the working quota to gain nothing. If a runtime refuses symlinks it copies instead and says
so.

1. **Build the bundle** locally, as for Colab: `python scripts/make_colab_bundle.py`.

2. **Upload it once as a Dataset.** kaggle.com → Datasets → New Dataset → upload
   `task1_colab_data.zip`. Kaggle extracts it on upload. This is the step you do *not* repeat
   per session — every later notebook just mounts it.

3. **New Notebook → File → Import Notebook**, and pick the one you want from this folder.

4. **+ Add Input** in the right-hand panel, and attach the Dataset from step 2. Attach any
   checkpoint hand-overs the job needs here too.

5. **Notebook options → Accelerator → GPU P100**, except `worker_hog_svm` and
   `worker_hogsearch`, which should stay on CPU. The control panel warns if it got the wrong
   one.

   Pick **P100, not `T4 x2`,** for an ordinary Run All. A notebook kernel is a single process
   and uses one GPU, so `T4 x2` bills two cards, trains on one, and that one T4 is slower than
   the P100 — the worst of both. `T4 x2` is worth choosing only if you are going to launch the
   job with torchrun, which is [its own section](#using-both-gpus-of-a-gpu-t4-x2-session). The
   control panel says so at the top of the run if it sees more than one card.

6. **Set `COLAB_ARM`**, then **Run All**. Leave `PROJECT_ROOT` on `"auto"`.

7. **When it finishes**, the last cell writes the zip into `/kaggle/working`. Take it from the
   Data panel on the right while the session is alive, or **Save Version** and take it from the
   run's Output.

### Handing checkpoints between Kaggle sessions

Two ways, both supported by the staging cell:

- **Publish the output as a Dataset** and `+ Add Input` it to the next notebook. The staging
  cell reads `checkpoints_*_<arm>.zip` out of `/kaggle/input/*/`, and also picks up loose
  `.pt` / `.joblib` / `.npz` files, since Kaggle extracts an uploaded zip and you may end up
  with either shape.
- **Download and re-upload**, which is the same manual round trip Colab needs.

### What is genuinely different on Kaggle

| | Colab | Kaggle |
|---|---|---|
| Data in | Drag the 590 MB zip in **every session** | Upload once as a Dataset, mount it read-only thereafter |
| Working root | `/content` | `/kaggle/working` |
| Results out | `files.download()` through the browser | `/kaggle/working` → Data panel, or Save Version → Output |
| Survives the session | Nothing does | `/kaggle/working` is kept as the version's output; interactive sessions keep it only with **Persistence** on |
| Internet | On | **Off by default** (Notebook options → Internet). Harmless here — Kaggle ships every dependency, so the check cell installs nothing — but a missing package cannot be fetched until you turn it on |
| GPU quota | Opaque, per-account | ~30 GPU-hours/week, shown as a number, which makes the schedule below much easier to plan |
| Session cap | ~12 h, idle-dropped | 12 h interactive, 9 h committed |

The quota being *visible* is the real advantage for this run: `lrsearch` at ~115 minutes and the
`resnet` chain at ~115 minutes are a known, plannable spend rather than a guess.

### Using both GPUs of a **GPU T4 x2** session

Picking `GPU T4 x2` in the accelerator menu gives the session two cards, but a notebook kernel
is a single process and reaches only the first. The notebooks detect this and say so at
start-up — *"2 GPUs visible but this is a single process"*. To use both, run the job through
the launcher in a shell cell instead of running the cells directly:

```python
!python scripts/task1_torchrun.py worker_resnet.ipynb
```

Both `scripts/task1_torchrun.py` and every `worker_*.ipynb` ride along in the bundle, because
torchrun needs the notebook as a *file* and the one open in the browser tab is not one — Kaggle
holds it server-side. The launcher flattens the notebook's code cells into a module and starts
one process per visible GPU. Output appears under the shell cell rather than under each cell,
and everything is written to the same places, so the export cell afterwards behaves as usual.

It roughly halves the wall clock on the GPU-bound jobs — `resnet`, `lrsearch`, `sweep`,
`stage2grid`. `hog_svm` and `hogsearch` are CPU jobs and gain nothing.

**The models are unchanged.** `BATCH_SIZE` stays the *global* batch and each GPU takes half of
it (128 → 64 × 2), so `RUN_FINGERPRINT` is untouched and the checkpoints stay interchangeable
with every single-GPU session in the same pass. Holding the global batch fixed is also what
makes DDP's averaged gradient identical to a one-GPU gradient; every BatchNorm is converted to
`SyncBatchNorm` so its statistics are pooled across both cards rather than computed per half.
The one thing that is not bit-identical is the augmentation RNG — each card augments its own
half, so the draw sequence differs. The policy and the distribution do not.

Worth knowing before you spend quota on it: **two cards for one hour costs two GPU-hours**, not
one. Against a ~30 GPU-hour weekly budget this buys wall-clock, not quota. It is worth it for a
job you are waiting on, and not worth it for one you would have committed and walked away from.

This is a Kaggle note rather than a Colab one. A free Colab runtime has a single GPU, so there
is nothing to distribute; on a multi-GPU Colab runtime the launcher works, but the export
cell's `files.download()` cannot run outside the kernel, so collect results from the file
browser instead.

---

## The parallel run

Eight jobs. The estimates come from `scripts/task1_layout.py` and were measured on Apple
Silicon MPS at fp32 — treat them as a reliable *ranking* and an order of magnitude, not as T4
figures, and re-measure before scheduling against them.

| Job | Runtime type | Estimate | Needs first |
|---|---|---:|---|
| `hog_svm` | **CPU** | ~3 min | — |
| `hogsearch` | **CPU** | ~16 min | — |
| `cnn` | GPU | ~9 min | — |
| `cnnsearch` | GPU | ~17 min | — |
| `resnet` | GPU | ~64 min | — |
| `sweep` | GPU | ~23 min | `model_resnet_stage1.pt` |
| `stage2grid` | GPU | ~28 min | `model_resnet_stage1.pt` |
| `lrsearch` | GPU | ~115 min | — |

Two floors bind any schedule, and no amount of hardware moves either: **`lrsearch` at ~115
minutes**, and **`resnet` + `sweep` + `stage2grid` at ~115 minutes**, which stay together in
one session so the stage-1 checkpoint dependency is satisfied locally with nothing to copy
mid-run.

### Four sessions — both arms in ~115 minutes

| Session | `COLAB_ARM` | Runtime | Jobs, in order | Total |
|---|---|---|---|---:|
| A | `"supplied"` | GPU | `lrsearch` | 115 min |
| B | `"supplied"` | GPU | `resnet`, `sweep`, `stage2grid` | 115 min |
| C | `"supplied"` | GPU | `cnnsearch`, `cnn` | 26 min |
| C′ | `"supplied"` | CPU | `hogsearch`, `hog_svm` | 19 min |
| D | `"enriched"` | GPU | `resnet`, `cnn`, then `hog_svm` | 76 min |

The enriched arm repeats only the three jobs that produce a *model*. The four tuning grids and
the sampler sweep are **not** repeated and must not be: hyper-parameters are tuned once on the
supplied arm and applied to both, because retuning per arm would move the data and the recipe
together and leave the difference unattributable. The notebook enforces this — it forces the
Section 7.4 grids off for any arm but `"supplied"`.

Session C′ is the CPU work. It can share an account with a GPU session where the quota allows
it, or be folded into session C at the cost of 19 minutes.

### Fewer sessions

| Sessions | Wall clock | Arms |
|---:|---:|---|
| 1 | ~275 min | supplied only |
| 2 | ~132 min | supplied only |
| 3 | ~115 min | supplied only |
| 4 | ~115 min | **both** |

Session C finishes first and is the natural place to run the combine notebook from.

### Combining

Unpack every session's `checkpoints_<job>_<arm>.zip` into
`models/task1/checkpoints/<arm>/` — files are keyed by name, so several sessions' zips merge
into one directory without collisions — then run `01_task1_article_type.ipynb` with
`COLAB_ARM` set to that arm.

With both arms banked, run the combine notebook **twice**, once per arm. The second pass
prints the Section 11 comparison table, because both `arm_results_*.json` files then exist.
Set `RUN_SAMPLER_SWEEP = False` on that second pass unless you want the 23-minute sampler
sweep repeated; it is an analysis of where the Section 6 gain comes from, not an input to the
experiment.

Then run `02_independent_evaluation.ipynb` against the exported model.

---

## Sessions and quota

**This is the practical limit on the parallel run, not the notebooks.** Colab caps how many
runtimes one account may hold at once, and the cap is neither published nor stable. In
practice a free account gets one GPU runtime at a time; Pro raises the ceiling but not to
four. So a four-session pass usually means **four Google accounts** — and, because the data
lives in the session rather than in an account, four uploads of the same 590 MB bundle.

If that is not available, the two-session or three-session schedules above give the same
results and only cost wall-clock. Nothing about the outcome depends on how many sessions were
used — the checkpoints are identical either way, which is the entire point of the fingerprint.

Moving checkpoints between accounts is what `checkpoints_<job>_<arm>.zip` is for, and with no
Drive in the picture the accounts never have to see each other: each session downloads its own
zip to your machine, and you drag the ones you need onto `/content` in the combine session.
They are small — tens of MB — so this is quick even though the bundle is not.

---

## Things that will bite

**Disconnects — read this one.** This is the price of not using Drive, and it is a real
price. Everything the run writes is on the runtime's local disk, so a dropped runtime loses
the whole job: there is no persistent copy to resume from, and `RESUME = True` cannot help
across a disconnect because the `epoch_*.pt` files it would resume from went with the disk.
Free Colab drops idle sessions after around 90 minutes and caps a session at around 12 hours.
`lrsearch` at ~115 minutes is the exposed one; `resnet` + `sweep` + `stage2grid` is the other.

What to do about it, in the order worth trying:

- **Keep the tab open and the machine awake.** Most losses are the idle timeout, not a real
  failure. Do not let the laptop sleep mid-run.
- **Bank partial work by hand on a long job.** `epoch_*.pt` is written to
  `models/task1/checkpoints/<arm>/` as training proceeds. Download those files from the
  sidebar file browser while the run is still going; if the runtime then drops, drag them back
  onto `/content` in a fresh session, move them into `models/task1/checkpoints/<arm>/`, and
  `RESUME = True` picks up from that epoch. The export cell deliberately excludes `epoch_*.pt`
  from the zip, so this is a manual step — nothing does it for you.
- **Prefer more, shorter sessions.** The four-session schedule above is also the one with the
  least exposure per session; the single-session run puts ~275 minutes behind one disconnect.

**scikit-learn version skew.** The SVM is persisted with joblib, and a scikit-learn pickle is
only reliably readable by the version that wrote it. Colab's version is not pinned by
`uv.lock`. Run `hog_svm`, `hogsearch` **and** the combine step all on Colab, or all locally —
do not mix. The `.pt` checkpoints carry no such constraint and move freely.

**Regenerating the manifest.** Don't. `train_manifest.csv` is hashed by byte into the
`task1_dataset1/<version>/` directory name, and re-running the EDA notebook would rename it
and change the split. It is tracked, and the bundle carries it.

**A truncated bundle.** The staging cell counts the training images and warns below 38,000. A
short count means a different split and therefore a different fingerprint, which the combine
step will reject rather than silently accept.

**`epoch_*.pt` files.** Resume state, not results, which is why the export cell excludes them
from the zip. Once the matching `model_*.pt` exists they carry nothing you need, and they die
with the runtime either way. They matter only as the manual mid-run backup described under
Disconnects.

---

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `task1_colab_data.zip not found` | The upload has not finished, or the file landed somewhere other than `/content`. Check the sidebar file browser; or set `UPLOAD_IF_MISSING = True` for a picker, or point `DATA_ZIP` at where it actually is |
| `Missing from /content: ...` | The bundle was built with `--no-test` or `--no-external` and this notebook needs what was dropped. Rebuild without the flag |
| "This notebook needs a GPU" | Runtime → Change runtime type → T4 GPU, then Run all again |
| "You are holding a GPU it will not use" | A HOG job on a GPU runtime. Harmless, but switch to CPU and free the card |
| Fingerprint differs between sessions | The manifest or the training images differ. Compare `train_manifest.csv` and the image count; re-stage from one bundle |
| Sweep or grid stops on an assertion | `model_resnet_stage1.pt` is not in `models/task1/checkpoints/<arm>/`. Unpack the `resnet` session's zip first |
| Unexpected retraining in the combine run | A checkpoint is missing or was written under a different fingerprint. Check the arm directory it landed in |
| Backbone has `num_batches_tracked=1` | That is the untrained shape-check probe, not the trained model. Use the real checkpoint |
| An hour of work lost on disconnect | Expected without Drive: the runtime's disk is the only copy. See [Disconnects](#things-that-will-bite) for the manual `epoch_*.pt` mitigation |
| A checkpoint zip you dragged in was ignored | It names the other arm. The staging cell lists these rather than merging them; `COLAB_ARM` and the zip's suffix must agree |
| **Kaggle:** `No data found` and `Datasets mounted right now: none` | The Dataset is not attached to this notebook. Right-hand panel → **+ Add Input** |
| **Kaggle:** `No data found` but the Dataset *is* attached | Its tree is nested deeper than one level, so `src/preprocessing.py` was not found. Rebuild the Dataset from the zip rather than from a folder |
| **Kaggle:** the data was copied instead of linked | That runtime refuses symlinks. It still works; it just spends the working quota. Nothing to fix |
| **Kaggle:** `pip install` fails in the dependency cell | Internet is off by default. Notebook options → Internet. Normally unnecessary — Kaggle ships every dependency |

---

## Differences from `notebooks/Task1/`

The complete list, all applied by `scripts/make_task1_colab.py`:

| Change | Why |
|---|---|
| Four cells prepended: header, control panel, dependency check, staging | Colab starts empty; the runtime has to be told where everything is |
| One cell appended: zip the results and download them | Nothing on a runtime's local disk outlives the runtime, and there is no Drive copy |
| The staging cell imports any `checkpoints_*_<arm>.zip` it finds | How a worker session's output reaches the sweep, grid and combine sessions without a shared filesystem. Searches `PROJECT_ROOT` and, on Kaggle, every mounted Dataset |
| `PROJECT_ROOT` and the platform are detected at runtime | The same ten files run on Colab and Kaggle with no edit. Kaggle mounts its data read-only, so the extracted tree is symlinked into the writable root instead of unpacked |
| `REPO_ROOT` resolves from `PROJECT_ROOT`, with the original walk-up as fallback | A Colab notebook is not inside a checkout. The fallback means these files still run locally |
| `ARM` and `ALLOW_CPU` read the control panel | So the arm is one string at the top rather than an edit buried in Section 1.1 |
| The arms CSV's `relative_path` goes through `resolve_external` | It records `notebooks/Task1/dataset1/...`; the CSV cannot be rewritten because its directory name is a content hash of the split. Both layouts are accepted |
| `ALLOW_CPU = True` in `worker_hog_svm` and `worker_hogsearch` | They are liblinear jobs with no GPU path |
| Stored outputs stripped | Colab renders its own run |
| **Evaluation only:** `notebooks/Task1/dataset{1,2}/` → `dataset{1,2}/` | The flat layout |
| **Evaluation only:** `CHECKPOINTS` gains the arm subdirectory | The source notebook still points at `models/task1/checkpoints/` with no arm, which has not existed since the Section 11 experiment moved checkpoints under `checkpoints/<arm>/`. Corrected here; **the source notebook still carries the original line** |

Nothing under `notebooks/Task1/`, `models/`, or `scripts/check_task1_workers.py` is modified
by any of this. `check_task1_workers.py` is scoped to `notebooks/Task1/` and is unaffected by
this folder.

## See also

- [`notebooks/Task1/PARALLEL_RUN.md`](../notebooks/Task1/PARALLEL_RUN.md) — the local parallel
  run, and the source of the job table and the schedules above
- [`docs/SUGGESTED_PIPELINE.md`](../docs/SUGGESTED_PIPELINE.md) — the grids and their result
  paths
- [`datasets/README.md`](../datasets/README.md) — counts and path details for the supplied data
- [`artifacts/README.md`](../artifacts/README.md) — what to hand over at the end
