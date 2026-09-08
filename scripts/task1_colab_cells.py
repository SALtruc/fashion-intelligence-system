"""The cells the Colab edition adds to each Task 1 notebook, as source text.

Kept apart from `make_task1_colab.py` for the same reason `task1_layout.py` is kept apart
from `make_task1_workers.py`: this file is *notebook source*, not generator logic, and the
two read very differently. Everything here is a raw string, because these templates contain
`\\n` sequences that must reach the generated notebook as an escape inside its own string
literals rather than being consumed on the way.

Placeholders are `__UPPERCASE__` tokens replaced by `str.replace`, not `str.format` fields.
The cells are dense with braces -- dict literals, f-strings, set literals -- and doubling
every one of them to survive `.format()` is unreadable and easy to get wrong.

No cell here touches Google Drive. The cells run on both Colab and Kaggle: the platform is
detected at runtime, and the only thing that really differs is how the data arrives -- a zip
you drag into a Colab session, an already-extracted read-only Dataset on Kaggle.
"""

# --- Markdown headers ----------------------------------------------------------------------

HEADER_COMBINE = """# Task 1 on Colab / Kaggle -- combine

Runs the **whole** notebook: restores every banked checkpoint, trains anything still
missing, then all of Section 7-11's analysis, the diagnostics, the model export and the
test-set predictions. This is the notebook that produces the deliverables.

Run it **after** the worker sessions have finished. Drag every session's
`checkpoints_<job>_<arm>.zip` into this session alongside the data bundle; the staging cell
unpacks them into `models/task1/checkpoints/<arm>/` and lists what it found. Anything without
a checkpoint is trained here instead, which is correct but slow.

**Runtime:** GPU (a T4 is enough). **Reads:** everything, including `datasets/test/`.

Get `task1_colab_data.zip` into the session, set `COLAB_ARM` in the control panel, then Run
all. With both arms banked, run it twice -- once per arm -- and the second
pass prints the Section 11 comparison table.

See `README.md` in this folder for the file manifest and the session schedule.
"""

HEADER_WORKER = """# Task 1 on Colab / Kaggle -- worker: {subtitle}

Trains **{job}** only, banks its checkpoints, and stops. Nothing here analyses, scores
against other models or writes predictions -- that is the combine notebook's job. Every
cell below the setup block is byte-identical to the combine notebook, which is what makes
the checkpoints interchangeable between sessions.

**Runtime:** {runtime_type}. **Estimated wall time:** {runtime} (measured on Apple Silicon
MPS at fp32; re-measure on your own GPU before scheduling against it).
{prerequisites}
**Before running:** get `task1_colab_data.zip` into the session -- on Colab drag it onto
`/content` with the sidebar file browser, on Kaggle add it as a Dataset with + Add Input --
then set `COLAB_ARM` in the control panel and Run all.

**After running:** the last cell writes `checkpoints_{job}_<arm>.zip`. On Colab it downloads
through the browser and you should **run it before closing the tab**, because the runtime's
disk is deleted with the session; on Kaggle it lands in `/kaggle/working` and is saved with
the notebook output. Either way that zip is how you hand this job to the next session.

See `README.md` in this folder for the file manifest and the session schedule.
"""

HEADER_EVAL = """# Task 1 on Colab / Kaggle -- independent evaluation

Scores the deployed model against the external cosmetics collections in `dataset1/` and
`dataset2/`. Trains nothing, takes a few minutes, writes figures 10 and 11.

Run it **after** the combine notebook. It loads `model_resnet_decoupled.pt` from the arm's
checkpoint directory and lifts the `SmallResNet` class straight out of `{combine}`, so both
have to be in the session: upload the data bundle, then drag in the checkpoint zip that
carries `model_resnet_decoupled.pt`.

**Runtime:** GPU preferred, CPU works. **Reads:** `dataset1/`, `dataset2/`, the checkpoints.

See `README.md` in this folder for the file manifest and the session schedule.
"""

SETUP_HEADING = """## 0. Colab / Kaggle setup

Three cells, none of them part of the assignment: the settings for this session, a dependency
check, and staging the data. They detect the platform themselves, so the same notebook runs on
both. Everything after them is the repository notebook, unchanged.
"""


# --- Code cells ------------------------------------------------------------------------------

CONTROL_PANEL = r'''# =========================================================================================
# Control panel -- the only cell in this notebook you edit
# =========================================================================================
# Everything below this cell is the assignment notebook as it stands in the repository.
# Nothing here enters RUN_FINGERPRINT, so a checkpoint trained under these settings is
# accepted by a machine running from an ordinary checkout, and the other way round.

# Which arm of the Section 11 external-data experiment this session trains.
#   "supplied"  the ordinary run: supplied rows only.
#   "enriched"  adds the 697 external crops from preprocessed_datasets/task1_dataset1_arms/.
# Every session in one pass must agree. The two arms write to separate checkpoint
# directories and carry different fingerprints, so neither can contaminate the other.
COLAB_ARM = "supplied"

# The runtime's writable working root, and what REPO_ROOT resolves to. "auto" picks
# /kaggle/working on Kaggle, /content on Colab, and the current directory anywhere else.
# Give it a path to override.
PROJECT_ROOT = "auto"

# The data bundle. On Colab, drag it into the session and leave this alone. On Kaggle you add
# it as a Dataset instead and Kaggle extracts it for you, so there is no zip to name -- the
# staging cell finds the extracted copy under /kaggle/input and this setting goes unused.
# Neither platform reads from or writes to Google Drive.
DATA_ZIP = "task1_colab_data.zip"

# Colab only: if the bundle is not already in the session, open a browser file picker instead
# of stopping. Convenient, but the picker is slower than the sidebar and drops large uploads
# more often, and this file is around 590 MB. Prefer the sidebar.
UPLOAD_IF_MISSING = False

# A silent CPU fallback on a GPU job presents as a hang rather than as an error, so the
# notebook refuses to start instead of running a hundred times too slowly.
__ALLOW_CPU_LINE__

# --- Runtime report ----------------------------------------------------------------------
import os
import shutil
import subprocess
import sys
from pathlib import Path

ON_KAGGLE = Path("/kaggle/working").is_dir()
IN_COLAB = not ON_KAGGLE and ("google.colab" in sys.modules or Path("/content").is_dir())
PLATFORM = "Kaggle" if ON_KAGGLE else "Colab" if IN_COLAB else "local"

if PROJECT_ROOT == "auto":
    PROJECT_ROOT = ("/kaggle/working" if ON_KAGGLE else
                    "/content" if IN_COLAB else str(Path.cwd()))

# The two automatic roots always exist, but a PROJECT_ROOT typed in by hand need not, and
# disk_usage below would then fail on a path nothing has created yet. The staging cell makes
# this directory anyway; doing it here as well costs nothing and moves the failure off a
# reporting line that has no business being the first thing to break.
Path(PROJECT_ROOT).mkdir(parents=True, exist_ok=True)

print(f"Platform         : {PLATFORM}")
print(f"PROJECT_ROOT     : {PROJECT_ROOT}")
print(f"Python           : {sys.version.split()[0]}")
print(f"CPU              : {os.cpu_count()} logical cores")
print(f"Disk free        : {shutil.disk_usage(PROJECT_ROOT).free / 1e9:.1f} GB")
print(f"Arm              : {COLAB_ARM}")
print(f"ALLOW_CPU        : {COLAB_ALLOW_CPU}")

_smi = shutil.which("nvidia-smi")
_gpu = ""
if _smi:
    _gpu = subprocess.run([_smi, "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
print("GPU              :", _gpu or "none visible")

# A second card is worse than useless under Run All: a notebook kernel is one process and
# uses one GPU, so a T4 x2 session bills two cards, trains on one, and that one T4 is slower
# than the single P100 the same menu offers. Said here, at the top, rather than after the
# imports -- by then the session is already committed.
_gpu_count = len([_line for _line in _gpu.splitlines() if _line.strip()])
if _gpu_count > 1:
    print(f"\n  {_gpu_count} GPUs are attached, but Run All uses ONE of them: a notebook kernel\n"
          "  is a single process. Two ways forward, and the first is usually right:\n"
          "    - For an ordinary Run All, switch to the single-GPU accelerator (P100 on\n"
          "      Kaggle). One P100 beats one T4, and you stop paying quota for an idle card.\n"
          "    - To actually use both, do not Run All. Run this instead, in a new cell:\n"
          "        !python scripts/task1_torchrun.py worker_<job>.ipynb\n"
          "      which starts one process per GPU. See 'Using both GPUs' in the README.")

if not _gpu and not COLAB_ALLOW_CPU:
    if ON_KAGGLE:
        print("\n  This notebook needs a GPU. Notebook options -> Accelerator -> GPU P100\n"
              "  (or GPU T4 x2 if you intend to use the torchrun launcher), then Run All again.")
    else:
        print("\n  This notebook needs a GPU. Runtime -> Change runtime type -> T4 GPU,\n"
              "  then Runtime -> Run all again.")
if _gpu and COLAB_ALLOW_CPU:
    print("\n  This job has no GPU path. You are holding a GPU it will not use;\n"
          "  switch this session to a CPU runtime and give the GPU to another job.")
'''

ENVIRONMENT = r'''# =========================================================================================
# Dependencies
# =========================================================================================
# Colab ships every one of these, so the ordinary path installs nothing and this cell is a
# version record rather than a setup step. It matters because the fitted SVM is persisted
# with joblib, and a scikit-learn pickle is only reliably readable by the version that wrote
# it: run hog_svm, hogsearch and the combine step on the same platform, or expect the
# restore to warn and possibly fail. The .pt checkpoints carry no such constraint.

# --- Thread limits, before anything loads a BLAS -----------------------------------------
# OpenMP, MKL and OpenBLAS each read their thread count once, when the shared library is
# first loaded, and ignore every later change to the environment. Section 1.0 below sets
# these and is the authority on them -- but it runs after this cell, and THIS is the cell
# that first imports numpy and torch. Setting them only there would be a no-op that looks
# like it worked, and the run would quietly use the library defaults instead of every core.
# The value matches Section 1.0's exactly, so the later assignment is a harmless restatement.
import os  # noqa: E402
import sys  # noqa: E402

try:
    _cores = len(os.sched_getaffinity(0))
except AttributeError:                 # not on Linux; no affinity mask to read
    _cores = os.cpu_count() or 1

for _variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_variable] = str(_cores)
os.environ["LOKY_MAX_CPU_COUNT"] = str(_cores)
print(f"Thread limits pinned to {_cores} cores before the first BLAS import.")

_required = {"numpy": "numpy", "pandas": "pandas", "torch": "torch",
             "sklearn": "scikit-learn", "skimage": "scikit-image",
             "matplotlib": "matplotlib", "seaborn": "seaborn", "joblib": "joblib"}

_missing = []
for _module, _package in _required.items():
    try:
        __import__(_module)
    except ImportError:
        _missing.append(_package)

if _missing:
    # Needs the runtime to have internet, which is on by default on Colab and off by default
    # on Kaggle.
    print("Installing:", " ".join(_missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_missing], check=True)
else:
    print("All dependencies already present; nothing installed.")

import numpy  # noqa: E402
import pandas  # noqa: E402
import skimage  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402

print(f"\nnumpy {numpy.__version__} | pandas {pandas.__version__} | "
      f"scikit-learn {sklearn.__version__} | scikit-image {skimage.__version__}")
print(f"torch {torch.__version__} | CUDA build {torch.version.cuda} | "
      f"CUDA available {torch.cuda.is_available()}")
print("\nRecord these versions alongside the run: uv.lock does not pin them here.")
'''

STAGING = r'''# =========================================================================================
# Staging the uploaded data
# =========================================================================================
# The layout this notebook expects, flattened so everything sits directly under
# PROJECT_ROOT rather than inside a repository checkout:
#
#     /content/
#     |- src/preprocessing.py
#     |- src/task1_ddp.py
#     |- preprocessed_datasets/train_manifest.csv
#     |- preprocessed_datasets/task1_dataset1_arms/<version>/     (enriched arm)
#     |- datasets/train/images_train/*.jpg
#     |- datasets/test/styles_prediction.csv, datasets/test/images_test/   (combine)
#     |- dataset1/, dataset2/                                     (enriched arm, evaluation)
#     |- models/, outputs/, predictions/                          (created here)
#
# Everything lives on the runtime's local disk. Nothing is read from or written to Google
# Drive, and nothing here outlives the runtime -- see the warning this cell prints.

PROJECT = Path(PROJECT_ROOT)
PROJECT.mkdir(parents=True, exist_ok=True)

KAGGLE_INPUT = Path("/kaggle/input")


def input_datasets():
    """The read-only dataset directories Kaggle has mounted, outermost first."""
    if not KAGGLE_INPUT.is_dir():
        return []
    return sorted(path for path in KAGGLE_INPUT.iterdir() if path.is_dir())


def extracted_bundle():
    """The bundle's flat tree, if the platform has already unpacked it for us.

    Kaggle unzips an added Dataset itself and mounts the result at /kaggle/input/<slug>/,
    read-only. So on that platform there is often no archive to unpack -- the tree is simply
    already there, on a filesystem this notebook cannot write to. Finding it here is what
    lets the staging step link it into the writable root instead of copying 590 MB into the
    working quota.

    The tree is located by its marker rather than by an assumed shape, because how deeply a
    Dataset nests depends on how it was built -- uploading the zip, a folder, or a folder of
    folders all give different answers.
    """
    for dataset in input_datasets():
        if (dataset / "src" / "preprocessing.py").is_file():
            return dataset
        for marker in sorted(dataset.rglob("src/preprocessing.py")):
            return marker.parent.parent
    return None


def bundle_archive():
    """The bundle still archived, wherever it is: the working root or a mounted Dataset.

    Kaggle usually extracts an uploaded archive, but not always -- a Dataset can carry the
    zip itself, and then /kaggle/input holds a file rather than a tree. Unpacking it into the
    writable root costs a minute and is better than failing.
    """
    explicit = Path(DATA_ZIP) if Path(DATA_ZIP).is_absolute() else (PROJECT / DATA_ZIP)
    if explicit.is_file():
        return explicit
    for dataset in input_datasets():
        for candidate in sorted(dataset.rglob(Path(DATA_ZIP).name)):
            return candidate
    # Any other archive that is plainly not a checkpoint hand-over.
    for dataset in input_datasets():
        for candidate in sorted(dataset.rglob("*.zip")):
            if not candidate.name.startswith("checkpoints_"):
                return candidate
    return None


def describe_inputs():
    """What is actually mounted, for an error message that ends the guessing."""
    lines = []
    for dataset in input_datasets():
        entries = sorted(dataset.iterdir())
        shown = ", ".join(entry.name for entry in entries[:8])
        if len(entries) > 8:
            shown += f", ... (+{len(entries) - 8} more)"
        lines.append(f"            {dataset.name}/  ->  {shown or '<empty>'}")
    return "\n".join(lines) or "            none"


# --- Stage the data, once per session ----------------------------------------------------
if (PROJECT / "src" / "preprocessing.py").is_file():
    print("Data already staged in this session; skipping the unpack.")

elif extracted_bundle() is not None:
    # Kaggle. Symlink rather than copy: /kaggle/input is read-only but perfectly readable, so
    # copying it would spend minutes and most of the working quota to gain nothing. The run
    # only ever writes to models/, outputs/ and predictions/, which are created below as real
    # directories and are not part of this tree.
    _source = extracted_bundle()
    print(f"Found the data already extracted at {_source}")
    _linked, _copied = 0, 0
    for _entry in sorted(_source.iterdir()):
        _target = PROJECT / _entry.name
        if _target.exists() or _target.is_symlink():
            continue
        try:
            _target.symlink_to(_entry, target_is_directory=_entry.is_dir())
            _linked += 1
        except OSError:
            # Some runtimes refuse symlinks. Copying is slower and spends the working
            # quota, but it is correct, so it is the fallback rather than a failure.
            if _entry.is_dir():
                shutil.copytree(_entry, _target)
            else:
                shutil.copy2(_entry, _target)
            _copied += 1
    print(f"Staged {_linked + _copied} entries into {PROJECT}: "
          f"{_linked} linked, {_copied} copied.")
    if _copied:
        print("  This runtime does not allow symlinks, so the data was copied instead. That "
              "is\n  correct but slower, and it spends the working quota.")

else:
    _zip = bundle_archive()

    if _zip is None and UPLOAD_IF_MISSING and IN_COLAB:
        from google.colab import files
        print(f"{DATA_ZIP} is not in this session. Choose it in the picker below.")
        _uploaded = files.upload()
        _zip = PROJECT / next(iter(_uploaded))

    if _zip is None:
        raise FileNotFoundError(
            "No data found. This notebook needs the bundle built by "
            "`python scripts/make_colab_bundle.py`\n"
            "from a checkout -- it is ~590 MB and contains src/, preprocessed_datasets/, "
            "datasets/,\ndataset1/ and dataset2/ at its top level.\n\n"
            f"  Colab   drag {Path(DATA_ZIP).name} onto {PROJECT} with the sidebar file "
            f"browser.\n"
            f"          Set UPLOAD_IF_MISSING = True for a picker instead, or point DATA_ZIP "
            f"at it.\n\n"
            "  Kaggle  add the bundle as a Dataset (+ Add Input in the right-hand panel).\n"
            "          Upload the ZIP ITSELF when creating the Dataset -- not the datasets/ "
            "folder,\n"
            "          and not the repository. Kaggle extracts it for you.\n\n"
            "          Looked for src/preprocessing.py at any depth, and for a bundle zip, "
            "in each\n"
            "          of these. What is mounted right now:\n\n"
            f"{describe_inputs()}\n\n"
            "See README.md in the task1-collab folder."
        )

    print(f"Unpacking {_zip.name} ({_zip.stat().st_size / 1e6:.0f} MB) into {PROJECT} ...")
    shutil.unpack_archive(str(_zip), str(PROJECT))
    print("Unpacked.")

# A bundle built before the multi-GPU support was added stages cleanly and then fails at the
# import in Section 1.3, an hour later on a Kaggle commit run. Naming it here instead.
if not (PROJECT / "src" / "task1_ddp.py").is_file():
    raise FileNotFoundError(
        f"{PROJECT}/src/task1_ddp.py is missing, so this bundle predates the multi-GPU "
        "support that Section 1.3 imports. "
        "Rebuild it with `python scripts/make_colab_bundle.py` and re-upload, or drop a copy "
        "of src/task1_ddp.py next to src/preprocessing.py if you only need this one session."
    )

for _name in ("models", "outputs", "predictions"):
    (PROJECT / _name).mkdir(parents=True, exist_ok=True)

# --- Import any checkpoints handed over from another session -----------------------------
# Bring a checkpoints_<job>_<arm>.zip from an earlier session into this one -- dragged onto
# PROJECT on Colab, added as a Dataset on Kaggle -- and it is unpacked into this arm's
# checkpoint directory here. That is how the stage-1 backbone reaches the sweep and grid jobs,
# and how every worker's output reaches the combine run. Files are keyed by name, so several
# sessions' zips merge into one directory without collisions.
CHECKPOINT_IMPORT_DIR = PROJECT / "models" / "task1" / "checkpoints" / COLAB_ARM
CHECKPOINT_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

# Kaggle extracts a zip added as a Dataset, so a checkpoint set may arrive there either still
# archived or already unpacked. Both are handled: the archives below, the loose files after.
_search_dirs = [PROJECT] + input_datasets()

_handed_over = [path for directory in _search_dirs
                for path in sorted(directory.glob("checkpoints_*_" + COLAB_ARM + ".zip"))]
for _archive_path in _handed_over:
    shutil.unpack_archive(str(_archive_path), str(CHECKPOINT_IMPORT_DIR))
    print(f"Imported {_archive_path.name}")

# Loose files carry no arm in their own names, so the Dataset has to say. A checkpoint from
# the other arm would be refused at load time by the fingerprint check -- but it would still
# be sitting in this arm's directory, and the export cell zips that directory wholesale, so
# the wrong-arm file would be handed on to the next session. Requiring the evidence up front
# makes that impossible rather than merely unlikely.
_unlabelled = []
for _directory in input_datasets():
    _loose_files = [path for path in sorted(_directory.glob("*"))
                    if path.suffix in {".pt", ".joblib", ".npz"}]
    if not _loose_files:
        continue

    # Evidence, in the order it is trusted: the Dataset's own name, or a marker file sitting
    # beside the checkpoints. An arm named nowhere is not guessed at.
    _names = f"{_directory.name} " + " ".join(path.name for path in _directory.glob("*.txt"))
    _says_this_arm = COLAB_ARM in _names.lower()
    _says_other_arm = any(arm in _names.lower()
                          for arm in ("supplied", "enriched") if arm != COLAB_ARM)

    if not _says_this_arm:
        _unlabelled.append((_directory.name, len(_loose_files), _says_other_arm))
        continue

    for _loose in _loose_files:
        _target = CHECKPOINT_IMPORT_DIR / _loose.name
        if not _target.exists():
            shutil.copy2(_loose, _target)
            print(f"Imported {_loose.name} from {_directory.name}")

for _name, _count, _other in _unlabelled:
    _why = (f"it names the other arm" if _other
            else f"nothing in it names an arm, and {COLAB_ARM!r} is not guessed at")
    print(f"\nSkipped {_count} loose checkpoint(s) in Dataset {_name!r}: {_why}.")
    if not _other:
        print(f"  If they are {COLAB_ARM!r}, rename the Dataset to include {COLAB_ARM!r} and "
              f"re-run,\n  or upload them as checkpoints_<job>_{COLAB_ARM}.zip, which carries "
              "the arm in its name.")

_banked_now = sorted(path.name for path in CHECKPOINT_IMPORT_DIR.iterdir()
                     if path.suffix in {".pt", ".joblib", ".npz"}
                     and not path.name.startswith("epoch_"))
if _banked_now:
    print(f"Checkpoints present for arm {COLAB_ARM!r}: {len(_banked_now)}")
    for _name in _banked_now:
        print("   ", _name)

# A zip for the other arm is a mistake worth naming: its checkpoints carry a different
# fingerprint and would be refused rather than used, after the session had already run.
_other_arm = [path.name for directory in _search_dirs
              for path in sorted(directory.glob("checkpoints_*.zip"))
              if path not in _handed_over]
if _other_arm:
    print("\nIgnored, they belong to the other arm:", ", ".join(_other_arm))


def resolve_external(relative):
    """Locate an external-collection image from the path recorded in the arms CSV.

    `external_train.csv` stores `notebooks/Task1/dataset1/images/<id>.jpg`, because that is
    where the collection sits in a checkout. Rewriting the CSV is not an option: its
    directory name is a content hash of the split and `make_dataset1_arms.py --check`
    verifies it byte for byte. So the path is resolved rather than the file rewritten, and
    both layouts are accepted -- the nested one a checkout has, and the flat one the bundle
    unpacks. REPO_ROOT is read at call time because it is defined below this cell.
    """
    candidate = REPO_ROOT / relative
    if candidate.is_file():
        return str(candidate)
    parts = Path(relative).parts
    for _collection in ("dataset1", "dataset2"):
        if _collection in parts:
            return str(REPO_ROOT.joinpath(*parts[parts.index(_collection):]))
    return str(candidate)


# --- Verify now, rather than an hour into the run ----------------------------------------
REQUIRED = __REQUIRED__
if COLAB_ARM == "enriched":
    REQUIRED = REQUIRED + __REQUIRED_ENRICHED__

_absent = [item for item in REQUIRED if not (PROJECT / item).exists()]
if _absent:
    raise FileNotFoundError(
        "Missing from " + str(PROJECT) + ":\n  " + "\n  ".join(_absent)
        + "\n\nThe bundle is incomplete for this notebook and this arm. Rebuild it with "
          "`python scripts/make_colab_bundle.py` from a checkout that has the data, and see "
          "the file manifest in README.md."
    )

_train_images = len(list((PROJECT / "datasets/train/images_train").glob("*.jpg")))
print(f"\nStaged: {len(REQUIRED)} required paths present | {_train_images:,} training images")
if _train_images < 38000:
    print(f"  WARNING: expected 38,612 training images, found {_train_images:,}. The bundle "
          "looks truncated,\n  and the split -- so the fingerprint -- will not match the other "
          "sessions.")

if ON_KAGGLE:
    print("\n  /kaggle/working becomes this notebook's output when you Save Version, and is\n"
          "  carried between interactive sessions only with Persistence on (Notebook options\n"
          "  -> Persistence). Run the last cell either way: it collects the results into one\n"
          "  zip, which is what the next session expects to be handed.")
else:
    print("\n  Everything this notebook writes lives on the runtime's local disk and is DELETED\n"
          "  when the runtime disconnects. Run the last cell to download the results before\n"
          "  closing the tab, and do not leave a finished session idle.")
'''

EXPORT_WORKER = r'''# =========================================================================================
# Hand the checkpoints back -- RUN THIS BEFORE CLOSING THE TAB
# =========================================================================================
# The runtime's disk is deleted when the session ends, so this zip is the only copy of the
# work that survives it. It downloads through the browser.
#
# One zip per session, named for the job and the arm, so several sessions' zips unpack into
# one directory without collisions. To use them in another session -- the combine run, or a
# sweep that needs this job's backbone -- drag the zips into that session alongside the data
# bundle and its staging cell imports them.

import zipfile

_zip_path = PROJECT / ("checkpoints___JOB___" + COLAB_ARM + ".zip")

# epoch_*.pt files are resume state, not results, and are excluded deliberately.
_banked = sorted(path for path in CHECKPOINT_DIR.iterdir()
                 if path.suffix in {".pt", ".joblib", ".npz"}
                 and not path.name.startswith("epoch_"))
if not _banked:
    raise FileNotFoundError(
        f"No banked model files in {CHECKPOINT_DIR}. The job did not complete, so there is "
        "nothing to hand back."
    )

# ZIP_STORED, not DEFLATED: these are already-compressed tensors, so deflating them spends
# minutes to save almost nothing.
with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as _archive:
    for _path in _banked:
        _archive.write(_path, _path.name)

print(f"Wrote {_zip_path}\n")
_total = 0.0
for _path in _banked:
    _size = _path.stat().st_size / 1e6
    _total += _size
    print(f"  {_path.name}  ({_size:.1f} MB)")
print(f"\n{_total:.0f} MB in {len(_banked)} files, arm {COLAB_ARM!r}.")
print("Fingerprint:", RUN_FINGERPRINT, "-- must match every session in this pass.")

_epochs = sorted(CHECKPOINT_DIR.glob("epoch_*.pt"))
if _epochs:
    print(f"\n{len(_epochs)} epoch_*.pt resume file(s) left behind; they are not results.")

if IN_COLAB:
    from google.colab import files
    print("\nStarting the download. Keep this tab open until the browser has the file.")
    files.download(str(_zip_path))
elif ON_KAGGLE:
    print(f"\nOn Kaggle: {_zip_path.name} is in /kaggle/working. Download it from the Data\n"
          "panel on the right while this session is alive, or Save Version and take it from\n"
          "the run's Output. To hand it to another Kaggle session, publish that output as a\n"
          "Dataset and add it there with + Add Input.")
else:
    print(f"\nCollect {_zip_path} yourself.")
'''

EXPORT_COMBINE = r'''# =========================================================================================
# Collect the deliverables -- RUN THIS BEFORE CLOSING THE TAB
# =========================================================================================
# The runtime's disk is deleted when the session ends. Everything the report and the
# submission need, in one archive, downloaded through the browser.

import zipfile

_zip_path = PROJECT / ("task1_outputs_" + COLAB_ARM + ".zip")

_wanted = []
for _directory, _patterns in ((ARTEFACT_DIR, ("*.csv", "*.json", "*.pt")),
                              (PREDICTION_DIR, ("*.csv", "*.npy")),
                              (FIGURE_DIR, ("*.png",))):
    for _pattern in _patterns:
        _wanted += sorted(_directory.glob(_pattern))

with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as _archive:
    for _path in _wanted:
        _archive.write(_path, str(_path.relative_to(PROJECT)))

print(f"Wrote {_zip_path} ({_zip_path.stat().st_size / 1e6:.0f} MB)\n")
for _path in _wanted:
    print(f"  {_path.relative_to(PROJECT)}  ({_path.stat().st_size / 1e6:.1f} MB)")
print(f"\n{len(_wanted)} files, arm {COLAB_ARM!r}.")
print("\nThe training checkpoints are NOT in this archive: they are inputs, not deliverables,\n"
      "and they are several hundred MB. If you want to keep them -- to re-run the combine\n"
      "step later without retraining -- download models/task1/checkpoints/ separately from\n"
      "the file browser before this runtime ends.")

if IN_COLAB:
    from google.colab import files
    print("\nStarting the download. Keep this tab open until the browser has the file.")
    files.download(str(_zip_path))
elif ON_KAGGLE:
    print(f"\nOn Kaggle: {_zip_path.name} is in /kaggle/working. Download it from the Data\n"
          "panel on the right while this session is alive, or Save Version and take it from\n"
          "the run's Output. To hand it to another Kaggle session, publish that output as a\n"
          "Dataset and add it there with + Add Input.")
else:
    print(f"\nCollect {_zip_path} yourself.")
'''

EXPORT_EVAL = r'''# =========================================================================================
# Collect the figures -- RUN THIS BEFORE CLOSING THE TAB
# =========================================================================================
import zipfile

_zip_path = PROJECT / "task1_independent_evaluation.zip"
_figures = sorted(FIGURES.glob("1[01]_*.png"))

with zipfile.ZipFile(_zip_path, "w", zipfile.ZIP_DEFLATED) as _archive:
    for _path in _figures:
        _archive.write(_path, _path.name)

print(f"Wrote {_zip_path}")
for _path in _figures:
    print(f"  {_path.name}")

if IN_COLAB:
    from google.colab import files
    files.download(str(_zip_path))
elif ON_KAGGLE:
    print(f"\nOn Kaggle: {_zip_path.name} is in /kaggle/working. Download it from the Data\n"
          "panel on the right while this session is alive, or Save Version and take it from\n"
          "the run's Output. To hand it to another Kaggle session, publish that output as a\n"
          "Dataset and add it there with + Add Input.")
else:
    print(f"\nCollect {_zip_path} yourself.")
'''

# The two spellings of the ALLOW_CPU line, and the runtime each implies.
ALLOW_CPU_GPU = (
    "COLAB_ALLOW_CPU = False    # this job needs a GPU and refuses to start without one."
)
ALLOW_CPU_CPU = (
    "COLAB_ALLOW_CPU = True     # this job is CPU-only: liblinear, with no GPU path at\n"
    "                           # all. Give this session a CPU runtime and leave the\n"
    "                           # GPU for a job that can use it."
)
RUNTIME_TYPE_GPU = "GPU, a T4 is enough"
RUNTIME_TYPE_CPU = "CPU only -- this job has no GPU path, so do not spend a GPU on it"
