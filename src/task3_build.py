# %% [markdown]
# # Task 3 — `gender` and `usage`
#
# COSC2753 Assignment 2 · group SG_G3
#
# The brief asks for a system that predicts **both** `gender` and `usage`. It does
# not say whether that is one model or two, which is the interesting part: that
# choice is the design decision this notebook is built to answer with evidence
# rather than preference.
#
# **What is compared**
#
# | | design | backbones | outputs |
# |---|---|---|---|
# | **A** | two independent models | 2 | 5 · 8 |
# | **B** | one model, joint label | 1 | 24 `gender × usage` pairs that occur |
# | **C** | one shared backbone, two heads | 1 | 5 + 8 |
#
# Identical convolutional body, identical seed, identical schedule. The only thing
# that changes between A, B and C is how the two labels are attached to it, so the
# difference between them **is** the effect of that choice and nothing else.
#
# **Three further questions, each one variable at a time**
#
# 1. does class-weighted loss help the long tail, or just trade the head for it?
# 2. does the externally collected data help? (it is predicted to *hurt* `usage` —
#    see §7, and the prediction is recorded here before the run)
# 3. how much of the score survives a split that mimics the real test set?
#
# **Metric.** macro-F1 throughout. Accuracy is reported beside it only to show why
# it must not be used: predicting `Casual` for every row scores **76.7%** accuracy
# on `usage` and a macro-F1 of **0.109**.

# %% [markdown]
# ## 0 · Setup
#
# Runs on Colab or locally. On Colab, mount Drive first; the cell below finds the
# data wherever it ended up.

# %%
import os, sys, json, time, math, random, zipfile, warnings
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")

SEED = 42
IMAGE_SIZE = (60, 80)          # width, height — the provided dataset's exact size
TARGETS = ["gender", "usage"]

# QUICK=True runs everything end to end in a few minutes on a smaller sample, to
# prove the notebook works before spending an hour on the real numbers.
QUICK = False
# Headless smoke test: A2_QUICK=1 flips this without editing the file, so a CI-style
# run (python task3_build.py, or nbconvert --execute) can prove the pipeline works
# before committing an hour of GPU. On Colab, ignore this and edit the line above.
if os.environ.get("A2_QUICK") == "1":
    QUICK = True
EPOCHS = 6 if QUICK else 20
SAMPLE = 4000 if QUICK else None

random.seed(SEED); np.random.seed(SEED)

# Stamped by build_notebook.py from a hash of the source. Its whole job is to make a
# stale upload obvious in the first five seconds: the handover note says which BUILD
# to expect, so a notebook that is a regeneration behind announces itself here rather
# than failing seventy minutes later. "dev" means running the .py directly.
BUILD = "dev"
print(f"colab={IN_COLAB}  quick={QUICK}  epochs={EPOCHS}  BUILD={BUILD}")

# %% [markdown]
# ### 0.1 Staging the data — Colab only, skipped locally
#
# Reading 37,745 individual JPEGs through the Drive mount is the slowest thing in
# this notebook: every file is a separate API call, so it costs roughly half an hour
# *per session*. Copying one zip and unpacking it onto Colab's local disk turns that
# into about a minute.
#
# Set the two paths below to match your Drive. A shortcut in *My Drive* pointing at a
# Shared Drive works fine — Colab mounts My Drive only, so the shortcut is the bridge.
#
# Re-running is cheap: if the data is already unpacked, this does nothing.

# %%
# Searched in order, first hit wins. The zip has lived in both places across runs and
# hardcoding one of them is what broke the first Colab attempt, so both are tried.
COLAB_ZIP_CANDIDATES = [
    "/content/drive/MyDrive/A2_ExternalData/ColabDataset.zip",
    "/content/drive/MyDrive/ColabDataset.zip",
]
EXTERNAL_ON_DRIVE = "/content/drive/MyDrive/A2_ExternalData"

if IN_COLAB:
    import shutil
    import subprocess

    marker = "preprocessed_datasets/train/styles_train.csv"
    roots = [Path("/content"), Path("/content/ColabDataset")]

    if not any((r / marker).exists() for r in roots):
        src = next((Path(c) for c in COLAB_ZIP_CANDIDATES if Path(c).exists()), None)
        if src is None:
            listing = "\n  ".join(sorted(p.name for p in Path("/content/drive/MyDrive").iterdir()))
            raise FileNotFoundError(
                "none of these exist:\n  " + "\n  ".join(COLAB_ZIP_CANDIDATES)
                + f"\nMy Drive contains:\n  {listing}\n"
                "Add the real path to COLAB_ZIP_CANDIDATES above.")
        print(f"copying {src.stat().st_size / 1e6:.0f} MB from Drive ...")
        shutil.copy(src, "/content/_data.zip")
        # -o overwrites without asking. Without it a re-run stops on a 'replace?'
        # prompt that Colab has no way to answer, and the cell hangs silently.
        subprocess.run(["unzip", "-qo", "/content/_data.zip", "-d", "/content/"], check=True)
        print("unpacked")

    root = next((r for r in roots if (r / marker).exists()), None)
    if root is None:
        raise FileNotFoundError(f"unpacked, but no {marker} under {[str(r) for r in roots]}")
    os.environ["A2_COLAB_DATASET"] = str(root)

    if Path(EXTERNAL_ON_DRIVE).is_dir():
        os.environ["A2_EXTERNAL_DATA"] = EXTERNAL_ON_DRIVE

    # Verify the unpack finished. A truncated unzip otherwise fails ten minutes later,
    # in the middle of image decoding, with an error that points somewhere else.
    n_img = len(list((root / "preprocessed_datasets/train/images_train").glob("*.jpg")))
    n_csv = len(pd.read_csv(root / marker))
    print(f"train images {n_img:,} | csv rows {n_csv:,}")
    assert n_img == n_csv, (
        f"{n_csv - n_img} images missing -- the unzip was incomplete. "
        "Delete /content/preprocessed_datasets and re-run this cell.")
    print("data staged OK")
else:
    print("not on Colab - using local paths")

# %% [markdown]
# ### 0.2 Finding the data
#
# Two datasets are involved and they are **not** the same thing:
#
# * **the provided catalogue** — Trực's `ColabDataset`, which is the provided
#   training data de-duplicated: 38,617 → 37,745 rows.
# * **the externally collected images** — `A2_ExternalData`, added to *training
#   only* in §7.
#
# On Colab a folder someone shared with you sits under *Shared with me*, which is
# **not mounted**. Right-click it in Drive → **Add shortcut to Drive** first.

# %%
def _first_dir(candidates, must_contain=None):
    """Return the first candidate that exists (and holds `must_contain`)."""
    tried = []
    for c in candidates:
        c = Path(c); tried.append(str(c))
        if c.is_dir() and (must_contain is None or (c / must_contain).exists()):
            return c
    raise FileNotFoundError(
        "not found. Tried:\n  " + "\n  ".join(tried)
        + "\n\nOn Colab: 'Shared with me' is NOT mounted — right-click the folder in "
          "Drive, 'Add shortcut to Drive', then re-run."
    )

DATA_ROOT = _first_dir([
    os.environ.get("A2_COLAB_DATASET", "/nonexistent"),
    "/content",                 # a zip that unpacked flat -- the common case
    "/content/ColabDataset",
    "/content/drive/MyDrive/ColabDataset",
    "/content/drive/MyDrive/A2/ColabDataset",
    "/content/drive/MyDrive/[ML] SG_G3/A2/ColabDataset",
    "D:/ColabDataset",
], must_contain="preprocessed_datasets/train/styles_train.csv")

TRAIN_CSV = DATA_ROOT / "preprocessed_datasets" / "train" / "styles_train.csv"
TRAIN_IMG = DATA_ROOT / "preprocessed_datasets" / "train" / "images_train"

try:
    EXTERNAL_ROOT = _first_dir([
        os.environ.get("A2_EXTERNAL_DATA", "/nonexistent"),
        "/content",
        "/content/A2_ExternalData",
        "/content/drive/MyDrive/A2_ExternalData",
        "/content/drive/MyDrive/A2/Nguyen/A2_ExternalData",
        "/content/drive/MyDrive/[ML] SG_G3/A2/Nguyen/A2_ExternalData",
        "D:/g2/Dataset",
    ], must_contain="ExternalCosmetics")
except FileNotFoundError as exc:
    EXTERNAL_ROOT = None
    print("external data not found — §7 will be skipped\n", exc)

print("catalogue :", DATA_ROOT)
print("external  :", EXTERNAL_ROOT)

# %% [markdown]
# ### 0.3 The frame, and the split rule
#
# `ColabDataset` ships the raw catalogue columns, so `filename`, `path` and
# `group_id` are added here. `group_id = id` is correct **because** this file is
# already one row per product — the de-duplication that produced it is what makes
# the assumption safe, and the assertion below is what checks it rather than
# trusting it.
#
# The split function is copied verbatim from `src/preprocessing.make_split` so the
# split matches the rest of the team's (rule 2 in the repo README) without needing
# to clone the private repo inside Colab. If `src/` happens to be importable, the
# real one is used instead and the two are checked to agree.

# %%
from sklearn.model_selection import train_test_split

RANDOM_STATE = SEED

def make_split(frame, target, validation_share=0.2, random_state=RANDOM_STATE):
    """Copy of src/preprocessing.make_split — whole groups stay on one side."""
    if target not in frame.columns:
        raise KeyError(f"{target!r} is not a column of the frame passed in.")
    frame = frame.loc[frame[target].notna()].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"No rows carry a {target!r} label.")
    assert frame["group_id"].is_unique, "multi-row group_id values found"

    group_label = frame.groupby("group_id")[target].agg(
        lambda v: v.value_counts().index[0])
    counts = group_label.value_counts()
    splittable = group_label[~group_label.isin(counts[counts < 2].index)]

    _, validation_groups = train_test_split(
        splittable.index, test_size=validation_share,
        stratify=splittable.values, random_state=random_state)
    is_val = frame["group_id"].isin(set(validation_groups))
    return (frame.loc[~is_val].reset_index(drop=True),
            frame.loc[is_val].reset_index(drop=True))


def make_split_forward(frame, target, validation_share=0.2):
    """Hold out the HIGHEST ids instead of a random draw.

    The graded test set is the 5,829 highest ids, so the real task is extrapolation
    forward along a catalogue that was added to over time, not interpolation inside
    a shuffled pool. A random split silently assumes the two are the same. §8
    measures how far apart they actually are.
    """
    frame = frame.loc[frame[target].notna()].sort_values("id").reset_index(drop=True)
    cut = int(len(frame) * (1 - validation_share))
    return frame.iloc[:cut].reset_index(drop=True), frame.iloc[cut:].reset_index(drop=True)


frame = pd.read_csv(TRAIN_CSV)
frame["filename"] = frame["id"].astype(str) + ".jpg"
frame["path"] = frame["filename"].map(lambda n: str(TRAIN_IMG / n))
frame["group_id"] = frame["id"].astype(str)
assert frame["group_id"].is_unique, "ColabDataset is meant to be one row per product"

if SAMPLE:
    frame = frame.sample(SAMPLE, random_state=SEED).reset_index(drop=True)

print(f"{len(frame):,} rows, {frame['articleType'].nunique()} articleTypes")
print(frame[["id", "gender", "usage", "articleType"]].head(3).to_string(index=False))

# %% [markdown]
# ## 1 · What the task actually is
#
# Before any model: the two targets look like one task but behave like two very
# different ones, and every design decision later follows from this section.

# %%
def class_table(frame, col):
    vc = frame[col].value_counts()
    out = pd.DataFrame({"n": vc, "share %": (100 * vc / len(frame)).round(2)})
    out.index.name = col
    return out

for col in TARGETS:
    t = class_table(frame, col)
    ratio = t["n"].max() / t["n"].min()
    print(f"=== {col} — {len(t)} classes, imbalance {ratio:,.0f}x ===")
    print(t.to_string(), "\n")

# %% [markdown]
# `gender` has five classes and the smallest holds **483** images. `usage` has
# eight and the smallest holds **one**. They are not two instances of the same
# problem; `gender` is a learnable imbalanced problem and `usage` is a long tail
# with classes that cannot be learned from the data at all.
#
# ### 1.1 The ceiling this puts on macro-F1
#
# macro-F1 averages over classes, so a class that can never be predicted
# contributes a hard zero and caps the score no matter how good the model is.

# %%
rows = []
for col in TARGETS:
    vc = frame[col].value_counts()
    for thr in (2, 10, 30, 100):
        dead = int((vc < thr).sum())
        rows.append({"target": col, "classes with fewer than": thr,
                     "such classes": f"{dead}/{len(vc)}",
                     "macro-F1 ceiling": round(1 - dead / len(vc), 3)})
ceiling = pd.DataFrame(rows)
print(ceiling.to_string(index=False))

# %% [markdown]
# Read the `usage` rows carefully. **Four of its eight classes have under 100
# training images** — `Home` 1, `Party` 13, `Travel` 25, `Smart Casual` 55. If
# those four are unlearnable then macro-F1 cannot exceed **0.500**, and a reported
# 0.5 would mean the model got *everything else perfect*.
#
# This is the single most important fact about Task 3, and it says the interesting
# work is not architecture search — it is what to do about four classes that the
# data does not contain enough of to teach.

# %% [markdown]
# ### 1.2 Why accuracy must not be the metric here

# %%
rows = []
for col in TARGETS:
    vc = frame[col].value_counts()
    p = vc.iloc[0] / len(frame)
    # every class but the majority scores F1 = 0; the majority scores 2p/(1+p)
    rows.append({"target": col, "majority class": vc.index[0],
                 "accuracy": f"{100 * p:.1f}%",
                 "macro-F1": round(2 * p / (1 + p) / len(vc), 3)})
print("predicting the majority class and nothing else:\n")
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# A model that has learned *nothing* scores 76.7% accuracy on `usage`. Any accuracy
# figure below that is worse than useless, and any figure near it says nothing at
# all. Every number from here on is macro-F1; accuracy is shown alongside only as
# evidence of this gap.

# %% [markdown]
# ### 1.3 "macro-F1" is two different numbers — which matters when comparing notebooks
#
# `usage` has classes that can vanish from validation entirely (`Home` has one image
# in the whole training set, so a grouped split puts it in train and nowhere else).
# When that happens the two usual conventions disagree:
#
# | | denominator | what it says |
# |---|---|---|
# | `macro_f1` | **all 8** classes | a label we cannot predict scores 0 — an honest description of the task |
# | `macro_f1_in_val` | classes **present in validation** (7) | sklearn's default when `labels=` is not passed |
#
# Eight versus seven is a **~14% difference on the same predictions**, and neither
# number is wrong. So every result below carries both, with the denominator printed
# next to it.
#
# This matters for more than tidiness: a teammate's Task 3 figures are reported as
# *"combined val macro-F1"* — 0.6312 for the combined-recipe CNN (gender 0.78,
# `usage` 0.49) and 0.6563 for BBN (`usage` 0.55). Putting a number from this
# notebook beside those is only meaningful if three things match: the **denominator**
# above, the **split**, and whether "combined" means the mean of the two targets.
# Confirm all three before claiming one approach beat another.

# %% [markdown]
# ## 2 · One model or two? — deciding before training, not after
#
# The brief leaves this open. It is tempting to settle it by training both and
# keeping the winner, but that answers "which scored higher on this seed", not
# "which is the right design". Two properties of the data decide it in advance,
# and the training runs in §5 then test the prediction.

# %%
g = sorted(frame["gender"].dropna().unique())
u = sorted(frame["usage"].dropna().unique())
tab = (frame.pivot_table(index="gender", columns="usage", values="id", aggfunc="count")
       .reindex(index=g, columns=u).fillna(0).astype(int))
occurring = int((tab > 0).sum().sum())

print(f"{len(g)} gender x {len(u)} usage = {len(g) * len(u)} possible pairs")
print(f"pairs that occur: {occurring}   ->   {len(g) * len(u) - occurring} never occur\n")
print(tab.to_string())

joint = frame.groupby(["gender", "usage"]).size().sort_values()
tiny = joint[joint < 10]
print(f"\njoint classes with fewer than 10 rows: {len(tiny)} of {occurring}")
print(tiny.to_string())

# %% [markdown]
# **Property 1 — a joint label multiplies the sparsity.** `usage` alone has one
# class under 10 images. Crossed with `gender` it has six. A combined model does
# not merely inherit the long tail, it lengthens it.
#
# **Property 2 — the two labels barely inform each other.** If knowing `gender`
# told you most of `usage`, a joint label would be capturing a real structure and
# the extra sparsity might be worth paying for. Measured as entropy:

# %%
def H(series):
    p = series.value_counts(normalize=True).values
    return float(-(p * np.log2(p)).sum())

def H_cond(frame, target, given):
    return float(sum(len(grp) / len(frame) * H(grp[target])
                     for _, grp in frame.groupby(given)))

rows = []
for t, given in [("usage", "gender"), ("gender", "usage"),
                 ("usage", "articleType"), ("gender", "articleType"),
                 ("usage", "subCategory"), ("gender", "subCategory")]:
    h, hc = H(frame[t]), H_cond(frame, t, given)
    rows.append({"target": t, "knowing": given, "H (bits)": round(h, 3),
                 "H | knowing": round(hc, 3),
                 "uncertainty removed": f"{100 * (1 - hc / h):.1f}%"})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# Knowing `gender` removes **11.5%** of the uncertainty in `usage`, and knowing
# `usage` removes **9.5%** of the uncertainty in `gender`. Nearly independent. So
# design **B** pays a large, measurable cost in sparsity to capture very little.
#
# **The same table also argues for design C.** `articleType` removes **66.8%** of
# the uncertainty in `usage` and **43.1%** in `gender` — far more than either
# target tells you about the other. The two tasks are not related to each other
# directly; they are both largely determined by a third thing, *what the garment
# is*, which is exactly what a convolutional backbone can see. That is a concrete
# reason to expect a **shared backbone with separate heads** to work: the sharing
# should happen in the features, not in the label.
#
# > **Prediction recorded before training:** C ≈ A > B, with B losing most on
# > `usage` macro-F1. §5 either confirms this or the reasoning above is wrong, and
# > both outcomes are worth reporting.

# %% [markdown]
# ## 3 · Baselines
#
# Two floors. Nothing below these counts as having learned anything.
#
# * **majority** — predict the most common class. Bounds *accuracy*.
# * **1-NN on downscaled pixels** — no training, no parameters. Bounds *macro-F1*:
#   whatever this scores is available from raw pixel similarity alone, so a CNN
#   that does not beat it has not earned its complexity.

# %%
from PIL import Image, ImageOps

def load_images(paths, size=IMAGE_SIZE, cache=None):
    """Decode once into one uint8 array. Re-reading JPEGs every epoch off Drive is
    the slowest thing in this notebook; 37,745 x 80 x 60 x 3 is only ~543 MB."""
    if cache is not None and Path(cache).exists():
        arr = np.load(cache)
        if len(arr) == len(paths):
            print(f"loaded {len(arr):,} images from cache {cache}")
            return arr
    out = np.zeros((len(paths), size[1], size[0], 3), dtype=np.uint8)
    t0, padded = time.time(), 0
    for i, p in enumerate(paths):
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size != size:
                # Pad on white rather than stretch: the catalogue background is white
                # and the portrait aspect ratio carries class cues top and bottom.
                im = ImageOps.pad(im, size, method=Image.Resampling.BILINEAR,
                                  color=(255, 255, 255), centering=(0.5, 0.5))
                padded += 1
            out[i] = np.asarray(im)
        if i and i % 5000 == 0:
            print(f"  {i:,}/{len(paths):,}  ({time.time() - t0:.0f}s)")
    print(f"decoded {len(out):,} images in {time.time() - t0:.0f}s "
          f"({padded} were not {size[0]}x{size[1]} and were padded)")
    if cache is not None:
        np.save(cache, out)
    return out

CACHE = Path("/content/train_images.npy") if IN_COLAB else Path("D:/ColabDataset/train_images.npy")
X_all = load_images(frame["path"].tolist(), cache=None if SAMPLE else CACHE)
print("image array:", X_all.shape, f"{X_all.nbytes / 1e6:.0f} MB")

# %%
# Positional index so a split frame can address rows of X_all without re-decoding.
frame["_row"] = np.arange(len(frame))

from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

def score(y_true, y_pred, labels=None):
    """Two macro-F1 numbers, on purpose.

    `usage` has a class with a single training image, so classes can be missing
    from validation entirely -- and then the two common conventions disagree:

      macro_f1         averages over ALL classes. A class absent from validation
                       contributes a hard 0, which is the honest description of the
                       task ("this label exists and we cannot predict it").
      macro_f1_in_val  averages only over classes actually present in y_true, which
                       is what sklearn does by default when `labels` is not passed.

    On this data the denominators are 8 vs 7 for `usage`, so the same model can be
    reported ~14% apart with neither number being wrong. Anyone comparing scores
    across notebooks needs to know which one they are holding, so both are kept.
    """
    all_labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    in_val = sorted(set(y_true))
    return {"macro_f1": f1_score(y_true, y_pred, average="macro",
                                 labels=all_labels, zero_division=0),
            "macro_f1_in_val": f1_score(y_true, y_pred, average="macro",
                                        labels=in_val, zero_division=0),
            "n_classes": len(all_labels),
            "n_in_val": len(in_val),
            "accuracy": accuracy_score(y_true, y_pred),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted",
                                    labels=all_labels, zero_division=0)}

RESULTS = []
def record(name, target, split, y_true, y_pred, labels=None, **extra):
    s = score(y_true, y_pred, labels)
    RESULTS.append({"model": name, "target": target, "split": split, **s, **extra})
    print(f"  {name:26} {target:7} {split:14} macro-F1 {s['macro_f1']:.4f} "
          f"(over {s['n_classes']}) | {s['macro_f1_in_val']:.4f} (over {s['n_in_val']} "
          f"in val) | acc {s['accuracy']:.4f}")
    return s

# %% [markdown]
# ### 3.0 The split — the team's frozen file, not one generated here
#
# The repo's rule 2 is that everyone evaluates on the same split. That split is a
# **file**, `splits/train_val_grouped_sha256.csv`, not a function call — and the
# difference matters more than it looks. Regenerating a stratified split from a seed
# reproduces it only if everyone runs the same scikit-learn version; the split this
# notebook generated for itself overlapped the team's by **15.6%**, so every number
# produced against it was incomparable with a teammate's despite both being "seed 42".
#
# So the file wins whenever it is present, and the generated split is only a fallback
# for running without it.
#
# Its shape, which differs from `preprocessing.make_split` in two ways worth knowing:
# **15% validation**, not 20%, and **one split shared by both targets** rather than a
# separate stratified draw per target. `Home` (1 training image) lands in train and is
# absent from validation, so `usage` averages over 7 of 8 classes unless the
# denominator is stated — see §1.3.

# %%
SHARED_SPLIT_NAME = "train_val_grouped_sha256.csv"

def find_shared_split():
    cands = [os.environ.get("A2_SHARED_SPLIT", ""),
             f"/content/drive/MyDrive/A2_ExternalData/{SHARED_SPLIT_NAME}",
             f"/content/{SHARED_SPLIT_NAME}",
             f"/content/splits/{SHARED_SPLIT_NAME}",
             str(Path.cwd() / SHARED_SPLIT_NAME),
             str(Path.cwd().parent / "splits" / SHARED_SPLIT_NAME),
             f"D:/g2/splits/{SHARED_SPLIT_NAME}",
             str(Path.cwd() / "task3" / SHARED_SPLIT_NAME)]
    for c in cands:
        if c and Path(c).is_file():
            return Path(c)
    print("shared split file NOT found - falling back to a locally generated split.\n"
          "Numbers will NOT be comparable with teammates'. Tried:\n  "
          + "\n  ".join(str(c) for c in cands if c))
    return None


SHARED = find_shared_split()
if SHARED is not None:
    _s = pd.read_csv(SHARED)
    SHARED_VAL = set(_s.loc[_s["split"] == "val", "id"])
    covered = frame["id"].isin(set(_s["id"])).mean()
    print(f"shared split: {SHARED}")
    print(f"  {len(_s):,} rows, val {len(SHARED_VAL):,} ({len(SHARED_VAL)/len(_s):.1%}), "
          f"covers {covered:.1%} of the frame")
    assert covered > 0.99, (
        f"the shared split only covers {covered:.1%} of the loaded frame - it was built "
        "on a different row set, so aligning to it would be meaningless")


def make_split_shared(frame, target):
    """Split by the team's frozen file. Rows outside the file go to training."""
    f = frame.loc[frame[target].notna()].reset_index(drop=True)
    is_val = f["id"].isin(SHARED_VAL)
    return f.loc[~is_val].reset_index(drop=True), f.loc[is_val].reset_index(drop=True)

# %%
frame["pair"] = frame["gender"].astype(str) + " x " + frame["usage"].astype(str)
PRIMARY = "shared val" if SHARED is not None else "generated val"

splits = {}
if SHARED is not None:
    tr, va = make_split_shared(frame, "gender")     # one file, both targets
else:
    tr, va = make_split(frame, "pair")              # fallback: stratify on the pair
for t in TARGETS:
    splits[(t, "primary")] = (tr, va)
    splits[(t, "forward")] = make_split_forward(frame, t)
    splits[(t, "perTarget")] = make_split(frame, t)   # preprocessing.make_split's rule

print(f"primary split ({PRIMARY}): train {len(tr):,}  val {len(va):,} "
      f"({len(va)/(len(tr)+len(va)):.1%})")
for t in TARGETS:
    a, b = splits[(t, "forward")]
    c, d = splits[(t, "perTarget")]
    print(f"{t:7} forward train {len(a):,} val {len(b):,}   "
          f"| per-target train {len(c):,} val {len(d):,}")

for t in TARGETS:
    held = sorted(set(frame[t].dropna()) - set(va[t]))
    if held:
        print()
        print(f"{t}: absent from validation entirely -> {held}")
        print(f"  macro-F1 over classes PRESENT in val therefore averages over "
              f"{va[t].nunique()} of {frame[t].nunique()} classes.")

# %%
print("=== baseline 1: majority class ===")
for t in TARGETS:
    tr, va = splits[(t, "primary")]
    guess = tr[t].value_counts().index[0]
    record("majority", t, PRIMARY, va[t], [guess] * len(va),
           labels=sorted(frame[t].dropna().unique()))

# %%
print("\n=== baseline 2: 1-NN on 15x20 grayscale pixels (no training at all) ===")
def thumbs(rows, w=15, h=20):
    a = X_all[rows].astype(np.float32).mean(axis=3)          # to grayscale
    a = a.reshape(len(rows), 4, h, 4, w).mean(axis=(1, 3))   # 80x60 -> 20x15
    return a.reshape(len(rows), -1)

def one_nn(tr, va, target, chunk=512):
    Xtr, Xva = thumbs(tr["_row"].values), thumbs(va["_row"].values)
    ytr = tr[target].to_numpy()
    tr_sq = (Xtr ** 2).sum(1)
    pred = np.empty(len(Xva), dtype=object)
    for i in range(0, len(Xva), chunk):
        b = Xva[i:i + chunk]
        d = tr_sq[None, :] - 2.0 * (b @ Xtr.T) + (b ** 2).sum(1)[:, None]
        pred[i:i + chunk] = ytr[np.argmin(d, axis=1)]
    return pred

for t in TARGETS:
    tr, va = splits[(t, "primary")]
    record("1-NN pixels", t, PRIMARY, va[t], one_nn(tr, va, t),
           labels=sorted(frame[t].dropna().unique()))

# %% [markdown]
# ### 3.1 Why the 1-NN floor is the number that matters
#
# 1-NN has no parameters and no training. Whatever it scores is obtainable from raw
# pixel similarity, so it separates *"the CNN learned something"* from *"the CNN
# rediscovered that similar-looking products share a label"*. Its per-class
# breakdown also says something the aggregate cannot: a class where 1-NN scores
# **zero despite having training images** is not starved, it is not visually
# distinguishable at 60×80 — and no amount of extra data will fix that one.

# %%
for t in TARGETS:
    tr, va = splits[(t, "primary")]
    pred = one_nn(tr, va, t)
    rep = pd.DataFrame(classification_report(
        va[t], pred, output_dict=True, zero_division=0)).T
    rep = rep.loc[[c for c in sorted(frame[t].dropna().unique()) if c in rep.index]]
    rep["train n"] = [int((tr[t] == c).sum()) for c in rep.index]
    print(f"\n=== {t}: 1-NN per class ===")
    print(rep[["train n", "support", "precision", "recall", "f1-score"]]
          .round(3).to_string())

# %% [markdown]
# ## 4 · One convolutional body, three ways of attaching labels to it
#
# Everything below shares this body. No pretrained weights (repo rule 4), so the
# comparison is between designs rather than between checkpoints someone else
# trained.
#
# 80×60 → 40×30 → 20×15 → 10×7 → global average → 128 features. Small on purpose:
# a 60×80 thumbnail does not carry enough detail to justify a deep network, and a
# small model makes eight training runs affordable, which is what actually buys the
# evidence.

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
print("device:", DEVICE, torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "")

X_t = torch.from_numpy(X_all)                      # uint8, NHWC
if DEVICE.type == "cuda" and X_t.numel() < 2_000_000_000:
    X_t = X_t.to(DEVICE)                           # ~543 MB, removes all host->device copies
print("images on", X_t.device)


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2))


class Backbone(nn.Module):
    OUT = 128
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(conv_block(3, 32), conv_block(32, 64), conv_block(64, 128))
        self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        return self.pool(self.body(x)).flatten(1)


class Net(nn.Module):
    """One backbone, one linear head per entry in `heads` ({name: n_classes}).

    A holds two of these with one head each; B holds one with a single 24-way head;
    C holds one with two heads. Same class, so the comparison cannot be confounded
    by an accidental difference in the body.
    """
    def __init__(self, heads, dropout=0.2):
        super().__init__()
        self.backbone = Backbone()
        self.drop = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({k: nn.Linear(Backbone.OUT, n) for k, n in heads.items()})
    def forward(self, x):
        f = self.drop(self.backbone(x))
        return {k: h(f) for k, h in self.heads.items()}


def n_params(m):
    return sum(p.numel() for p in m.parameters())

print(f"backbone + one 8-way head: {n_params(Net({'usage': 8})):,} parameters")

# %%
CLASSES = {t: sorted(frame[t].dropna().unique()) for t in TARGETS}
CLASSES["pair"] = sorted(frame["pair"].dropna().unique())
IDX = {k: {c: i for i, c in enumerate(v)} for k, v in CLASSES.items()}
PAIR_TO_PARTS = [tuple(p.split(" x ")) for p in CLASSES["pair"]]
for k, v in CLASSES.items():
    print(f"{k:7} {len(v)} classes")


def encode(series, key):
    return torch.tensor([IDX[key][v] for v in series], dtype=torch.long)


# Normalisation from the TRAINING rows only — using all rows would leak validation
# statistics into the model's input scaling.
_tr, _va = splits[("gender", "primary")]
# A 5,000-row sample, not all 30,196: the full float32 copy would be ~1.7 GB and the
# channel means are identical to three decimals either way.
_rows = _tr["_row"].values
_sample = np.random.RandomState(SEED).choice(_rows, size=min(5000, len(_rows)), replace=False)
_stats = X_all[_sample].astype(np.float32) / 255.0

# dtype=np.float64 is NOT optional here. Reducing over (0,1,2) sums 24 million
# float32 values per channel, and float32 stops resolving +1 past 2**24 =
# 16,777,216 — so the running sum silently saturates and every channel returns
# 16777216/24000000 = 0.699, identical to the last digit. That equal-across-channels
# result is what exposed it; the numbers are simply wrong, not merely imprecise.
_mean = _stats.mean(axis=(0, 1, 2), dtype=np.float64)
_std = _stats.std(axis=(0, 1, 2), dtype=np.float64)
assert len(set(np.round(_mean, 6))) > 1, (
    "all three channel means are identical — the accumulator saturated again")

MEAN = torch.tensor(_mean, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor(_std, dtype=torch.float32, device=DEVICE).view(1, 3, 1, 1)
del _stats
print("channel mean", MEAN.flatten().tolist(), "\nchannel std ", STD.flatten().tolist())

# %%
def batch_x(rows, train=False):
    """uint8 NHWC -> normalised float NCHW, with a random horizontal flip in training.

    Flipping is safe for these two targets: a mirrored garment is the same garment,
    the same `gender` and the same `usage`. (It would not be safe for a target that
    depended on printed text, which is unreadable at 60x60 anyway.)
    """
    x = X_t[rows]
    if x.device != DEVICE:
        x = x.to(DEVICE, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    if train:
        flip = torch.rand(x.shape[0], device=DEVICE) < 0.5
        x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
    return (x - MEAN) / STD


@torch.no_grad()
def predict(model, fr, heads, bs=512):
    model.eval()
    rows = torch.as_tensor(fr["_row"].values, device=X_t.device)
    raw = {k: [] for k in heads}
    for i in range(0, len(rows), bs):
        out = model(batch_x(rows[i:i + bs]))
        for k in heads:
            raw[k].append(out[k].argmax(1).cpu())
    raw = {k: torch.cat(v).numpy() for k, v in raw.items()}

    if "pair" in heads:                       # design B: decode the joint label
        parts = [PAIR_TO_PARTS[i] for i in raw["pair"]]
        return {"gender": np.array([p[0] for p in parts]),
                "usage": np.array([p[1] for p in parts])}
    return {k: np.array([CLASSES[k][i] for i in v]) for k, v in raw.items()}


def evaluate(model, fr, heads):
    pred = predict(model, fr, heads)
    return {t: f1_score(fr[t], pred[t], average="macro",
                        labels=CLASSES[t], zero_division=0) for t in pred}


def class_weights(series, key, scheme="inverse_sqrt"):
    """Weight rare classes up. sqrt rather than 1/n on purpose: with `Home` at a
    single image, a pure 1/n weight gives that one row ~29,000x the pull of a
    Casual row and the model chases it into noise. sqrt keeps the direction and
    tempers the magnitude — itself a choice worth reporting, and §6 measures it."""
    counts = np.array([max((series == c).sum(), 1) for c in CLASSES[key]], dtype=np.float64)
    w = 1.0 / np.sqrt(counts) if scheme == "inverse_sqrt" else 1.0 / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


def train_model(name, heads, tr, va, epochs=EPOCHS, weighted=False, lr=1e-3,
                bs=256, verbose=True, init_backbone=None):
    """Train one Net and return (best state_dict, history). Selection on mean
    macro-F1 over the reported targets, never on accuracy."""
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    model = Net(heads).to(DEVICE)
    if init_backbone is not None:
        # Start from a backbone trained on another task (§10.3). Only the body is
        # copied; the heads stay randomly initialised because their label spaces
        # differ, and the whole thing is then fine-tuned rather than frozen -- with
        # 32,000 rows there is enough data to move the features, and freezing would
        # test a different question.
        model.backbone.load_state_dict(init_backbone.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = {k: nn.CrossEntropyLoss(
                weight=class_weights(tr[k], k) if weighted else None) for k in heads}

    rows = torch.as_tensor(tr["_row"].values, device=X_t.device)
    y = {k: encode(tr[k], k).to(DEVICE) for k in heads}
    n = len(rows)
    best, best_state, hist = -1.0, None, []

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=rows.device)
        total = 0.0
        for i in range(0, n, bs):
            sel = perm[i:i + bs]
            out = model(batch_x(rows[sel], train=True))
            loss = sum(crit[k](out[k], y[k][sel.to(DEVICE)]) for k in heads)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            total += float(loss.detach()) * len(sel)
        sched.step()

        f1 = evaluate(model, va, heads)
        mean_f1 = float(np.mean(list(f1.values())))
        hist.append({"epoch": ep, "loss": total / n, **{f"val_{k}": v for k, v in f1.items()}})
        improved = mean_f1 > best
        if improved:
            best, best_state = mean_f1, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}
        if verbose:
            print(f"    epoch {ep:2}/{epochs}  loss {total / n:.4f}  " +
                  "  ".join(f"{k} F1 {v:.4f}" for k, v in f1.items()) +
                  ("   <- best" if improved else ""))

    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist)

# %% [markdown]
# ## 5 · A vs B vs C
#
# Same body, same seed, same 20 epochs, same rows. The only difference is how the
# labels attach.

# %%
tr, va = splits[("gender", "primary")]
MODELS, HIST = {}, {}

print("A · two independent models")
for t in TARGETS:
    print(f"  --- {t} ---")
    MODELS[("A", t)], HIST[("A", t)] = train_model(f"A_{t}", {t: len(CLASSES[t])}, tr, va)

print("\nB · one model, 24 joint gender x usage classes")
MODELS[("B", "pair")], HIST[("B", "pair")] = train_model(
    "B_pair", {"pair": len(CLASSES["pair"])}, tr, va)

print("\nC · one shared backbone, two heads")
MODELS[("C", "multi")], HIST[("C", "multi")] = train_model(
    "C_multi", {t: len(CLASSES[t]) for t in TARGETS}, tr, va)

# %%
print("=== design comparison, random val ===")
DESIGNS = {"A two models": [(MODELS[("A", t)], {t: len(CLASSES[t])}) for t in TARGETS],
           "B joint label": [(MODELS[("B", "pair")], {"pair": len(CLASSES["pair"])})],
           "C shared body": [(MODELS[("C", "multi")], {t: len(CLASSES[t]) for t in TARGETS})]}

def design_predict(design, fr):
    """Merge the predictions of however many models a design uses."""
    out = {}
    for model, heads in DESIGNS[design]:
        out.update(predict(model, fr, heads))
    return out

rows = []
for design in DESIGNS:
    pred = design_predict(design, va)
    both = np.mean([(pred["gender"][i] == va["gender"].iloc[i]) and
                    (pred["usage"][i] == va["usage"].iloc[i]) for i in range(len(va))])
    r = {"design": design,
         "params": sum(n_params(m) for m, _ in DESIGNS[design]),
         "both correct": round(float(both), 4)}
    for t in TARGETS:
        s = score(va[t], pred[t], labels=CLASSES[t])
        r[f"{t} macro-F1"] = round(s["macro_f1"], 4)
        r[f"{t} acc"] = round(s["accuracy"], 4)
        record(design, t, PRIMARY, va[t], pred[t], labels=CLASSES[t])
    rows.append(r)

comparison = pd.DataFrame(rows)
print()
print(comparison.to_string(index=False))

# %% [markdown]
# ### 5.1 What the comparison showed — and what running it four times did to the claim
#
# §2 registered **C ≈ A > B, with B losing most on `usage`**. It was wrong, and the shape
# of how it was wrong is the finding.
#
# | design | params | gender | usage | both correct |
# |---|---|---|---|---|
# | A — two models | 576,589 | **0.7488** | 0.4081 | **0.8075** |
# | B — joint label | 290,552 | 0.7271 | **0.4083** | 0.8071 |
# | C — shared body | 289,133 | 0.7174 | 0.4066 | 0.8037 |
#
# **Why §2's reasoning came out backwards.** It measured that `gender` and `usage` share
# only ~10% of each other's entropy, and concluded a *joint label* would pay sparsity for
# nothing. That was half the implication. Near-independence argues against sharing the
# **features** too: if two tasks have little in common, one backbone serving both mostly
# suffers interference, and A's two separate backbones avoid it entirely. The measurement
# was right and the inference from it was incomplete — which is more useful to report
# than a prediction that happened to land.
#
# #### One table cannot rank three designs
#
# This notebook has been run end to end four times on this split — three on Colab, one on
# different hardware and a different library stack. §9.2 does the arithmetic. What matters
# here is that **the ranking in any single table is partly an artefact of the run**:
#
# * **A leads on macro-F1 in all four runs, on both targets — 8 of 8.** Its margin over
#   B on `gender` also keeps its size across platforms (+0.025 mean, spread 0.020). That
#   is the one design claim this notebook will defend.
# * **Its margin on `usage` is real in sign and unreliable in size**: +0.0054, +0.0059,
#   +0.0069 on Colab, then **+0.0015** on other hardware. Note that in *this* table B is
#   ahead of A on `usage` by 0.0002 — which is what an unreliable margin looks like.
# * **B versus C is not resolvable.** The sign flips on both targets. Do not quote a
#   three-way ordering from this table; an earlier version of this notebook quoted
#   "A > B > C, 4 out of 4" from two runs, and the next two runs disagreed with it and
#   with each other.
# * **`both correct` ranks the designs differently again**, and it is not an independent
#   measurement — §10.5 shows it is the product of the two accuracies to within 0.0023.
#
# The cheapest defence against all of this was available from the start: run it more than
# once, and preferably somewhere else. §9.4 asks for a deliberate seed sweep, which is
# wider still.
# %%
fig_hist = pd.concat([h.assign(run=f"{k[0]} {k[1]}") for k, h in HIST.items()])
try:
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for k, h in HIST.items():
        for ax, t in zip(axes, TARGETS):
            col = f"val_{t}"
            # design A trains one head per model, so each of its runs carries a
            # curve for one target only; B and C carry both.
            if col not in h.columns:
                continue
            ax.plot(h["epoch"], h[col], label=f"{k[0]} {k[1]}", marker="o", ms=3)
    for ax, t in zip(axes, TARGETS):
        ax.set_title(f"{t} — validation macro-F1 by epoch")
        ax.set_xlabel("epoch"); ax.set_ylabel("macro-F1"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.tight_layout(); plt.show()
except ImportError:
    print(fig_hist.to_string(index=False))

# %% [markdown]
# ## 6 · Class-weighted loss — does it help the tail or just move the damage?
#
# The obvious response to `usage` having a class with one image is to weight the
# loss. One variable changes: the winning design from §5, trained again with
# `weight = 1/sqrt(n)` per class.
#
# Weighting cannot manufacture information. What it can do is trade head-class
# precision for tail-class recall — and because macro-F1 counts a 1-image class
# exactly as much as a 29,000-image one, that trade can raise macro-F1 while making
# the model worse at almost every row it will actually see. Both numbers are
# reported so the trade is visible instead of hidden inside one figure.

# %%
# Sections 6-9 all carry design C forward, because §2 predicted it and §5 is expected
# to confirm it. If the §5 table says otherwise, swap the head spec on the next line
# for the design that won -- {"pair": len(CLASSES["pair"])} for B, or run the two
# single-target models separately for A -- and re-run from here. Nothing below depends
# on C specifically; it only depends on `heads_multi` naming the winner's heads.
heads_multi = {t: len(CLASSES[t]) for t in TARGETS}

MODELS[("C", "weighted")], HIST[("C", "weighted")] = train_model(
    "C_weighted", heads_multi, tr, va, weighted=True)

# %%
print("=== unweighted vs class-weighted (design C) ===")
rows = []
for tag, model in [("C unweighted", MODELS[("C", "multi")]),
                   ("C weighted", MODELS[("C", "weighted")])]:
    pred = predict(model, va, heads_multi)
    for t in TARGETS:
        s = score(va[t], pred[t], labels=CLASSES[t])
        head = va[t].value_counts().index[0]
        tail = [c for c in CLASSES[t] if c != head]
        rows.append({"model": tag, "target": t,
                     "macro-F1": round(s["macro_f1"], 4),
                     "accuracy": round(s["accuracy"], 4),
                     f"F1 on majority": round(f1_score(
                         va[t], pred[t], labels=[head], average="macro", zero_division=0), 4),
                     "mean F1 on the rest": round(f1_score(
                         va[t], pred[t], labels=tail, average="macro", zero_division=0), 4)})
        record(tag, t, PRIMARY, va[t], pred[t], labels=CLASSES[t])
print()
print(pd.DataFrame(rows).to_string(index=False))

# %%
print("=== per-class detail, usage, design C unweighted vs weighted ===")
for tag, model in [("unweighted", MODELS[("C", "multi")]), ("weighted", MODELS[("C", "weighted")])]:
    pred = predict(model, va, heads_multi)
    rep = pd.DataFrame(classification_report(va["usage"], pred["usage"],
                                             labels=CLASSES["usage"], output_dict=True,
                                             zero_division=0)).T
    rep = rep.loc[[c for c in CLASSES["usage"] if c in rep.index]]
    rep["train n"] = [int((tr["usage"] == c).sum()) for c in rep.index]
    print(f"\n--- {tag} ---")
    print(rep[["train n", "support", "precision", "recall", "f1-score"]].round(3).to_string())

# %% [markdown]
# ## 7 · The externally collected data — a prediction, then the measurement
#
# 1,899 extra images were collected for this assignment (`docs/EXTERNAL_DATA_USAGE.md`).
# Every one of them is `gender=Women`, `usage=Casual`.
#
# **Recorded before the run:** this should do nothing for `gender` and should
# *hurt* `usage`.
#
# * `Women` is already the second-largest `gender` class (≈37%), so the rows land
#   where there is no shortage — imbalance stays at ~31×.
# * `Casual` is already the `usage` **majority** at 76.7%. Adding 1,899 more moves
#   it to ~78% and the imbalance ratio from ~29,000× to ~31,000×. The whole
#   difficulty of `usage` is its tail, and this makes the tail *relatively rarer*.
#
# Collecting data that turns out not to help is not a wasted experiment — it is a
# measured negative result, and reporting it is worth more than quietly dropping
# it. What follows tests whether the reasoning above is right.

# %%
EXT_OK = EXTERNAL_ROOT is not None
if EXT_OK:
    ext = []
    for folder, csv_name in [("ExternalCosmetics", "external_cosmetics.csv"),
                             ("ExternalCosmetics2", "external_cosmetics2.csv")]:
        base = Path(EXTERNAL_ROOT) / folder
        e = pd.read_csv(base / csv_name)
        e["path"] = e["id"].astype(str).map(lambda i: str(base / "images" / f"{i}.jpg"))
        e["group_id"] = "ext:" + e["id"].astype(str)
        ext.append(e)
    ext = pd.concat(ext, ignore_index=True)
    ext = ext[ext[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    print(f"{len(ext):,} external rows")
    for t in TARGETS:
        print(f"  {t}: {dict(ext[t].value_counts())}")

    X_ext = load_images(ext["path"].tolist())
    ext["_row"] = np.arange(len(ext)) + len(X_all)
    ext["pair"] = ext["gender"].astype(str) + " x " + ext["usage"].astype(str)

    X_all = np.concatenate([X_all, X_ext], axis=0)
    X_t = torch.from_numpy(X_all)
    if DEVICE.type == "cuda" and X_t.numel() < 2_000_000_000:
        X_t = X_t.to(DEVICE)
    del X_ext
    print("image array now:", X_all.shape)
else:
    print("external data unavailable — section skipped")

# %% [markdown]
# The external rows join **training only**. Validation stays 100% provided
# catalogue imagery, so the with/without comparison measures the data and not a
# change of yardstick. Normalisation constants are also left at their catalogue
# values, for the same reason.
#
# > **Note the padding count printed above.** `ExternalCosmetics` (batch 1) is stored
# > at native crop size, not 60×80, so all 1,200 of its images are padded with white
# > on load. `ExternalCosmetics2` and the evaluation set are already 60×80 and pass
# > through untouched. That padding is not cosmetic: it raises batch 1's measured
# > border brightness from 101.5 to **201.2** against the catalogue's 247.2, so those
# > crops look far more catalogue-like to the model than the raw files suggest.
# > Reproduce with `python src/verify_external_data.py --domain-gap`.

# %%
if EXT_OK:
    tr_ext = pd.concat([tr, ext[[c for c in tr.columns if c in ext.columns]]],
                       ignore_index=True)
    assert tr_ext["group_id"].is_unique
    assert not set(ext["group_id"]) & set(va["group_id"])
    print(f"training {len(tr):,} -> {len(tr_ext):,}   validation unchanged at {len(va):,}\n")

    for t in TARGETS:
        b, a = tr[t].value_counts(), tr_ext[t].value_counts()
        print(f"{t:7} imbalance max/min  {b.max() / b.min():>10,.0f}x  ->  "
              f"{a.max() / a.min():>10,.0f}x    majority share "
              f"{100 * b.max() / b.sum():.1f}% -> {100 * a.max() / a.sum():.1f}%")

    MODELS[("C", "external")], HIST[("C", "external")] = train_model(
        "C_external", heads_multi, tr_ext, va)

# %%
if EXT_OK:
    print("=== with vs without the external data (design C) ===")
    rows = []
    for tag, model in [("C without external", MODELS[("C", "multi")]),
                       ("C with external", MODELS[("C", "external")])]:
        pred = predict(model, va, heads_multi)
        for t in TARGETS:
            s = score(va[t], pred[t], labels=CLASSES[t])
            rows.append({"model": tag, "target": t,
                         "macro-F1": round(s["macro_f1"], 4),
                         "accuracy": round(s["accuracy"], 4)})
            record(tag, t, PRIMARY, va[t], pred[t], labels=CLASSES[t])
    ablation = pd.DataFrame(rows).pivot(index="target", columns="model", values="macro-F1")
    ablation["delta"] = (ablation["C with external"] - ablation["C without external"]).round(4)
    print()
    print(ablation.to_string())
    # Do NOT hardcode the direction. The first run measured usage -0.0063 and this
    # sentence asserted "a negative delta confirms the prediction". The second run,
    # on the team's split, measured +0.0050 and the same sentence printed anyway.
    # A conclusion that survives its own data changing sign is not a conclusion.
    print()
    for _tgt, _v in ablation["delta"].items():
        _verdict = ('worse with external data' if _v < -0.002 else
                    'better with external data' if _v > 0.002 else
                    'indistinguishable from zero at this scale')
        print(f'    {_tgt}: delta {_v:+.4f}  ({_verdict})')
    print()
    # A print statement must not editorialise. Four times now a hardcoded sentence
    # here has been falsified by the numbers printed directly above it -- the last
    # asserted 'negative in all three runs' beside a measured delta of +0.0124. Every
    # cross-run claim now lives in section 9.3, which is prose and can be revised
    # without touching this cell or discarding its output.
    print('    Section 7 predicted no gain on usage, because every external row',
          'carries the majority class of both targets.')
    print('    One run cannot size this effect. Section 9.2 pools all four runs and',
          'section 9.3 finding 6 draws')
    print('    the verdict -- do not read a sign off this single table.')

# %% [markdown]
# ## 8 · Two harder tests than the random validation split
#
# ### 8.1 The forward split — what the graded test set actually looks like
#
# The provided test set is the **5,829 highest ids**. The catalogue was built up
# over time, so predicting it is extrapolation forward, not interpolation inside a
# shuffled pool. A random split quietly assumes those are the same problem. They
# are not:

# %%
tr_f, va_f = splits[("gender", "forward")]
print(f"forward split: train ids {tr_f['id'].min()}-{tr_f['id'].max()} ({len(tr_f):,}),  "
      f"val ids {va_f['id'].min()}-{va_f['id'].max()} ({len(va_f):,})\n")
for t in TARGETS:
    a = tr_f[t].value_counts(normalize=True) * 100
    b = va_f[t].value_counts(normalize=True) * 100
    d = pd.DataFrame({"train %": a.round(2), "val %": b.round(2)}).fillna(0)
    d["drift"] = (d["val %"] - d["train %"]).round(2)
    d["train n"] = tr_f[t].value_counts()
    print(f"--- {t} ---"); print(d.fillna(0).to_string(), "\n")

# %% [markdown]
# `Women` goes from 33% of training to 53% of validation, `Men` the other way by
# 14 points, `Sports` collapses from 12% to 3.7%. The label distribution is not
# stationary along the id axis, so a score measured on a random split is measured
# on a distribution the graded test set does not have.

# %%
MODELS[("C", "forward")], HIST[("C", "forward")] = train_model(
    "C_forward", heads_multi, tr_f, va_f)

print("\n=== the same design, scored two ways ===")
rows = []
for tag, model, vv in [(PRIMARY, MODELS[("C", "multi")], va),
                       ("forward split", MODELS[("C", "forward")], va_f)]:
    pred = predict(model, vv, heads_multi)
    for t in TARGETS:
        s = score(vv[t], pred[t], labels=CLASSES[t])
        rows.append({"evaluated on": tag, "target": t,
                     "macro-F1": round(s["macro_f1"], 4),
                     "accuracy": round(s["accuracy"], 4)})
        record("C shared body", t, tag, vv[t], pred[t], labels=CLASSES[t])
print()
print(pd.DataFrame(rows).pivot(index="target", columns="evaluated on",
                               values="macro-F1").to_string())

# %% [markdown]
# ### 8.2 The independent evaluation set
#
# 261 openly-licensed photographs collected from outside the provided data
# entirely, gated to zero overlap with both the provided train and test sets, and
# hand-labelled for all four targets. This is the brief's §3.3 — *"data collected
# completely outside of the scope of your original training and evaluation"*.
#
# No model in this project has seen them, and that is their whole value. **The gap
# between this score and the validation score is the finding**: it separates what
# the model learned about clothing from what it learned about Myntra's photography
# convention (white background, single centred product, consistent lighting).
#
# Two limits to carry into the report: the labels come from a single annotator, and
# `usage=Home` has 4 images here against **1** in the entire provided training set —
# so this set can measure a class the provided data cannot teach.
#
# > **The label distribution of this set is very different from the catalogue's, and
# > that is partly deliberate.** Rare classes were over-collected on purpose, because
# > a class with one training image cannot be measured otherwise:
# >
# > | | catalogue train | this set | ratio |
# > |---|---|---|---|
# > | `gender=Unisex` | 5.4% | **32.6%** | 6.0x |
# > | `gender=Men` | 54.4% | 16.9% | 0.3x |
# > | `usage=Party` | 0.03% | **9.2%** | 267x |
# > | `usage=Home` | 0.003% | **1.5%** | 578x |
# >
# > So a macro-F1 measured here mixes two different effects: the **image** domain gap
# > (background, framing, lighting) and a **label prior** shift, much of it created by
# > the annotation choices above. A model trained on 5% `Unisex` will rarely predict
# > `Unisex`, and would score badly on a set that is a third `Unisex` even if the
# > photographs were perfect catalogue cut-outs.
# >
# > The two can be partly separated. `usage=Casual` is the one class whose share
# > barely moves (76.7% -> 63.6%), and its F1 falls **0.932 -> 0.731** — that drop is
# > attributable to the images, not the prior. Quote that number when the claim is
# > about domain gap, and the macro-F1 when the claim is about the long tail. Do not
# > use the macro-F1 collapse alone as a measure of generalisation.

# %%
EVAL_OK = EXT_OK and (Path(EXTERNAL_ROOT) / "ExternalEval" / "external_eval.csv").exists()
if EVAL_OK:
    base = Path(EXTERNAL_ROOT) / "ExternalEval"
    ev = pd.read_csv(base / "external_eval.csv")
    ev["path"] = ev["id"].astype(str).map(lambda i: str(base / "images" / f"{i}.jpg"))
    ev = ev[ev[TARGETS].notna().all(axis=1)].reset_index(drop=True)
    X_ev = load_images(ev["path"].tolist())
    ev["_row"] = np.arange(len(ev)) + len(X_all)
    X_all = np.concatenate([X_all, X_ev], axis=0)
    X_t = torch.from_numpy(X_all)
    if DEVICE.type == "cuda" and X_t.numel() < 2_000_000_000:
        X_t = X_t.to(DEVICE)
    del X_ev
    print(f"{len(ev)} independent evaluation images")
    for t in TARGETS:
        print(f"  {t}: {dict(ev[t].value_counts())}")
else:
    print("independent evaluation set unavailable — section skipped")

# %%
if EVAL_OK:
    print("=== validation (catalogue) vs independent evaluation (in the wild) ===")
    rows = []
    pred_va = predict(MODELS[("C", "multi")], va, heads_multi)
    pred_ev = predict(MODELS[("C", "multi")], ev, heads_multi)
    for t in TARGETS:
        shared = sorted(set(ev[t]) | set(va[t]))
        s_va = score(va[t], pred_va[t], labels=CLASSES[t])
        s_ev = score(ev[t], pred_ev[t], labels=sorted(set(ev[t])))
        rows.append({"target": t,
                     "val macro-F1": round(s_va["macro_f1"], 4),
                     "independent macro-F1": round(s_ev["macro_f1"], 4),
                     "val acc": round(s_va["accuracy"], 4),
                     "independent acc": round(s_ev["accuracy"], 4)})
        record("C shared body", t, "independent eval", ev[t], pred_ev[t],
               labels=sorted(set(ev[t])))
    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nper-class on the independent set:")
    for t in TARGETS:
        print(f"\n--- {t} ---")
        print(pd.DataFrame(classification_report(
            ev[t], pred_ev[t], output_dict=True, zero_division=0)).T.round(3).to_string())

# %% [markdown]
# ## 9 · Results, and what they say
#
# ### 9.1 On the per-target split, as a robustness check
#
# Everything above used the team's frozen split file, so the numbers sit directly
# beside a teammate's. This section retrains the chosen design under
# `preprocessing.make_split(frame, target)` instead — a 20% validation share drawn
# separately per target, rather than 15% shared by both.
#
# It is a robustness check, not a second headline. If a conclusion from §5–§7 flips
# when the split changes, that conclusion was an artefact of one draw and the report
# should say so; if the numbers move but the ordering holds, the finding survives a
# change that touches every row.

# %%
for t in TARGETS:
    tr_t, va_t = splits[(t, "perTarget")]
    m, _ = train_model(f"C_perTarget_{t}", heads_multi, tr_t, va_t, verbose=False)
    pred = predict(m, va_t, heads_multi)
    record("C shared body", t, "per-target split", va_t[t], pred[t], labels=CLASSES[t])
    MODELS[("C", f"team_{t}")] = m

# %%
try:
    import matplotlib.pyplot as plt
    pred = predict(MODELS[("C", "multi")], va, heads_multi)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    for ax, t in zip(axes, TARGETS):
        cm = confusion_matrix(va[t], pred[t], labels=CLASSES[t])
        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES[t]))); ax.set_yticks(range(len(CLASSES[t])))
        ax.set_xticklabels(CLASSES[t], rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels([f"{c} ({n})" for c, n in
                            zip(CLASSES[t], [int((va[t] == c).sum()) for c in CLASSES[t]])],
                           fontsize=8)
        ax.set_title(f"{t} — row-normalised (true label, n in val)")
        for i in range(len(CLASSES[t])):
            for j in range(len(CLASSES[t])):
                if norm[i, j] > 0.01:
                    ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center",
                            fontsize=7, color="white" if norm[i, j] > .5 else "black")
    plt.tight_layout(); plt.show()
except ImportError:
    pass

# %%
results = pd.DataFrame(RESULTS).round(4)
results = results.sort_values(["target", "split", "macro_f1"], ascending=[True, True, False])
print(results.to_string(index=False))

def results_path(name):
    """Write results to DRIVE on Colab, never to /content.

    /content dies with the VM, and the VM is recycled whenever the browser stays
    disconnected -- so a run that finishes while the laptop is asleep can leave
    nothing behind. That already happened once: the first full run's CSV was gone
    on reconnect and had to be read back out of the notebook's printed output.
    Writing to Drive means the numbers outlive the tab, the VM and the machine.
    """
    if not IN_COLAB:
        return Path(name)
    for base in (Path(EXTERNAL_ON_DRIVE), Path("/content/drive/MyDrive")):
        if base.is_dir():
            return base / name
    return Path("/content") / name          # last resort, better than crashing


OUT = results_path("task3_results.csv")
print(f"results will be written to: {OUT}")
results.to_csv(OUT, index=False)
print(f"\nsaved -> {OUT}")

# %% [markdown]
# ### 9.2 Reproducibility — three runs to estimate, a fourth to test
#
# This notebook has been run end to end four times on the team's frozen split, with the
# same seed, the same 20 epochs and the same validation rows. The runs are deliberately
# not equivalent:
#
# | run | where | GPU | stack |
# |---|---|---|---|
# | 1, 2, 3 | Colab | Tesla T4 | Colab's pinned torch |
# | **4** | a local machine | RTX 4070 Laptop | torch 2.14.0+cu126, Windows |
#
# Runs 1–3 isolate **GPU non-determinism alone**. Run 4 additionally changes the GPU
# architecture, the operating system and the whole library stack, so it is used here as a
# **held-out portability test**: effect sizes are estimated from runs 1–3, and run 4 asks
# whether they survive a change of platform. That is a stricter question than "did it
# reproduce", and it is the one that matters to anyone re-running this from the repo.
#
# The controls make the comparison legitimate. The two untrained baselines — majority
# class and 1-NN on pixels — return **0.1412, 0.5335, 0.1080 and 0.3265 in all four runs,
# identical to four decimal places**. The data, the split and the metric are therefore
# provably the same on both platforms, so every difference below belongs to training.
#
# Two kinds of spread matter here, and conflating them is easy:
#
# * **Marginal spread** — how much *one* configuration's score moves between runs.
# * **Paired spread** — how much the *difference between two* configurations moves, when
#   both were trained inside the same run.
#
# The second is smaller, because two designs trained in one run share the data ordering
# that the non-determinism perturbs, so the common part cancels. An earlier version of
# this notebook used the marginal spread to dismiss every effect under 0.01, which was
# too blunt.
#
# The portability test then punished the opposite error. On runs 1–3 the `A − C`
# difference on `usage` was +0.0054, +0.0059, +0.0069 — a spread of 0.0015 — and this
# notebook called it the most reproducible effect it had. Run 4 measured **+0.0015**: the
# sign held, the magnitude fell fourfold. **A tight spread across runs on one machine is
# not a small error bar; it is a small sample of one platform.** That is the single most
# useful thing the fourth run bought.
# %%
# The three runs' shared-val macro-F1, transcribed from task3_results.csv of each.
# Literal on purpose: a past run cannot be recomputed, and pretending otherwise by
# reading a file that may have been overwritten would be worse than writing it down.
RUNS = {
    "A two models":             {"gender": [0.7542, 0.7525, 0.7347], "usage": [0.4092, 0.4106, 0.4076]},
    "B joint label":            {"gender": [0.7237, 0.7181, 0.7205], "usage": [0.4060, 0.4014, 0.4047]},
    "C shared body":            {"gender": [0.7169, 0.7297, 0.7284], "usage": [0.4038, 0.4047, 0.4007]},
    "C weighted":               {"gender": [0.7332, 0.7186, 0.7230], "usage": [0.4657, 0.5020, 0.4853]},
    "C with external":          {"gender": [0.7153, 0.7087, 0.7113], "usage": [0.4088, 0.4021, 0.4071]},
    "C + logit adj":            {"gender": [None,   0.7297, 0.7361], "usage": [None,   0.4103, 0.4109]},
    "C + logit adj + TTA":      {"gender": [None,   0.7286, 0.7462], "usage": [None,   0.4143, 0.4126]},
    "D articleType-pretrained": {"gender": [None,   0.7485, 0.7315], "usage": [None,   0.4095, 0.4079]},
    "majority":                 {"gender": [0.1412, 0.1412, 0.1412], "usage": [0.1080, 0.1080, 0.1080]},
    "1-NN pixels":              {"gender": [0.5335, 0.5335, 0.5335], "usage": [0.3265, 0.3265, 0.3265]},
}

# This run compared against the last column above. A NON-ZERO delta here is not an
# error -- it is exactly the phenomenon this section documents, seen live: if you are
# reading a fresh execution, these are run 4's numbers against run 3's. Deliberately
# not an assert, because a run that reproduced its predecessor to four decimal places
# would contradict the section it sits in.
_this = {(r["model"], r["target"]): r["macro_f1"]
         for r in RESULTS if r["split"] == PRIMARY}
rows = []
for m, d in RUNS.items():
    for t in TARGETS:
        if (m, t) in _this and d[t][-1] is not None:
            rows.append({"configuration": m, "target": t,
                         "last transcribed": d[t][-1],
                         "this run": round(_this[(m, t)], 4),
                         "delta": round(_this[(m, t)] - d[t][-1], 4)})
_live = pd.DataFrame(rows)
print("=== this run vs the last transcribed run ===")
print(_live.to_string(index=False))
_moved = _live[_live.delta.abs() > 5e-4]
if len(_moved) == 0:
    print("\n  identical -- so this IS the transcribed run, re-read from its own file.")
else:
    print(f"\n  {len(_moved)} of {len(_live)} configurations moved, worst "
          f"{_moved.delta.abs().max():.4f}.")
    if QUICK:
        print("  QUICK is on, so this run trained on a fraction of the rows for a")
        print("  fraction of the epochs. These deltas measure THAT, not run-to-run")
        print("  variation -- ignore them and do not add this run to RUNS.")
    else:
        print("  This is the run-to-run spread the section is about, seen live.")
        print("  Add a column to RUNS rather than editing one: the point is the")
        print("  spread across runs, not the latest value.")

print("\n=== marginal spread: one configuration, three runs ===")
rows = []
for m, d in RUNS.items():
    for t in TARGETS:
        v = [x for x in d[t] if x is not None]
        rows.append({"configuration": m, "target": t, "runs": len(v),
                     "min": min(v), "max": max(v), "range": round(max(v) - min(v), 4)})
marg = pd.DataFrame(rows)
print(marg.to_string(index=False))
_trained = marg[~marg.configuration.isin(["majority", "1-NN pixels"])]["range"]
_untrained = marg[marg.configuration.isin(["majority", "1-NN pixels"])]["range"]
print(f"\n  untrained baselines: range {_untrained.max():.4f} "
      "-- the data, the split and the metric are identical across runs")
print(f"  trained models:      range median {_trained.median():.4f}, "
      f"worst {_trained.max():.4f}")

print("\n=== paired spread: the difference between two configurations, same run ===")
PAIRS = [("A two models", "C shared body", "A - C"),
         ("A two models", "B joint label", "A - B"),
         ("C shared body", "B joint label", "C - B"),
         ("C weighted", "C shared body", "class weighting (S6)"),
         ("C with external", "C shared body", "external data (S7)"),
         ("D articleType-pretrained", "C shared body", "articleType transfer (S10.3)"),
         ("C + logit adj", "C shared body", "logit adjustment (S10.1)"),
         ("C + logit adj + TTA", "C + logit adj", "TTA on top (S10.2)")]
rows = []
for a, b, name in PAIRS:
    for t in TARGETS:
        d = [round(x - y, 4) for x, y in zip(RUNS[a][t], RUNS[b][t])
             if x is not None and y is not None]
        mean, spread = float(np.mean(d)), max(d) - min(d)
        if not (all(x > 0 for x in d) or all(x < 0 for x in d)):
            verdict = "sign flips -- not resolvable"
        elif spread < abs(mean):
            verdict = "consistent"
        else:
            verdict = "same sign, wide"
        rows.append({"comparison": name, "target": t, "n": len(d),
                     "deltas": " ".join(f"{x:+.4f}" for x in d),
                     "mean": round(mean, 4), "spread": round(spread, 4),
                     "verdict": verdict})
paired = pd.DataFrame(rows)
print(paired.to_string(index=False))
print("\n  'consistent' = same sign every run AND spread smaller than the mean.")
print("  Those are the only effects §9.3 states as findings; the rest are reported")
print("  as unresolved, which is a result too.")
# %% [markdown]
# ### 9.3 What this notebook found
#
# Numbers are from run 4 unless stated, all on the team's frozen split
# `train_val_grouped_sha256.csv` (37,745 rows, 15% validation), so they sit directly
# beside a teammate's. Effects are called by the test in §9.2: **same sign in every run,
# spread below the mean, and the sign holding on the held-out platform.** Anything else
# is reported as unresolved, which is also a result.
#
# One reading note. The table printed by §9.2 labels an effect `consistent` using
# **runs 1–3 only**, because those are the estimate; the portability test is applied
# here, in the findings. So an effect can be `consistent` in that table and still be
# demoted below — `A − C` on `usage` is exactly that case, and finding 3 says why.
#
# **1. `gender` and `usage` are not one task.** `gender` has five classes, smallest 483;
# `usage` has eight, smallest **1**. Best `gender` macro-F1 **0.7488**, best `usage`
# **0.4674**. The gap is not model quality — half of `usage`'s classes have almost no
# training data.
#
# **2. Accuracy is unusable here.** Predicting `Casual` everywhere scores **76.1%**
# accuracy and **0.108** macro-F1. The best `usage` model reaches **87.4%**, eleven
# points above a model that has learned nothing.
#
# **3. Two separate models beat both single-model designs on `gender`.** A led every
# design comparison in all four runs, 8 of 8.
#
# | | Colab mean | Colab spread | run 4 | verdict |
# |---|---|---|---|---|
# | A − B, `gender` | +0.0264 | 0.0202 | +0.0217 | **holds, in sign and size** |
# | A − C, `gender` | +0.0221 | 0.0310 | +0.0314 | sign holds, size unmeasured |
# | A − C, `usage` | +0.0061 | 0.0015 | +0.0015 | sign holds, size collapses |
# | A − B, `usage` | +0.0051 | 0.0063 | **−0.0002** | **fails portability** |
#
# A beats B on `gender` by a margin that keeps its size across platforms, and beats
# everything else by a margin whose *sign* is reliable and whose *size* is not. It pays
# **2× the parameters** for that, and finding 8 shows a 289k-parameter alternative that
# matches it. **B versus C stays unresolved** — the sign flips on both targets.
#
# §2's entropy measurement was right and the inference drawn from it was incomplete:
# near-independent targets argue against sharing **features**, not only against merging
# labels.
#
# **4. Class weighting is the only large effect here, and the only one that survives
# everything.** On `usage` it is **+0.0813** on Colab (+0.0619, +0.0973, +0.0846) and
# **+0.0608** on the held-out platform — four runs, two platforms, one sign, and an
# order of magnitude above anything in §10. On `gender` it is a wash; the sign flips. Which tail class it rescues
# is unstable too:
#
# | usage class | train n | val n | unweighted | weighted | weighted, run 3 |
# |---|---|---|---|---|---|
# | `Party` | 10 | 3 | 0.000 | 0.000 | 0.000 |
# | `Travel` | 22 | 3 | 0.000 | **0.286** | 0.500 |
# | `Smart Casual` | 46 | 9 | 0.000 | **0.250** | 0.167 |
# | `Home` | 1 | 0 | — | — | — |
# | `Casual` | 24,662 | 4,306 | 0.933 | 0.918 | 0.919 |
# | `Sports` | 3,300 | 614 | 0.669 | 0.654 | 0.666 |
#
# The aggregate is solid; the per-class rescues are not. With **3 to 9 validation items**
# in those classes, one item changing hands moves that class's F1 by 0.1–0.3.
#
# **5. §1.1's ceiling arithmetic predicted the outcome.** It put `usage` macro-F1 at
# **0.500** if the four classes under 100 images stayed unlearnable. Four runs measured
# 0.4657, 0.5020, 0.4853 and **0.4674** — mean **0.4801**. Because macro-F1 divides by 8,
# each class that never comes alive costs **0.125** whatever the model does, and `Home`
# and `Party` never came alive in any run.
#
# **6. The external data has no measurable effect on either target.** §7 predicted no
# gain, because all 1,899 rows carry the majority class of both targets and adding them
# makes `usage` imbalance *worse* (24,662× → 26,561×). Four runs agree with that
# prediction and disagree with each other about the sign:
#
# | | split generated here | team split, runs 1–3 | team split, run 4 |
# |---|---|---|---|
# | `usage` | −0.0063 | +0.0050, −0.0026, +0.0064 | −0.0007 |
# | `gender` | +0.0092 | −0.0016, −0.0210, −0.0171 | **+0.0124** |
#
# Two earlier readings of this table are withdrawn. The first called the data *harmful*
# from a single negative `usage` delta. The second, written when all three Colab runs
# came out negative on `gender`, called it a "possible small cost on `gender`" — and run
# 4, on different hardware, measured **+0.0124**. Three same-signed runs on one platform
# were not evidence of a sign. The defensible claim is **"collected, tested, no
# effect"** — which is exactly what §7 predicted before any of it was measured, and is
# why that section records the prediction first.
#
# **7. The random split flatters the model; the independent photographs demolish it —
# but only the first half of that is a clean measurement.**
#
# | | `gender` | `usage` |
# |---|---|---|
# | team split (15% val) | 0.7174 | 0.4066 |
# | per-target split (20% val) | 0.6873 | 0.4056 |
# | forward split (highest ids, like the graded test) | 0.5791 | 0.3585 |
# | **261 independent photographs** | **0.1306** | **0.1164** |
#
# The forward split costs **0.138** on `gender` and is the honest estimate of the graded
# score, because it *is* the high-id region. It is also the steadiest number in the
# notebook: 0.5815, 0.5729, 0.5813, 0.5791 across four runs — a range of 0.0086, on two
# platforms.
#
# The independent set costs **0.587**, putting `gender` below its majority baseline
# (0.1306 against 0.1412) with accuracy falling 0.8968 → **0.2605**. The collapse is real
# but **not purely a domain gap**: this set is 32.6% `Unisex` against the catalogue's
# 5.4%, and 9.2% `Party` against 0.03%, because rare classes were deliberately
# over-collected so they could be measured at all. A model trained on 5% `Unisex` rarely
# predicts it — F1 **0.000 on 85 items** — and that one class alone removes 0.2 from a
# five-class macro average.
#
# The cleanest number separating the two effects is `usage=Casual`, the one class whose
# share barely moves (76.7% → 63.6%): F1 **0.933 → 0.746**. That **0.19** is attributable
# to the photographs. Use it for claims about generalisation, and the macro-F1 for claims
# about the long tail — not the other way round.
#
# **8. A 289k-parameter stack beats the 577k-parameter design on `usage` every time it
# was measured.** `C + logit adjustment + TTA` adds two forward passes and no retraining.
# Against A on `usage`: **0.4143 vs 0.4106, 0.4126 vs 0.4076, 0.4133 vs 0.4081** — three
# for three. On `gender` A wins two of three. Given finding 3 — A's margin is reliable in
# sign but not in size — **the 2× parameter cost is not justified by this evidence.** The
# caveat in §9.4 about how τ is chosen applies. See §10.5.
#
# ### 9.4 What would be worth doing next
#
# * **Vary the seed deliberately, not just the platform.** Four runs came from GPU
#   non-determinism plus one platform change; none varies the seed explicitly. The claim
#   that needs it is the `gender` A − C margin, whose spread (0.031) still exceeds its
#   mean (0.024).
# * **Add a third split, because every number here is selected on the one it is
#   reported on.** Two mechanisms do this. `train_model` returns the checkpoint with the
#   best mean validation F1, so each reported score is a peak by construction — visible
#   in the gap between `C weighted`'s best gender epoch (0.7399) and its reported score
#   (0.7360), which came from the best *mean* epoch instead. And §10.1 picks τ by argmax
#   on that same validation set, so its gain **cannot be negative by construction**. The
#   bias is small and applies to every design equally, so the comparisons above stand;
#   the absolute values are upper bounds, and findings 8 and §10.1 lean on it most.
# * **Combine class weighting with the §10 levers.** Weighting is +0.081 on `usage`,
#   logit adjustment +0.008, and both correct the class prior — one in the loss, one at
#   inference. They have never been measured together, and they may not add.
# * **Re-score the independent set with matched label priors**, by reweighting or by
#   sampling a subset with the catalogue's distribution. That separates the image domain
#   gap from the annotation prior and would settle how much of the 0.587 is real.
# * **`Home` and `Party` are unfixable by modelling** — 1 and 10 training images, 0 and 3
#   in validation, 0.000 F1 in all four runs. Either collect images, merge them into a
#   documented "other" class, or report them as a known **0.250** of the macro-F1 that no
#   model can earn. Choosing openly beats a quiet zero.
# %% [markdown]
# ## 10 · Trying to improve it, and finding where the improvement stops
#
# §9 established two things that decide what is worth attempting here.
#
# **`usage` accuracy is already at the ceiling its labels allow.** An oracle told the true
# `articleType` and nothing else scores **0.8945**; giving it all six metadata fields
# moves it to 0.8928, i.e. nowhere; the best CNN in this run scores **0.8972** from
# 60×80 pixels, slightly *above* the oracle. The residual is products of the same type
# carrying different labels — `Tshirts` are 86% `Casual` and 14% `Sports`, and at this
# resolution those are frequently the same picture. No architecture recovers that.
# §10.5 measures both ceilings properly.
#
# **More epochs cannot help, though not because of dramatic overfitting.** Training loss
# falls monotonically (1.3203 → 0.4404 for design C) while validation F1 flattens after
# roughly epoch 15: across the eight training runs in run 4, the peak epoch is 17–20 and
# the peak-to-final gap is **0.0000 to 0.0070**. An earlier version of this section called
# that overfitting and cited a 0.003 drop; the flat curve is the more accurate reading.
# And since `train_model` already returns the best checkpoint by mean validation F1, the
# reported score is a peak by construction — so lengthening the schedule cannot move it.
#
# So accuracy is spent. **macro-F1 is not:** with `Home` and `Party` unreachable the
# ceiling is 6/8 = **0.750** against a measured 0.4674, and the slack sits in the classes
# class weighting already partly rescues — `Smart Casual`, `Travel`, `Sports`, `Formal`.
# And because accuracy has nothing left to gain, **spending accuracy to buy macro-F1 is
# close to free when macro-F1 is the graded metric**. Three attempts follow, cheapest
# first, and §10.5 explains why each one is small.
# %% [markdown]
# ### 10.1 Logit adjustment — the whole trade-off curve without retraining
#
# Class weighting attacks the imbalance during training, so every setting costs a run.
# The same trade-off can be made at *inference*: shift each class's logit by its log
# prior before taking the argmax.
#
# ```
# prediction = argmax( logit − τ · log P(class) )
# ```
#
# τ = 0 is the untouched model; τ = 1 fully removes the training prior. One trained
# model gives the entire curve for the cost of a few forward passes, which is why this
# is first: if the curve is flat there is nothing to buy and the retraining ideas
# below are not worth starting.

# %%
def train_prior(frame_tr, target):
    counts = np.array([max((frame_tr[target] == c).sum(), 1) for c in CLASSES[target]],
                      dtype=np.float64)
    return torch.tensor(counts / counts.sum(), dtype=torch.float32)


@torch.no_grad()
def logits_of(model, fr, heads, bs=512, tta=False):
    """Raw logits, optionally averaged with the mirrored image (see §10.2)."""
    model.eval()
    rows = torch.as_tensor(fr["_row"].values, device=X_t.device)
    acc = {k: [] for k in heads}
    for i in range(0, len(rows), bs):
        x = batch_x(rows[i:i + bs])
        out = model(x)
        if tta:
            flipped = model(torch.flip(x, dims=[-1]))
            out = {k: ((out[k].softmax(1) + flipped[k].softmax(1)) / 2).log()
                   for k in heads}
        for k in heads:
            acc[k].append(out[k].float().cpu())
    return {k: torch.cat(v) for k, v in acc.items()}


def adjusted(logits, target, tau, prior):
    idx = (logits - tau * prior.log()).argmax(1).numpy()
    return np.array(CLASSES[target])[idx]


TAUS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
PRIOR = {t: train_prior(tr, t) for t in TARGETS}
base_logits = logits_of(MODELS[("C", "multi")], va, heads_multi)

rows_out = []
for t in TARGETS:
    for tau in TAUS:
        s = score(va[t], adjusted(base_logits[t], t, tau, PRIOR[t]), labels=CLASSES[t])
        rows_out.append({"target": t, "tau": tau,
                         "macro_f1": round(s["macro_f1"], 4),
                         "macro_f1_in_val": round(s["macro_f1_in_val"], 4),
                         "accuracy": round(s["accuracy"], 4)})
frontier = pd.DataFrame(rows_out)
for t in TARGETS:
    sub = frontier[frontier.target == t]
    best = sub.loc[sub.macro_f1.idxmax()]
    print(f"=== {t} ===")
    print(sub.to_string(index=False))
    print(f"  best macro-F1 at tau={best.tau}: {best.macro_f1:.4f} "
          f"(tau=0 gives {sub.iloc[0].macro_f1:.4f}, "
          f"accuracy {sub.iloc[0].accuracy:.4f} -> {best.accuracy:.4f})\n")

for t in TARGETS:
    sub = frontier[frontier.target == t]
    best = sub.loc[sub.macro_f1.idxmax()]
    record("C + logit adj", t, PRIMARY,
           va[t], adjusted(base_logits[t], t, float(best.tau), PRIOR[t]),
           labels=CLASSES[t], tau=float(best.tau))

# %% [markdown]
# ### 10.2 Test-time augmentation
#
# One extra forward pass per image: average the softmax over the image and its mirror.
# It cannot fix a label ambiguity, so the expected gain is small — but §4 already
# trains with random horizontal flips, so the model has been taught that a mirrored
# garment is the same garment, and averaging over both views is the matching decision
# rule. Cheap enough that not testing it would be the odd choice.

# %%
tta_logits = logits_of(MODELS[("C", "multi")], va, heads_multi, tta=True)
rows_out = []
for t in TARGETS:
    tau = float(frontier[frontier.target == t].loc[
        frontier[frontier.target == t].macro_f1.idxmax(), "tau"])
    for name, lg in (("single view", base_logits), ("mirror-averaged", tta_logits)):
        s = score(va[t], adjusted(lg[t], t, tau, PRIOR[t]), labels=CLASSES[t])
        rows_out.append({"target": t, "view": name, "tau": tau,
                         "macro_f1": round(s["macro_f1"], 4),
                         "accuracy": round(s["accuracy"], 4)})
tta_tbl = pd.DataFrame(rows_out)
print(tta_tbl.to_string(index=False))
for t in TARGETS:
    sub = tta_tbl[tta_tbl.target == t]
    d = sub.iloc[1].macro_f1 - sub.iloc[0].macro_f1
    print(f"  {t}: TTA delta {d:+.4f}" +
          ("  (keep)" if d > 0.002 else "  (not worth the extra pass)"))

for t in TARGETS:
    tau = float(tta_tbl[tta_tbl.target == t].iloc[0].tau)
    record("C + logit adj + TTA", t, PRIMARY,
           va[t], adjusted(tta_logits[t], t, tau, PRIOR[t]), labels=CLASSES[t])

# %% [markdown]
# ### 10.3 Transfer from `articleType` — the one idea §2 predicted should work
#
# §2 measured how much each thing explains about the targets:
#
# | | explains `usage` | explains `gender` |
# |---|---|---|
# | the other target | 11.5% | 9.5% |
# | **`articleType`** | **66.8%** | **43.1%** |
#
# §5 then found that sharing a backbone *between* `gender` and `usage` **hurt** — C
# came last on both splits. That is consistent with the top row: two targets with ~10%
# mutual information have almost nothing to share, so one body serving both mostly
# suffers interference.
#
# The bottom row is a different proposition. `articleType` explains six times as much
# about `usage` as `usage`'s partner target does, and it has 121 classes and 32,000
# labelled rows to learn from. So: train the backbone on `articleType` first, then
# fine-tune it for `gender` and `usage`.
#
# This is the same "share the features" idea that failed in §5, aimed at a task that
# the measurements say actually shares something. If it fails too, then the 60×80
# backbone simply has no transferable capacity left, and that is the finding.

# %%
CLASSES["articleType"] = sorted(frame["articleType"].dropna().unique())
IDX["articleType"] = {c: i for i, c in enumerate(CLASSES["articleType"])}
print(f"pretraining task: articleType, {len(CLASSES['articleType'])} classes")

pretrained, pre_hist = train_model(
    "pretrain_articleType", {"articleType": len(CLASSES["articleType"])}, tr, va,
    epochs=max(8, EPOCHS // 2))
print(f"\npretraining reached articleType macro-F1 "
      f"{pre_hist['val_articleType'].max():.4f} — only a means to an end, not a Task 1 entry")

MODELS[("D", "transfer")], HIST[("D", "transfer")] = train_model(
    "D_transfer", heads_multi, tr, va, init_backbone=pretrained.backbone)

# %%
print("=== does an articleType-pretrained backbone beat training from scratch? ===")
rows_out = []
for tag, model in [("C from scratch", MODELS[("C", "multi")]),
                   ("D articleType-pretrained", MODELS[("D", "transfer")])]:
    pred = predict(model, va, heads_multi)
    for t in TARGETS:
        s = score(va[t], pred[t], labels=CLASSES[t])
        rows_out.append({"model": tag, "target": t,
                         "macro_f1": round(s["macro_f1"], 4),
                         "accuracy": round(s["accuracy"], 4)})
        record(tag, t, PRIMARY, va[t], pred[t], labels=CLASSES[t])
transfer_tbl = pd.DataFrame(rows_out).pivot(index="target", columns="model",
                                            values="macro_f1")
transfer_tbl["delta"] = (transfer_tbl["D articleType-pretrained"]
                         - transfer_tbl["C from scratch"]).round(4)
print()
print(transfer_tbl.to_string())

# %% [markdown]
# ### 10.4 Everything, in one table
#
# §9 wrote `task3_results.csv` before this section existed, so it is rewritten here
# with the §10 rows included. Sorted by target then macro-F1, so the top row per
# target is the best configuration found anywhere in the notebook.

# %%
results = pd.DataFrame(RESULTS).round(4)
results = results.sort_values(["target", "split", "macro_f1"], ascending=[True, True, False])
print(results.to_string(index=False))
results.to_csv(OUT, index=False)
print(f"\nrewritten -> {OUT}  ({len(results)} rows)")

HIST_OUT = results_path("task3_history.csv")
pd.concat([h.assign(run=f"{k[0]} {k[1]}") for k, h in HIST.items()],
          ignore_index=True).to_csv(HIST_OUT, index=False)
print(f"per-epoch curves -> {HIST_OUT}")

print("\n=== best per target, on the primary split ===")
prim = results[results.split == PRIMARY]
for t in TARGETS:
    sub = prim[prim.target == t].sort_values("macro_f1", ascending=False)
    b, base = sub.iloc[0], sub[sub.model == "C shared body"]
    ref = base.iloc[0].macro_f1 if len(base) else float("nan")
    print(f"  {t:7} best {b.macro_f1:.4f} ({b.model})   "
          f"vs plain C {ref:.4f}   gain {b.macro_f1 - ref:+.4f}   acc {b.accuracy:.4f}")

# %% [markdown]
# ### 10.5 Why the improvements are small — the ceiling, measured two ways
#
# §10.1–10.3 tried three standard levers. Pooled over the runs that measured them
# (§9.2), each is worth between **+0.003 and +0.008** on `usage` and nothing that
# survives a sign test on `gender` — against class weighting's +0.081, which §6 had
# already found. Before calling that a failure of effort, it is worth measuring what
# the task allows at all. Two measurements do that, and neither needs a GPU.
#
# **Measurement 1 — metadata oracles.** Give a predictor one metadata field for free,
# let it memorise the modal target per value on training rows only, and score it on
# validation. It cannot be beaten by any model that sees only that field, so it bounds
# how much the field says about the target. The interesting comparison is against the
# CNN, which sees none of it and only ever sees 60×80 pixels.

# %%
print("=== metadata oracles: what the catalogue alone determines ===")
_tr_o, _va_o = splits[("gender", "primary")]
print(f"train {len(_tr_o):,}  val {len(_va_o):,}   (same split as everything above)")

ORACLE_FEATURES = {
    "articleType": ["articleType"],
    "subCategory": ["subCategory"],
    "baseColour": ["baseColour"],
    "season": ["season"],
    "productDisplayName": ["productDisplayName"],
    "all metadata": ["articleType", "subCategory", "masterCategory",
                     "baseColour", "season", "year"],
}

# The notebook only ever touched articleType and subCategory before this section, so
# whether the de-duplicated CSV still carries the other four fields is unverified here.
# Degrade to a smaller table rather than crashing seventy minutes into a run.
_have = set(frame.columns)
_want = sorted({c for cols in ORACLE_FEATURES.values() for c in cols})
_missing = [c for c in _want if c not in _have]
if _missing:
    print(f"  columns absent from this CSV, dropped from the oracles: {_missing}")
    ORACLE_FEATURES = {k: [c for c in v if c in _have] for k, v in ORACLE_FEATURES.items()}
    ORACLE_FEATURES = {k: v for k, v in ORACLE_FEATURES.items() if v}
    print("  NOTE: the figures quoted in the 10.5 prose below were measured with all six")
    print("  metadata fields present. This run has fewer, so read the table, not the prose.")
assert "articleType" in _have, "articleType is the load-bearing oracle and is missing"

oracle_tables = {}
for t in TARGETS:
    fallback = _tr_o[t].mode().iloc[0]
    rows = []
    feats = dict(ORACLE_FEATURES)
    feats["the other target"] = [o for o in TARGETS if o != t]
    for name, cols in feats.items():
        lut = (_tr_o.groupby(cols, observed=True)[t]
                    .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else fallback))
        key = _va_o[cols[0]] if len(cols) == 1 else pd.MultiIndex.from_frame(_va_o[cols])
        pred = pd.Series(lut.reindex(key).to_numpy(), index=_va_o.index)
        seen = float(pred.notna().mean())
        pred = pred.fillna(fallback)
        s = score(_va_o[t], pred.to_numpy(), labels=CLASSES[t])
        rows.append({"oracle knows": name, "groups": len(lut),
                     "val in a seen group": round(seen, 4),
                     "accuracy": round(s["accuracy"], 4),
                     "macro-F1": round(s["macro_f1"], 4)})
    oracle_tables[t] = pd.DataFrame(rows)
    # The CNN row is the same split and the same metric, so it belongs in the table.
    _r = pd.DataFrame(RESULTS)
    best = (_r[(_r["split"] == PRIMARY) & (_r["target"] == t)]
            .sort_values("accuracy", ascending=False).iloc[0])
    print(f"\n--- {t} ---")
    print(oracle_tables[t].to_string(index=False))
    print(f"  CNN, pixels only, best accuracy on this split: {best.accuracy:.4f} "
          f"({best.model})")
    _ceiling = oracle_tables[t]["accuracy"].max()
    print(f"  best oracle accuracy: {_ceiling:.4f}   "
          f"CNN minus oracle: {best.accuracy - _ceiling:+.4f}")

# %% [markdown]
# **Measurement 2 — is `both correct` a wall or a product?** The team's headline number
# is the fraction of validation items where *both* labels are right. If the two targets'
# errors were independent, that number would simply be the product of the two
# accuracies, and "raising it" would mean raising one of the factors — there would be
# nothing joint to optimise. §2 measured the targets as nearly independent, so this is a
# testable prediction rather than a guess.

# %%
print("=== is 'both correct' just the product of the two accuracies? ===")
_chk = comparison.assign(
    predicted=lambda d: (d["gender acc"] * d["usage acc"]).round(4),
    residual=lambda d: (d["both correct"] - d["gender acc"] * d["usage acc"]).round(4))
print(_chk[["design", "gender acc", "usage acc", "predicted",
            "both correct", "residual"]].to_string(index=False))
print(f"\n  largest residual {_chk.residual.abs().max():.4f} across "
      f"{len(_chk)} designs — the two heads' errors are close to independent,")
print("  so 'both correct' has no slack of its own: it moves only when a factor moves.")

# %% [markdown]
# #### What the two measurements say
#
# **`usage` accuracy is finished.** An oracle handed the true `articleType` — 121 groups,
# every validation row covered — reaches **0.8945**. Adding five more metadata fields
# moves it to **0.8928**, i.e. nowhere. The CNN, from 60×80 pixels alone, reaches
# **0.8972**, *above* the oracle. The network has already recovered as much of the
# garment's identity as the label depends on, and the oracle numbers are deterministic:
# they came out identical on both platforms. The residual 10% is not a modelling gap —
# two identical T-shirts are `Casual` or `Sports` by a marketing decision that is not
# photographed. `usage` is partly a label about the product page, not the product.
#
# **`gender` accuracy is not finished, and that is where the pixels earn their keep.**
# The best oracle, given all six metadata fields, reaches **0.8002**. The CNN reaches
# **0.9057**. The image is worth **+10.6 accuracy points** over the entire catalogue
# description, and +0.235 macro-F1 over the best oracle's 0.5142. Further effort on this
# task belongs here, not on `usage`.
#
# **So the 80% is arithmetic, not a barrier.** `both correct` is the product of the two
# accuracies to within **0.0023** across all three designs. One factor is pinned at a
# ceiling set by the labels, so raising the headline means raising `gender` accuracy and
# nothing else.
#
# **What the three levers are worth.** Judged by §9.2's test — same sign every run,
# spread below the mean, sign holding on the held-out platform — they split cleanly by
# target:
#
# | lever | `usage`: Colab → run 4 | `gender`: Colab → run 4 |
# |---|---|---|
# | logit adjustment (§10.1) | **+0.0079 → +0.0044** | +0.0039 → +0.0214, one run gained nothing |
# | mirror TTA (§10.2) | **+0.0029 → +0.0023** | sign flips on Colab |
# | `articleType` transfer (§10.3) | +0.0060 → **+0.0001** | +0.0109 → +0.0109 |
# | *class weighting (§6, for scale)* | **+0.0813 → +0.0608** | sign flips on Colab |
#
# On `usage` all three keep their sign across the platform change and all three are
# **tiny** — transfer nearly vanishes there, +0.0060 → +0.0001. Stacked, logit
# adjustment plus TTA moved plain C by +0.0096 and +0.0119 on Colab and **+0.0067** on
# the held-out platform. On `gender` not one survives a sign test on Colab.
# And the honest comparison is the last row — class weighting is worth **ten times**
# the best of them on the same target, and §6 had it before this section began.
#
# One caveat belongs on §10.1 specifically: τ is chosen by argmax on the validation set
# it is then scored on, so its gain **cannot come out negative** and these figures are
# upper bounds. TTA and transfer are fixed rules with no such selection, so their smaller
# numbers are the more trustworthy ones.
#
# Two of the three levers are **inference-time corrections of the class prior** — which
# is what class weighting does during training. Measured against each other on the same
# split: in the loss, **+0.081**; at inference, **+0.008**. Same idea, tenfold
# difference, and the cheap version loses.
#
# Logit adjustment is also visibly a trade rather than a gain. Past τ=0.5 on `usage`,
# accuracy falls off a cliff — 0.8799, then 0.6315, then 0.3061 — while macro-F1 declines
# too, so no operating point buys tail recall cheaply.
#
# **But the cheap stack does replace the expensive design.** `C + logit adjustment + TTA`
# is 289k parameters plus two forward passes and no retraining; A is 577k parameters and
# two full models. The stack beat A on `usage` in all three runs that measured both, and
# on `gender` in one of three. Since §9.3 finding 3 shows A's `gender` margin is reliable
# in sign but not in size, **the 2× parameter cost is not justified by this evidence.**
#
# **The honest summary of §10: the levers work on the target that needed them least.**
# `usage` gains a reproducible +0.01 from all three combined, against +0.081 from class
# weighting and the 0.125-per-class that `Home` and `Party` cost unconditionally. The
# gains that would matter are not in the optimiser, the loss or the inference rule — they
# are in the 79 training images spread across `Home`, `Party`, `Travel` and `Smart
# Casual`, and §9.4 says what to do about that. Reporting a measured ceiling is a more
# useful result than a fourth lever, which is why this section is titled *finding where
# the improvement stops*.