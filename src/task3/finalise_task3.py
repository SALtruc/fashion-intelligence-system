"""Train the ultimate judgement for Task 3, save it, and predict the test set.

The notebook compares designs; it does not leave a model on disk, and the assignment
asks for one. So this script produces the two artefacts the notebook is missing:

    artifacts/task3/task3_gender_usage_C_weighted.pt   the model, plus everything
                                                       needed to run it
    predictions/task3_gender_usage.csv                 styles_prediction.csv format

**The ultimate judgement is design C with a class-weighted loss.** One shared
convolutional body, two heads, 289k parameters. The reasoning is entirely from the
notebook's measurements, not preference:

  * Class weighting is the only effect that survived every check -- four runs across
    two platforms, +0.0813 on Colab and +0.0608 locally on `usage` macro-F1. Nothing
    else in the notebook is that robust.
  * It costs `gender` about 0.013-0.025 against the best design (A, two models) and
    gains `usage` about 0.06-0.10. Both are real: three genuinely-seeded runs put
    `gender`'s spread at only 0.007 and `usage`'s at 0.047, so A's gender edge is not
    noise -- it is simply the smaller of the two effects. Giving up ~0.02 on one
    target to gain ~0.08 on the other lifts the mean of the two by about +0.03, and
    that is the argument.
  * A would cost 2x the parameters for the `gender` margin, and section 9.3 finding 3
    shows that margin is reliable in sign but not in size (+0.037, +0.023, +0.006,
    +0.031 across four runs).

Mirror TTA is measured here rather than assumed: the notebook only ever tested it on
the *unweighted* model, so this script scores the weighted model both ways on
validation and uses whichever wins, saying which.

Preprocessing and architecture are not reimplemented. The script executes the
notebook's own cells up to the end of section 4 and then uses the definitions it
finds, so there is no second copy of the decode/pad/normalise path to drift.

    python finalise_task3.py
"""
import datetime
import hashlib
import inspect
import io
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _results_dir():
    """Outputs go to predictions/task3, not next to the source. This file lives in
    src/task3 now, so HERE is the wrong place to write."""
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            d = base / "predictions" / "task3"
            d.mkdir(parents=True, exist_ok=True)
            return d
    return HERE

SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "          # everything before section 5 defines what we need


def run_prefix():
    """Execute the notebook's cells up to section 5 and return its namespace."""
    text = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    parts = re.split(r"(?m)^# %%(.*)$", text)
    cells = [("markdown" if "markdown" in parts[k] else "code", parts[k + 1])
             for k in range(1, len(parts), 2)]
    ns = {"__name__": "__notebook__"}
    for i, (kind, body) in enumerate(cells):
        if kind == "markdown":
            if body.strip().startswith(PREFIX_STOP):
                print(f"\n--- prefix stopped at cell {i} (section 5) ---")
                return ns
            continue
        exec(compile(body, f"<task3_build cell {i}>", "exec"), ns)
    raise SystemExit("section 5 marker not found -- did the notebook's structure change?")


print("executing the notebook prefix so preprocessing is shared, not copied ...")
t0 = time.time()
ns = run_prefix()
print(f"prefix done in {time.time() - t0:.0f}s")

# Names the prefix defines, pulled out so failures are loud and immediate.
need = ["frame", "splits", "TARGETS", "CLASSES", "DEVICE", "MEAN", "STD", "X_t",
        "train_model", "predict", "score", "Net", "batch_x", "X_all",
        "load_images", "IMAGE_SIZE", "SEED", "EPOCHS", "PRIMARY", "record"]
missing = [n for n in need if n not in ns]
if missing:
    raise SystemExit(f"the prefix did not define {missing} -- section boundaries moved")
g = ns
TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]

import numpy as np
import pandas as pd
import torch

heads_multi = {t: len(CLASSES[t]) for t in TARGETS}
tr, va = g["splits"][("gender", "primary")]
print(f"\ntrain {len(tr):,}  val {len(va):,}  |  heads {heads_multi}")

SEEDS = [42, 43, 44]        # labels for three repeats, NOT three seeds -- see below
print(f"\n=== ultimate judgement: design C, class-weighted loss, {len(SEEDS)} repeats ===")
print("Three repeat runs, and the SHIPPED model is the MEDIAN -- not the best.")
print("The notebook measured this configuration four times, 0.4657-0.5020 on usage,")
print("but every one of those trained it fifth in the same script order, so they")
print("share an RNG history and are a correlated sample, not four independent draws.")
print("A fresh independent draw came out 0.045 below that band -- which is exactly")
print("section 9.2's lesson landing on this notebook's own headline number.")
print("Shipping the best of three would be selecting on the validation set again.")

# These are three REPEATS, not three seeds, and the distinction is worth stating
# because the variable is named `sd`. `train_model` re-seeds from the module-level
# SEED as its first statement, so seeding here would be overwritten and is not
# attempted; the three runs share initialisation and batch order, and differ only
# through nondeterministic GPU kernels. That is still a valid noise estimate -- it
# is the irreducible variation of the training procedure itself -- and shipping the
# median of three of them is still not selecting on validation. It is simply a
# narrower source of variation than changing the seed would be.
#
# Measured afterwards, with the seed genuinely varied (see
# experiment_party_external.py, whose base arm does this properly): `gender` spreads
# 0.007 across three seeds and `usage` 0.047. So this loop's spread understates the
# `usage` band and overstates the `gender` one; section 9.3 carries both numbers.
trained = []
for sd in SEEDS:
    m, _ = g["train_model"](f"C_weighted_repeat{sd}", heads_multi, tr, va,
                            weighted=True, verbose=False)
    trained.append((sd, m))
    print(f"  repeat {sd} done")

print(f"\n=== each repeat, and whether mirror TTA helps the WEIGHTED model ===")
print("(the notebook only ever tested TTA on the UNWEIGHTED model, so it is measured",
      "here rather than assumed)")
# Same arithmetic as the notebook's logits_of, which lives in section 10 and so is
# not in the prefix: average the two softmaxes, then take logs back.
@torch.no_grad()
def predict_rows(model, fr, heads, bs=512, tta=False):
    model.eval()
    rows_t = torch.as_tensor(fr["_row"].values, device=g["X_t"].device)
    acc = {k: [] for k in heads}
    for i in range(0, len(rows_t), bs):
        x = g["batch_x"](rows_t[i:i + bs])
        o = model(x)
        if tta:
            f = model(torch.flip(x, dims=[-1]))
            o = {k: ((o[k].softmax(1) + f[k].softmax(1)) / 2).log() for k in heads}
        for k in heads:
            acc[k].append(o[k].argmax(1).cpu())
    return {k: np.asarray(CLASSES[k])[torch.cat(v).numpy()] for k, v in acc.items()}


rows = []
for sd, m in trained:
    for tta in (False, True):
        p = predict_rows(m, va, heads_multi, tta=tta)
        s = {t: g["score"](va[t], p[t], labels=CLASSES[t]) for t in TARGETS}
        rows.append({"repeat": sd, "view": "mirror" if tta else "single", "tta": tta,
                     **{f"{t} macro-F1": round(s[t]["macro_f1"], 4) for t in TARGETS},
                     "mean": round(float(np.mean([s[t]["macro_f1"]
                                                  for t in TARGETS])), 4)})
allr = pd.DataFrame(rows)
print(allr.drop(columns="tta").to_string(index=False))

# One TTA decision, taken from the mean over repeats rather than cherry-picked per run.
by_tta = allr.groupby("tta")["mean"].mean()
best_tta = bool(by_tta[True] > by_tta[False])
print(f"\n  TTA averaged over repeats: {by_tta[False]:.4f} -> {by_tta[True]:.4f} "
      f"({by_tta[True] - by_tta[False]:+.4f}) -> "
      f"{'USING TTA' if best_tta else 'NOT using TTA'}")

sel = allr[allr.tta == best_tta].sort_values("mean").reset_index(drop=True)
median_run = int(sel.iloc[len(sel) // 2]["repeat"])
model = dict(trained)[median_run]
_u = sorted(sel["usage macro-F1"].tolist())
_gd = sorted(sel["gender macro-F1"].tolist())
print(f"\n  usage macro-F1 across repeats:  {_u}   spread {max(_u) - min(_u):.4f}")
print(f"  gender macro-F1 across repeats: {_gd}   spread {max(_gd) - min(_gd):.4f}")
print(f"  SHIPPING repeat {median_run} -- the median by mean macro-F1, not the best")

pred_val = predict_rows(model, va, heads_multi, tta=best_tta)
val_scores = {t: g["score"](va[t], pred_val[t], labels=CLASSES[t]) for t in TARGETS}
print(f"  shipped model: gender {val_scores['gender']['macro_f1']:.4f}  "
      f"usage {val_scores['usage']['macro_f1']:.4f}")

# ---------------------------------------------------------------- save the model
def _repo_root():
    """Walk up to the .git entry. Beats a hardcoded absolute path, which only ever
    worked on the machine it was typed on."""
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    return HERE.parent.parent


ROOT = _repo_root()
# artifacts/task3, which is gitignored, and from there to the team Drive. The
# checkpoint spent an evening under models/task3/checkpoints because three of the four
# tasks keep theirs under models/; the team lead's call is that models/ counts as
# artifacts and does not get pushed, so this follows the written convention rather
# than the majority practice.
ART = ROOT / "artifacts" / "task3"
CKPT = ART
ART.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ART / "task3_gender_usage_C_weighted.pt"
torch.save({
    "state_dict": model.state_dict(),
    "heads": heads_multi,
    "classes": {t: list(CLASSES[t]) for t in TARGETS},
    "image_size": g["IMAGE_SIZE"],
    "channel_mean": g["MEAN"].flatten().tolist(),
    "channel_std": g["STD"].flatten().tolist(),
    "use_tta": best_tta,
    "design": "C - one shared conv body, one head per target, class-weighted loss",
    "trained_on": {"split_file": "train_val_grouped_sha256.csv",
                   "n_train": int(len(tr)), "n_val": int(len(va)),
                   "epochs": int(g["EPOCHS"]), "repeats_tried": SEEDS,
                   "repeat_shipped": median_run,
                   "selection": "median by mean macro-F1, not the best"},
    "val_macro_f1": {t: round(val_scores[t]["macro_f1"], 4) for t in TARGETS},
    "val_macro_f1_in_val_classes": {t: round(val_scores[t]["macro_f1_in_val"], 4)
                                    for t in TARGETS},
    "val_accuracy": {t: round(val_scores[t]["accuracy"], 4) for t in TARGETS},
}, MODEL_PATH)
print(f"\nmodel -> {MODEL_PATH}  ({MODEL_PATH.stat().st_size / 1e6:.1f} MB)")

# ------------------------------------------------------------ predict the test set
# `predict` and `logits_of` index X_t, the training images already on the GPU, so
# neither works on the test set. The three lines of normalisation below are the only
# thing duplicated from batch_x, and they must stay identical to it: same /255, same
# MEAN and STD -- which are computed from TRAINING images and reused here rather than
# recomputed on the test set, because recomputing would fit a statistic to the data
# being predicted.
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))
from src import data_paths                                  # noqa: E402

template_path, img_dir = data_paths.test_template(), data_paths.test_images()
if not template_path.is_file() or not img_dir.is_dir():
    data_paths.check()

sample = pd.read_csv(template_path)
print(f"\n=== test set: {template_path.parent} ===")
print(f"{len(sample):,} rows, columns {list(sample.columns)}")
paths = [str(img_dir / f"{i}.jpg") for i in sample["id"]]
absent = [p for p in paths if not Path(p).exists()]
if absent:
    raise SystemExit(f"{len(absent)} test images missing, first: {absent[0]}")

overlap = set(sample["id"]) & set(g["frame"]["id"])
assert not overlap, f"{len(overlap)} test ids also appear in training -- leakage"
print("leakage check: no test id appears in the training frame")

X_test = g["load_images"](paths, cache=None)
MEAN, STD, DEVICE = g["MEAN"], g["STD"], g["DEVICE"]


@torch.no_grad()
def predict_paths(model, arr, heads, bs=512, tta=False):
    model.eval()
    out = {k: [] for k in heads}
    for i in range(0, len(arr), bs):
        x = torch.from_numpy(arr[i:i + bs]).to(DEVICE)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = (x - MEAN) / STD
        o = model(x)
        if tta:
            f = model(torch.flip(x, dims=[-1]))
            o = {k: ((o[k].softmax(1) + f[k].softmax(1)) / 2).log() for k in heads}
        for k in heads:
            out[k].append(o[k].argmax(1).cpu())
    return {k: np.asarray(CLASSES[k])[torch.cat(v).numpy()] for k, v in out.items()}


test_pred = predict_paths(model, X_test, heads_multi, tta=best_tta)

# styles_prediction.csv is "a sample file showing the required submission format.
# Please do not change the format" -- so fill the two columns Task 3 owns, keep the
# other columns and the row order exactly as given, and leave the teammates' targets
# untouched for whoever merges the four tasks.
out = sample.copy()
out["gender"] = test_pred["gender"]
out["usage"] = test_pred["usage"]
assert list(out.columns) == list(sample.columns), "column set changed"
assert len(out) == len(sample) and (out["id"].values == sample["id"].values).all(), \
    "row order changed"

PRED_DIR = ROOT / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH = PRED_DIR / "task3_gender_usage.csv"
out.to_csv(PRED_PATH, index=False)
print(f"\npredictions -> {PRED_PATH}")

print("\npredicted class distribution on the test set, against training:")
for t in TARGETS:
    a = out[t].value_counts(normalize=True).mul(100).round(2).rename("test %")
    b = g["frame"][t].value_counts(normalize=True).mul(100).round(2).rename("train %")
    print(f"\n--- {t} ---")
    print(pd.concat([a, b], axis=1).fillna(0).to_string())

def _rel(p):
    """Repo-relative, so a committed path means something on another machine."""
    p = Path(p)
    for anchor in ("models", "artifacts", "predictions"):
        if anchor in p.parts:
            return "/".join(p.parts[p.parts.index(anchor):])
    return p.name


def _git(*a):
    """The commit is what makes the weights reproducible; a missing one is worth
    recording as missing rather than silently omitting."""
    try:
        return subprocess.run(("git", *a), cwd=str(HERE), capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


# Read the two knobs off the function signature rather than retyping them, so the
# metadata cannot drift from the code the way a hardcoded number would.
_sig = inspect.signature(g["train_model"]).parameters
_dirty = _git("status", "--porcelain")

meta = {
    # -- the fields artifacts/README.md lists as required --------------------------
    "artifact": _rel(MODEL_PATH),
    "task": "task3",
    "created_at": datetime.date.today().isoformat(),
    "git_commit": _git("rev-parse", "HEAD"),
    "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
    "git_clean": (_dirty == "") if _dirty is not None else None,
    "dataset_version": ("A2_FashionDataset as provided, images 60x80 RGB, "
                        f"{len(g['frame']):,} labelled rows"),
    "split": f"splits/task3/{g['SHARED_SPLIT_NAME']}",
    "preprocessing": ("embedded in the checkpoint: image_size, channel_mean, "
                      "channel_std. No separate transformer file."),
    "labels": "embedded in the checkpoint under 'classes'; head order is that list",
    "framework": f"PyTorch {torch.__version__}",
    "sha256": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest(),
    "size_bytes": MODEL_PATH.stat().st_size,
    "validation_metrics": {t: {"macro_f1": round(val_scores[t]["macro_f1"], 4),
                               "accuracy": round(val_scores[t]["accuracy"], 4)}
                           for t in TARGETS},
    "notes": (f"python src/task3/finalise_task3.py -- design C, class-weighted "
              f"loss, {'mirror TTA' if best_tta else 'no TTA'}, {g['EPOCHS']} epochs, "
              f"batch {_sig['bs'].default}, Adam lr {_sig['lr'].default}. Shipped "
              f"checkpoint is the MEDIAN of {len(SEEDS)} repeats by mean macro-F1, "
              f"not the best. train_model() re-seeds from the module-level SEED on "
              f"entry, so those repeats differ by cuDNN nondeterminism only -- they "
              f"are repeats, not independent seeds. macro-F1 uses labels= over all "
              f"{len(CLASSES['usage'])} usage classes; dropping the class absent from "
              f"validation would read about 0.07 higher."),

    # -- Task 3 specifics, kept from the previous version --------------------------
    "model": _rel(MODEL_PATH),
    "predictions": _rel(PRED_PATH),
    "design": "C shared body, class-weighted loss",
    "tta_used": best_tta,
    "val_macro_f1": {t: round(val_scores[t]["macro_f1"], 4) for t in TARGETS},
    "val_macro_f1_over_classes_present": {
        t: round(val_scores[t]["macro_f1_in_val"], 4) for t in TARGETS},
    "val_accuracy": {t: round(val_scores[t]["accuracy"], 4) for t in TARGETS},
    "n_test": int(len(out)),
    "columns_filled": ["gender", "usage"],
    "columns_left_for_teammates": [c for c in out.columns
                                   if c not in ("id", "gender", "usage")],
}
# artifacts/** is gitignored by team convention -- weights go to the team Drive, not
# into the repo. So the metadata is written twice: beside the model for whoever picks
# it up from Drive, and into the tracked notebook folder so the numbers describing the
# submitted model survive in git even though the weights do not.
# Written from one dict to every location, so the copies cannot drift apart.
_meta_json = json.dumps(meta, indent=2)
print()
for _d in (ART, _results_dir()):
    (_d / "task3_final_metadata.json").write_text(_meta_json, encoding="utf-8")
    print(f"metadata -> {_d / 'task3_final_metadata.json'}")
print(f"\nUPLOAD {MODEL_PATH.name} to the team Drive: artifacts/** is gitignored.")
print("\ndone. Remaining for the team: merge articleType and season into this file.")
