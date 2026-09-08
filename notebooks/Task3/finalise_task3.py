"""Train the ultimate judgement for Task 3, save it, and predict the test set.

The notebook compares designs; it does not leave a model on disk, and the assignment
asks for one. So this script produces the two artefacts the notebook is missing:

    artifacts/task3/task3_gender_usage_C_weighted.pt   the model, plus everything
                                                       needed to run it
    predictions/task3_gender_usage_nguyen.csv          styles_prediction.csv format

**The ultimate judgement is design C with a class-weighted loss.** One shared
convolutional body, two heads, 289k parameters. The reasoning is entirely from the
notebook's measurements, not preference:

  * Class weighting is the only effect that survived every check -- four runs across
    two platforms, +0.0813 on Colab and +0.0608 locally on `usage` macro-F1. Nothing
    else in the notebook is that robust.
  * It costs `gender` about 0.013 against the best design (A, two models), which is
    inside the run-to-run band of 0.036. It gains `usage` about 0.059, which is far
    outside it. Trading a difference you cannot measure for one you can is the whole
    argument.
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
import io
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
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

SEEDS = [42, 43, 44]
print(f"\n=== ultimate judgement: design C, class-weighted loss, seeds {SEEDS} ===")
print("Three seeds, and the SHIPPED model is the MEDIAN -- not the best.")
print("The notebook measured this configuration four times, 0.4657-0.5020 on usage,")
print("but every one of those trained it fifth in the same script order, so they")
print("share an RNG history and are a correlated sample, not four independent draws.")
print("A fresh independent draw came out 0.045 below that band -- which is exactly")
print("section 9.2's lesson landing on this notebook's own headline number.")
print("Shipping the best of three would be selecting on the validation set again.")

trained = []
for sd in SEEDS:
    torch.manual_seed(sd)
    torch.cuda.manual_seed_all(sd)
    m, _ = g["train_model"](f"C_weighted_seed{sd}", heads_multi, tr, va,
                            weighted=True, verbose=False)
    trained.append((sd, m))
    print(f"  seed {sd} done")

print(f"\n=== each seed, and whether mirror TTA helps the WEIGHTED model ===")
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
        rows.append({"seed": sd, "view": "mirror" if tta else "single", "tta": tta,
                     **{f"{t} macro-F1": round(s[t]["macro_f1"], 4) for t in TARGETS},
                     "mean": round(float(np.mean([s[t]["macro_f1"]
                                                  for t in TARGETS])), 4)})
allr = pd.DataFrame(rows)
print(allr.drop(columns="tta").to_string(index=False))

# One TTA decision, taken from the mean over seeds rather than cherry-picked per seed.
by_tta = allr.groupby("tta")["mean"].mean()
best_tta = bool(by_tta[True] > by_tta[False])
print(f"\n  TTA averaged over seeds: {by_tta[False]:.4f} -> {by_tta[True]:.4f} "
      f"({by_tta[True] - by_tta[False]:+.4f}) -> "
      f"{'USING TTA' if best_tta else 'NOT using TTA'}")

sel = allr[allr.tta == best_tta].sort_values("mean").reset_index(drop=True)
median_seed = int(sel.iloc[len(sel) // 2]["seed"])
model = dict(trained)[median_seed]
_u = sorted(sel["usage macro-F1"].tolist())
_gd = sorted(sel["gender macro-F1"].tolist())
print(f"\n  usage macro-F1 across seeds:  {_u}   spread {max(_u) - min(_u):.4f}")
print(f"  gender macro-F1 across seeds: {_gd}   spread {max(_gd) - min(_gd):.4f}")
print(f"  SHIPPING seed {median_seed} -- the median by mean macro-F1, not the best")

pred_val = predict_rows(model, va, heads_multi, tta=best_tta)
val_scores = {t: g["score"](va[t], pred_val[t], labels=CLASSES[t]) for t in TARGETS}
print(f"  shipped model: gender {val_scores['gender']['macro_f1']:.4f}  "
      f"usage {val_scores['usage']['macro_f1']:.4f}")

# ---------------------------------------------------------------- save the model
ART = HERE.parent / "artifacts" / "task3"
for cand in (Path("D:/g2_pr/artifacts/task3"), ART):
    if cand.parent.is_dir():
        ART = cand
        break
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
                   "epochs": int(g["EPOCHS"]), "seeds_tried": SEEDS,
                   "seed_shipped": median_seed,
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
TEST_DIR = None
for cand in [HERE.parent / "A2_FashionDataset" / "FashionDataset" / "test",
             Path("D:/g2/Dataset/FashionDataset/test")]:
    if (cand / "styles_prediction.csv").exists():
        TEST_DIR = cand
        break
if TEST_DIR is None:
    raise SystemExit("styles_prediction.csv not found -- point TEST_DIR at the test set")

sample = pd.read_csv(TEST_DIR / "styles_prediction.csv")
print(f"\n=== test set: {TEST_DIR} ===")
print(f"{len(sample):,} rows, columns {list(sample.columns)}")

img_dir = TEST_DIR / "images_test"
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

PRED_DIR = None
for cand in (Path("D:/g2_pr/predictions"), HERE.parent / "predictions"):
    if cand.parent.is_dir():
        PRED_DIR = cand
        break
PRED_DIR.mkdir(parents=True, exist_ok=True)
PRED_PATH = PRED_DIR / "task3_gender_usage_nguyen.csv"
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
    for anchor in ("artifacts", "predictions"):
        if anchor in p.parts:
            return "/".join(p.parts[p.parts.index(anchor):])
    return p.name


meta = {
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
_meta_json = json.dumps(meta, indent=2)
(ART / "task3_final_metadata.json").write_text(_meta_json)
(HERE / "task3_final_metadata.json").write_text(_meta_json)
print(f"\nmetadata -> {ART / 'task3_final_metadata.json'}")
print(f"         -> {HERE / 'task3_final_metadata.json'}  (tracked in git)")
print(f"\nUPLOAD {MODEL_PATH.name} to the team Drive: artifacts/** is gitignored.")
print("\ndone. Remaining for the team: merge articleType and season into this file.")
