"""Does Truc's external Party data raise `usage` macro-F1? A paired A/B.

Design of the experiment, and why each choice is forced:

  * `Home` is DROPPED. It has 0 validation images and 0 in the forward split, so its
    F1 is 0 whatever we train, and in this dataset `usage=Home` means homeware
    (its one provided instance is a cushion cover), not sleepwear. Keeping it could
    only add false positives. Only the 215 `Party` training rows are used.
  * PAIRED on the seed: each seed trains both arms, so the shared initialisation
    cancels and we compare like with like. Section 9.2's lesson.
  * The gender head is MASKED on external rows. The manifest says "Do not train
    gender or subCategory heads on this manifest", and design C shares one body
    between two heads. Setting the external gender label to the sentinel -100 makes
    CrossEntropyLoss skip those rows for that head only (its default ignore_index),
    so they train `usage` and the shared body without inventing a gender label.
  * We report `k` and `c`, not just the macro. With 3 Party images in validation the
    macro-F1 delta is inside the 0.036 run-to-run band by construction, but "how
    often does the model now say Party, and how often is it right" is a direct count
    and is interpretable at this sample size.
  * Truc's own 41-image Party holdout is scored too. If the model cannot recognise
    held-out external Party dresses either, it learned nothing from the set; if it
    can, but validation does not move, the failure is domain transfer.

    python task3/experiment_party_external.py [--epochs N] [--seeds 42,43,44]
"""
import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--ext", required=True, help="folder holding task3_usage_external.csv")
ap.add_argument("--out", default=str(HERE / "experiment_party_external.csv"))
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]
EXT = Path(args.ext)


def run_prefix():
    text = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    parts = re.split(r"(?m)^# %%(.*)$", text)
    cells = [("markdown" if "markdown" in parts[k] else "code", parts[k + 1])
             for k in range(1, len(parts), 2)]
    ns = {"__name__": "__notebook__"}
    for i, (kind, body) in enumerate(cells):
        if kind == "markdown":
            if body.strip().startswith(PREFIX_STOP):
                print(f"--- prefix stopped at cell {i} (section 5) ---")
                return ns
            continue
        exec(compile(body, f"<task3_build cell {i}>", "exec"), ns)
    raise SystemExit("section 5 marker not found")


print("executing the notebook prefix so preprocessing is shared, not copied ...")
t0 = time.time()
g = run_prefix()
print(f"prefix done in {time.time() - t0:.0f}s")

need = ["frame", "splits", "TARGETS", "CLASSES", "DEVICE", "X_t", "X_all", "IDX",
        "train_model", "predict", "score", "load_images", "SEED", "EPOCHS",
        "PRIMARY", "batch_x", "class_weights"]
missing = [n for n in need if n not in g]
if missing:
    raise SystemExit(f"the prefix did not define {missing} -- section boundaries moved")

import torch

TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
tr, va = g["splits"][("gender", "primary")]
EPOCHS = args.epochs if args.epochs else g["EPOCHS"]
print(f"train {len(tr):,}  val {len(va):,}  heads {heads}  epochs {EPOCHS}")

# ---------------------------------------------------------------- external rows
man = pd.read_csv(EXT / "task3_usage_external.csv")
print()
print(f"manifest: {len(man)} rows  {dict(man['usage'].value_counts())}")
party = man[man["usage"] == "Party"].reset_index(drop=True)
n_home = int((man["usage"] == "Home").sum())
ptr = party[party["split"] == "train"].reset_index(drop=True)
pho = party[party["split"] == "holdout"].reset_index(drop=True)
print(f"Party only: {len(ptr)} train + {len(pho)} holdout   (Home dropped: {n_home} rows)")

both = pd.concat([ptr, pho], ignore_index=True)
paths = [str(EXT / str(p)) for p in both["path"]]
X_ext = g["load_images"](paths)
base_n = len(g["X_all"])
both["_row"] = np.arange(len(both)) + base_n
both["group_id"] = both["id"]
both["gender"] = "<ignore>"          # masked: see the docstring

g["X_all"] = np.concatenate([g["X_all"], X_ext], axis=0)
Xt = torch.from_numpy(g["X_all"])
if g["DEVICE"].type == "cuda" and Xt.numel() < 2_000_000_000:
    Xt = Xt.to(g["DEVICE"])
g["X_t"] = Xt
del X_ext
print("image array now:", g["X_all"].shape)

# The sentinel. CrossEntropyLoss ignores index -100 by default, so external rows
# contribute to `usage` and to the shared body but never to the gender head.
g["IDX"]["gender"]["<ignore>"] = -100

ext_tr = both[both["split"] == "train"].copy()
ext_ho = both[both["split"] == "holdout"].copy()
cols = ["_row", "group_id", "gender", "usage"]
tr_party = pd.concat([tr[cols], ext_tr[cols]], ignore_index=True)
assert tr_party["group_id"].is_unique, "external ids collide with catalogue group_ids"
assert not set(ext_tr["group_id"]) & set(va["group_id"])
assert (tr_party["_row"].values < len(g["X_all"])).all()
print(f"training {len(tr):,} -> {len(tr_party):,}   validation unchanged at {len(va):,}")

_pi = CLASSES["usage"].index("Party")
_b = tr["usage"].value_counts()
_a = tr_party["usage"].value_counts()
print(f"  Party train rows {_b.get('Party', 0)} -> {_a.get('Party', 0)}")
print("  class weight on Party "
      f"{float(g['class_weights'](tr['usage'], 'usage')[_pi]):.3f} -> "
      f"{float(g['class_weights'](tr_party['usage'], 'usage')[_pi]):.3f}")
assert bool((g["class_weights"](tr_party["gender"], "gender")
             == g["class_weights"](tr["gender"], "gender")).all()), "gender weights moved"

# ---------------------------------------------------------------------- the runs
from sklearn.metrics import precision_recall_fscore_support

VA_PARTY = int((va["usage"] == "Party").sum())
print()
print(f"validation contains {VA_PARTY} Party images out of {len(va):,}")


def measure(model, arm, seed):
    pred = g["predict"](model, va, heads)
    row = {"arm": arm, "seed": seed}
    for t in TARGETS:
        s = g["score"](va[t], pred[t], labels=CLASSES[t])
        row[f"{t} macroF1"] = round(s["macro_f1"], 4)
        row[f"{t} acc"] = round(s["accuracy"], 4)
    pu = pred["usage"]
    yu = va["usage"].values
    _, _, f, _ = precision_recall_fscore_support(
        yu, pu, labels=CLASSES["usage"], zero_division=0)
    for c, fc in zip(CLASSES["usage"], f):
        row[f"F1 {c}"] = round(float(fc), 4)
    row["k"] = int((pu == "Party").sum())
    row["c"] = int(((pu == "Party") & (yu == "Party")).sum())
    row["PartyFP from Casual"] = int(((pu == "Party") & (yu == "Casual")).sum())
    # Truc's held-out external Party images: can the model see them at all?
    ph = g["predict"](model, ext_ho, heads)
    row["extHO recall"] = round(float((ph["usage"] == "Party").mean()), 4)
    row["extHO n"] = len(ext_ho)
    return row


# Three arms, not two. Adding the data changes TWO things at once: the model sees
# 215 more Party images, and `class_weights` recomputes, so the inverse-sqrt weight
# on Party falls from 1.450 to 0.357 -- the data arrives and the pressure to chase
# the class is removed in the same step. `party` is the honest as-shipped
# comparison; `party_fixedw` holds the weight vector at the base arm's values so the
# data effect is separated from the reweighting effect.
_cw = g["class_weights"]
_frozen = {k: _cw(tr[k], k) for k in heads}

ARMS = [("base", tr, None), ("party", tr_party, None), ("party_fixedw", tr_party, _frozen)]

rows = []
for sd in SEEDS:
    for arm, frame_arm, frozen in ARMS:
        g["SEED"] = sd            # train_model re-seeds from this global itself
        g["class_weights"] = (_cw if frozen is None
                              else (lambda s, k, _f=frozen, **kw: _f[k]))
        t1 = time.time()
        m, _ = g["train_model"](f"{arm}_s{sd}", heads, frame_arm, va,
                                epochs=EPOCHS, weighted=True, verbose=False)
        g["class_weights"] = _cw
        row = measure(m, arm, sd)
        row["minutes"] = round((time.time() - t1) / 60, 1)
        rows.append(row)
        print(f"  seed {sd} {arm:5}  usage {row['usage macroF1']:.4f}  "
              f"gender {row['gender macroF1']:.4f}  "
              f"PartyF1 {row['F1 Party']:.4f}  k={row['k']:4d} c={row['c']}/{VA_PARTY}  "
              f"extHO {row['extHO recall']:.2f}  [{row['minutes']}m]")
        del m
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(args.out, index=False)

res = pd.DataFrame(rows)
res.to_csv(args.out, index=False)
print()
print(f"wrote {args.out}")

piv = res.pivot(index="seed", columns="arm")
base_spread = piv["usage macroF1"]["base"].max() - piv["usage macroF1"]["base"].min()

for other in ("party", "party_fixedw"):
    print()
    print(f"=== paired deltas, per seed ({other} minus base) ===")
    for col in ["usage macroF1", "gender macroF1", "F1 Party", "k", "c", "extHO recall"]:
        d = piv[col][other] - piv[col]["base"]
        vals = "  ".join(f"s{s}:{v:+.4f}" for s, v in d.items())
        print(f"  {col:18s} {vals}    mean {d.mean():+.4f}")
    d = piv["usage macroF1"][other] - piv["usage macroF1"]["base"]
    print(f"  -> usage macro-F1 mean {d.mean():+.4f}  "
          f"range {d.min():+.4f} to {d.max():+.4f}   "
          f"same sign in all {len(d)}: {bool((d > 0).all() or (d < 0).all())}   "
          f"|mean| > base spread ({base_spread:.4f}): {bool(abs(d.mean()) > base_spread)}")

print()
print("=== arm means over seeds ===")
print(res.groupby("arm")[["usage macroF1", "gender macroF1", "F1 Party", "k", "c",
                          "PartyFP from Casual", "extHO recall"]].mean().round(4).to_string())
print()
print(f"run-to-run spread of the BASE arm alone: {base_spread:.4f}")
print("Any delta smaller than that is not distinguishable from training noise.")
print()
print(res.to_string(index=False))
