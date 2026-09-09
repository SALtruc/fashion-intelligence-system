"""Does the `Task3CatalogCandidates` set raise `usage` macro-F1? A paired A/B.

Run for Truc, whose Colab quota is exhausted, on the machine every other number in
the Task 3 report was measured on -- so the result is directly comparable.

**Prediction recorded before the run, so the test can fail.** `Formal` should DROP.
The set teaches 250 formal shoes as `Smart Casual`, and the provided training data
labels 585 of its 610 formal shoes `Formal` against only 9 `Smart Casual` -- 65:1 the
other way. `Formal` carries 343 validation images and currently scores F1 0.755, so it
is one of the few classes stable enough for the damage to show. The hoped-for gain sits
on `Party` (3 validation images), `Smart Casual` (9) and `Travel` (3).

Design, and why each choice is forced:

  * The audit is unreviewed (119/119 rows `review_status=pending`), so there is no
    accepted manifest. This runs the **unaudited** set at face value: all 753 rows whose
    image downloaded, labelled by `usage_suggested`. That is the most-volume,
    least-precision case. If it does not help here it will not help from 119 reviewed
    rows either, and if it does help the audit is worth finishing.
  * PAIRED on the seed: each seed trains both arms, so shared initialisation cancels.
  * The gender head is MASKED on external rows -- the manifest sets
    `supervise_gender=False` on all 840 -- via the sentinel -100, which
    CrossEntropyLoss ignores by default. The rows still train `usage` and the shared body.
  * 15% of the external rows are held out from training and scored separately. That
    separates "the model never learned this data" from "it learned it and none of it
    transferred". On the previous external set that number was the informative one:
    recall went 0.00 -> 0.88 on its own holdout while the provided `Party` F1 never
    left 0.0000.
  * Per-class F1 is reported for all eight classes. With 15 validation images across the
    three target classes the macro cannot resolve the effect; `Formal` and `Sports`
    (343 and 614 images) can.

    python task3/experiment_catalog_external.py --ext D:/tmp_cat/Task3CatalogCandidates
"""
import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def _results_dir():
    """Outputs go to results/task3, not next to the source. This file lives in
    src/task3 now, so HERE is the wrong place to write."""
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            d = base / "results" / "task3"
            d.mkdir(parents=True, exist_ok=True)
            return d
    return HERE

SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--seeds", default="42,43,44")
ap.add_argument("--ext", required=True)
ap.add_argument("--holdout", type=float, default=0.15)
ap.add_argument("--out", default=str(_results_dir() / "experiment_catalog_external.csv"))
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
        "batch_x", "class_weights"]
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
man = pd.read_csv(EXT / "task3_catalog_candidates.csv")
print()
print(f"manifest: {len(man)} rows   sources {dict(man['source'].value_counts())}")
have = man[man["image_exists"].astype(str).str.lower() == "true"].copy()
print(f"rows whose image downloaded: {len(have)}  "
      f"(dropped {len(man) - len(have)} with image_exists=False)")
have["path"] = have["external_id"].str.replace(":", "_", regex=False).map(
    lambda s: str(EXT / "images" / f"{s}.jpg"))
missing_files = [p for p in have["path"] if not Path(p).exists()]
if missing_files:
    raise SystemExit(f"{len(missing_files)} manifest rows point at absent files, "
                     f"e.g. {missing_files[:3]}")
have["usage"] = have["usage_suggested"]
assert set(have["usage"]) <= set(CLASSES["usage"]), \
    f"unknown usage values: {set(have['usage']) - set(CLASSES['usage'])}"
print(f"usage_suggested: {dict(have['usage'].value_counts())}")
print(f"mapping groups : {dict(have['mapping_group'].value_counts())}")

# Hold out whole external rows, stratified by the taught class, so the holdout has
# every class in it. Seeded here and not per-arm: both arms see the same split.
rng = np.random.RandomState(0)
hold = []
for _c, grp in have.groupby("usage"):
    idx = grp.index.to_numpy()
    rng.shuffle(idx)
    hold.extend(idx[:max(1, int(round(args.holdout * len(idx))))])
have["ext_split"] = np.where(have.index.isin(hold), "holdout", "train")
print(f"external split : {dict(have['ext_split'].value_counts())}")

X_ext = g["load_images"](have["path"].tolist())
base_n = len(g["X_all"])
have = have.reset_index(drop=True)
have["_row"] = np.arange(len(have)) + base_n
have["group_id"] = have["external_id"]
have["gender"] = "<ignore>"          # supervise_gender=False on every manifest row

g["X_all"] = np.concatenate([g["X_all"], X_ext], axis=0)
Xt = torch.from_numpy(g["X_all"])
if g["DEVICE"].type == "cuda" and Xt.numel() < 2_000_000_000:
    Xt = Xt.to(g["DEVICE"])
g["X_t"] = Xt
del X_ext
print("image array now:", g["X_all"].shape)

g["IDX"]["gender"]["<ignore>"] = -100

ext_tr = have[have["ext_split"] == "train"].copy()
ext_ho = have[have["ext_split"] == "holdout"].copy()
cols = ["_row", "group_id", "gender", "usage"]
tr_cat = pd.concat([tr[cols], ext_tr[cols]], ignore_index=True)
assert tr_cat["group_id"].is_unique, "external ids collide with catalogue group_ids"
assert not set(ext_tr["group_id"]) & set(va["group_id"])
assert (tr_cat["_row"].values < len(g["X_all"])).all()
print(f"training {len(tr):,} -> {len(tr_cat):,}   validation unchanged at {len(va):,}")

print()
print("  class counts and inverse-sqrt weights, before -> after:")
for c in CLASSES["usage"]:
    i = CLASSES["usage"].index(c)
    wb = float(g["class_weights"](tr["usage"], "usage")[i])
    wa = float(g["class_weights"](tr_cat["usage"], "usage")[i])
    nb, na = int((tr["usage"] == c).sum()), int((tr_cat["usage"] == c).sum())
    flag = "  <-- changed" if na != nb else ""
    print(f"    {c:13s} n {nb:6d} -> {na:6d}   weight {wb:.3f} -> {wa:.3f}{flag}")
assert bool((g["class_weights"](tr_cat["gender"], "gender")
             == g["class_weights"](tr["gender"], "gender")).all()), "gender weights moved"

# ---------------------------------------------------------------------- the runs
from sklearn.metrics import precision_recall_fscore_support

TARGET_CLASSES = sorted(have["usage"].unique())
print()
print("validation support for the classes this data targets:")
for c in TARGET_CLASSES:
    print(f"    {c:13s} {int((va['usage'] == c).sum())} images")


def measure(model, arm, seed):
    pred = g["predict"](model, va, heads)
    row = {"arm": arm, "seed": seed}
    for t in TARGETS:
        s = g["score"](va[t], pred[t], labels=CLASSES[t])
        row[f"{t} macroF1"] = round(s["macro_f1"], 4)
        row[f"{t} macroF1_inval"] = round(s["macro_f1_in_val"], 4)
        row[f"{t} acc"] = round(s["accuracy"], 4)
    pu, yu = pred["usage"], va["usage"].values
    _, _, f, _ = precision_recall_fscore_support(
        yu, pu, labels=CLASSES["usage"], zero_division=0)
    for c, fc in zip(CLASSES["usage"], f):
        row[f"F1 {c}"] = round(float(fc), 4)
    for c in TARGET_CLASSES:
        row[f"k {c}"] = int((pu == c).sum())
        row[f"c {c}"] = int(((pu == c) & (yu == c)).sum())
    # Did it learn the external concept at all? Score the held-out external rows.
    ph = g["predict"](model, ext_ho, heads)["usage"]
    row["extHO acc"] = round(float((ph == ext_ho["usage"].values).mean()), 4)
    row["extHO n"] = len(ext_ho)
    return row


_cw = g["class_weights"]
ARMS = [("base", tr), ("catalog", tr_cat)]
rows = []
for sd in SEEDS:
    for arm, frame_arm in ARMS:
        g["SEED"] = sd
        t1 = time.time()
        m, _ = g["train_model"](f"{arm}_s{sd}", heads, frame_arm, va,
                                epochs=EPOCHS, weighted=True, verbose=False)
        row = measure(m, arm, sd)
        row["minutes"] = round((time.time() - t1) / 60, 1)
        rows.append(row)
        print(f"  seed {sd} {arm:8}  usage {row['usage macroF1']:.4f}  "
              f"gender {row['gender macroF1']:.4f}  |  "
              f"Formal {row['F1 Formal']:.4f}  Sports {row['F1 Sports']:.4f}  |  "
              f"Party {row['F1 Party']:.4f}  SmartCas {row['F1 Smart Casual']:.4f}  "
              f"Travel {row['F1 Travel']:.4f}  |  extHO {row['extHO acc']:.2f}  "
              f"[{row['minutes']}m]")
        del m
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(args.out, index=False)

res = pd.DataFrame(rows)
res.to_csv(args.out, index=False)
print()
print(f"wrote {args.out}")

piv = res.pivot(index="seed", columns="arm")
base_spread = piv["usage macroF1"]["base"].max() - piv["usage macroF1"]["base"].min()

print()
print("=== paired deltas per seed (catalog minus base) ===")
watch = ["usage macroF1", "gender macroF1"] + [f"F1 {c}" for c in CLASSES["usage"]] \
        + ["extHO acc"]
for col in watch:
    d = piv[col]["catalog"] - piv[col]["base"]
    print(f"  {col:20s} " + "  ".join(f"s{s}:{v:+.4f}" for s, v in d.items())
          + f"    mean {d.mean():+.4f}")

print()
print("=== the recorded prediction: did Formal drop? ===")
df = piv["F1 Formal"]["catalog"] - piv["F1 Formal"]["base"]
print(f"  Formal F1 delta: mean {df.mean():+.4f}  range {df.min():+.4f} to {df.max():+.4f}"
      f"   negative in all {len(df)}: {bool((df < 0).all())}")
du = piv["usage macroF1"]["catalog"] - piv["usage macroF1"]["base"]
print(f"  usage macro-F1 : mean {du.mean():+.4f}  range {du.min():+.4f} to {du.max():+.4f}"
      f"   same sign in all {len(du)}: {bool((du > 0).all() or (du < 0).all())}")
print(f"  base arm's own seed spread on usage macro-F1: {base_spread:.4f}")
print("  -- any delta smaller than that is not distinguishable from training noise.")

print()
print("=== arm means ===")
show = ["usage macroF1", "usage macroF1_inval", "gender macroF1"] \
       + [f"F1 {c}" for c in CLASSES["usage"]] + ["extHO acc"]
print(res.groupby("arm")[show].mean().round(4).T.to_string())
print()
print(res.to_string(index=False))
