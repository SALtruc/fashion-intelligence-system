"""Design D + the class-weighted loss: the two best ingredients, never yet combined.

Across the notebook's recorded repeats, `D articleType-pretrained` is the strongest
`gender` model (0.7400 mean of two runs) and `C weighted` the strongest `usage` one
(0.4843 mean of three). They have only ever been measured apart. This measures them
together.

Why it is worth a run rather than a guess -- there is a mechanism, not just a knob.
diagnose_gender_ceiling.py localised `gender`'s loss to ONE class: Unisex, F1 0.529,
precision 0.417, recall 0.721, with 178 Men and 119 Women wrongly tagged Unisex.
Perfecting Unisex alone would take macro-F1 from 0.7202 to 0.8144. And Unisex is the
only class where an `articleType -> modal gender` lookup BEATS the CNN (0.609 vs
0.529) while losing badly everywhere else. Unisex-ness is a catalogue convention about
accessories more than a visual property, so the article-type signal is exactly what a
pixel-only model lacks there -- and design D is the one that carries it.

Run ON THE FORWARD SPLIT, not the random validation split, and this is the whole
design of the experiment. `epochs=30` won the hyper-parameter sweep on random
validation by +0.019 and then measured -0.002 on the forward split: the sweep had been
measuring its own selection. A candidate tested only where it was chosen cannot be
distinguished from that. The forward split -- train on the low ids, hold out the 5,661
highest, the way the graded test set was actually cut -- has never selected anything
here, so one pass over it needs no second confirmation stage.

The pretraining uses the forward split's TRAINING rows only. Pretraining on the random
split's rows would leak: the forward holdout sits mostly inside them.

    THE RULE, fixed before running:
      ADOPT D+weighted as the submitted model if
        1. `gender` improves in ALL THREE repeats, and
        2. the mean improvement exceeds the C arm's own spread, and
        3. `usage` does not fall by more than the C arm's own `usage` spread.
      KEEP C+weighted otherwise, and report this as sensitivity.

    A recorded prediction: REJECT on condition 1. Two reasons. The base rate for
    "beat the noise band on one split" surviving a split that did not select it is
    currently 0 for 1. And across repeats the effect is only about +0.015 -- 1.3x the
    0.0114 band -- which is not the size of thing that holds its sign three times out
    of three. What I do expect is a positive MEAN on `gender` and a flat `usage`,
    because class weighting moved `gender` by +0.0000 across three repeats (C shared
    body 0.7250, C weighted 0.7249) and all of `usage`'s gain came from it.

    The per-class table is the real test: if the mechanism above is right, Unisex F1
    rises. If `gender` improves without Unisex improving, the story is wrong even if
    the number is up.

    python notebooks/Task3/experiment_transfer_weighted.py

Roughly 75 minutes on an RTX 4070: per repeat, ~5 min pretraining, ~10 min for each
of the two arms.
"""
import argparse
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
SRC = next(p for p in (ROOT / "src" / "task3_build.py", HERE / "task3_build.py")
           if p.is_file())
PREFIX_STOP = "# ## 5 "

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="51,52,53")
ap.add_argument("--out", default=str(HERE / "experiment_transfer_weighted.csv"))
args = ap.parse_args()
SEEDS = [int(s) for s in args.seeds.split(",")]


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

for n in ("splits", "TARGETS", "CLASSES", "IDX", "train_model", "score", "frame",
          "batch_x", "X_t", "EPOCHS", "SEED"):
    if n not in g:
        raise SystemExit(f"the prefix did not define {n}")

import torch

TARGETS, CLASSES, IDX = g["TARGETS"], g["CLASSES"], g["IDX"]
heads = {t: len(CLASSES[t]) for t in TARGETS}

# Design D pretrains on articleType, which lives outside the prefix's label set.
CLASSES["articleType"] = sorted(g["frame"]["articleType"].dropna().unique())
IDX["articleType"] = {c: i for i, c in enumerate(CLASSES["articleType"])}
N_ART = len(CLASSES["articleType"])

tr_f, va_f = g["splits"][("gender", "forward")]
tr_p, _ = g["splits"][("gender", "primary")]
assert not set(tr_f["id"]) & set(va_f["id"]), "the forward split leaks into itself"
_overlap = len(set(va_f["id"]) & set(tr_p["id"])) / len(va_f)
print(f"\nforward split: train {len(tr_f):,}  holdout {len(va_f):,}")
print(f"  {_overlap:.1%} of this holdout sits in the RANDOM split's training rows,")
print(f"  which is why the pretraining below uses the FORWARD split's train rows only.")
print(f"pretraining task: articleType, {N_ART} classes")

PRE_EPOCHS = max(8, g["EPOCHS"] // 2)


@torch.no_grad()
def predict_tta(model, fr, bs=512):
    """Mirror TTA, the same arithmetic the shipped model uses."""
    model.eval()
    rows_t = torch.as_tensor(fr["_row"].values, device=g["X_t"].device)
    acc = {k: [] for k in heads}
    for i in range(0, len(rows_t), bs):
        x = g["batch_x"](rows_t[i:i + bs])
        o = model(x)
        f = model(torch.flip(x, dims=[-1]))
        o = {k: ((o[k].softmax(1) + f[k].softmax(1)) / 2).log() for k in heads}
        for k in heads:
            acc[k].append(o[k].argmax(1).cpu())
    return {k: np.asarray(CLASSES[k])[torch.cat(v).numpy()] for k, v in acc.items()}


from sklearn.metrics import classification_report

rows, per_class = [], []
print(f"\n=== C+weighted vs D+weighted, paired on repeats {SEEDS}, forward split ===")
for sd in SEEDS:
    pretrained = None
    for arm in ("C_weighted", "D_weighted"):
        t1 = time.time()
        if arm == "D_weighted":
            # train_model re-seeds from the module-level SEED on entry, so set it
            # rather than seeding here -- the lesson from the "three seeds" that were
            # three repeats.
            g["SEED"] = sd
            pre, pre_hist = g["train_model"](
                f"pretrain_art_s{sd}", {"articleType": N_ART}, tr_f, va_f,
                epochs=PRE_EPOCHS, verbose=False)
            pretrained = pre.backbone
            print(f"  repeat {sd}  pretrain articleType macro-F1 "
                  f"{pre_hist['val_articleType'].max():.4f}  "
                  f"[{(time.time() - t1) / 60:.1f}m]")
            t1 = time.time()

        g["SEED"] = sd
        m, _ = g["train_model"](f"{arm}_s{sd}", heads, tr_f, va_f, weighted=True,
                                verbose=False,
                                init_backbone=pretrained if arm == "D_weighted"
                                else None)
        pred = predict_tta(m, va_f)

        row = {"arm": arm, "repeat": sd}
        for t in TARGETS:
            s = g["score"](va_f[t], pred[t], labels=CLASSES[t])
            row[f"{t} macroF1"] = round(s["macro_f1"], 4)
            row[f"{t} acc"] = round(s["accuracy"], 4)
        row["minutes"] = round((time.time() - t1) / 60, 1)
        rows.append(row)

        rep = classification_report(va_f["gender"], pred["gender"],
                                    labels=CLASSES["gender"], output_dict=True,
                                    zero_division=0)
        for c in CLASSES["gender"]:
            per_class.append({"arm": arm, "repeat": sd, "class": c,
                              "val_n": int(rep[c]["support"]),
                              "precision": round(rep[c]["precision"], 4),
                              "recall": round(rep[c]["recall"], 4),
                              "f1": round(rep[c]["f1-score"], 4)})

        print(f"  repeat {sd}  {arm:11}  gender {row['gender macroF1']:.4f}  "
              f"usage {row['usage macroF1']:.4f}   Unisex F1 "
              f"{rep['Unisex']['f1-score']:.4f}   [{row['minutes']}m]")
        del m
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(args.out, index=False)
        pd.DataFrame(per_class).to_csv(
            args.out.replace(".csv", "_perclass.csv"), index=False)
    del pretrained
    torch.cuda.empty_cache()

res = pd.DataFrame(rows)
pc = pd.DataFrame(per_class)
res.to_csv(args.out, index=False)
pc.to_csv(args.out.replace(".csv", "_perclass.csv"), index=False)
print(f"\nwrote {args.out}")

piv = res.pivot(index="repeat", columns="arm")
dg = piv["gender macroF1"]["D_weighted"] - piv["gender macroF1"]["C_weighted"]
du = piv["usage macroF1"]["D_weighted"] - piv["usage macroF1"]["C_weighted"]
band_g = (piv["gender macroF1"]["C_weighted"].max()
          - piv["gender macroF1"]["C_weighted"].min())
band_u = (piv["usage macroF1"]["C_weighted"].max()
          - piv["usage macroF1"]["C_weighted"].min())

print("\n=== paired deltas on the forward split (D+weighted minus C+weighted) ===")
print("  gender  " + "  ".join(f"r{s}:{v:+.4f}" for s, v in dg.items())
      + f"    mean {dg.mean():+.4f}")
print("  usage   " + "  ".join(f"r{s}:{v:+.4f}" for s, v in du.items())
      + f"    mean {du.mean():+.4f}")
print(f"\n  the C arm's own spread here: gender {band_g:.4f}  usage {band_u:.4f}")

print("\n=== the mechanism: Unisex, the class the hypothesis is about ===")
u = pc[pc["class"] == "Unisex"].pivot(index="repeat", columns="arm", values="f1")
u["delta"] = u["D_weighted"] - u["C_weighted"]
print(u.round(4).to_string())
print("\n  gender F1 by class, mean over repeats:")
print(pc.pivot_table(index="class", columns="arm", values="f1")
        .assign(delta=lambda d: d["D_weighted"] - d["C_weighted"])
        .round(4).to_string())

c1 = bool((dg > 0).all())
c2 = bool(dg.mean() > band_g)
c3 = bool(du.mean() >= -band_u)
print("\n=== the rule, as written before the run ===")
print(f"  1. gender improves in all {len(dg)} repeats              -> {c1}")
print(f"  2. mean gender {dg.mean():+.4f} > C's own spread {band_g:.4f}   -> {c2}")
print(f"  3. usage {du.mean():+.4f} >= -{band_u:.4f}                    -> {c3}")
adopt = c1 and c2 and c3
print(f"\n  VERDICT: {'ADOPT D+weighted' if adopt else 'KEEP C+weighted'}")
print(f"  recorded prediction was REJECT on condition 1 -> "
      f"{'WRONG' if adopt else 'CORRECT' if not c1 else 'PARTLY -- c1 held'}")

print("\n=== what follows ===")
if adopt:
    print("  It survives a split that never selected it, and the notebook's own")
    print("  ingredients explain why. Shipping it costs: finalise_task3.py with")
    print("  init_backbone (~30 min), a full notebook re-run for outputs (~90 min),")
    print("  regenerated metadata, the gate, and a rewrite of the report's section 11")
    print("  ultimate judgement. Roughly 2.5 hours. Deadline is 12/09.")
else:
    print("  Keep C+weighted, and keep the reported model exactly as it is. This goes")
    print("  into the report as sensitivity beside the epochs=30 result -- and if the")
    print("  per-class table shows Unisex flat, that is the more interesting finding:")
    print("  the article-type signal does not transfer through a shared backbone even")
    print("  when a lookup on the same signal beats the model.")
print()
print(res.to_string(index=False))
