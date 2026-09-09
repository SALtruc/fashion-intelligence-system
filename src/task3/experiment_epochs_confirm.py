"""Confirm or refute `epochs=30` on a split that was never used to select it.

`experiment_hyperparams.py` moved five knobs and `epochs=30` came out best on `gender`:
both its seeds landed above all six baseline runs, an exact rank-sum p of 0.0357. But it
was the best of five, and correcting for that -- 1 - (1 - 0.0357)^5 = 0.166, Bonferroni
0.179 -- leaves nothing significant. A sweep can only nominate; it cannot confirm its own
nomination, because the nomination used the same validation set.

So this tests one hypothesis, fixed in advance, on the **forward split**: train on the
low ids, evaluate on the 5,661 highest, which is how the graded test set was actually
cut. That split has never been used to select anything here, and section 8.1 measures it
costing 0.138 on `gender` against the random split -- it is the honest estimate of the
graded score, not a second bite at the same apple.

Note why the forward split needs its own training runs rather than a second evaluation
of the shipped model: its holdout ids sit mostly inside the *random* split's training
set, so scoring a primary-trained model on them would be scoring it on data it saw.

Fresh seeds (45, 46, 47), not the 42/43 the sweep already drew.

    THE RULE, fixed before running:
      ADOPT  if  `gender` improves on the forward split in ALL THREE seeds
             AND the mean improvement exceeds +0.010
             AND `usage` does not fall by more than the forward split's own base spread
      KEEP 20 otherwise.

    A recorded prediction: ADOPT. Both sweep seeds cleared the entire baseline range by
    +0.008 and +0.017, and 20 epochs of cosine annealing plausibly stops a little early.
    If the forward split disagrees, the sweep was measuring selection and this says so.

    python task3/experiment_epochs_confirm.py
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
ap.add_argument("--seeds", default="45,46,47")
ap.add_argument("--out", default=str(_results_dir() / "experiment_epochs_confirm.csv"))
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

need = ["splits", "TARGETS", "CLASSES", "train_model", "predict", "score", "SEED"]
missing = [n for n in need if n not in g]
if missing:
    raise SystemExit(f"the prefix did not define {missing}")

import torch

TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
tr_f, va_f = g["splits"][("gender", "forward")]
tr_p, va_p = g["splits"][("gender", "primary")]

# The reason this file trains its own models rather than reusing the shipped one.
_overlap = len(set(va_f["id"]) & set(tr_p["id"])) / len(va_f)
print(f"\nforward split: train {len(tr_f):,}  holdout {len(va_f):,}")
print(f"  {_overlap:.1%} of the forward holdout sits in the RANDOM split's training set")
print("  -- which is why it needs its own runs, not a second evaluation of the ")
print("     model that was trained on the random split.")
assert not set(tr_f["id"]) & set(va_f["id"]), "forward split leaks into itself"

print(f"\n=== epochs 20 vs 30, paired on seeds {SEEDS}, forward split ===")
rows = []
for sd in SEEDS:
    for ep in (20, 30):
        g["SEED"] = sd
        t1 = time.time()
        m, _ = g["train_model"](f"fwd_e{ep}_s{sd}", heads, tr_f, va_f,
                                epochs=ep, weighted=True, verbose=False)
        pred = g["predict"](m, va_f, heads)
        row = {"epochs": ep, "seed": sd}
        for t in TARGETS:
            s = g["score"](va_f[t], pred[t], labels=CLASSES[t])
            row[f"{t} macroF1"] = round(s["macro_f1"], 4)
            row[f"{t} acc"] = round(s["accuracy"], 4)
        row["minutes"] = round((time.time() - t1) / 60, 1)
        rows.append(row)
        print(f"  seed {sd}  epochs {ep}   gender {row['gender macroF1']:.4f}  "
              f"usage {row['usage macroF1']:.4f}   [{row['minutes']}m]")
        del m
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(args.out, index=False)

res = pd.DataFrame(rows)
res.to_csv(args.out, index=False)
print(f"\nwrote {args.out}")

piv = res.pivot(index="seed", columns="epochs")
dg = piv["gender macroF1"][30] - piv["gender macroF1"][20]
du = piv["usage macroF1"][30] - piv["usage macroF1"][20]
band_g = piv["gender macroF1"][20].max() - piv["gender macroF1"][20].min()
band_u = piv["usage macroF1"][20].max() - piv["usage macroF1"][20].min()

print("\n=== paired deltas on the forward split (30 epochs minus 20) ===")
print("  gender  " + "  ".join(f"s{s}:{v:+.4f}" for s, v in dg.items())
      + f"    mean {dg.mean():+.4f}")
print("  usage   " + "  ".join(f"s{s}:{v:+.4f}" for s, v in du.items())
      + f"    mean {du.mean():+.4f}")
print(f"\n  the 20-epoch arm's own spread here: gender {band_g:.4f}  usage {band_u:.4f}")

c1 = bool((dg > 0).all())
c2 = bool(dg.mean() > 0.010)
c3 = bool(du.mean() >= -band_u)
print("\n=== the rule, as written before the run ===")
print(f"  1. gender improves in all {len(dg)} seeds            -> {c1}")
print(f"  2. mean gender improvement {dg.mean():+.4f} > +0.010   -> {c2}")
print(f"  3. usage delta {du.mean():+.4f} >= -{band_u:.4f}          -> {c3}")
adopt = c1 and c2 and c3
print(f"\n  VERDICT: {'ADOPT epochs=30' if adopt else 'KEEP epochs=20'}")
print(f"  recorded prediction was ADOPT -> {'CORRECT' if adopt else 'WRONG'}")

print("\n=== what follows ===")
if adopt:
    print("  The gain survives a split that never selected it, so it is not the sweep")
    print("  picking its own winner. Ship it: finalise_task3.py with epochs=30, three")
    print("  runs, median as before. The NOTEBOOK stays at 20 epochs -- holding epochs")
    print("  fixed across A/B/C/D is what makes section 5 a comparison of designs, and")
    print("  changing it there would confound the thing the notebook exists to measure.")
else:
    print("  The sweep's nomination does not survive a split it did not select on, which")
    print("  is what selection bias looks like from the inside. Keep 20 epochs, keep the")
    print("  shipped model, and report the sweep as sensitivity only -- which is how")
    print("  appendix B6 already words it.")
print()
print(res.to_string(index=False))
