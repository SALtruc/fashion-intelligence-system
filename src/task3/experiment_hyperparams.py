"""Hyper-parameter sensitivity for the shipped Task 3 configuration.

The notebook fixes `lr=1e-3`, `dropout=0.2`, `bs=256`, `epochs=20` for every design so
that the A/B/C/D comparison measures the design and not a tuning difference. That is the
right call for the comparison and the wrong answer to the spec, which lists
"Hyper-parameter setting and tuning to refine the model" among the elements a thorough
investigation should contain. This fills the gap.

**Reported as a sensitivity analysis, not as a search.** No configuration is selected on
the strength of its validation score -- picking the best of N on the same validation set
we then report is the selection bias section 10.1 already warns about, and with N small
and the noise band wide the winner would mostly be luck. What the table answers is "how
much does the score move when this knob moves", which is a claim about the model, not a
claim about the best setting.

**Read it on `gender` first.** Three genuinely-seeded runs put the run-to-run spread at
**0.007 on `gender`** but **0.047 on `usage`**, because all of `usage`'s instability lives
in classes holding 3 to 9 validation images. So `gender` can resolve a 0.01-0.02 effect
here and `usage` cannot resolve anything below ~0.05. `gender` is also the target with
headroom left: `usage` is capped near 0.5 by its own label noise (section 10.5).

The default configuration is NOT retrained. It has already been measured six times, by
the `base` arms of `experiment_party_external.py` and `experiment_catalog_external.py`,
on this same split, hardware and code. Those runs are the baseline and its noise band, so
the sweep only trains the configurations that differ from the default.

    python task3/experiment_hyperparams.py [--seeds 42,43]
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

DEFAULT = {"lr": 1e-3, "dropout": 0.2, "bs": 256, "epochs": 20}

# One knob moved at a time, around the default. Not a grid: a full grid over four knobs
# is 18+ configurations and three hours, and with the noise band this wide the extra
# cells would buy resolution we cannot use.
VARIANTS = [
    {"lr": 3e-4},
    {"lr": 3e-3},
    {"dropout": 0.0},
    {"dropout": 0.4},
    {"epochs": 30},
]

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", default="42,43")
ap.add_argument("--out", default=str(_results_dir() / "experiment_hyperparams.csv"))
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

need = ["splits", "TARGETS", "CLASSES", "train_model", "predict", "score", "Net",
        "SEED", "EPOCHS"]
missing = [n for n in need if n not in g]
if missing:
    raise SystemExit(f"the prefix did not define {missing} -- section boundaries moved")

import torch

TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
tr, va = g["splits"][("gender", "primary")]
print(f"train {len(tr):,}  val {len(va):,}  heads {heads}")

# Confirm the notebook's defaults really are what this script calls the default, so the
# baseline reused below belongs to the same configuration.
import inspect
_sig = inspect.signature(g["train_model"]).parameters
assert _sig["lr"].default == DEFAULT["lr"], f"train_model lr default moved: {_sig['lr'].default}"
assert _sig["bs"].default == DEFAULT["bs"], f"train_model bs default moved: {_sig['bs'].default}"
assert g["EPOCHS"] == DEFAULT["epochs"], f"EPOCHS is {g['EPOCHS']}, not {DEFAULT['epochs']}"
_dp = inspect.signature(g["Net"].__init__).parameters["dropout"].default
assert _dp == DEFAULT["dropout"], f"Net dropout default moved: {_dp}"
print(f"defaults confirmed: {DEFAULT}")

# --------------------------------------------------- the already-measured baseline
BASE_FILES = ["experiment_party_external.csv", "experiment_catalog_external.csv"]
base_rows = []
for f in BASE_FILES:
    p = HERE / f
    if not p.exists():
        print(f"  note: {f} not present, its baseline runs are unavailable")
        continue
    d = pd.read_csv(p)
    d = d[d["arm"] == "base"]
    base_rows.append(d[["seed", "gender macroF1", "usage macroF1"]].assign(source=f))
if not base_rows:
    raise SystemExit("no baseline runs found -- run experiment_party_external.py first, "
                     "or this sweep has nothing to compare against")
base = pd.concat(base_rows, ignore_index=True)
BASE = {t: float(base[f"{t} macroF1"].mean()) for t in TARGETS}
NOISE = {t: float(base[f"{t} macroF1"].max() - base[f"{t} macroF1"].min()) for t in TARGETS}
print()
print(f"=== baseline: the default configuration, {len(base)} runs already measured ===")
print(base.to_string(index=False))
for t in TARGETS:
    print(f"  {t:7s} mean {BASE[t]:.4f}   spread {NOISE[t]:.4f}   "
          f"({base[f'{t} macroF1'].min():.4f} to {base[f'{t} macroF1'].max():.4f})")
print()
print("Those spreads are the resolution limit. A knob that moves the score by less than")
print("its target's spread has not been shown to do anything.")

# ----------------------------------------------------------------------- the sweep
OrigNet = g["Net"]


def measure(model, label, cfg, seed):
    pred = g["predict"](model, va, heads)
    row = {"config": label, "seed": seed, **{k: cfg[k] for k in DEFAULT}}
    for t in TARGETS:
        s = g["score"](va[t], pred[t], labels=CLASSES[t])
        row[f"{t} macroF1"] = round(s["macro_f1"], 4)
        row[f"{t} acc"] = round(s["accuracy"], 4)
    row["mean macroF1"] = round(float(np.mean([row[f"{t} macroF1"] for t in TARGETS])), 4)
    return row


rows = []
for var in VARIANTS:
    cfg = {**DEFAULT, **var}
    label = ", ".join(f"{k}={var[k]}" for k in var)
    for sd in SEEDS:
        g["SEED"] = sd
        # train_model builds Net(heads) with the class's own default, so dropout is
        # varied by swapping the class the notebook's namespace resolves.
        g["Net"] = (OrigNet if cfg["dropout"] == DEFAULT["dropout"]
                    else (lambda h, _d=cfg["dropout"]: OrigNet(h, dropout=_d)))
        t1 = time.time()
        m, _ = g["train_model"](f"hp_{label}_s{sd}", heads, tr, va,
                                epochs=cfg["epochs"], weighted=True,
                                lr=cfg["lr"], bs=cfg["bs"], verbose=False)
        g["Net"] = OrigNet
        row = measure(m, label, cfg, sd)
        row["minutes"] = round((time.time() - t1) / 60, 1)
        rows.append(row)
        print(f"  {label:16s} seed {sd}  gender {row['gender macroF1']:.4f} "
              f"({row['gender macroF1'] - BASE['gender']:+.4f})  "
              f"usage {row['usage macroF1']:.4f} "
              f"({row['usage macroF1'] - BASE['usage']:+.4f})  [{row['minutes']}m]")
        del m
        torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(args.out, index=False)

res = pd.DataFrame(rows)
res.to_csv(args.out, index=False)
print()
print(f"wrote {args.out}")

print()
print("=== sensitivity: mean over seeds, against the default and its noise band ===")
print(f"{'configuration':18s} {'gender':>8s} {'delta':>9s} {'>noise?':>8s}"
      f" {'usage':>8s} {'delta':>9s} {'>noise?':>8s}")
summary = []
for label, grp in res.groupby("config", sort=False):
    line = f"{label:18s}"
    rec = {"config": label, "n_seeds": len(grp)}
    for t in TARGETS:
        v = float(grp[f"{t} macroF1"].mean())
        d = v - BASE[t]
        beats = abs(d) > NOISE[t]
        line += f" {v:8.4f} {d:+9.4f} {('YES' if beats else 'no'):>8s}"
        rec[f"{t}"] = round(v, 4)
        rec[f"{t} delta"] = round(d, 4)
        rec[f"{t} beats_noise"] = bool(beats)
    print(line)
    summary.append(rec)
print(f"{'DEFAULT':18s} {BASE['gender']:8.4f} {0.0:+9.4f} {'-':>8s}"
      f" {BASE['usage']:8.4f} {0.0:+9.4f} {'-':>8s}")
print(f"\nnoise bands: gender {NOISE['gender']:.4f}   usage {NOISE['usage']:.4f}")

pd.DataFrame(summary).to_csv(
    Path(args.out).with_name(Path(args.out).stem + "_summary.csv"), index=False)

print()
print("=== verdict ===")
any_gender = [r["config"] for r in summary if r["gender beats_noise"]]
any_usage = [r["config"] for r in summary if r["usage beats_noise"]]
print(f"  configurations moving gender beyond its {NOISE['gender']:.4f} band: "
      f"{any_gender if any_gender else 'none'}")
print(f"  configurations moving usage  beyond its {NOISE['usage']:.4f} band: "
      f"{any_usage if any_usage else 'none'}")
if not any_gender and not any_usage:
    print()
    print("  No knob moved either target beyond its own run-to-run spread. That is a")
    print("  result, not a failure: the shipped configuration is insensitive to lr over")
    print("  a factor of 10, to dropout over 0.0-0.4, and to training half again as long,")
    print("  so the score is not being left on the table by an untuned hyper-parameter.")
print()
print("  Reported as sensitivity. No configuration is adopted on the strength of a")
print("  validation score measured on the same split the report quotes.")
print()
print(res.to_string(index=False))
