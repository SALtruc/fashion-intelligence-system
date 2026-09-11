"""Turn the ensemble members' disagreement into data, from the saved probabilities.

An ensemble score without a disagreement rate beside it is unreadable: averaging models
that predict the same thing everywhere is arithmetic on identical vectors, and a gain
reported without showing the members differ could be a bug rather than an effect. The
first ensemble run printed these rates to its log and nowhere else, which puts them out
of reach of a generator -- and a number a report generator cannot read is a number
somebody eventually types in by hand and gets wrong.

Cheap enough to be worth doing properly: disagreement is a comparison of two argmax
vectors, so it needs the saved probabilities and no labels, no images and no GPU. The
probabilities live under artifacts/, which is gitignored, so the rates are written to
predictions/task3 where the consolidation can fold them into the one tracked table.

    python src/task3/measure_ensemble_diversity.py
"""
import itertools
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
PROBS = ROOT / "artifacts" / "task3" / "probs_cache"
OUT = ROOT / "predictions" / "task3" / "ensemble_diversity.csv"
TARGETS = ("gender", "usage")

if not PROBS.is_dir():
    raise SystemExit(f"{PROBS} not found -- run experiment_ensemble_confirm.py first")

rows = []
for split in ("forward", "random"):
    files = sorted(PROBS.glob(f"{split}_e20_seed*.npz"))
    if len(files) < 2:
        print(f"{split}: {len(files)} member(s) saved, need at least two -- skipped")
        continue
    P = {int(f.stem.split("seed")[1]): dict(np.load(f)) for f in files}
    seeds = sorted(P)
    print(f"{split}: {len(seeds)} members, seeds {seeds}")
    for t in TARGETS:
        ds = []
        for a, b in itertools.combinations(seeds, 2):
            # The plain view only. Mirror-averaging first would blur the very thing
            # being measured, which is whether two independently initialised models
            # put different rows in different classes.
            pa = P[a][f"{t}_plain"].argmax(1)
            pb = P[b][f"{t}_plain"].argmax(1)
            d = float((pa != pb).mean())
            ds.append(d)
            rows.append({"split": split, "target": t, "pair": f"{a}v{b}",
                         "metric": "disagreement", "value": round(d, 5)})
        rows.append({"split": split, "target": t, "pair": "mean",
                     "metric": "disagreement", "value": round(float(np.mean(ds)), 5)})
        rows.append({"split": split, "target": t, "pair": "n_rows",
                     "metric": "val_rows",
                     "value": int(P[seeds[0]][f"{t}_plain"].shape[0])})
        print(f"  {t:7} pairwise disagreement  mean {np.mean(ds):.3%}  "
              f"min {min(ds):.3%}  max {max(ds):.3%}   over {len(ds)} pairs")

if not rows:
    raise SystemExit("no member probabilities found for either split")
pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT.relative_to(ROOT)} ({len(rows)} rows)")
