"""Confirm the ensemble result, with a criterion that fixes a flaw in the first one.

experiment_ensemble.py measured two three-model ensembles on the forward split and its
pre-registered rule produced an outcome that does not survive reading:

    arm A (3 runs at SEED 42)      ensemble 0.6040, median member 0.5966,
                                   member spread 0.0072  -> +0.0074 > spread, PASSES
    arm B (SEEDs 42, 43, 44)       ensemble 0.6107, median member 0.5915,
                                   member spread 0.0216  -> +0.0192 < spread, FAILS

Arm B scores higher, beats its best member by +0.0168 against arm A's +0.0029, and
fails; arm A passes by 0.0002. The rule divides the gain by the members' spread, and
member spread is not only noise -- with different seeds it is also genuine diversity,
which is the mechanism an ensemble runs on. So the rule penalised the arm for having
the property that makes ensembling work, and passed the other one by a margin smaller
than any quantity in the table. It was a reasonable rule for the from-scratch question,
where spread really was noise, and it is the wrong rule here. Saying so before running
the replacement is the only way that admission means anything.

THE CORRECTED CRITERION, fixed before this run
    Train `--members` models with genuinely different seeds, then score every one of
    the C(n,3) three-model ensembles that pool allows. ADOPT requires all three:
      1. the MEAN of all triple ensembles exceeds the BEST single member. The best of n
         members is a maximum over n draws and therefore biased upward, so clearing it
         with an average is a deliberately conservative test.
      2. the WORST triple ensemble is at or above the MEDIAN single member -- adopting
         must not be able to make things worse than the current protocol does.
      3. the gain holds on `gender` AND `usage`, both reported, neither used alone.
    A criterion that needs no spread in its denominator cannot be gamed by member
    diversity in either direction.

THE PREDICTION, fixed before this run
    Condition 1 passes: averaging softmax over independently seeded members is one of
    the few interventions in this project with a mechanism rather than a hope behind it,
    and the first run already beat its best member by +0.0168. Condition 2 passes as
    well, but by less than +0.005 on `usage`, where four classes carry half the macro
    average on a handful of validation rows. I expect the mean triple to land 0.603 to
    0.610 on `gender`, that is +0.015 to +0.022 over our three-run reference of 0.5883,
    and I expect the spread across triples to be wider than most write-ups of ensembles
    would suggest, on the order of 0.01.

Member probabilities are saved to disk this time. The first run discarded them, so the
distribution over triples could not be computed without retraining five models -- an
hour lost to not saving four megabytes.

    python src/task3/experiment_ensemble_confirm.py
    python src/task3/experiment_ensemble_confirm.py --split random   # if forward passes
"""
import argparse
import io
import itertools
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
RES = ROOT / "predictions" / "task3"
RES.mkdir(parents=True, exist_ok=True)
SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

REF = {"forward": {"gender": [0.5877, 0.5937, 0.5834],
                   "usage": [0.3471, 0.3430, 0.3399]}}

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="forward", choices=["forward", "random"])
ap.add_argument("--members", type=int, default=6)
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--probs-dir", default="")
ap.add_argument("--out", default="")
args = ap.parse_args()
OUT = Path(args.out) if args.out else RES / f"ensemble_confirm_{args.split}.csv"
# artifacts/** is gitignored by team convention, which is the right home for a
# few megabytes of intermediate probabilities: reusable, not a result.
PROBS = (Path(args.probs_dir) if args.probs_dir
         else ROOT / "artifacts" / "task3" / "probs_cache")
PROBS.mkdir(parents=True, exist_ok=True)


def run_prefix():
    text = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    parts = re.split(r"(?m)^# %%(.*)$", text)
    cells = [("markdown" if "markdown" in parts[k] else "code", parts[k + 1])
             for k in range(1, len(parts), 2)]
    ns = {"__name__": "__notebook__"}
    for i, (kind, body) in enumerate(cells):
        if kind == "markdown":
            if body.strip().startswith(PREFIX_STOP):
                return ns
            continue
        exec(compile(body, f"<task3_build cell {i}>", "exec"), ns)
    raise SystemExit("section 5 marker not found")


print("executing the notebook prefix ...")
t0 = time.time()
g = run_prefix()
print(f"prefix done in {time.time() - t0:.0f}s")

import torch

TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
X_t, batch_x, score = g["X_t"], g["batch_x"], g["score"]
key = ("gender", "primary" if args.split == "random" else "forward")
tr, va = g["splits"][key]
SEEDS = [42 + i for i in range(args.members)]
print(f"  {args.split:8} train {len(tr):,}  val {len(va):,}   "
      f"{args.members} members, seeds {SEEDS[0]}-{SEEDS[-1]}, {args.epochs} epochs")


@torch.no_grad()
def probs(model, frame, bs=512):
    model.eval()
    rows = torch.as_tensor(frame["_row"].values, device=X_t.device)
    out = {(k, v): [] for k in heads for v in ("plain", "mirror")}
    for i in range(0, len(rows), bs):
        x = batch_x(rows[i:i + bs])
        for view, xx in (("plain", x), ("mirror", x.flip(-1))):
            o = model(xx)
            for k in heads:
                out[(k, view)].append(torch.softmax(o[k].float(), 1).cpu())
    return {f"{k}_{v}": torch.cat(out[(k, v)]).numpy() for k in heads
            for v in ("plain", "mirror")}


def scores_of(p, tta=True):
    """macro-F1 per target from a probability dict, mirror-averaged when tta."""
    out = {}
    for k in heads:
        pk = (p[f"{k}_plain"] + p[f"{k}_mirror"]) / 2 if tta else p[f"{k}_plain"]
        yp = np.array([CLASSES[k][i] for i in pk.argmax(1)])
        out[k] = round(score(va[k], yp, labels=CLASSES[k])["macro_f1"], 4)
    return out


# --------------------------------------------------------------------- train the pool
P = {}
for sd in SEEDS:
    f = PROBS / f"{args.split}_e{args.epochs}_seed{sd}.npz"
    if f.is_file():
        P[sd] = dict(np.load(f))
        print(f"  seed {sd}: probabilities reused from {f.name}")
        continue
    t1 = time.time()
    g["SEED"] = sd
    m, _ = g["train_model"](f"C_weighted_s{sd}", heads, tr, va,
                            epochs=args.epochs, weighted=True, verbose=False)
    P[sd] = probs(m, va)
    np.savez_compressed(f, **P[sd])
    s = scores_of(P[sd])
    print(f"  seed {sd}: gender {s['gender']:.4f}  usage {s['usage']:.4f}"
          f"   [{(time.time() - t1) / 60:.1f}m]  probs -> {f.name}")
    del m
    torch.cuda.empty_cache()

MEM = {sd: scores_of(P[sd]) for sd in SEEDS}
rows = [{"kind": "member", "id": str(sd), "n": 1,
         **{f"{k} macroF1": MEM[sd][k] for k in heads}} for sd in SEEDS]

# ------------------------------------------------------- every triple the pool allows
TRIPLES = list(itertools.combinations(SEEDS, 3))
print(f"\n=== all {len(TRIPLES)} three-model ensembles from {args.members} members ===")
tri = {}
for c in TRIPLES:
    avg = {kk: np.mean([P[sd][kk] for sd in c], axis=0) for kk in P[c[0]]}
    tri[c] = scores_of(avg)
    rows.append({"kind": "ensemble3", "id": "+".join(map(str, c)), "n": 3,
                 **{f"{k} macroF1": tri[c][k] for k in heads}})
    print(f"  {'+'.join(map(str, c)):14} gender {tri[c]['gender']:.4f}  "
          f"usage {tri[c]['usage']:.4f}")

allm = {kk: np.mean([P[sd][kk] for sd in SEEDS], axis=0) for kk in P[SEEDS[0]]}
s_all = scores_of(allm)
rows.append({"kind": f"ensemble{args.members}", "id": "all", "n": args.members,
             **{f"{k} macroF1": s_all[k] for k in heads}})

res = pd.DataFrame(rows)
res.insert(0, "split", args.split)
res.insert(0, "epochs", args.epochs)
res.to_csv(OUT, index=False)
print(f"\nwrote {OUT.name}")

# ------------------------------------------------------------------------- the verdict
print("\n=== the corrected criterion, as written before the run ===")
cond = {}
for k in heads:
    ms = [MEM[sd][k] for sd in SEEDS]
    ts = [tri[c][k] for c in TRIPLES]
    best, med = max(ms), float(np.median(ms))
    c1, c2 = float(np.mean(ts)) > best, min(ts) >= med
    cond[k] = c1 and c2
    print(f"  {k}")
    print(f"    members  n={len(ms)}  best {best:.4f}  median {med:.4f}  "
          f"worst {min(ms):.4f}  spread {max(ms) - min(ms):.4f}")
    print(f"    triples  n={len(ts)}  mean {np.mean(ts):.4f}  best {max(ts):.4f}  "
          f"worst {min(ts):.4f}  spread {max(ts) - min(ts):.4f}")
    print(f"    all-{args.members} ensemble {s_all[k]:.4f}")
    print(f"    1. mean triple {np.mean(ts):.4f} > best member {best:.4f} -> {c1}")
    print(f"    2. worst triple {min(ts):.4f} >= median member {med:.4f} -> {c2}")
    if args.split in REF:
        r = float(np.mean(REF[args.split][k]))
        print(f"    vs our 3-run reference {r:.4f}: mean triple "
              f"{np.mean(ts) - r:+.4f}, single member mean "
              f"{float(np.mean(ms)) - r:+.4f}")

adopt = all(cond.values())
print(f"\n  ADOPT (both targets) -> {adopt}    "
      + "  ".join(f"{k} {cond[k]}" for k in heads))
if adopt:
    print("  VERDICT: worth putting to the team. It changes inference only, so every")
    print("           experiment in the catalog stands, but it makes the submission")
    print("           three checkpoints and requires finalise_task3.py to set the seed")
    print("           per member, which the current three repeats do not do.")
else:
    print("  VERDICT: design C ships unchanged, one model.")
print("\n  recorded prediction: condition 1 passes, condition 2 passes but by under")
print("  +0.005 on usage, mean triple 0.603-0.610 on gender, triple spread near 0.01.")
