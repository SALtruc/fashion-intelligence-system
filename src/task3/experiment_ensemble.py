"""Does averaging three design-C models help, and does the help survive the forward split?

Task 1's best result is an ensemble of three plus flip TTA (0.7811 -> 0.8010). We have
never measured one. The reason it is worth measuring here, when almost nothing else has
transferred, is mechanical: an ensemble reduces *variance*, and the forward split is
where extra *capacity* has just been shown to evaporate (the from-scratch ResNet18 kept
only 22% of its random-split gender gain). Variance reduction and capacity are not the
same lever, so the base rate of "random-split gain fails to transfer" does not
automatically apply.

There is a second reason, specific to our pipeline: finalise_task3.py ALREADY trains
three models and ships the median, discarding two. If averaging helps, the training
cost is already paid and the only change is the inference step.

But that same fact hides a trap, and it is the point of the two arms. Those three
"repeats" share initialisation and batch order -- train_model() re-seeds from the
module-level SEED on entry -- so they differ only through nondeterministic GPU kernels.
Three models with correlated errors cannot be averaged into much. An ensemble pays off
when its members are wrong in *different places*, so this script measures the
disagreement rate as well as the score; a gain with no disagreement behind it would be
a measurement error, not an ensemble.

    arm A   3 runs at SEED 42          the free version, no pipeline change
    arm B   runs at SEED 42, 43, 44    genuinely different initialisations
            (the SEED-42 member is shared, so this costs 5 trainings, not 6)

THE RULE, fixed before the run
    ADOPT_A  arm A's ensemble beats the MEDIAN of its own members on `gender` by more
             than those members' spread. The median, not the best, because that is what
             finalise_task3.py ships -- beating the best member would be a different and
             easier-to-fake claim.
    ADOPT_B  arm B's ensemble clears the same bar against its own members AND beats
             arm A's ensemble. Both conditions: a seeded ensemble that only matches the
             free one does not justify changing how the shipped model is trained.
    `usage` is reported beside `gender` in every table and is not used to decide,
    because `gender` is where the reference is tightest (spread 0.0103 over three runs
    against `usage`'s 0.0072 on a much noisier quantity). Reporting only the target that
    happened to win is the failure mode this project has been guarding against since
    section 9.2.

THE PREDICTION, fixed before the run
    Arm A: members disagree on under 5% of validation rows and the ensemble gains under
    +0.005 on `gender` -- ADOPT_A fails. Arm B: members disagree on 8-15% of rows and
    the ensemble gains +0.005 to +0.020, so ADOPT_B is genuinely uncertain and is the
    reason to spend the fifty minutes. Both arms gain less than Task 1's +0.020, because
    their three models differ by more than an initialisation.
    On `usage` I predict nothing directional: the four rare classes carry half the macro
    average on a handful of validation rows, so that column is noise-dominated either
    way.

    python src/task3/experiment_ensemble.py                  # forward split
    python src/task3/experiment_ensemble.py --split random   # only if forward passes
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


def _results_dir():
    d = _repo_root() / "results" / "task3"
    d.mkdir(parents=True, exist_ok=True)
    return d


ROOT = _repo_root()
SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

# Our own C_weighted, three runs, 20 epochs, class-weighted, no TTA, from
# experiment_epochs_confirm.csv. Kept only as a cross-session sanity check: every
# comparison this script decides on is measured inside this run.
REF = {"forward": {"gender": [0.5877, 0.5937, 0.5834],
                   "usage": [0.3471, 0.3430, 0.3399]},
       "random": {"gender": [0.7202], "usage": [0.4676]}}

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="forward", choices=["forward", "random"])
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--out", default="")
args = ap.parse_args()
OUT = Path(args.out) if args.out else _results_dir() / f"ensemble_{args.split}.csv"
PC_OUT = Path(str(OUT).replace(".csv", "_perclass.csv"))


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
from sklearn.metrics import classification_report

TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
X_t, batch_x, score = g["X_t"], g["batch_x"], g["score"]
key = ("gender", "primary" if args.split == "random" else "forward")
tr, va = g["splits"][key]
print(f"  {args.split:8} train {len(tr):,}  val {len(va):,}   "
      f"{args.epochs} epochs, class-weighted")


@torch.no_grad()
def probs(model, frame, bs=512):
    """Softmax per target for the plain and the mirrored view, on CPU.

    Two views are returned rather than one averaged pair so that a member can be
    scored with and without TTA from the same forward passes. Mirroring after
    normalisation is the same tensor as mirroring before it: MEAN and STD are
    per-channel scalars, so the two operations commute.
    """
    model.eval()
    rows = torch.as_tensor(frame["_row"].values, device=X_t.device)
    out = {(k, v): [] for k in heads for v in ("plain", "mirror")}
    for i in range(0, len(rows), bs):
        x = batch_x(rows[i:i + bs])
        for view, xx in (("plain", x), ("mirror", x.flip(-1))):
            o = model(xx)
            for k in heads:
                out[(k, view)].append(torch.softmax(o[k].float(), 1).cpu())
    return {kv: torch.cat(v).numpy() for kv, v in out.items()}


def labels_from(p, k):
    return np.array([CLASSES[k][i] for i in p.argmax(1)])


rows_out, pc_out = [], []


def record(tag, kind, p, n_members, minutes, extra=""):
    """One row per (tag, tta) pair; per-class rows alongside."""
    got = {}
    for tta in (False, True):
        r = {"arm": tag, "kind": kind, "split": args.split, "tta": tta,
             "members": n_members, "epochs": args.epochs,
             "minutes": round(minutes, 1), "note": extra}
        for k in heads:
            pk = (p[(k, "plain")] + p[(k, "mirror")]) / 2 if tta else p[(k, "plain")]
            yp = labels_from(pk, k)
            s = score(va[k], yp, labels=CLASSES[k])
            r[f"{k} macroF1"] = round(s["macro_f1"], 4)
            r[f"{k} acc"] = round(s["accuracy"], 4)
            rep = classification_report(va[k], yp, labels=CLASSES[k],
                                        output_dict=True, zero_division=0)
            for c in CLASSES[k]:
                pc_out.append({"arm": tag, "kind": kind, "tta": tta, "target": k,
                               "class": c, "val_n": int(rep[c]["support"]),
                               "f1": round(rep[c]["f1-score"], 4)})
        rows_out.append(r)
        got[tta] = r
    pd.DataFrame(rows_out).to_csv(OUT, index=False)
    pd.DataFrame(pc_out).to_csv(PC_OUT, index=False)
    print(f"  {tag:16} {kind:8} tta=False gender {got[False]['gender macroF1']:.4f}"
          f"  usage {got[False]['usage macroF1']:.4f}"
          f"   | tta=True gender {got[True]['gender macroF1']:.4f}"
          f"  usage {got[True]['usage macroF1']:.4f}   [{round(minutes, 1)}m]")
    return got


# ------------------------------------------------------------------ train the members
# Keyed by label; the SEED-42 member belongs to both arms and is trained once.
MEMBERS = [("r0", 42), ("r1", 42), ("r2", 42), ("s43", 43), ("s44", 44)]
P = {}
for name, sd in MEMBERS:
    t1 = time.time()
    g["SEED"] = sd            # train_model() reads this global on its first line
    m, _ = g["train_model"](f"C_weighted_{name}", heads, tr, va,
                            epochs=args.epochs, weighted=True, verbose=False)
    P[name] = probs(m, va)
    record(f"member_{name}", "single", P[name], 1, (time.time() - t1) / 60,
           f"SEED={sd}")
    del m
    torch.cuda.empty_cache()

ARMS = {"A_repeats": ["r0", "r1", "r2"], "B_seeds": ["r0", "s43", "s44"]}


def disagreement(names):
    """Mean pairwise fraction of validation rows where two members' argmax differ.

    The number an ensemble score cannot do without: averaging models that agree
    everywhere is arithmetic on identical vectors.
    """
    out = {}
    for k in heads:
        ds = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = P[names[i]][(k, "plain")].argmax(1)
                b = P[names[j]][(k, "plain")].argmax(1)
                ds.append(float((a != b).mean()))
        out[k] = float(np.mean(ds))
    return out


print("\n=== how different are the members, before any score is quoted ===")
DIS = {}
for arm, names in ARMS.items():
    DIS[arm] = disagreement(names)
    print(f"  {arm:10} pairwise disagreement   "
          + "  ".join(f"{k} {DIS[arm][k]:.3%}" for k in heads))

print("\n=== the ensembles ===")
ENS = {}
for arm, names in ARMS.items():
    avg = {kv: np.mean([P[n][kv] for n in names], axis=0)
           for kv in P[names[0]]}
    ENS[arm] = record(f"ens_{arm}", "ensemble", avg, len(names), 0.0,
                      "mean of member softmax: " + "+".join(names))

# ------------------------------------------------------------------------- the verdict
print(f"\nwrote {OUT.name} and {PC_OUT.name}")
print()
res = pd.DataFrame(rows_out)
print(res.to_string(index=False))


def member_scores(names, k, tta=True):
    col = f"{k} macroF1"
    return [float(res[(res.arm == f"member_{n}") & (res.tta == tta)][col].iloc[0])
            for n in names]


print("\n=== the rule, as written before the run ===")
verdict = {}
for arm, names in ARMS.items():
    ms = member_scores(names, "gender")
    med, spread = float(np.median(ms)), max(ms) - min(ms)
    ens = ENS[arm][True]["gender macroF1"]
    gain = ens - med
    verdict[arm] = gain > spread
    print(f"  {arm}: ensemble {ens:.4f} vs median member {med:.4f} "
          f"-> {gain:+.4f} vs member spread {spread:.4f} -> {verdict[arm]}")
    print(f"      members {['%.4f' % x for x in ms]}  best {max(ms):.4f} "
          f"(ensemble beats best by {ens - max(ms):+.4f})")
    us = member_scores(names, "usage")
    print(f"      usage: ensemble {ENS[arm][True]['usage macroF1']:.4f} vs median "
          f"{float(np.median(us)):.4f} -> "
          f"{ENS[arm][True]['usage macroF1'] - float(np.median(us)):+.4f} "
          f"(reported, not used to decide)")

adopt_a = verdict["A_repeats"]
adopt_b = (verdict["B_seeds"]
           and ENS["B_seeds"][True]["gender macroF1"]
           > ENS["A_repeats"][True]["gender macroF1"])
print(f"\n  ADOPT_A (free, no pipeline change) -> {adopt_a}")
print(f"  ADOPT_B (vary the seed as well)     -> {adopt_b}")
if adopt_a or adopt_b:
    print("  VERDICT: a legal candidate. It changes inference only, not the "
          "architecture,\n           so the experiment catalog stands -- but it needs "
          "a confirmation run\n           before it goes anywhere near the submission.")
else:
    print("  VERDICT: design C ships unchanged, one model, as it has all along.")
print(f"\n  recorded prediction: arm A disagrees <5% and gains <0.005 (fails); arm B "
      f"disagrees\n  8-15% and gains +0.005..+0.020; both under Task 1's +0.020.")
print(f"  measured: A disagreement {DIS['A_repeats']['gender']:.3%}, "
      f"B {DIS['B_seeds']['gender']:.3%}")
ref = REF[args.split]["gender"]
print(f"\n  cross-session check: our 3-run reference on this split is "
      f"{float(np.mean(ref)):.4f}; this run's members averaged "
      f"{float(np.mean(member_scores(ARMS['A_repeats'], 'gender', tta=False))):.4f} "
      f"without TTA.")
