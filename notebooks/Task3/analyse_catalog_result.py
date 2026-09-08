"""Score `experiment_catalog_external.py`'s output against a rule fixed in advance.

The decision this automates is: does Truc's `Task3CatalogCandidates` set earn a place
in the notebook and in the shipped model, or does it belong in the report as a measured
negative result?

Deciding that after seeing the numbers is how a noise band becomes a discovery. So the
rule is written here, before the third seed finished, and the script only applies it:

    ADOPT  if   the paired mean `usage` macro-F1 delta exceeds +0.047
                -- the run-to-run spread the base arm itself shows, so a smaller
                   delta is not distinguishable from training noise --
           AND  `Party` F1 is above zero in at least 2 of the 3 seeds
                -- the macro can drift upward on `Travel`/`Smart Casual` jitter alone;
                   requiring the targeted class to actually move stops us crediting the
                   data for noise in a class it never touched --
           AND  `Formal` F1 does not fall by more than its own base-arm spread
                -- `Formal` carries 343 validation images against the 15 held by all
                   three target classes combined. Trading a measurable class for an
                   unmeasurable one is not an improvement.

    REJECT otherwise.

A recorded prediction, also fixed in advance: **REJECT, and `Formal` falls.** The set
teaches 250 formal shoes as `Smart Casual` while the provided data labels 585 of 610
`Formal` -- 65:1 against. If `Formal` holds steady the prediction was wrong and that is
worth saying plainly.

    python task3/analyse_catalog_result.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "experiment_catalog_external.csv"
VERDICT = HERE / "experiment_catalog_verdict.txt"

NOISE_GATE = 0.047          # the base arm's own seed spread on usage macro-F1
TARGETS3 = ["Party", "Smart Casual", "Travel"]
VAL_N = {"Casual": 4306, "Ethnic": 383, "Formal": 343, "Home": 0,
         "Party": 3, "Smart Casual": 9, "Sports": 614, "Travel": 3}

if not CSV.exists():
    raise SystemExit(f"{CSV} not present -- job 1 has not written anything yet")

res = pd.read_csv(CSV)
n_seeds = res["seed"].nunique()
if len(res) < 2 * n_seeds or set(res["arm"]) != {"base", "catalog"}:
    raise SystemExit(f"{CSV} holds {len(res)} rows over {n_seeds} seeds -- incomplete")

out = []


def say(s=""):
    print(s)
    out.append(s)


piv = res.pivot(index="seed", columns="arm")
say(f"=== catalog external arm, {n_seeds} paired seeds ===")
say()

d_usage = piv["usage macroF1"]["catalog"] - piv["usage macroF1"]["base"]
d_gender = piv["gender macroF1"]["catalog"] - piv["gender macroF1"]["base"]
base_spread = piv["usage macroF1"]["base"].max() - piv["usage macroF1"]["base"].min()
formal_spread = piv["F1 Formal"]["base"].max() - piv["F1 Formal"]["base"].min()

say("per-seed paired deltas (catalog minus base):")
for name, d in [("usage macroF1", d_usage), ("gender macroF1", d_gender)]:
    say(f"  {name:16s} " + "  ".join(f"s{s}:{v:+.4f}" for s, v in d.items())
        + f"    mean {d.mean():+.4f}")
say()
say(f"  base arm's own seed spread: usage {base_spread:.4f}   Formal {formal_spread:.4f}")
say()

say("per-class F1, arm means, and where the macro delta actually comes from:")
say(f"  {'class':13s} {'val n':>6s} {'base':>8s} {'catalog':>8s} {'delta':>9s} {'contrib':>9s}")
contrib = {}
for c in VAL_N:
    col = f"F1 {c}"
    if col not in piv:
        continue
    b, a = piv[col]["base"].mean(), piv[col]["catalog"].mean()
    contrib[c] = (a - b) / 8
    mark = "  <- targeted" if c in TARGETS3 else ""
    say(f"  {c:13s} {VAL_N[c]:6d} {b:8.4f} {a:8.4f} {a - b:+9.4f} {contrib[c]:+9.4f}{mark}")
say(f"  {'':13s} {'':6s} {'':8s} {'':8s} {'TOTAL':>9s} {sum(contrib.values()):+9.4f}")
say()

party_pos = int((piv["F1 Party"]["catalog"] > 0).sum())
d_formal = (piv["F1 Formal"]["catalog"] - piv["F1 Formal"]["base"]).mean()
if "extHO acc" in piv:
    say(f"external holdout accuracy: base {piv['extHO acc']['base'].mean():.4f}"
        f"  ->  catalog {piv['extHO acc']['catalog'].mean():.4f}")
    say("  (high here with a flat validation is the signature of a domain gap: the set")
    say("   was learned, and none of it crossed to the provided catalogue's images)")
    say()

say("=== the rule, as written before the run ===")
c1 = d_usage.mean() > NOISE_GATE
c2 = party_pos >= 2
c3 = d_formal >= -formal_spread
say(f"  1. usage paired mean {d_usage.mean():+.4f} > +{NOISE_GATE:.3f}          -> {c1}")
say(f"  2. Party F1 > 0 in >=2 of {n_seeds} seeds (actual {party_pos})   -> {c2}")
say(f"  3. Formal delta {d_formal:+.4f} >= -{formal_spread:.4f} (its own spread) -> {c3}")
adopt = bool(c1 and c2 and c3)
say()
say(f"  VERDICT: {'ADOPT' if adopt else 'REJECT'}")
say()

say("=== the recorded prediction ===")
say(f"  predicted REJECT   -> {'CORRECT' if not adopt else 'WRONG'}")
say(f"  predicted Formal falls, measured {d_formal:+.4f}   -> "
    f"{'CORRECT' if d_formal < 0 else 'WRONG, Formal held or rose'}")
per_seed = "  ".join(f"s{s}:{v:+.4f}" for s, v in
                     (piv["F1 Formal"]["catalog"] - piv["F1 Formal"]["base"]).items())
say(f"  Formal per seed: {per_seed}")
if d_formal < 0 and not bool(((piv["F1 Formal"]["catalog"]
                               - piv["F1 Formal"]["base"]) < 0).all()):
    say("  note: negative on the mean but not on every seed, so the direction is")
    say("        indicated, not established. Say so rather than claiming the prediction.")
say()

say("=== what follows ===")
if adopt:
    say("  ADOPT: the data earns a notebook section (a 7.2 beside the cosmetics arm) and")
    say("  a re-run of finalise_task3.py to reship the model. Both are code changes, so")
    say("  the notebook's outputs must be regenerated afterwards -- tonight's notebook")
    say("  run becomes stale and has to be repeated.")
else:
    say("  REJECT: handled the way the imaterialist set was -- this standalone script and")
    say("  its CSV are the evidence, the report carries one paragraph, and the notebook is")
    say("  NOT modified. Tonight's notebook run therefore stands, and the shipped model")
    say("  is unchanged.")

VERDICT.write_text("\n".join(out) + "\n", encoding="utf-8")
print(f"\nwrote {VERDICT}")
sys.exit(0 if not adopt else 3)      # exit 3 = ADOPT, so the runner can branch on it
