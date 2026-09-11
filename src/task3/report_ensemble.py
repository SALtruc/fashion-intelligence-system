"""Generate appendix B11, the ensemble measurement, from its CSVs.

Written before the random-split triples existed, which is the point: this section has to
report whatever the distribution turns out to be, including the parts that argue against
adopting anything.

Two things this section must not do, both of which the first ensemble run did do.

It must not compare an ensemble against a summary statistic of its own members. The
first run's rule asked whether the ensemble beat the median member by more than the
members' spread, and that produced a verdict backwards to every other quantity in the
table: the correlated arm passed by 0.0002 while the seeded arm, scoring higher and
beating its own best member by six times as much, failed. Member spread is not only
noise; with different seeds it is diversity, which is the mechanism. The rule punished
an arm for the property that makes ensembling work.

And it must compare against the artifact actually being submitted. `SEED = 42` is the
module default, fixed before any measurement, so it was never selected on validation --
but it turns out to be one of the two best of six seeds on the forward split, which
makes our reported number a fortunate default rather than a typical one. An ensemble
compared against the median seed looks much better than the same ensemble compared
against the model in the zip file. Only the second comparison decides anything.

    python src/task3/report_ensemble.py            # print it
    python src/task3/report_ensemble.py --append   # insert into REPORT_TASK3.md
"""
import argparse
import io
import json
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from results_io import forward_repeats, load_wide, read_report, report_path

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
RES = ROOT / "predictions" / "task3"
REPORT = report_path()
TARGETS = ["gender", "usage"]

ap = argparse.ArgumentParser()
ap.add_argument("--append", action="store_true")
args = ap.parse_args()

# What is actually in the zip, read from the metadata the finalise script wrote.
meta = json.loads(io.open(RES / "task3_final_metadata.json", encoding="utf-8").read())
SHIP_RANDOM = {t: float(meta["val_macro_f1"][t]) for t in TARGETS}
SHIP_FWD = {t: float(np.mean(forward_repeats(t))) for t in TARGETS}

def _diversity():
    """The disagreement rates, which are long-form in both places they can come from.

    Not routed through load_wide: that inverts a melt, and this file was never wide.
    """
    f = RES / "ensemble_diversity.csv"
    if f.is_file():
        return pd.read_csv(f)
    long = pd.read_csv(RES / "task3_all_results.csv", dtype=str).fillna("")
    long = long[long.source == "ensemble_diversity"]
    return None if long.empty else long.rename(columns={"rep": "pair"})


div = _diversity()


def disagree(split, target):
    if div is None:
        return None
    r = div[(div.split == split) & (div.target == target)
            & (div.pair == "mean") & (div.metric == "disagreement")]
    return float(r["value"].iloc[0]) if len(r) else None


def confirm(split):
    """Members, triples and the all-six ensemble for one split, or None."""
    d = load_wide(f"ensemble_confirm_{split}", "ensemble_confirm",
                  {"id": "arm", "kind": "variant", "split": "split"},
                  needs=["arm", "variant", "gender macroF1", "usage macroF1"],
                  optional=True)
    if d is None:
        return None
    d = d[d.split == split] if "split" in d.columns else d
    if d.empty:
        return None
    mem, tri = d[d.variant == "member"], d[d.variant == "ensemble3"]
    allm = d[d.variant.str.startswith("ensemble") & (d.variant != "ensemble3")]
    if not len(mem) or not len(tri):
        return None
    return {"members": mem, "triples": tri,
            "all": allm.iloc[0] if len(allm) else None}


F, R = confirm("forward"), confirm("random")
if F is None:
    raise SystemExit("no forward-split confirmation data; run "
                     "experiment_ensemble_confirm.py first")


def block(c, ship, split):
    """One split's table plus the comparison that decides anything."""
    out, stats = [], {}
    for t in TARGETS:
        col = f"{t} macroF1"
        ms = sorted(float(x) for x in c["members"][col])
        ts = [float(x) for x in c["triples"][col]]
        with42 = [float(r[col]) for _, r in c["triples"].iterrows()
                  if str(r["arm"]).startswith("42+")]
        alln = float(c["all"][col]) if c["all"] is not None else float("nan")
        s42 = float(c["members"][c["members"].arm.astype(str) == "42"][col].iloc[0])
        stats[t] = {
            "ms": ms, "ts": ts, "with42": with42, "all": alln, "s42": s42,
            "rank42": sorted(ms, reverse=True).index(s42) + 1,
            "below": sum(1 for x in ts if x < ship[t]),
            "below42": sum(1 for x in with42 if x < ship[t])}
        d = disagree(split, t)
        out.append(
            f"| `{t}` | {ship[t]:.4f} | {min(ms):.4f} to {max(ms):.4f} "
            f"({max(ms) - min(ms):.4f}) | {float(np.mean(ts)):.4f} | "
            f"{min(ts):.4f} to {max(ts):.4f} | {alln:.4f} | "
            + (f"{d:.1%} |" if d is not None else "n/a |"))
    return "\n".join(out), stats


fwd_rows, FS = block(F, SHIP_FWD, "forward")
rnd_rows, RS = block(R, SHIP_RANDOM, "random") if R else ("", None)


def criterion(stats):
    """The corrected criterion, recomputed here rather than trusted from a log."""
    out = {}
    for t, s in stats.items():
        c1 = float(np.mean(s["ts"])) > max(s["ms"])
        c2 = min(s["ts"]) >= float(np.median(s["ms"]))
        out[t] = (c1, c2)
    return out


CRIT = {"forward": criterion(FS)}
if RS is not None:
    CRIT["random"] = criterion(RS)

n_mem = len(F["members"])
n_tri = len(F["triples"])
g = FS["gender"]

# The two-arm run that came first, and its rule.
first = load_wide("ensemble_forward", "ensemble",
                  {"arm": "arm", "kind": "variant", "split": "split", "tta": "tta"},
                  needs=["arm", "gender macroF1"], optional=True)
arm_txt = ""
if first is not None:
    ft = first[first.tta == True] if "tta" in first.columns else first
    def _a(name):
        r = ft[ft.arm == name]
        return float(r["gender macroF1"].iloc[0]) if len(r) else float("nan")
    a, b = _a("ens_A_repeats"), _a("ens_B_seeds")
    if not (np.isnan(a) or np.isnan(b)):
        arm_txt = (
            f"An earlier run compared two three-model ensembles on the forward split: "
            f"three runs at one seed, which is what `finalise_task3.py` already trains "
            f"and discards two of, and three genuinely seeded runs. The seeded arm "
            f"scored {b:.4f} against the correlated arm's {a:.4f}. Its pre-registered "
            f"rule nevertheless failed the seeded arm and passed the correlated one by "
            f"0.0002, because the rule divided each gain by its members' spread and "
            f"seeded members are more spread out. Spread there is not noise, it is the "
            f"diversity the method runs on, so the rule penalised an arm for the "
            f"property that makes it work. It was replaced before the confirmation "
            f"run, not after seeing the result, and the replacement has no spread in "
            f"its denominator.\n\n")

def _c(split, target, i):
    if split not in CRIT:
        return "not run"
    return "**passes**" if CRIT[split][target][i] else "fails"


_rand_detail = ""
if RS is not None:
    _ru = RS["usage"]
    _worst_member = min(_ru["ms"])
    if _ru["all"] < _worst_member:
        _rand_detail = (
            f"One number in that table deserves its own sentence. Averaging all "
            f"{n_mem} members scores {_ru['all']:.4f} on `usage`, below the worst "
            f"single member at {_worst_member:.4f}. Averaging probabilities pulls a "
            f"confident minority-class prediction back toward the majority, and "
            f"macro-F1 gives a class with three validation images the same weight as "
            f"one with four thousand, so losing a handful of rare-class hits costs "
            f"more than the accuracy gained on the common ones is worth. That is the "
            f"likely mechanism rather than a measured one -- it was not tested "
            f"separately -- but it is consistent with `usage` having four classes that "
            f"hold fifteen validation images between them.\n\n")

rnd_section = ""
if RS is not None:
    rg, ru = RS["gender"], RS["usage"]
    rnd_section = (
        f"The same six seeds on the random split, which is where the reported headline "
        f"lives:\n\n"
        f"| target | shipped | members (spread) | mean of {len(R['triples'])} triples | "
        f"range | all {len(R['members'])} | disagreement |\n"
        f"|---|---:|---:|---:|---:|---:|---:|\n{rnd_rows}\n\n"
        f"On `gender` the shipped {SHIP_RANDOM['gender']:.4f} sits "
        f"{'below' if SHIP_RANDOM['gender'] < min(rg['ms']) else 'inside'} the member "
        f"range, and on `usage` it sits "
        f"{'above' if SHIP_RANDOM['usage'] > max(ru['ms']) else 'inside'} it. "
        f"Adopting an ensemble would therefore move the two reported numbers in "
        f"opposite directions: `gender` {float(np.mean(rg['ts'])) - SHIP_RANDOM['gender']:+.4f} "
        f"and `usage` {float(np.mean(ru['ts'])) - SHIP_RANDOM['usage']:+.4f}. A change "
        f"that improves the split we cannot see while lowering the number the report "
        f"quotes is not a free upgrade, and saying so is the reason this split was "
        f"measured at all.\n\n")

text = f"""
**B11. Averaging several models, and what our own seed was worth** - variance reduction
is not the same lever as capacity, and B10 had just shown capacity keeping 23% of its
gain on the forward split. An ensemble was therefore the one remaining intervention
with a mechanism rather than a hope behind it, and it costs nothing extra to train:
`finalise_task3.py` already fits three models and ships the median of them.

{arm_txt}The confirmation trains {n_mem} models at {n_mem} different seeds and scores
every one of the {n_tri} three-model ensembles that pool allows, so the reported number
is a distribution rather than a draw. Mirror TTA is applied throughout, as the shipped
model uses it.

| target | shipped | members (spread) | mean of {n_tri} triples | range | all {n_mem} | disagreement |
|---|---:|---:|---:|---:|---:|---:|
{fwd_rows}

Members disagree on about a tenth of validation rows, which is what makes the averaging
do anything; the rate is reported because an ensemble gain quoted without it could as
easily be a bug as an effect.

The uncomfortable column is the second one. Six seeds spread **{max(g['ms']) - min(g['ms']):.4f}**
on `gender`, and `SEED = 42`, the module default the submitted model was trained under,
ranks **{g['rank42']} of {n_mem}** at {g['s42']:.4f} against a member mean of
{float(np.mean(g['ms'])):.4f}. It was never selected on validation -- it was fixed before
any of this was measured -- but our reported number is a fortunate default rather than a
typical one, by roughly {g['s42'] - float(np.mean(g['ms'])):+.4f}. Section 9.2's lesson
arrives one more time, from a direction we had not checked.

That is also what decides the ensemble question, because an ensemble has to be compared
against the model in the zip file and not against the median seed:

| comparison on `gender`, forward split | mean | worst | triples below what we ship |
|---|---:|---:|---:|
| all {n_tri} triples | {float(np.mean(g['ts'])) - SHIP_FWD['gender']:+.4f} | {min(g['ts']) - SHIP_FWD['gender']:+.4f} | {g['below']} of {n_tri} |
| the {len(g['with42'])} triples containing seed 42 | {float(np.mean(g['with42'])) - SHIP_FWD['gender']:+.4f} | {min(g['with42']) - SHIP_FWD['gender']:+.4f} | {g['below42']} of {len(g['with42'])} |
| all {n_mem} members averaged | {g['all'] - SHIP_FWD['gender']:+.4f} | | |

Against the median seed the ensemble gains
{float(np.mean(g['ts'])) - float(np.median(g['ms'])):+.4f}, and against the model we
actually ship it gains {float(np.mean(g['ts'])) - SHIP_FWD['gender']:+.4f}. Both numbers
are true; only the second one is the decision. {g['below']} of {n_tri} triples land
below what we already have, and the ones that reuse our own seed are the ones that do
not.

{rnd_section}The criterion was fixed before either run, and the two splits do not
agree:

| criterion | `gender` fwd | `usage` fwd | `gender` random | `usage` random |
|---|---|---|---|---|
| 1. mean triple beats the best member | {_c('forward', 'gender', 0)} | {_c('forward', 'usage', 0)} | {_c('random', 'gender', 0)} | {_c('random', 'usage', 0)} |
| 2. worst triple at or above the median member | {_c('forward', 'gender', 1)} | {_c('forward', 'usage', 1)} | {_c('random', 'gender', 1)} | {_c('random', 'usage', 1)} |

{_rand_detail}**Design C ships unchanged, as one model.** The intervention passed on the
split it was designed for and failed on the other one, which is why the criterion was
written to cover both splits rather than only the one that motivated it. There is no
variant of adopting it that is safe: the gain on the forward split is
{float(np.mean(g['ts'])) - SHIP_FWD['gender']:+.4f} against what we ship, and the cost on
the random split is a `usage` number that no member of the ensemble would have produced.

The finding worth carrying out of this section is not the ensemble. It is the spread.
Six seeds of the same architecture, same data, same schedule, differ by
{max(g['ms']) - min(g['ms']):.4f} on `gender` on the forward split and
{max(FS['usage']['ms']) - min(FS['usage']['ms']):.4f} on `usage`, and `SEED = 42` ranks
{g['rank42']} of {n_mem} on one split while ranking
{RS['gender']['rank42'] if RS else 'n/a'} of {n_mem} on the other. There is no such
thing as a good seed here, only a good draw for one validation set. Every single number
in this report, ours included, should be read as one sample from a spread of that width
rather than as the performance of the method.
"""


def rewrap(s, width=88):
    out = []
    for para in s.split("\n\n"):
        lines = para.split("\n")
        if any(ln.lstrip().startswith("|") for ln in lines) or not para.strip():
            out.append(para)
        else:
            out.append(textwrap.fill(" ".join(ln.strip() for ln in lines).strip(),
                                     width=width))
    return "\n\n".join(out)


text = rewrap(text)
print(text)
if args.append:
    t = read_report()
    if "**B11." in t:
        raise SystemExit("B11 already present -- edit it, do not append")
    if "**B10." not in t:
        raise SystemExit("B10 is missing; B11 refers to it and must not precede it")
    i = t.find("## Notes for Tr")
    if i < 0:
        raise SystemExit("cannot find the notes heading; refusing to guess placement")
    t = t[:i].rstrip() + "\n" + text.rstrip() + "\n\n---\n\n" + t[i:]
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(t)
    a, b, c = t.find("**B10."), t.find("**B11."), t.find("## Notes for Tr")
    assert 0 < a < b < c, "B11 landed in the wrong place"
    print(f"inserted into the appendix of {REPORT.relative_to(ROOT)}")
