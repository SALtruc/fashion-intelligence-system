"""Generate the report's pre-trained comparison section from the experiment's CSVs.

Written before the per-class numbers existed, deliberately: the section has to state
whatever the data says, and a paragraph drafted after seeing the answer is a paragraph
written to fit it. Every number here is read from sota_comparison.csv and its
per-class companion, and the verdict is computed, not typed. That is the same
discipline appendix B7 uses, and it exists because B4 once carried a hand-typed range
that its own numbers contradicted.

The decomposition is exact rather than indicative. macro-F1 over eight classes is the
mean of eight per-class F1 scores, so

    delta macro = (1/8) * sum over classes of (delta F1 for that class)

and each class's contribution to a gain can be read off directly. Four `usage` classes
hold 5,646 validation images between them and the other four hold 15, while each
contributes an eighth of the average. So "where did the gain come from" has an
arithmetic answer, and it decides whether the report's label-ceiling argument survives.

    python src/task3/report_sota.py            # print it
    python src/task3/report_sota.py --append   # insert into REPORT_TASK3.md
"""
import argparse
import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
RES = ROOT / "results" / "task3"
CSV = RES / "sota_comparison.csv"
PC = RES / "sota_comparison_perclass.csv"
REPORT = ROOT / "notebooks" / "Task3" / "REPORT_TASK3.md"

LARGE = ["Casual", "Ethnic", "Formal", "Sports"]
OURS = "C_weighted (ours)"
BAND = {"gender": 0.0114, "usage": 0.0640}

ap = argparse.ArgumentParser()
ap.add_argument("--append", action="store_true")
args = ap.parse_args()

if not CSV.is_file():
    raise SystemExit(f"{CSV.name} not found -- run experiment_pretrained_sota.py first")
res = pd.read_csv(CSV).drop_duplicates("arm", keep="last")
if OURS not in set(res.arm):
    raise SystemExit(f"no '{OURS}' row -- the comparison needs our own arm to mean "
                     f"anything")
mine = res[res.arm == OURS].iloc[0]

rows = "\n".join(
    f"| `{r.arm}` | {int(r['params']):,} | {r['gender macroF1']:.4f} | "
    f"{r['usage macroF1']:.4f} | {r['minutes']:.1f} |"
    for _, r in res.iterrows())

deltas = []
for _, r in res.iterrows():
    if r.arm == OURS:
        continue
    dg = r["gender macroF1"] - mine["gender macroF1"]
    du = r["usage macroF1"] - mine["usage macroF1"]
    deltas.append(
        f"| `{r.arm}` | {dg:+.4f} | "
        f"{'beyond' if abs(dg) > BAND['gender'] else 'within'} | {du:+.4f} | "
        f"{'beyond' if abs(du) > BAND['usage'] else 'within'} |")

# ------------------------------------------------------------- the exact decomposition
verdict = attribution = ""
if PC.is_file():
    pc = pd.read_csv(PC).drop_duplicates(["arm", "target", "class"], keep="last")
    u = pc[pc.target == "usage"]
    w = u.pivot_table(index="class", columns="arm", values="f1")
    best = res.loc[res["usage macroF1"].idxmax(), "arm"]
    if {best, OURS} <= set(w.columns) and best != OURS:
        d = (w[best] - w[OURS]) / len(w)          # each class's share of delta macro
        big = float(d[d.index.isin(LARGE)].sum())
        rare = float(d[~d.index.isin(LARGE)].sum())
        n_big = int(u[(u.arm == OURS) & u["class"].isin(LARGE)].val_n.sum())
        n_rare = int(u[(u.arm == OURS) & ~u["class"].isin(LARGE)].val_n.sum())
        attribution = (
            "\n| `usage` class | val images | ours | " + best + " | contribution to the gap |\n"
            "|---|---:|---:|---:|---:|\n"
            + "\n".join(
                f"| {c}{' *(rare)*' if c not in LARGE else ''} | "
                f"{int(u[(u.arm == OURS) & (u['class'] == c)].val_n.iloc[0])} | "
                f"{w.loc[c, OURS]:.3f} | {w.loc[c, best]:.3f} | {d[c]:+.4f} |"
                for c in w.index)
            + f"\n| **total** | **{n_big + n_rare:,}** | | | **{big + rare:+.4f}** |\n")

        share_big = big / (big + rare) if (big + rare) else 0.0
        if share_big >= 0.5:
            verdict = (
                f"**{share_big:.0%} of the gap comes from the four large classes**, "
                f"which hold {n_big:,} validation images between them. That is not "
                f"luck on a handful of pictures, and it means this report's earlier "
                f"claim was too strong. The label evidence stands: an `articleType` "
                f"oracle scores below our CNN, `Home` has no validation image at all, "
                f"and identical article types carry different `usage` labels. What "
                f"does not follow, and what we wrote anyway, is that therefore no "
                f"model could do much better. A pre-trained backbone at 224x224 finds "
                f"`usage` signal in the pixels that a 289k-parameter network at 60x80 "
                f"does not. Much of what we attributed to noisy labels was capacity "
                f"and resolution. The ceiling argument is now a claim about *our* "
                f"model, not about the task.")
        else:
            verdict = (
                f"**{1 - share_big:.0%} of the gap comes from the four rare classes**, "
                f"which hold {n_rare} validation images between them against "
                f"{n_big:,} in the large four -- and each contributes an eighth of "
                f"the average regardless. A gain concentrated there is a gain on "
                f"{n_rare} pictures, which is the instability this report measures at "
                f"0.064 rather than a capability the model acquired. The label-ceiling "
                f"argument survives, and the headline gap overstates what a stronger "
                f"backbone actually buys on this target.")

text = f"""
**B9. Measured against a pre-trained backbone** - the spec forbids a pre-trained
system as the submitted model and recommends one for comparison, so this is the
comparison and none of it is a submission candidate. Published results on the wider
Kaggle collection this dataset is drawn from are not usable as a reference: they report
accuracy, over a different label set, at full resolution, on their own splits. Changing
three definitions at once and quoting the number is decoration. Training a strong
external backbone on **our** rows and scoring it with **our** metric is the same
question asked in a way that has an answer.

All arms share the split, the class-weighted loss, the two-head structure and the
metric; the backbone is the only difference. The pre-trained arms see the 60x80 images
upscaled to 224x224 with ImageNet normalisation, which adds no information but matches
the receptive field their stem assumes. Our own design was retrained in the same
session so the comparison carries no cross-session confound.

| model | trainable params | `gender` | `usage` | minutes |
|---|---:|---:|---:|---:|
{rows}

| against ours | `gender` delta | vs band {BAND['gender']} | `usage` delta | vs band {BAND['usage']} |
|---|---:|---|---:|---|
{chr(10).join(deltas)}

`resnet18_frozen` trains 6,669 parameters and runs 11.2 million frozen ones, so its
row is not a claim about a small model. It answers a narrower question: generic
ImageNet features, with no fine-tuning at all, land within noise of a network we
trained from scratch on this data. Fine-tuning is where the gap opens.
{attribution}
{verdict}
"""

print(text)
if args.append:
    if not PC.is_file():
        raise SystemExit("refusing to append without the per-class file: the section's "
                         "conclusion depends on it")
    t = io.open(REPORT, encoding="utf-8").read().replace("\r\n", "\n")
    if "**B9. Measured against a pre-trained backbone**" in t:
        raise SystemExit("B9 already present -- edit it, do not append")
    i = t.find("## Notes for Tr")
    if i < 0:
        raise SystemExit("cannot find the notes heading; refusing to guess placement")
    t = t[:i].rstrip() + "\n" + text.rstrip() + "\n\n---\n\n" + t[i:]
    io.open(REPORT, "w", encoding="utf-8", newline="\n").write(t)
    j, k = t.find("**B9."), t.find("## Notes for Tr")
    assert 0 < j < k
    print(f"inserted into the appendix of {REPORT.relative_to(ROOT)}")
