"""Rebuild results/task3/task3_all_results.csv as one tidy table, and fix two defects.

Defect 1, the one that matters for the submission. .gitignore keeps
`results/task3/*.csv` out of git and admits only this consolidated file, which was the
right call when it was made -- ten CSVs in a notebook folder is sprawl -- but the file
was last built before the pre-trained comparison, the scratch-vs-pretrained experiment
and the two ensemble runs existed. So appendices B9 and B10 are generated from CSVs that
live on one laptop. A fresh checkout gets "sota_comparison.csv not found", and the spec
requires that the assignment run from the code in the zip. Every experiment CSV present
on disk is folded in here, so the one tracked file carries the data behind every number
the report quotes.

Defect 2. In the previous build the `run` column packed a composite key into a single
string: "base 42", "lr=0.0003 42", "C shared body forward split", "A gender 1",
"20.0 45.0". 1,226 of 1,326 rows could not be filtered by their own key -- reading the
forward-split reference for B10 needed a whitespace split to get at an epoch count. The
measurements were never wrong, only unusable.

The unpacking rule is deliberately the most conservative one available: nothing moves
between existing columns. Where `run` begins with the row's own `arm` value, the
remainder is split off into new columns; where `arm` is empty, `run` is split on
whitespace and each token assigned by an explicit per-source rule. Any pattern not on
the allowlist raises rather than being guessed at. Then, for every row, the parsed
pieces are rejoined in the documented order and compared to the original `run` string
character for character -- if the rebuild does not match, nothing is written.

Text columns (`note`) are dropped rather than stored in `value`. A `value` column that
sometimes holds prose is what made figure 4 plot index positions instead of numbers.

    python src/task3/consolidate_results.py --check   # verify, write nothing
    python src/task3/consolidate_results.py           # rebuild the file
"""
import argparse
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
OUT = RES / "task3_all_results.csv"
COLS = ["source", "arm", "variant", "rep", "split", "tta",
        "target", "class", "metric", "value"]
TARGETS = ("gender", "usage", "pair")

# Every source name produced by the melt calls further down. Kept here so the guard
# above can tell "a source I rebuild" from "a source only this file still holds".
MELTED = ("pretrained_comparison", "pretrained_comparison_perclass",
          "scratch_vs_pretrained", "scratch_vs_pretrained_perclass",
          "ensemble", "ensemble_perclass", "ensemble_confirm",
          "gender_ceiling_diagnostic", "gender_confusion",
          "ensemble_diversity")

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true", help="verify only, write nothing")
args = ap.parse_args()


# --------------------------------------------------------------- unpack the old rows
# One rule per source. `rest` is what remains of `run` after its own `arm` prefix is
# removed; `tokens` is `run` split on whitespace when `arm` is empty. A rule returns
# the fields to fill, and the rejoin check below proves it lost nothing.
def unpack(source, run, arm):
    run, arm = (run or "").strip(), (arm or "").strip()
    if not run:
        return {}
    if arm and run == arm:
        return {}
    if arm and run.startswith(arm + " "):
        rest = run[len(arm) + 1:]
        if source in ("external_catalog", "external_party", "hyperparam_sweep",
                      "transfer_weighted", "transfer_weighted_perclass"):
            if not rest.isdigit():
                raise SystemExit(f"{source}: expected a seed or repeat, got {rest!r}")
            return {"rep": rest}
        if source == "model_comparison":
            # Four evaluation protocols, not four splits, and stored verbatim: these
            # are single labels that happen to contain a space, not composite keys.
            if rest not in ("forward split", "per-target split", "shared val",
                            "independent eval"):
                raise SystemExit(f"{source}: unknown protocol {rest!r}")
            return {"split": rest}
        raise SystemExit(f"no rule for {source} with remainder {rest!r}")
    if not arm:
        tok = run.split()
        if source == "epochs_confirmation":                 # "<epochs> <repeat>"
            if len(tok) != 2:
                raise SystemExit(f"{source}: expected 2 tokens, got {tok}")
            return {"variant": f"epochs={tok[0]}", "rep": tok[1]}
        if source == "training_history":                 # "<design> <variant> <epoch>"
            # The middle token is a variant label, not a target: alongside gender,
            # usage and pair it also takes external, forward, multi, weighted and
            # transfer. Filing it under `target` would have been wrong for five of
            # the eight configurations.
            if len(tok) != 3 or not tok[2].isdigit():
                raise SystemExit(f"{source}: expected '<design> <variant> <epoch>', "
                                 f"got {tok}")
            return {"arm": tok[0], "variant": tok[1], "rep": tok[2]}
        raise SystemExit(f"no rule for {source} with run {run!r} and no arm")
    raise SystemExit(f"{source}: run {run!r} does not start with arm {arm!r}")


def rebuild(source, row, arm_original):
    """Reconstruct the original `run` string from the parsed fields."""
    if source == "epochs_confirmation":
        return f"{row['variant'].split('=')[1]} {row['rep']}"
    if source == "training_history":
        return f"{row['arm']} {row['variant']} {row['rep']}"
    if source == "model_comparison":
        return f"{arm_original} {row['split']}" if row["split"] else arm_original
    if row["rep"]:
        return f"{arm_original} {row['rep']}"
    return arm_original


# Sources whose own CSV no longer exists on disk: they were consolidated and the
# per-experiment files removed, so this file is their only copy and its rows are
# carried forward. Everything else is re-derived from the CSVs below, which is what
# makes a second run safe -- an earlier version of this script read its own output as
# input, so re-running it would have doubled every melted row and then failed on the
# missing `run` column.
LEGACY = ("epochs_confirmation", "external_catalog", "external_party",
          "gender_ceiling", "hyperparam_summary", "hyperparam_sweep",
          "model_comparison", "training_history", "transfer_weighted",
          "transfer_weighted_perclass")

old = pd.read_csv(OUT, dtype=str).fillna("")
print(f"read {OUT.name}: {len(old):,} rows, sources {old.source.nunique()}")
unknown = sorted(set(old.source) - set(LEGACY) - set(MELTED))
if unknown:
    raise SystemExit(f"unrecognised sources in {OUT.name}: {unknown}. Add them to "
                     f"LEGACY if their CSV is gone, or to the melt list if it is not.")
dropped = len(old) - int(old.source.isin(LEGACY).sum())
old = old[old.source.isin(LEGACY)]
if "run" not in old.columns:
    old = old.assign(run="")
    print(f"  input already in the new schema; {len(old):,} legacy rows kept verbatim")
elif dropped:
    print(f"  {dropped:,} re-derivable rows dropped, to be rebuilt from their CSVs")
rows, mismatched = [], 0
for _, r in old.iterrows():
    out = {c: "" for c in COLS}
    # Every schema column the input actually has, not a fixed six. Copying only
    # source/arm/target/class/metric/value silently blanked `variant`, `rep` and
    # `split` for all 1,326 legacy rows on a second run -- the columns this script
    # exists to populate -- and the damage was invisible until a generator went
    # looking for the epoch count and found nothing.
    out.update({c: r[c] for c in COLS if c in old.columns})
    out.update(unpack(r["source"], r["run"], r["arm"]))
    if (r["run"] or "").strip() and             rebuild(r["source"], out, r["arm"].strip()) != (r["run"] or "").strip():
        mismatched += 1
        if mismatched <= 5:
            print(f"  MISMATCH {r['source']}: {r['run']!r} != "
                  f"{rebuild(r['source'], out, r['arm'].strip())!r}")
    rows.append(out)
if mismatched:
    raise SystemExit(f"{mismatched} rows did not rebuild; refusing to write")
print(f"  unpacked and rebuilt {len(rows):,}/{len(rows):,} rows exactly")


# ------------------------------------------------------- melt the per-experiment CSVs
def melt(name, source, keys, skip=()):
    """Long-form rows from a wide CSV.

    A column named "<target> <metric>" becomes target + metric; anything else numeric
    becomes a metric with no target. `keys` maps the file's own columns onto the
    schema's, so nothing is renamed by guesswork.
    """
    f = RES / f"{name}.csv"
    if not f.is_file():
        print(f"  {name}.csv absent, skipped")
        return []
    d = pd.read_csv(f)
    out = []
    valuecols = [c for c in d.columns if c not in keys and c not in skip]
    for _, r in d.iterrows():
        base = {c: "" for c in COLS}
        base["source"] = source
        for src_col, dst in keys.items():
            base[dst] = "" if pd.isna(r[src_col]) else str(r[src_col])
        for c in valuecols:
            if pd.isna(r[c]):
                continue
            row = dict(base)
            first = c.split(" ")[0]
            if first in TARGETS and " " in c:
                row["target"], row["metric"] = first, c.split(" ", 1)[1]
            else:
                row["metric"] = c
            row["value"] = str(r[c])
            out.append(row)
    print(f"  {name}.csv -> {len(out):,} rows")
    return out


new = []
new += melt("sota_comparison", "pretrained_comparison",
            {"arm": "arm"}, skip=("note",))
new += melt("sota_comparison_perclass", "pretrained_comparison_perclass",
            {"arm": "arm", "target": "target", "class": "class"})
new += melt("scratch_vs_pretrained", "scratch_vs_pretrained",
            {"arm": "arm", "split": "split", "tta": "tta"})
new += melt("scratch_vs_pretrained_perclass", "scratch_vs_pretrained_perclass",
            {"arm": "arm", "split": "split", "tta": "tta",
             "target": "target", "class": "class"})
new += melt("ensemble_forward", "ensemble",
            {"arm": "arm", "kind": "variant", "split": "split", "tta": "tta"},
            skip=("note",))
new += melt("ensemble_forward_perclass", "ensemble_perclass",
            {"arm": "arm", "kind": "variant", "tta": "tta",
             "target": "target", "class": "class"})
for sp in ("forward", "random"):
    new += melt(f"ensemble_confirm_{sp}", "ensemble_confirm",
                {"id": "arm", "kind": "variant", "split": "split"})
new += melt("diagnose_gender_ceiling", "gender_ceiling_diagnostic",
            {"target": "target", "class": "class"})

# Already long-form, so it is copied across rather than melted: the member pair goes
# into `rep`, which is where a repeat identifier belongs.
f = RES / "ensemble_diversity.csv"
if f.is_file():
    d = pd.read_csv(f)
    n = 0
    for _, r in d.iterrows():
        row = {k: "" for k in COLS}
        row.update({"source": "ensemble_diversity", "split": str(r["split"]),
                    "target": str(r["target"]), "rep": str(r["pair"]),
                    "metric": str(r["metric"]), "value": str(r["value"])})
        new.append(row)
        n += 1
    print(f"  ensemble_diversity.csv -> {n:,} rows")

# The confusion matrix is the one file that is not one row per measurement: its columns
# are predicted classes. Encoded as metric="predicted_<class>" so the truth row stays
# in `class` and nothing is lost.
f = RES / "gender_confusion.csv"
if f.is_file():
    d = pd.read_csv(f)
    n = 0
    for _, r in d.iterrows():
        for c in d.columns:
            if c == "truth":
                continue
            row = {k: "" for k in COLS}
            row.update({"source": "gender_confusion", "target": "gender",
                        "class": str(r["truth"]), "metric": f"predicted_{c}",
                        "value": str(r[c])})
            new.append(row)
            n += 1
    print(f"  gender_confusion.csv -> {n:,} rows")

allrows = pd.DataFrame(rows + new, columns=COLS)
print(f"\ntotal {len(allrows):,} rows, {allrows.source.nunique()} sources")
# `arm` and `split` legitimately hold multi-word labels ("C shared body",
# "independent eval"). `variant` and `rep` must not: a space there would mean a
# composite key survived the unpacking, which is the defect being fixed.
for c in ("variant", "rep"):
    bad = allrows[allrows[c].astype(str).str.contains(" ", na=False)]
    if len(bad):
        raise SystemExit(f"column {c} still packs a key in {len(bad)} rows: "
                         f"{sorted(set(bad[c]))[:5]}")
# Self-check: the unpacked columns must actually be populated. Without this the
# corruption above is a blank column nobody notices until a report fails to build.
EXPECT = {"epochs_confirmation": ("variant", "rep"),
          "training_history": ("arm", "variant", "rep"),
          "model_comparison": ("split",),
          "transfer_weighted": ("rep",),
          "hyperparam_sweep": ("rep",)}
for src, cols in EXPECT.items():
    part = allrows[allrows.source == src]
    if part.empty:
        raise SystemExit(f"source {src} vanished from the rebuild")
    for c in cols:
        if (part[c].astype(str) == "").any():
            raise SystemExit(f"{src}: column {c} is blank in "
                             f"{int((part[c].astype(str) == '').sum())} of "
                             f"{len(part)} rows -- the unpacking lost data")
print(f"self-check: {len(EXPECT)} legacy sources keep their unpacked keys")

nonnum = allrows[pd.to_numeric(allrows.value, errors="coerce").isna()]
print(f"non-numeric values: {len(nonnum)}"
      + (f" -- {sorted(set(nonnum.metric))[:6]}" if len(nonnum) else ""))

if args.check:
    print("\n--check given, nothing written")
else:
    allrows.to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print(allrows.groupby("source").size().to_string())
