#!/usr/bin/env python
"""Fill the missing `season` and `usage` columns on the external cosmetics crops.

Why this is label propagation and not invention
-----------------------------------------------
`prepare_external_cosmetics.py` wrote `gender`, `masterCategory`, `subCategory` and
`articleType` but left `season` and `usage` out, because the source archive carries
neither. That made the 1,200 crops useless for Task 2 and for Task 3's usage head.

They do not have to stay useless. In the PROVIDED data these two labels are not
visual properties of a cosmetic at all - they are a catalogue convention, and the
convention is close to deterministic:

    every one of the 12 cosmetics articleTypes is 100% season=Spring
    masterCategory 'Personal Care' is 99.4% Spring (710 of 714 train rows)
    'Personal Care' is 96.6% usage=Casual
    per class, usage is 83-100% Casual wherever it is not null

So the honest move is to copy the convention across, per articleType, and say so.
This is inference from the provided labels, not a guess about the photographs.

What this changes, stated plainly for the report
------------------------------------------------
Spring is the rarest season: 1,093 of 27,596 train rows (4.0%). Tagging these 1,200
crops as Spring takes it to 2,293 - it slightly more than DOUBLES the class.

That is a real gain and a real risk in the same move. Spring is already 65% Personal
Care; afterwards it is 83.3%. The model is being taught, harder, that a cosmetic
implies Spring.

That is defensible because it is the actual rule generating the label - but it means
any Task 2 improvement must be reported honestly: it comes from cosmetics, not from
the model learning seasonality. The 383 Spring rows that are NOT Personal Care
(Casual Shoes 77, Sports Shoes 60, Tshirts 45, ...) get no help at all.

The check worth running: measure Spring recall separately on Personal Care and on
non-Personal-Care validation rows. If only the former improves, the gain is the
categorical shortcut and should be described as such.

Usage
-----
    python add_cosmetics_season_usage.py            # writes the two columns
    python add_cosmetics_season_usage.py --dry-run  # show what it would do
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# Written where the loader looks, rather than at a path that happened to work on
# one laptop. Override with A2_EXTERNAL_DATA.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_data import default_data_root  # noqa: E402

CSV = default_data_root() / "ExternalCosmetics" / "external_cosmetics.csv"
PROVIDED = Path("A2_FashionDataset/FashionDataset/train/styles_train.csv")

# The propagated values, with the evidence for each. Derived from the provided train
# split, not chosen by hand - see the module docstring.
RULE = {
    "Eyeshadow":   ("Spring", "Casual"),
    "Lipstick":    ("Spring", "Casual"),
    "Nail Polish": ("Spring", "Casual"),
}


def evidence(provided: pd.DataFrame) -> None:
    """Print the provided-data support for the rule, so it is never taken on trust."""
    print("Support in the PROVIDED training data:\n")
    print(f"  {'articleType':16} {'n':>4}  {'season':>18}  {'usage':>18}")
    for at in RULE:
        s = provided[provided["articleType"] == at]
        if s.empty:
            print(f"  {at:16} {0:>4}  (absent)")
            continue
        se = s["season"].value_counts(dropna=True)
        us = s["usage"].value_counts(dropna=True)
        se_txt = f"{se.index[0]} {100*se.iloc[0]/se.sum():.0f}%" if len(se) else "-"
        us_txt = f"{us.index[0]} {100*us.iloc[0]/us.sum():.0f}%" if len(us) else "-"
        print(f"  {at:16} {len(s):>4}  {se_txt:>18}  {us_txt:>18}")

    pc = provided[provided["masterCategory"] == "Personal Care"]
    if not pc.empty:
        se = pc["season"].value_counts()
        us = pc["usage"].value_counts()
        print(f"\n  masterCategory 'Personal Care' (n={len(pc)}):")
        print(f"    season {se.index[0]} {100*se.iloc[0]/se.sum():.1f}%"
              f"   usage {us.index[0]} {100*us.iloc[0]/us.sum():.1f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not CSV.exists():
        sys.exit(f"not found: {CSV}")
    df = pd.read_csv(CSV)
    print(f"{CSV}\n  {len(df)} rows, columns: {list(df.columns)}\n")

    if PROVIDED.exists():
        evidence(pd.read_csv(PROVIDED))
    else:
        print(f"  (skipping evidence table: {PROVIDED} not found)")

    unknown = sorted(set(df["articleType"]) - set(RULE))
    if unknown:
        sys.exit(f"\nno propagation rule for: {unknown} - refusing to guess.")

    df["season"] = df["articleType"].map(lambda a: RULE[a][0])
    df["usage"] = df["articleType"].map(lambda a: RULE[a][1])
    # Record HOW these two came about, so nobody later mistakes them for source data.
    df["label_source"] = "season+usage propagated per articleType from provided train"

    cols = ["id", "gender", "masterCategory", "subCategory", "articleType",
            "season", "usage", "source", "label_source"]
    df = df[[c for c in cols if c in df.columns]]

    print("\nWould write:" if a.dry_run else "\nWriting:")
    for at in sorted(RULE):
        n = int((df["articleType"] == at).sum())
        print(f"  {at:16} {n:>5} rows -> season={RULE[at][0]}, usage={RULE[at][1]}")

    if a.dry_run:
        print("\n--dry-run: nothing written")
        return

    backup = CSV.with_suffix(".csv.bak")
    if not backup.exists():
        shutil.copy2(CSV, backup)
        print(f"\n  backup -> {backup}")
    df.to_csv(CSV, index=False)
    print(f"  wrote {CSV}  ({len(df)} rows, {len(df.columns)} columns)")
    print("\n  NOTE: train_task1_condition_resnet.py selects an explicit column list")
    print("  when it concatenates this file. Adding columns does not break it, but a")
    print("  Task 2 / Task 3-usage run must be updated to read the new ones.")


if __name__ == "__main__":
    main()
