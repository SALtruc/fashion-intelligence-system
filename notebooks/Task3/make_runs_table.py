"""Generate the RUNS literal in section 9.2 from the runs' own result CSVs.

Transcribing four runs x ten configurations x two targets by hand is 80 numbers, and
a single slip there would quietly corrupt the reproducibility analysis that the rest
of section 9.3 now rests on. This reads the CSVs and prints the block to paste, so the
only manual step is choosing which files are runs 1..N.

    python make_runs_table.py run1.csv run2.csv run3.csv task3_results.csv

Each CSV is a task3_results.csv from a completed run. Order matters: oldest first.
"""
import sys
from pathlib import Path

import pandas as pd

if len(sys.argv) < 2:
    raise SystemExit(__doc__)

paths = [Path(p) for p in sys.argv[1:]]
for p in paths:
    if not p.exists():
        raise SystemExit(f"missing: {p}")

# The order the notebook lists them in, so a diff of the generated block is readable.
ORDER = ["A two models", "B joint label", "C shared body", "C weighted",
         "C with external", "C + logit adj", "C + logit adj + TTA",
         "D articleType-pretrained", "majority", "1-NN pixels"]
TARGETS = ["gender", "usage"]

runs = []
for p in paths:
    d = pd.read_csv(p)
    d = d[d["split"] == "shared val"].drop_duplicates(["model", "target"])
    runs.append(d.set_index(["model", "target"])["macro_f1"].to_dict())

width = max(len(m) for m in ORDER) + 2
print(f"# {len(runs)} runs, read from: " + ", ".join(p.name for p in paths))
print("RUNS = {")
for m in ORDER:
    cells = []
    for t in TARGETS:
        vals = [r.get((m, t)) for r in runs]
        inner = ", ".join("None  " if v is None else f"{v:.4f}" for v in vals)
        cells.append(f'"{t}": [{inner}]')
    print(f'    {(chr(34) + m + chr(34) + ":"):<{width + 1}} ' + "{" + ", ".join(cells) + "},")
print("}")

missing = [(m, t) for m in ORDER for t in TARGETS
           if all(r.get((m, t)) is None for r in runs)]
if missing:
    print("\n# configurations absent from every run (drop them from ORDER?):", missing)
