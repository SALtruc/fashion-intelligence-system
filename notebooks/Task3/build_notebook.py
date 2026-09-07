"""Generate 03_task3_gender_usage_nguyen.ipynb from task3_build.py.

The .py is the source of truth; the .ipynb is a build artefact. Keeping it that way is
what makes the notebook reviewable -- a diff of the .py is readable, a diff of a .ipynb
is not.

Outputs are transplanted from a donor notebook (a completed Colab run) by matching the
EXACT text of each code cell. A cell whose code changed deliberately loses its stale
output, because showing an output that the current code did not produce is worse than
showing none.

    python build_notebook.py                       # rebuild, keep outputs from the .ipynb
    python build_notebook.py path/to/donor.ipynb   # take outputs from a fresh Colab run

Every run stamps BUILD with a hash of the source, so a stale upload is caught by the
notebook's own first cell instead of seventy minutes later.
"""
import hashlib
import io
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

def _find(name, *rel):
    """Work in both layouts: everything in one folder (local), or the repo's
    notebooks/Task3 + src split."""
    for c in (HERE / name, *(HERE.joinpath(*r) / name for r in rel)):
        if c.exists():
            return c
    raise SystemExit(f"cannot find {name} near {HERE}")

SRC = _find("task3_build.py", ("..", "..", "src"))
NB = _find("03_task3_gender_usage_nguyen.ipynb", ("..", "notebooks", "Task3"))
DONOR = Path(sys.argv[1]) if len(sys.argv) > 1 else NB

PLACEHOLDER = 'BUILD = "dev"'


def split_cells(text):
    """Split the source on '# %%' markers into (kind, source) pairs."""
    parts = re.split(r"(?m)^# %%(.*)$", text)
    if parts[0].strip():
        raise SystemExit("source has content before the first '# %%' marker")
    cells = []
    for k in range(1, len(parts), 2):
        kind = "markdown" if "markdown" in parts[k] else "code"
        body = parts[k + 1].strip("\n")
        if kind == "markdown":
            body = "\n".join(re.sub(r"^# ?", "", l) for l in body.split("\n"))
        cells.append((kind, body))
    return cells


src = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
if PLACEHOLDER not in src:
    raise SystemExit(f"{SRC.name} is missing the {PLACEHOLDER} line")

# Hash the source with the stamp still a placeholder, so the hash is a function of the
# real content and does not depend on itself.
build = hashlib.sha256(src.encode()).hexdigest()[:8]
stamped = src.replace(PLACEHOLDER, f'BUILD = "{build}"')

cells = split_cells(stamped)

donor_outputs = {}
if DONOR.exists():
    donor = json.loads(io.open(DONOR, encoding="utf-8").read())
    for c in donor["cells"]:
        if c["cell_type"] != "code":
            continue
        key = "".join(c["source"]).strip("\n")
        # Ignore the stamp line when matching, or every rebuild would drop every output.
        key = re.sub(r'BUILD = "[0-9a-f]{8}"', PLACEHOLDER, key)
        if c.get("outputs"):
            donor_outputs[key] = (c["outputs"], c.get("execution_count"))

out, kept, dropped = [], 0, []
for i, (kind, body) in enumerate(cells):
    if kind == "markdown":
        out.append({"cell_type": "markdown", "metadata": {},
                    "source": body.splitlines(keepends=True)})
        continue
    cell = {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": body.splitlines(keepends=True)}
    key = re.sub(r'BUILD = "[0-9a-f]{8}"', PLACEHOLDER, body)
    if key in donor_outputs:
        cell["outputs"], cell["execution_count"] = donor_outputs[key]
        kept += 1
    elif body.strip():
        dropped.append(i)
    out.append(cell)

nb = {"cells": out,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"},
                   "colab": {"provenance": []}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 0}

io.open(NB, "w", encoding="utf-8", newline="\n").write(
    json.dumps(nb, indent=1, ensure_ascii=False) + "\n")

n_code = sum(1 for k, _ in cells if k == "code")
print(f"wrote {NB.name}: {len(cells)} cells ({n_code} code), {NB.stat().st_size / 1024:.0f} KB")
print(f"BUILD = {build}   <- the notebook's first cell must print this")
print(f"outputs transplanted: {kept}/{n_code}")
if dropped:
    print(f"no output for code cells {dropped} (code changed since the donor run, "
          "or never run)")
