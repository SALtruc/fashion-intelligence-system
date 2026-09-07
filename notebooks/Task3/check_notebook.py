"""Gate the shipped .ipynb before telling anyone to upload it.

Exists because a previous handover shipped a notebook that was a regeneration behind
its source: the table of contents still showed the old section names, and the Colab run
failed ten minutes in. Eyeballing did not catch it; this does.

Two of these checks were added after the second full run, each for a defect that had
already cost a run:

  * "notebook code == source code" -- the notebook that was actually executed was
    missing the Drive-persistence fix, so 70 minutes of results were written to
    /content instead of Drive. The .ipynb on disk had the fix; the uploaded copy did
    not, and nothing compared them.
  * "BUILD is stamped" -- the fingerprint that lets the person running it see, in the
    first cell, whether they uploaded the current file.

    python check_notebook.py                  # gate the notebook against its source
    python check_notebook.py ran_copy.py      # also diff against code that was executed
"""
import ast
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

NB = _find("03_task3_gender_usage_nguyen.ipynb", ("..", "notebooks", "Task3"))
SRC = _find("task3_build.py", ("..", "..", "src"))
RAN = Path(sys.argv[1]) if len(sys.argv) > 1 else None

nb = json.loads(io.open(NB, encoding="utf-8").read())
code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
code = "\n".join("".join(c["source"]) for c in code_cells)

fails = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  -- ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


print(f"checking {NB.name}  ({len(nb['cells'])} cells, {NB.stat().st_size / 1024:.0f} KB)\n")

try:
    ast.parse(code)
    check("every code cell parses", True)
except SyntaxError as e:
    check("every code cell parses", False, str(e))

headings = re.findall(r"^#{2,3} (.+)$", md, re.M)
want = ["0.1 Staging the data", "0.2 Finding the data", "0.3 The frame",
        "1.3 \"macro-F1\" is two different numbers",
        "10.5 Why the improvements are small"]
for w in want:
    check(f"section present: {w}", any(h.startswith(w) for h in headings))

check("/content in the catalogue search", '"/content",' in code)
check("unzip uses -o (a re-run must not hang on a prompt)", '"-qo"' in code)
check("file-count assertion after unpack", "images missing" in code)
check("both macro-F1 conventions reported", "macro_f1_in_val" in code)
check("QUICK ships as False", re.search(r"^QUICK = False$", code, re.M) is not None)

stamp = re.search(r'BUILD = "([0-9a-f]{8}|dev)"', code)
check("BUILD is stamped, not 'dev'",
      stamp is not None and stamp.group(1) != "dev",
      "run build_notebook.py" if not stamp or stamp.group(1) == "dev"
      else f"BUILD = {stamp.group(1)}")

# The check that matters most: the notebook must be this source, not an older build.
src = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
parts = re.split(r"(?m)^# %%(.*)$", src)
src_code = [parts[k + 1].strip("\n") for k in range(1, len(parts), 2)
            if "markdown" not in parts[k]]
nb_code = [re.sub(r'BUILD = "[0-9a-f]{8}"', 'BUILD = "dev"', "".join(c["source"]).strip("\n"))
           for c in code_cells]
if len(src_code) != len(nb_code):
    check("notebook code == source code", False,
          f"{len(src_code)} code cells in source, {len(nb_code)} in notebook")
else:
    bad = [i for i, (a, b) in enumerate(zip(src_code, nb_code)) if a != b]
    check("notebook code == source code", not bad,
          "" if not bad else f"cells differ: {bad} -- rebuild with build_notebook.py")

n_out = sum(1 for c in code_cells if c.get("outputs"))
print(f"\n  note: {n_out}/{len(code_cells)} code cells carry output"
      f"{' -- a full re-run is needed before submitting' if n_out < len(code_cells) else ''}")

if RAN and RAN.exists():
    keep, lines = True, []
    for line in io.open(RAN, encoding="utf-8").read().split("\n"):
        if line.startswith("# %%"):
            keep = "[markdown]" not in line
            continue
        if keep:
            lines.append(line)
    ran = "\n".join(lines)
    for a, b in [("QUICK = True", "QUICK = False"),
                 ("EPOCHS = 2 if QUICK", "EPOCHS = 6 if QUICK"),
                 ("SAMPLE = 3000 if QUICK", "SAMPLE = 4000 if QUICK")]:
        ran = ran.replace(a, b)
    norm = lambda s: re.sub(r"\n{2,}", "\n", s).strip()
    check("notebook == the code that was executed", norm(code) == norm(ran))

print()
if fails:
    print(f"{len(fails)} CHECK(S) FAILED -- do not ship")
    sys.exit(1)
print("ALL CHECKS PASSED -- safe to upload")
