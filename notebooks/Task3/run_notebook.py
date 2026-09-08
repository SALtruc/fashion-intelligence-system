"""Execute the notebook headlessly on a local GPU, and save it with its outputs.

Written when Colab quota ran out. This machine has an RTX 4070 Laptop, which is
faster than the T4 the earlier runs used, and the whole dataset is already staged at
D:/ColabDataset -- so a local run needs nothing from Drive.

It replaces "upload to Colab, click Run all, hope the tab survives" with one command
that cannot be run against a stale file: the notebook is rebuilt from task3_build.py
first, so what executes is always the current source.

    python run_notebook.py                 # full run, writes outputs into the .ipynb
    python run_notebook.py --quick         # ~2 min sanity pass, writes to a scratch copy
    python run_notebook.py --kernel NAME   # default a2torch, see --help

Register the kernel once, bound to the interpreter that has torch:

    python -m ipykernel install --user --name a2torch --display-name "A2 (torch cu126)"
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

HERE = Path(__file__).resolve().parent
NB = HERE / "03_task3_gender_usage_nguyen.ipynb"

ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--quick", action="store_true",
                help="A2_QUICK=1 -- 4,000 rows and 6 epochs, to prove it runs")
ap.add_argument("--kernel", default="a2torch", help="kernelspec name (default a2torch)")
ap.add_argument("--no-rebuild", action="store_true",
                help="skip regenerating the notebook from task3_build.py")
args = ap.parse_args()

if not args.no_rebuild:
    print("rebuilding the notebook from task3_build.py ...")
    subprocess.run([sys.executable, str(HERE / "build_notebook.py")], check=True)

target = NB
if args.quick:
    target = HERE / "_quick_run.ipynb"
    shutil.copy(NB, target)
    print(f"quick mode: executing a scratch copy, {target.name}, "
          "so the real notebook keeps its outputs")

env = dict(os.environ, MPLBACKEND="Agg", PYTHONIOENCODING="utf-8")
if args.quick:
    env["A2_QUICK"] = "1"

nb = nbformat.read(target, as_version=4)
n_code = sum(1 for c in nb.cells if c.cell_type == "code")
print(f"executing {n_code} code cells with kernel '{args.kernel}' "
      f"(cwd {HERE.name}, no timeout) ...")

client = NotebookClient(
    nb, timeout=None, kernel_name=args.kernel, allow_errors=False,
    resources={"metadata": {"path": str(HERE)}},
)
# The kernel inherits this process's environment, so A2_QUICK and MPLBACKEND land
# inside the notebook without editing a cell.
os.environ.update(env)

t0 = time.time()
failed = None
try:
    client.execute()
except CellExecutionError as exc:
    failed = exc
mins = (time.time() - t0) / 60

nbformat.write(nb, target)
done = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
print(f"\nwrote {target.name} after {mins:.1f} min -- "
      f"{done}/{n_code} code cells carry output")

if failed is not None:
    print("\nA CELL FAILED. The notebook was still saved, so the traceback is in it.")
    print(str(failed)[:2000])
    sys.exit(1)

print("finished cleanly.")
if not args.quick:
    print("Next: python check_notebook.py, then commit the .ipynb and "
          "task3_results.csv.")
