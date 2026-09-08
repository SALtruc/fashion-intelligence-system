"""Run a Task 1 notebook across every GPU on the machine.

A notebook cannot reach more than one GPU on its own. `torch.distributed` needs one process
per device, the process group has to be established before any of them builds a model, and
`mp.spawn` cannot pickle a function defined in a notebook cell -- so the launcher has to be
outside the kernel. This script is that launcher: it flattens the notebook's code cells into
a module and hands it to `torchrun`, which starts N copies and sets the environment variables
`src/task1_ddp.py` reads.

    python scripts/task1_torchrun.py notebooks/Task1/worker_resnet.ipynb

By default it uses every visible GPU. `--nproc` overrides that, and `--dry-run` writes the
flattened script without running it, which is the quickest way to see what will execute.

Nothing here changes what is learned. The notebook run interactively and the notebook run
through this launcher train the same model to the same global batch; the launcher only
decides how many processes share the work. See the module docstring of `src/task1_ddp.py`
for why the global batch is held fixed rather than multiplied by the device count.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def flatten(notebook):
    """The notebook's code cells as one module, in order.

    IPython's line magics and shell escapes are dropped rather than translated: `%matplotlib`
    and `!pip install` have no meaning outside a kernel, and the alternative -- depending on
    IPython to interpret them -- would put a second execution model between the notebook and
    the training run. Everything dropped is display or environment setup, never modelling.
    """
    document = json.loads(Path(notebook).read_text(encoding="utf-8"))
    parts, dropped = [], 0

    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in "".join(cell["source"]).splitlines():
            if line.lstrip().startswith(("%", "!", "%%")):
                dropped += 1
                continue
            lines.append(line)
        body = "\n".join(lines).strip("\n")
        if body:
            parts.append(f"# ---- cell {index} " + "-" * 60 + f"\n{body}\n")

    header = (
        '"""Generated from '
        f'{Path(notebook).as_posix()} by scripts/task1_torchrun.py. Do not edit."""\n'
        "import matplotlib\n"
        "matplotlib.use('Agg')          # no display under torchrun; figures still get written\n"
    )
    footer = (
        "\n# ---- launcher epilogue " + "-" * 52 + "\n"
        "# Leaving the process group cleanly. Without this a rank that finishes first can\n"
        "# exit while another is mid-collective, which surfaces as a NCCL timeout rather\n"
        "# than as the successful run it actually was.\n"
        "try:\n"
        "    ddp.shutdown()\n"
        "except Exception:\n"
        "    pass\n"
    )
    return header + "\n".join(parts) + footer, dropped


def gpu_count():
    """Visible GPUs, asked for in a subprocess.

    Importing torch in the launcher would initialise CUDA in a process that is about to fork
    children which each need their own context, so the question is asked and answered out of
    process instead.
    """
    probe = "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)"
    try:
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                              timeout=180)
        return int(done.stdout.strip() or 0)
    except (subprocess.SubprocessError, ValueError):
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("notebook", help="the .ipynb to run")
    parser.add_argument("--nproc", type=int, default=None,
                        help="processes to start (default: every visible GPU)")
    parser.add_argument("--out", default=None,
                        help="where to write the flattened script "
                             "(default: alongside the notebook, as _run_<name>.py)")
    parser.add_argument("--dry-run", action="store_true",
                        help="write the script and stop, without launching")
    args = parser.parse_args()

    notebook = Path(args.notebook).resolve()
    if not notebook.is_file():
        raise SystemExit(f"No such notebook: {notebook}")

    script, dropped = flatten(notebook)
    out = Path(args.out) if args.out else notebook.parent / f"_run_{notebook.stem}.py"
    out.write_text(script, encoding="utf-8")
    print(f"Flattened {notebook.name} -> {out}  "
          f"({len(script.splitlines())} lines, {dropped} magic/shell lines dropped)")

    available = gpu_count()
    nproc = args.nproc if args.nproc is not None else max(available, 1)

    if args.dry_run:
        print(f"Dry run: would launch {nproc} process(es) over {available} visible GPU(s).")
        return

    if available == 0:
        print("No CUDA device visible. Running the script directly in one process; "
              "the notebook's own ALLOW_CPU guard still applies.")
        command = [sys.executable, str(out)]
    elif nproc == 1:
        # torchrun for a single process adds a rendezvous and buys nothing.
        print(f"{available} GPU(s) visible, running 1 process. "
              f"{'Pass --nproc to use more.' if available > 1 else ''}")
        command = [sys.executable, str(out)]
    else:
        print(f"Launching {nproc} processes over {available} visible GPU(s) via torchrun. "
              "Rank 0's output is the run log; the other ranks are silenced.")
        command = [sys.executable, "-m", "torch.distributed.run",
                   "--standalone", f"--nproc_per_node={nproc}", str(out)]

    # From the repository root, because the notebook locates REPO_ROOT by walking up from the
    # working directory for src/preprocessing.py.
    environment = dict(os.environ, MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
    raise SystemExit(subprocess.call(command, cwd=str(ROOT), env=environment))


if __name__ == "__main__":
    main()
