#!/usr/bin/env python
"""Generate the Task 1 worker notebooks from the combine notebook.

Each worker is the combine notebook with the other jobs' cells removed, the analysis that
reads every model removed, and three cells substituted: a header, the execution-mode cell
with JOB_FILTER set, and a done cell. Everything else is copied verbatim, which is what
makes every machine agree on RUN_FINGERPRINT.

    python scripts/make_task1_workers.py              # write the workers
    python scripts/make_task1_workers.py --check      # report what would change, write nothing
    python scripts/make_task1_workers.py --legacy --check
                                                      # prove this reproduces the hand-derived
                                                      # workers before trusting it

The --legacy gate is the reason to believe the cell-selection rule in task1_layout is the
real one. In that mode the header and done cells are taken from the existing files rather
than rendered, so the comparison tests the risky part -- which cells are kept, in what order,
and how the mode cell is patched -- rather than prose that is trivial to eyeball.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task1_layout as layout  # noqa: E402

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "Task1"
COMBINE = NOTEBOOK_DIR / "01_task1_article_type.ipynb"

HEADER = """# Worker notebook — {subtitle}

Trains **{job}** only, banks its checkpoints, and stops. Runtime {runtime} on a mid-range CUDA GPU.

**Before running:** copy `preprocessed_datasets/` from the machine that ran notebook 00 —
do not regenerate it here. The split and the normalisation constants come from those files
and feed `RUN_FINGERPRINT`; regenerating risks a different split and checkpoints that
silently do not match the other machines'.

**After running:** copy everything in `models/task1/checkpoints/` to the combine machine's
`models/task1/checkpoints/`, then run `01_task1_article_type.ipynb` there.

Nothing in this notebook analyses, scores against other models, or writes predictions —
that is the combine notebook's job. Setup and definitions are byte-identical to it."""

DONE = '''# --- Done -------------------------------------------------------------------------------
print("Worker complete: {job}")
print("\\nCheckpoints in", CHECKPOINT_DIR.resolve())
_total = 0
for _path in sorted(CHECKPOINT_DIR.glob("*.pt")) + sorted(CHECKPOINT_DIR.glob("*.joblib")):
    _size = _path.stat().st_size / 1e6
    _total += _size
    print(f"  {{_path.name}}  ({{_size:.1f}} MB)")
print(f"\\n{{_total:.0f}} MB total. Copy these to the combine machine, then run the "
      f"final notebook there with JOB_FILTER = None.")
print("Fingerprint:", RUN_FINGERPRINT, "-- must match on every machine.")'''


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sources(document):
    return [(c["cell_type"], "".join(c["source"])) for c in document["cells"]]


def as_lines(text):
    """Notebook source format: one string per line, each keeping its own newline.

    splitlines rather than split("\\n"): the latter turns a trailing newline into an extra
    empty element, which reads back identically but is not the same JSON.
    """
    return text.splitlines(keepends=True)


def make_cell(template, kind, text):
    """A cell carrying `text`, reusing `template`'s id and metadata so the diff stays small."""
    cell = {"cell_type": kind, "metadata": copy.deepcopy(template.get("metadata", {})),
            "source": as_lines(text)}
    if "id" in template:
        cell["id"] = template["id"]
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def clean(cell):
    """A copy of a combine cell with any recorded run stripped out."""
    cell = copy.deepcopy(cell)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def build(document, job, runtime, subtitle, legacy_owned=None, legacy=False):
    """The worker notebook for `job` as a parsed document."""
    cells = sources(document)
    title_index = layout.resolve(cells, layout.TITLE)
    mode_index = layout.resolve(cells, layout.JOB_FILTER_CELL)
    kept = layout.worker_cells(cells, job, legacy=legacy)

    # Anchored to the start of a line: the same text appears in the comment above the
    # assignment, and rewriting that too would document every worker as combine mode.
    mode_source, substitutions = re.subn(
        f"^{re.escape(layout.JOB_FILTER_LINE)}$", f'JOB_FILTER = {{"{job}"}}',
        cells[mode_index][1], flags=re.MULTILINE,
    )
    if substitutions != 1:
        raise RuntimeError(
            f"{layout.JOB_FILTER_LINE!r} matched {substitutions} assignments, expected 1"
        )

    if legacy_owned is not None:
        header_text, done_text = legacy_owned
    else:
        header_text = HEADER.format(subtitle=subtitle, job=job, runtime=runtime)
        done_text = DONE.format(job=job)

    body = [(mode_index, make_cell(document["cells"][mode_index], "code", mode_source))]
    body += [(index, clean(document["cells"][index])) for index in kept]
    body.sort(key=lambda pair: pair[0])

    worker = copy.deepcopy(document)
    worker["cells"] = (
        [make_cell(document["cells"][title_index], "markdown", header_text)]
        + [cell for _, cell in body]
        + [make_cell(document["cells"][-1], "code", done_text)]
    )
    return worker


def legacy_header_and_done(path):
    """The header and done cells of an existing worker, verbatim."""
    cells = sources(load(path))
    return cells[0][1], cells[-1][1]


def legacy_runtime_and_subtitle(text):
    subtitle = re.search(r"# Worker notebook — (.+)", text).group(1)
    runtime = re.search(r"Runtime (~\d+ min)", text).group(1)
    return runtime, subtitle


def comparable(document):
    """The document reduced to what actually matters, so formatting noise cannot fail a diff."""
    return json.dumps(
        {"metadata": document.get("metadata"),
         "cells": [{k: v for k, v in cell.items() if k != "id"} for cell in document["cells"]]},
        sort_keys=True, indent=1,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true",
                        help="compare against the files on disk, write nothing")
    parser.add_argument("--legacy", action="store_true",
                        help="reproduce the hand-derived workers exactly")
    parser.add_argument("--jobs", nargs="*", default=None,
                        help="limit to these job names")
    args = parser.parse_args()

    document = load(COMBINE)
    jobs = args.jobs or list(layout.JOBS)
    unknown = set(jobs) - set(layout.JOBS)
    if unknown:
        sys.exit(f"unknown job(s): {sorted(unknown)}")

    differences = 0
    for job in jobs:
        subtitle, _ = layout.JOBS[job]
        path = NOTEBOOK_DIR / f"worker_{job}.ipynb"
        runtime = layout.RUNTIMES.get(job, "~? min")
        legacy_owned = None

        if args.legacy:
            if not path.exists():
                print(f"  skip  {path.name}: no existing file to reproduce")
                continue
            legacy_owned = legacy_header_and_done(path)

        worker = build(document, job, runtime, subtitle,
                       legacy_owned=legacy_owned, legacy=args.legacy)

        if args.check:
            if not path.exists():
                print(f"  new   {path.name}: would be created "
                      f"({len(worker['cells'])} cells)")
                differences += 1
                continue
            before, after = comparable(load(path)), comparable(worker)
            if before == after:
                print(f"  same  {path.name}: {len(worker['cells'])} cells, identical")
            else:
                existing, produced = sources(load(path)), sources(worker)
                print(f"  DIFF  {path.name}: {len(existing)} cells on disk, "
                      f"{len(produced)} produced")
                for index in range(max(len(existing), len(produced))):
                    a = existing[index][1] if index < len(existing) else None
                    b = produced[index][1] if index < len(produced) else None
                    if a != b:
                        head = (b or a or "").split("\n")[0][:70]
                        print(f"          cell {index}: {head}")
                differences += 1
        else:
            path.write_text(json.dumps(worker, indent=1, ensure_ascii=False) + "\n",
                            encoding="utf-8")
            print(f"  wrote {path.name} ({len(worker['cells'])} cells)")

    if args.check:
        print(f"\n{differences} of {len(jobs)} worker(s) differ.")
        return 1 if differences else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
