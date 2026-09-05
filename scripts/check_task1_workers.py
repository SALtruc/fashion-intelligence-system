#!/usr/bin/env python
"""Structural checks for the Task 1 parallel-run notebooks.

Four properties the parallel scheme depends on, none of which any single notebook can
enforce on its own:

  1. anchors      every anchor in task1_layout resolves to exactly one combine cell
  2. identity     every cell a worker shares with the combine notebook is byte-identical
                  to it, so all machines agree on RUN_FINGERPRINT
  3. static       no undefined names, so stripping another job's cells cannot leave a
                  worker referring to something that no longer exists
  4. fingerprint  RUN_FINGERPRINT still hashes to the value the checkpoints on disk were
                  written under

Plus a non-fatal layout report: which cells each worker holds against what
task1_layout.JOBS says it should. Run with --strict to make that fatal too.

    python scripts/check_task1_workers.py [--strict]

Exit status is 0 when every check passes.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task1_layout as layout  # noqa: E402

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "Task1"
COMBINE = NOTEBOOK_DIR / "01_task1_article_type.ipynb"

failures: list[str] = []
notes: list[str] = []


def fail(message):
    failures.append(message)
    print(f"  FAIL  {message}")


def ok(message):
    print(f"  ok    {message}")


def read_cells(path):
    """(cell_type, source) for every cell, with the source joined exactly as stored."""
    document = json.loads(path.read_text(encoding="utf-8"))
    return [(c["cell_type"], "".join(c["source"])) for c in document["cells"]]


def digest(source):
    return hashlib.sha1(source.encode("utf-8")).hexdigest()


# --- 1. anchors --------------------------------------------------------------------------
def check_anchors(combine):
    print("\n[1] anchors resolve uniquely")
    anchors = [layout.TITLE, layout.JOB_FILTER_CELL, layout.WORKER_STOP]
    anchors += layout.COMBINE_ONLY
    anchors += layout.LEGACY_TRAILING
    for _, job_anchors in layout.JOBS.values():
        anchors += job_anchors
    for anchor in anchors:
        try:
            layout.resolve(combine, anchor)
        except LookupError as error:
            fail(str(error))
    if not failures:
        ok(f"{len(anchors)} anchors, each matching exactly one cell")


# --- 2. cell identity --------------------------------------------------------------------
def check_identity(combine, workers):
    print("\n[2] worker cells are byte-identical to the combine notebook")
    combine_digests = {digest(source) for _, source in combine}
    for path, cells in workers.items():
        owned = []
        for index, (kind, source) in enumerate(cells):
            if digest(source) not in combine_digests:
                first = next((line for line in source.split("\n") if line.strip()), "")
                owned.append((index, kind, first[:64]))
        # A worker is allowed to own exactly three cells: its header, its JOB_FILTER cell,
        # and its done cell. Anything else is drift from the combine notebook.
        if len(owned) != 3:
            fail(f"{path.name}: owns {len(owned)} cells, expected 3")
            for index, kind, first in owned:
                print(f"          cell {index} [{kind}] {first}")
        else:
            positions = [index for index, _, _ in owned]
            if positions[0] != 0 or positions[-1] != len(cells) - 1:
                fail(f"{path.name}: owned cells at {positions}, expected first/middle/last")
            else:
                ok(f"{path.name}: {len(cells) - 3} shared cells identical, 3 owned")


# --- 3. static sanity --------------------------------------------------------------------
def check_static(paths):
    print("\n[3] no undefined names")
    try:
        subprocess.run([sys.executable, "-m", "pyflakes", "--version"],
                       capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        notes.append("pyflakes is not installed; check 3 skipped (pip install pyflakes)")
        print("  skip  pyflakes not installed")
        return

    with tempfile.TemporaryDirectory() as workspace:
        extracted = []
        for path in paths:
            parts = []
            for kind, source in read_cells(path):
                if kind != "code":
                    continue
                # Line magics and shell escapes are notebook syntax, not Python.
                parts.append("\n".join(
                    "#" + line if line.lstrip().startswith(("%", "!")) else line
                    for line in source.split("\n")
                ))
            target = Path(workspace) / (path.stem + ".py")
            target.write_text("\n\n".join(parts), encoding="utf-8")
            extracted.append(target)

        result = subprocess.run(
            [sys.executable, "-m", "pyflakes", *map(str, extracted)],
            capture_output=True, text=True,
        )
        # Unused imports are expected: each worker carries the full shared import block but
        # exercises one job's worth of it. Undefined names are not.
        problems = [line for line in result.stdout.splitlines()
                    if line.strip() and "imported but unused" not in line]
        if problems:
            fail(f"pyflakes reported {len(problems)} non-import problems")
            for line in problems[:20]:
                print(f"          {line}")
        else:
            ok(f"{len(extracted)} notebooks clean (unused imports ignored)")


# --- 4. fingerprint ----------------------------------------------------------------------
def module_level_constants(cells):
    """Literal assignments at module level, i.e. not inside `if QUICK_RUN:` and friends.

    Restricting to module level is what makes this read the configured value rather than an
    override that only fires in a branch the run does not take.
    """
    values = {}
    for kind, source in cells:
        if kind != "code":
            continue
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if not isinstance(target, ast.Name):
                    continue
                try:
                    values[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return values


def check_fingerprint(combine):
    print("\n[4] RUN_FINGERPRINT unchanged")
    constants = module_level_constants(combine)

    missing = [name for name in
               (*layout.FINGERPRINT_SCALARS, *layout.FINGERPRINT_AUG, *layout.FINGERPRINT_STAGE2)
               if name not in constants]
    if missing:
        fail(f"could not read {missing} from the notebook; fingerprint not checked")
        return

    payload = {key: constants[name] for name, key in layout.FINGERPRINT_SCALARS.items()}
    payload["aug"] = [constants[name] for name in layout.FINGERPRINT_AUG]
    payload["stage2"] = [constants[name] for name in layout.FINGERPRINT_STAGE2]
    payload.update(layout.RECORDED_RUN)

    actual = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()[:12]

    if actual != layout.EXPECTED_FINGERPRINT:
        fail(f"fingerprint is {actual}, expected {layout.EXPECTED_FINGERPRINT}; "
             "every checkpoint on disk would be refused")
        print(f"          payload: {json.dumps(payload, sort_keys=True)}")
    else:
        ok(f"{actual} (checkpoints on disk stay valid)")


# --- 5. layout report --------------------------------------------------------------------
def expected_layout(combine, job, legacy):
    """Combine-cell indices a worker for `job` should hold, in order."""
    stop = layout.resolve(combine, layout.WORKER_STOP)
    owners = {}
    for name, (_, anchors) in layout.JOBS.items():
        for anchor in anchors:
            owners[layout.resolve(combine, anchor)] = name
    if legacy:
        note = layout.resolve(combine, "#### On `P2_FINER_MAP`")
        owners[note] = layout.LEGACY_P2_NOTE_OWNER

    combine_only = {layout.resolve(combine, anchor) for anchor in layout.COMBINE_ONLY}
    trailing = {layout.resolve(combine, anchor) for anchor in layout.LEGACY_TRAILING}
    # The mode cell is rewritten per worker, so it is never one of the shared cells this
    # comparison can see.
    owned_by_worker = {stop, layout.resolve(combine, layout.JOB_FILTER_CELL)}

    kept = []
    for index in range(len(combine)):
        if index >= stop and not (legacy and index in trailing):
            continue
        if index in combine_only or index in owned_by_worker:
            continue
        if index in owners and owners[index] != job:
            continue
        kept.append(index)
    return kept


def check_layout(combine, workers, strict, legacy):
    label = "legacy" if legacy else "target"
    print(f"\n[5] worker composition against the {label} layout")
    index_of = {digest(source): i for i, (_, source) in enumerate(combine)}
    for path, cells in workers.items():
        job = path.stem.replace("worker_", "")
        if job not in layout.JOBS:
            notes.append(f"{path.name}: no job named {job!r} in task1_layout.JOBS")
            continue
        held = [index_of[digest(source)] for _, source in cells if digest(source) in index_of]
        want = expected_layout(combine, job, legacy)
        extra = sorted(set(held) - set(want))
        absent = sorted(set(want) - set(held))
        if extra or absent:
            message = f"{path.name}: holds extra {extra}, missing {absent}"
            if strict:
                fail(message)
            else:
                print(f"  diff  {message}")
                notes.append(message)
        else:
            ok(f"{path.name}: {len(held)} cells match")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="treat layout deviations as failures")
    parser.add_argument("--legacy", action="store_true",
                        help="expect the hand-derived layout (P2 note on lrsearch, "
                             "trailing cells retained)")
    args = parser.parse_args()

    if not COMBINE.exists():
        sys.exit(f"combine notebook not found: {COMBINE}")

    combine = read_cells(COMBINE)
    worker_paths = sorted(NOTEBOOK_DIR.glob("worker_*.ipynb"))
    workers = {path: read_cells(path) for path in worker_paths}

    print(f"combine: {COMBINE.name} ({len(combine)} cells)")
    print(f"workers: {len(workers)}")

    check_anchors(combine)
    if failures:
        print("\nAnchors did not resolve; the remaining checks would be meaningless.")
        return 1

    check_identity(combine, workers)
    check_static([COMBINE, *worker_paths])
    check_fingerprint(combine)
    check_layout(combine, workers, args.strict, args.legacy)

    print()
    for note in notes:
        print(f"note: {note}")
    if failures:
        print(f"\n{len(failures)} check(s) failed.")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
