#!/usr/bin/env python
"""Derive the Google Colab edition of the Task 1 notebooks from the checked-in originals.

`task1-collab/` is a *derivation*, not a fork. Every notebook in it is produced from the
matching file under `notebooks/Task1/` by the small, explicit set of edits below, so an
edit to the combine notebook reaches the Colab edition by re-running this script rather
than by being re-applied by hand. A hand-maintained copy of a 260 KB notebook diverges the
first time anyone forgets, and the divergence presents as a fingerprint mismatch an hour
into a training run.

Why this edition needs to exist at all: the source notebooks resolve `REPO_ROOT` by walking
up for `src/preprocessing.py`, which assumes the notebook sits *inside* a checkout. A hosted
notebook does not -- it runs with `/content` (Colab) or `/kaggle/working` (Kaggle) as the
working directory, with the data flat beneath it. That is one cell. Everything else here is
housekeeping: dependency checks, staging the data, importing checkpoints handed over from
another session, and zipping the results back out before the runtime is reclaimed.

The generated notebooks run unmodified on **both Colab and Kaggle** -- the setup cells detect
the platform and resolve `PROJECT_ROOT` themselves. Nothing touches Google Drive: the data is
brought into the session and the results are taken out of it.

What is deliberately NOT changed:

  * Every hyper-parameter, the split, the manifests and the augmentation policy. The Colab
    edition prints the same `RUN_FINGERPRINT` as the source (`e6b15f5c51de` on the supplied
    arm), so its checkpoints interchange with ones trained from a local checkout. `ARM`,
    `ALLOW_CPU`, `RESUME` and every path are outside that fingerprint.
  * The cell order, and the derivation of workers from the combine notebook. This script
    runs *after* `make_task1_workers.py`, over its output.

Each patch asserts its own substitution count. A source edit that moves one of these lines
therefore fails this script loudly instead of silently producing a Colab notebook that is
missing the edit and fails on a runtime an hour later.

    python scripts/make_task1_colab.py            # write task1-collab/
    python scripts/make_task1_colab.py --check    # verify it matches, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import task1_layout as layout  # noqa: E402
import task1_colab_cells as cells  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "notebooks" / "Task1"
DEST_DIR = ROOT / "task1-collab"

COMBINE_NAME = "01_task1_article_type.ipynb"
EVAL_NAME = "02_independent_evaluation.ipynb"

# The two CPU-only jobs. PARALLEL_RUN.md: ALLOW_CPU stays False everywhere else, because a
# silent CPU fallback on a GPU job presents as a hang rather than as an error. These two
# genuinely have no GPU path, so on Colab they are given a CPU runtime and must be allowed
# to use it.
CPU_ONLY_JOBS = {"hog_svm", "hogsearch"}

# The jobs the enriched arm repeats: the three that produce a model. The four tuning grids
# and the sampler sweep are tuned once on the supplied arm and shared, so their workers
# never see external data and have no external path to resolve.
MODEL_JOBS = {"hog_svm", "cnn", "resnet"}


# --- Cell helpers, matching make_task1_workers.py -----------------------------------------

def as_lines(text):
    """Notebook source format: one string per line, each keeping its own newline."""
    return text.splitlines(keepends=True)


def make_cell(kind, text, cell_id):
    cell = {"cell_type": kind, "id": cell_id, "metadata": {}, "source": as_lines(text)}
    if kind == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def cell_text(cell):
    return "".join(cell["source"])


def find_cell(document, needle, description):
    """The index of the one cell containing `needle`.

    Exactly one: zero means the source moved and this patch would be silently skipped;
    more than one means the anchor stopped identifying a single cell and the patch would
    land somewhere unintended.
    """
    hits = [i for i, cell in enumerate(document["cells"]) if needle in cell_text(cell)]
    if len(hits) != 1:
        raise RuntimeError(
            f"{description}: anchor {needle!r} matched {len(hits)} cells, expected exactly 1. "
            "The source notebook has moved; update scripts/make_task1_colab.py."
        )
    return hits[0]


def substitute(text, pattern, replacement, description, expected=1, flags=0):
    new, count = re.subn(pattern, replacement, text, flags=flags)
    if count != expected:
        raise RuntimeError(
            f"{description}: pattern {pattern!r} matched {count} times, expected {expected}. "
            "The source notebook has moved; update scripts/make_task1_colab.py."
        )
    return new


def strip_outputs(document):
    """Colab renders its own run; a stored one is 500 KB of noise in a file people re-upload."""
    for cell in document["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return document


# --- Required-path manifests ---------------------------------------------------------------
# What each notebook actually reads. Kept minimal on purpose: datasets/train/styles_train.csv
# is absent because no Task 1 notebook opens it -- only notebook 00 does, and the manifest it
# writes is tracked, so the EDA run is not repeated per session.

REQUIRED_BASE = [
    "src/preprocessing.py",
    "preprocessed_datasets/train_manifest.csv",
    "datasets/train/images_train",
]
REQUIRED_ENRICHED = [
    "preprocessed_datasets/task1_dataset1_arms",
    "dataset1/images",
]
REQUIRED_COMBINE = REQUIRED_BASE + [
    "datasets/test/styles_prediction.csv",
    "datasets/test/images_test",
    "preprocessed_datasets/task1_dataset1_arms",
    "dataset1/images",
]
REQUIRED_EVAL = REQUIRED_BASE + [
    "dataset1/external_cosmetics.csv",
    "dataset1/images",
    "dataset2/external_cosmetics2.csv",
    "dataset2/images",
    COMBINE_NAME,
]


def as_literal(items, indent=""):
    """A readable multi-line list literal for the staging cell.

    `indent` is the column the literal is opened at, so a list spliced into an indented
    statement closes under it rather than at column zero.
    """
    if not items:
        return "[]"
    body = ",\n".join(f'{indent}    "{item}"' for item in items)
    return "[\n" + body + f",\n{indent}]"


# --- The patches ---------------------------------------------------------------------------

REPO_ROOT_REPLACEMENT = (
    "# Colab edition: the repository is flattened into PROJECT_ROOT, so there is no parent\n"
    "# directory to walk up to. The original walk is kept as the fallback, which is what\n"
    "# lets this same file run unchanged inside an ordinary checkout.\n"
    "REPO_ROOT = Path(PROJECT_ROOT).resolve()\n"
    'if not (REPO_ROOT / "src" / "preprocessing.py").is_file():\n'
    "    REPO_ROOT = next(\n"
    "        (parent for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents)\n"
    '         if (parent / "src" / "preprocessing.py").is_file()),\n'
    "        None,\n"
    "    )\n"
)


def patch_repo_root(document, description):
    """Resolve REPO_ROOT from PROJECT_ROOT, keeping the walk-up as the off-Colab fallback.

    The combine notebook and the evaluation notebook spell the same walk two ways -- the
    former over several lines with a named `parent`, the latter compressed onto two with `p`
    -- so both spellings are handled rather than one of them being quietly missed.
    """
    index = find_cell(document, "REPO_ROOT = next(", description)
    text = cell_text(document["cells"][index])

    # Balanced parentheses rather than a line pattern. The two notebooks wrap this call
    # differently, and a regex that matches one spelling silently fails on the other -- which
    # is a Colab notebook that resolves REPO_ROOT by walking up from /content and finds
    # nothing.
    start = text.index("REPO_ROOT = next(")
    depth = 0
    for position in range(text.index("(", start), len(text)):
        if text[position] == "(":
            depth += 1
        elif text[position] == ")":
            depth -= 1
            if depth == 0:
                end = text.index("\n", position) + 1
                break
    else:
        raise RuntimeError(
            f"{description}: the REPO_ROOT walk-up call is unbalanced. "
            "Update scripts/make_task1_colab.py."
        )

    text = text[:start] + REPO_ROOT_REPLACEMENT + text[end:]

    # The comment above the call explains why the root is found by walking up rather than by
    # naming a parent directory. That is no longer the primary rule here, and leaving it in
    # place puts a paragraph arguing against the code directly above the code. The first two
    # sentences still hold and stay.
    stale = re.search(
        r"# The root is found by walking up for the marker(?:.|\n)*?"
        r"# exactly what the import below needs\.\n", text)
    if stale is not None:
        text = text.replace(stale.group(0), "", 1)

    document["cells"][index]["source"] = as_lines(text)
    return document


def patch_config(document, description):
    """ARM and ALLOW_CPU come from the control panel instead of being edited in place."""
    # Anchored on EXTERNAL_ARMS_DIR rather than on the ARM assignment itself: the assignment
    # is what a worker machine edits by hand to switch arms, so the working tree may hold
    # either value, and the Colab control panel echoes the string as well. This line is in the
    # same cell, unique, and nobody edits it.
    index = find_cell(document, 'EXTERNAL_ARMS_DIR = REPO_ROOT /', description)
    text = cell_text(document["cells"][index])

    text = substitute(
        text, r"^ALLOW_CPU = False$",
        "ALLOW_CPU = COLAB_ALLOW_CPU   # from the Colab control panel at the top",
        f"{description}: ALLOW_CPU", flags=re.MULTILINE)
    # Either value: whichever arm the source file was last run under, the Colab edition takes
    # its arm from the control panel, so the generated file is the same either way.
    text = substitute(
        text, r'^ARM = "(?:supplied|enriched)"$',
        "ARM = COLAB_ARM   # from the Colab control panel at the top",
        f"{description}: ARM", flags=re.MULTILINE)

    document["cells"][index]["source"] = as_lines(text)
    return document


def patch_external_paths(document, description):
    """Route the arms CSV's recorded paths through the layout-tolerant resolver."""
    index = find_cell(document, 'EXTERNAL_TRAIN["relative_path"].map(', description)
    text = cell_text(document["cells"][index])
    text = substitute(
        text,
        r'EXTERNAL_TRAIN\["relative_path"\]\.map\(\n'
        r'\s*lambda relative: str\(REPO_ROOT / relative\)\)',
        'EXTERNAL_TRAIN["relative_path"].map(resolve_external)',
        f"{description}: external path map")
    document["cells"][index]["source"] = as_lines(text)
    return document


def patch_eval_paths(document, description):
    """The evaluation notebook is the only one that names notebooks/Task1/ paths."""
    index = find_cell(document, 'COMBINE = REPO_ROOT / "notebooks/Task1/', description)
    text = cell_text(document["cells"][index])

    text = substitute(
        text, r'REPO_ROOT / "notebooks/Task1/01_task1_article_type\.ipynb"',
        f'REPO_ROOT / "{COMBINE_NAME}"', f"{description}: combine notebook path")

    # The arm subdirectory. models/task1/checkpoints/ has held per-arm subdirectories since
    # the Section 11 experiment landed, and this line was not updated with it, so the load
    # below it cannot find model_resnet_decoupled.pt. Corrected here; the source notebook
    # under notebooks/Task1/ still carries the original line.
    text = substitute(
        text, r'CHECKPOINTS = REPO_ROOT / "models/task1/checkpoints"',
        'CHECKPOINTS = REPO_ROOT / "models/task1/checkpoints" / COLAB_ARM',
        f"{description}: checkpoint arm directory")

    document["cells"][index]["source"] = as_lines(text)

    index = find_cell(document, '"dataset1": (REPO_ROOT / "notebooks/Task1/', description)
    text = cell_text(document["cells"][index])
    text = substitute(text, r'REPO_ROOT / "notebooks/Task1/dataset(\d)/',
                      r'REPO_ROOT / "dataset\1/', f"{description}: collection paths",
                      expected=4)
    document["cells"][index]["source"] = as_lines(text)
    return document


def prerequisite_note(job):
    """The cross-job checkpoint dependency, rendered for the Colab header."""
    if job not in layout.PREREQUISITES:
        return ""
    _severity, files, _why = layout.PREREQUISITES[job]
    names = ", ".join(f"`{name}` (from the `{producer}` session)"
                      for name, producer in files.items())
    return (f"\n**Needs first:** {names}. Unpack that session's zip into\n"
            f"`models/task1/checkpoints/<arm>/` before starting, or this job stops on an\n"
            f"assertion having done nothing.\n")


def build(source_path, kind, job=None):
    document = strip_outputs(json.loads(source_path.read_text(encoding="utf-8")))
    description = source_path.name

    if job in CPU_ONLY_JOBS:
        allow_cpu_line = cells.ALLOW_CPU_CPU
        runtime_type = cells.RUNTIME_TYPE_CPU
    else:
        allow_cpu_line = cells.ALLOW_CPU_GPU
        runtime_type = cells.RUNTIME_TYPE_GPU

    if kind == "combine":
        header = cells.HEADER_COMBINE
        required, required_enriched = REQUIRED_COMBINE, []
        export = cells.EXPORT_COMBINE
    elif kind == "worker":
        subtitle, _ = layout.JOBS[job]
        header = cells.HEADER_WORKER.format(
            subtitle=subtitle, job=job, runtime=layout.RUNTIMES[job],
            runtime_type=runtime_type, prerequisites=prerequisite_note(job),
            combine=COMBINE_NAME)
        required, required_enriched = REQUIRED_BASE, REQUIRED_ENRICHED
        export = cells.EXPORT_WORKER.replace("__JOB__", job)
    else:
        header = cells.HEADER_EVAL.format(combine=COMBINE_NAME)
        required, required_enriched = REQUIRED_EVAL, []
        export = cells.EXPORT_EVAL

    # Patch before the setup cells go in. The anchors are distinctive lines of the source
    # notebook, and the control panel deliberately echoes some of them -- COLAB_ARM =
    # "supplied" contains the ARM anchor verbatim -- so searching a document that already
    # carries the injected cells finds two matches and fails.
    document = patch_repo_root(document, description)
    if kind in ("combine", "worker"):
        document = patch_config(document, description)
        # Only the combine notebook and the three model-producing jobs carry the split cell's
        # external-data append; the tuning grids never see external data.
        if kind == "combine" or job in MODEL_JOBS:
            document = patch_external_paths(document, description)
    else:
        document = patch_eval_paths(document, description)

    document["cells"][0] = make_cell("markdown", header, "colab-header")

    setup = [
        make_cell("markdown", cells.SETUP_HEADING, "colab-setup-heading"),
        make_cell("code", cells.CONTROL_PANEL.replace("__ALLOW_CPU_LINE__", allow_cpu_line),
                  "colab-control-panel"),
        make_cell("code", cells.ENVIRONMENT, "colab-environment"),
        make_cell("code",
                  cells.STAGING.replace("__REQUIRED__", as_literal(required))
                               .replace("__REQUIRED_ENRICHED__",
                                        as_literal(required_enriched, "    ")),
                  "colab-staging"),
    ]
    document["cells"] = [document["cells"][0]] + setup + document["cells"][1:]

    document["cells"].append(make_cell("code", export, "colab-export"))
    return document


def targets():
    """(source path, destination name, kind, job) for every notebook in the Colab edition."""
    yield SOURCE_DIR / COMBINE_NAME, COMBINE_NAME, "combine", None
    for job in sorted(layout.JOBS):
        name = f"worker_{job}.ipynb"
        yield SOURCE_DIR / name, name, "worker", job
    yield SOURCE_DIR / EVAL_NAME, EVAL_NAME, "eval", None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="compare against the files on disk, write nothing")
    arguments = parser.parse_args()

    DEST_DIR.mkdir(parents=True, exist_ok=True)
    differences = []
    written = 0

    for source_path, name, kind, job in targets():
        if not source_path.is_file():
            raise FileNotFoundError(
                f"{source_path} is missing. Run scripts/make_task1_workers.py first."
            )
        built = json.dumps(build(source_path, kind, job), indent=1, ensure_ascii=False) + "\n"
        destination = DEST_DIR / name

        if arguments.check:
            current = destination.read_text(encoding="utf-8") if destination.is_file() else None
            if current != built:
                differences.append(name)
                print(f"  DIFFERS  {name}")
            else:
                print(f"  ok       {name}")
        else:
            destination.write_text(built, encoding="utf-8")
            written += 1
            print(f"  wrote    {name}  ({len(built) / 1024:.0f} KB)")

    if arguments.check and differences:
        print(f"\n{len(differences)} notebook(s) differ from what this script generates. "
              "Re-run it without --check.")
        return 1
    print(f"\n{DEST_DIR.name}/: {written or len(differences) or 10} notebooks "
          f"{'verified' if arguments.check else 'written'}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
