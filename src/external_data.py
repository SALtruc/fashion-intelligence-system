"""Externally collected images, for the COSC2753 A2 task notebooks.

Companion to `preprocessing.py`. That module owns the provided catalogue; this one
owns everything collected from outside it, and nothing else changes.

There are TWO external sets and they exist for opposite reasons. Mixing them up
invalidates the report, so they are reached through different functions and the one
that must never be trained on refuses to look like training data.

    add_to_training(...)  ->  extra TRAINING rows (1,899 cosmetics crops)
    load_eval_set()       ->  the INDEPENDENT EVALUATION set (261 images)

Why `add_to_training` takes the training frame rather than the whole manifest
------------------------------------------------------------------------------
External rows must reach `train` only. If they were appended before `make_split`,
some would land in validation, validation would stop being purely provided
catalogue imagery, and every "with vs without external data" comparison would be
meaningless - the thing the data was collected to demonstrate.

Rather than document that rule and hope, the signature enforces it: this function
takes the output of `make_split`, so there is no way to call it with a frame that
validation is still inside.

    training, validation = pp.make_split(frame, target)
    training = add_to_training(training, target)      # validation untouched

Provenance
----------
Both training batches are cosmetics crops, gated against the provided data
(0 matches against train, 0 against the test set). `gender`, `season` and `usage`
are propagated per articleType from the provided training labels, because those are
catalogue conventions rather than visual properties and are deterministic within
these classes. Every row carries `label_source` recording that, so a propagated
label is never mistaken for an observed one.

Consequence worth stating in the report: every cosmetics class is `season=Spring`,
the rarest season (4.0% of train). Adding these roughly triples it. That is a real
gain and a real bias - the model is taught harder that a cosmetic implies Spring,
which is the actual rule generating the label but is not seasonality. Measure Spring
recall separately on Personal Care and non-Personal-Care rows before claiming
Task 2 improved.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

TARGETS = ["articleType", "season", "gender", "usage"]

# Each training batch: folder name, its CSV, and what it was collected for.
TRAINING_SETS = {
    "cosmetics1": ("ExternalCosmetics", "external_cosmetics.csv",
                   "400 each of Eyeshadow, Lipstick, Nail Polish"),
    "cosmetics2": ("ExternalCosmetics2", "external_cosmetics2.csv",
                   "~100 each of Compact, Concealer, Foundation and Primer, "
                   "Highlighter and Blush, Kajal and Eyeliner, Lip Gloss, Lip Liner"),
}
EVAL_SET = ("ExternalEval", "external_eval.csv")

# group_id must not collide with the catalogue's. External images are single products
# with no near-duplicate partners, so one group per image is correct.
_GROUP_PREFIX = "ext:"


def _candidate_roots() -> list[Path]:
    """Where the external data might live, most explicit first.

    Colab is the reason this is a search rather than a constant: the notebooks run on
    a fresh clone with the data mounted from Drive, so no single hard-coded path is
    right both there and on a teammate's laptop.
    """
    roots = []
    env = os.environ.get("A2_EXTERNAL_DATA")
    if env:
        roots.append(Path(env))
    roots += [
        _REPO_ROOT / "Dataset",                      # local clone, alongside datasets/
        _REPO_ROOT / "datasets",                     # if dropped in beside the provided data
        _REPO_ROOT / "external_datasets",
        # Colab mounts MyDrive only. A folder someone shared with you appears under
        # "Shared with me", which is NOT mounted - it has to be right-clicked in
        # Drive and added with "Add shortcut to Drive" first. These cover the usual
        # places that shortcut ends up.
        Path("/content/drive/MyDrive/A2_ExternalData"),
        Path("/content/drive/MyDrive/Nguyen/A2_ExternalData"),
        Path("/content/drive/MyDrive/A2/Nguyen/A2_ExternalData"),
        Path("/content/drive/MyDrive/[ML] SG_G3/A2/Nguyen/A2_ExternalData"),
        Path("/content/drive/MyDrive/ExternalData"),
        Path("/content/drive/MyDrive/A2/ExternalData"),
        Path("/content/external_data"),
        Path("/content/Dataset"),
    ]
    return roots


def default_data_root() -> Path:
    """Where a collector script should WRITE a newly built set.

    The same rule `find_external_root` reads with, so a set written by
    `prepare_external_cosmetics2.py` is somewhere `add_to_training` will look.
    Keeping both in one module is the point: two copies of "where the data lives"
    drift, and the failure is silent - the loader simply reports nothing found.

    Override with the A2_EXTERNAL_DATA environment variable.
    """
    return Path(os.environ.get("A2_EXTERNAL_DATA", _REPO_ROOT / "Dataset"))


def find_external_root(required: str | None = None) -> Path:
    """Locate the folder holding the external sets, or explain how to point at it.

    Args:
        required: a subfolder that must exist, e.g. "ExternalCosmetics".

    Raises:
        FileNotFoundError: with the paths tried and the one-line fix.
    """
    tried = []
    for root in _candidate_roots():
        tried.append(str(root))
        if not root.is_dir():
            continue
        if required is None or (root / required).is_dir():
            return root

    raise FileNotFoundError(
        "External data not found"
        + (f" (looking for {required!r})" if required else "")
        + ".\nTried:\n  " + "\n  ".join(tried)
        + "\n\nOn Colab: the team folder lives under 'Shared with me', which Colab "
          "does NOT mount.\nOpen Drive, right-click A2_ExternalData -> 'Add shortcut "
          "to Drive', then re-run.\n"
          "\nOtherwise place the folders next to datasets/ in a Dataset/ folder, or "
          "point at them directly:\n"
          "    import os; os.environ['A2_EXTERNAL_DATA'] = "
          "'/content/drive/MyDrive/<folder holding ExternalCosmetics etc>'\n"
    )


def _load_one(key: str) -> pd.DataFrame:
    folder, csv_name, _ = TRAINING_SETS[key]
    root = find_external_root(folder)
    base = root / folder
    frame = pd.read_csv(base / csv_name)

    missing = [c for c in TARGETS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{base / csv_name} is missing {missing}. Batch 1 needs "
            "add_cosmetics_season_usage.py run against it before it can be used for "
            "Task 2 or Task 3."
        )

    frame = frame.copy()
    frame["filename"] = frame["id"].astype(str) + ".jpg"
    frame["path"] = frame["filename"].map(lambda n: str(base / "images" / n))
    frame["group_id"] = _GROUP_PREFIX + frame["id"].astype(str)
    frame["external_set"] = key

    present = frame["path"].map(lambda p: Path(p).is_file())
    if not present.all():
        raise FileNotFoundError(
            f"{(~present).sum()} of {len(frame)} images listed in {csv_name} are missing "
            f"under {base / 'images'}. Re-download that folder from Drive."
        )
    return frame


def load_external_training(sets=("cosmetics1", "cosmetics2"), target=None) -> pd.DataFrame:
    """Load the extra training rows. Prefer `add_to_training` unless inspecting."""
    unknown = [s for s in sets if s not in TRAINING_SETS]
    if unknown:
        raise KeyError(f"unknown set(s) {unknown}; available: {list(TRAINING_SETS)}")

    frames = [_load_one(s) for s in sets]
    out = pd.concat(frames, ignore_index=True)
    if target is not None:
        if target not in out.columns:
            raise KeyError(f"{target!r} is not a column of the external data.")
        out = out.loc[out[target].notna()].reset_index(drop=True)
    return out


def add_to_training(training: pd.DataFrame, target: str,
                    sets=("cosmetics1", "cosmetics2"),
                    verbose: bool = True) -> pd.DataFrame:
    """Append the external rows to a TRAINING frame from `make_split`.

    Args:
        training: the first element returned by `preprocessing.make_split`.
        target: the task's label column, used to drop external rows without it.
        sets: which batches to include.
        verbose: print what was added, per class.

    Returns:
        A new frame; `training` is not modified.
    """
    if target not in training.columns:
        raise KeyError(f"{target!r} is not a column of the training frame.")

    external = load_external_training(sets=sets, target=target)
    shared = [c for c in training.columns if c in external.columns]
    for needed in ("path", "group_id", target):
        if needed not in shared:
            raise ValueError(
                f"the training frame has no {needed!r} column, so the external rows "
                "cannot be aligned to it. Pass the frame returned by make_split()."
            )

    merged = pd.concat(
        [training, external[shared]], ignore_index=True
    )

    overlap = set(training["group_id"]) & set(external["group_id"])
    assert not overlap, f"group_id collision with the catalogue: {sorted(overlap)[:5]}"

    if verbose:
        counts = external[target].value_counts()
        print(f"external rows added to TRAINING only: +{len(external):,} "
              f"({len(training):,} -> {len(merged):,})")
        for label, n in counts.items():
            print(f"    {target}={label:<24} +{n}")
        _warn_if_imbalance_worsens(training, merged, target)
    return merged


def _warn_if_imbalance_worsens(before: pd.DataFrame, after: pd.DataFrame,
                               target: str) -> None:
    """Say plainly whether these rows helped or hurt this target's balance.

    The external rows are entirely `Women` / `Spring` / `Casual`, which lands very
    differently depending on the task:

        season  Spring is the RAREST class  -> balance improves a lot
        gender  Women is the 2nd LARGEST    -> nothing gained
        usage   Casual is the MAJORITY 77%  -> imbalance gets WORSE

    Adding majority-class rows to a task whose whole difficulty is a long tail is a
    real mistake, and it is invisible unless someone prints it. So print it.
    """
    b = before[target].value_counts()
    a = after[target].value_counts()
    if len(b) < 2:
        return
    r_before, r_after = b.max() / b.min(), a.max() / a.min()
    top_before = b.index[0]
    added_to_majority = set(after[target].value_counts().index[:1]) == {top_before} \
        and a[top_before] > b[top_before]

    if r_after > r_before * 1.001:
        print(f"\n    !! WARNING: this made {target!r} MORE imbalanced, "
              f"not less: max/min {r_before:,.0f}x -> {r_after:,.0f}x")
        if added_to_majority:
            print(f"       Every external row is {target}={top_before!r}, which is "
                  f"already the majority class "
                  f"({100 * b[top_before] / b.sum():.1f}% -> "
                  f"{100 * a[top_before] / a.sum():.1f}%).")
        print("       The rare classes gained nothing. Consider training this target "
              "WITHOUT the external data,")
        print("       and report the comparison - a negative result measured is worth "
              "more than one avoided.")
    elif r_after < r_before * 0.999:
        print(f"    balance improved: max/min {r_before:,.0f}x -> {r_after:,.0f}x")
    else:
        print(f"    balance unchanged: max/min {r_before:,.0f}x. These rows land in a "
              f"class that was not starved, so expect little effect on {target!r}.")

    if target == "season":
        print("    NOTE: every external row is season=Spring. Before claiming Task 2 "
              "improved, measure Spring")
        print("       recall separately on Personal Care and non-Personal-Care rows - "
              "see this module's docstring.")


def load_eval_set(verbose: bool = True) -> pd.DataFrame:
    """Load the INDEPENDENT EVALUATION set. Never train on these images.

    261 openly-licensed in-the-wild photographs, gated to zero overlap with the
    provided train and test sets, labelled for all four targets by hand.

    Their whole value is that no model in this project has seen them. Training on
    them once destroys that permanently and makes the report's independent-evaluation
    claim false, so they are returned by a separate function that no training path
    calls, and every row is tagged `is_eval_only=True`.

    Known limits, to carry into the report: `season` skews Summer (74%, against 49.6%
    in the provided data) because season-agnostic accessories were defaulted to the
    modal value - so `season` here measures the annotator's prior as much as the
    model. There are no Spring images at all. Conversely `usage=Home` has 4 images
    against 1 in the entire provided training set.
    """
    folder, csv_name = EVAL_SET
    base = find_external_root(folder) / folder
    frame = pd.read_csv(base / csv_name)
    frame["filename"] = frame["id"].astype(str) + ".jpg"
    frame["path"] = frame["filename"].map(lambda n: str(base / "images" / n))
    frame["group_id"] = _GROUP_PREFIX + frame["id"].astype(str)
    frame["is_eval_only"] = True

    present = frame["path"].map(lambda p: Path(p).is_file())
    if not present.all():
        raise FileNotFoundError(
            f"{(~present).sum()} of {len(frame)} evaluation images are missing under "
            f"{base / 'images'}. Re-download that folder from Drive."
        )

    if verbose:
        print(f"independent evaluation set: {len(frame)} images, "
              f"{frame['articleType'].nunique()} articleType classes")
        print("    EVALUATION ONLY - do not train on these.")
    return frame


def summary() -> pd.DataFrame:
    """One row per external set, for a report table. Never raises if data is absent."""
    rows = []
    for key, (folder, csv_name, purpose) in TRAINING_SETS.items():
        try:
            frame = _load_one(key)
            rows.append({"set": key, "role": "training", "images": len(frame),
                         "classes": frame["articleType"].nunique(), "purpose": purpose})
        except (FileNotFoundError, ValueError) as exc:
            rows.append({"set": key, "role": "training", "images": None,
                         "classes": None, "purpose": f"UNAVAILABLE: {exc}"})
    try:
        ev = load_eval_set(verbose=False)
        rows.append({"set": "eval", "role": "evaluation only", "images": len(ev),
                     "classes": ev["articleType"].nunique(),
                     "purpose": "independent evaluation, never trained on"})
    except FileNotFoundError as exc:
        rows.append({"set": "eval", "role": "evaluation only", "images": None,
                     "classes": None, "purpose": f"UNAVAILABLE: {exc}"})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(summary().to_string(index=False))
