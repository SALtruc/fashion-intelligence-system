"""Read a Task 3 experiment's numbers, from its own CSV or from the consolidated table.

The report generators used to read the per-experiment CSVs directly, which works on the
laptop that ran the experiments and nowhere else: .gitignore admits only
task3_all_results.csv, so a fresh checkout has none of them and the appendices cannot be
regenerated. The spec requires the assignment to run from the code in the submitted zip,
so "it works here" is not a defence.

`load_wide` returns the same wide frame either way. When the per-experiment CSV is
present it is read unchanged, so nothing about the generators' output moves. When it is
absent the frame is rebuilt from the consolidated long table by inverting the melt.

Column naming, which is the only subtle part. The melt turns a column called
"gender macroF1" into target="gender", metric="macroF1", and leaves a column called
"f1" on a per-class row as metric="f1" with target="gender" already occupying the key.
So the inverse depends on whether `target` is one of the frame's keys: if it is, the
column name is the metric alone; if it is not, target and metric are joined back
together. Getting this backwards silently produces a frame with the right shape and the
wrong column names, which is why load_wide checks its own output against the columns
the caller says it needs.
"""
import io
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
RES = ROOT / "predictions" / "task3"
CONSOLIDATED = RES / "task3_all_results.csv"
BOOLS = ("tta",)


def load_wide(name, source, keys, needs=(), optional=False):
    """The experiment's own CSV if present, else rebuilt from the consolidated table.

    `keys` maps the experiment CSV's own key columns onto the schema's, exactly as the
    melt in consolidate_results.py does -- {"id": "arm", "kind": "variant"} and so on.
    A plain list means the names already agree. Passing it matters because the raw CSV
    calls a column `id` where the consolidated table calls it `arm`; without the
    rename the two paths would return frames of the same shape under different names,
    which is precisely the failure this module exists to prevent.

    `needs` are column names the caller will index by name; if the frame cannot
    provide one of them it raises rather than failing later with a KeyError somewhere
    less informative.
    """
    keymap = {k: k for k in keys} if not isinstance(keys, dict) else dict(keys)
    f = RES / f"{name}.csv"
    if f.is_file():
        d = pd.read_csv(f).rename(columns=keymap)
    else:
        # optional=True is for data a section can do without: the caller decides what
        # to leave out rather than dying, and still refuses to publish a section whose
        # conclusion depends on the missing half.
        if not CONSOLIDATED.is_file():
            if optional:
                return None
            raise SystemExit(f"neither {name}.csv nor {CONSOLIDATED.name} is present")
        long = pd.read_csv(CONSOLIDATED, dtype=str).fillna("")
        if "source" not in long.columns:
            raise SystemExit(f"{CONSOLIDATED.name} has no `source` column; it predates "
                             f"consolidate_results.py -- run that first")
        long = long[long.source == source]
        if long.empty and optional:
            return None
        if long.empty:
            raise SystemExit(f"{name}.csv is absent and {CONSOLIDATED.name} holds no "
                             f"rows for source {source!r}; run "
                             f"consolidate_results.py")
        target_is_key = "target" in keymap.values()
        long = long.assign(_col=[
            m if (target_is_key or not t) else f"{t} {m}"
            for t, m in zip(long["target"], long["metric"])])
        idx = [k for k in keymap.values() if (long[k] != "").any()]
        if not idx:
            raise SystemExit(f"none of {sorted(keymap.values())} is populated "
                             f"for source {source!r}")
        d = (long.pivot_table(index=idx, columns="_col", values="value",
                              aggfunc="last")
             .reset_index().rename_axis(None, axis=1))

        # pivot_table sorts its index, which silently reorders the rows of every table
        # a generator prints. B9's table runs frozen, fine-tuned, ours -- weakest to
        # strongest, with the prose underneath referring to "its row" -- and sorting
        # alphabetically puts ours first. So the rows are restored to the order they
        # first appear in the long table, which is the order the experiment recorded
        # them. Done before the dtype coercion below, while both sides are strings.
        first = {}
        for i, tup in enumerate(zip(*[long[k].astype(str) for k in idx])):
            first.setdefault(tup, i)
        d = (d.assign(_ord=[first.get(tuple(str(v) for v in row), len(first))
                            for row in d[idx].values])
             .sort_values("_ord", kind="stable")
             .drop(columns="_ord").reset_index(drop=True))

        for c in d.columns:
            if c in BOOLS:
                d[c] = d[c].map({"True": True, "False": False, True: True,
                                 False: False})
            elif c not in idx or c in ("rep",):
                num = pd.to_numeric(d[c], errors="coerce")
                if num.notna().all():
                    d[c] = num
    missing = [c for c in needs if c not in d.columns]
    if missing:
        raise SystemExit(f"{name}: rebuilt frame is missing {missing}; it has "
                         f"{sorted(d.columns)}")
    return d


def forward_repeats(target, epochs=20):
    """The three 20-epoch forward-split repeats of design C, from either schema.

    Reads the consolidated table, which has held these rows since the per-experiment
    CSV was folded in and removed. Accepts both the current schema, where the epoch
    count sits in `variant` as "epochs=20.0", and the previous one, where `run` packed
    "<epochs> <repeat>" into a single string -- an old checkout should still be able to
    regenerate the appendix that quotes these numbers.
    """
    d = pd.read_csv(CONSOLIDATED, dtype=str).fillna("")
    d = d[d.source == "epochs_confirmation"]
    if d.empty:
        raise SystemExit(f"{CONSOLIDATED.name} holds no epochs_confirmation rows")
    if "variant" in d.columns and (d["variant"] != "").any():
        ep = d["variant"].str.replace("epochs=", "", regex=False)
    elif "run" in d.columns:
        ep = d["run"].str.split().str[0]
    else:
        raise SystemExit("cannot find the epoch count in either schema")
    keep = d[ep.astype(float) == float(epochs)]
    vals = [float(v) for v in keep[keep.metric == f"{target} macroF1"]["value"]]
    if len(vals) != 3:
        raise SystemExit(f"expected 3 forward repeats for {target} at {epochs} "
                         f"epochs, found {len(vals)}")
    return vals

def report_path():
    """Where the appendix file lives, tolerating either capitalisation.

    The notebook folder was renamed Task3 -> task3 on main. Windows checkouts have
    core.ignorecase set, so git tracks the lowercase name while the filesystem still
    reports the old one; a hardcoded path is wrong on one platform or the other. The
    directory is resolved by looking, not by assuming.
    """
    nb = ROOT / "notebooks"
    for name in ("task3", "Task3"):
        if (nb / name).is_dir():
            return nb / name / "REPORT_TASK3.md"
    return nb / "task3" / "REPORT_TASK3.md"


def read_report():
    """The appendix file's text, or an instruction instead of a traceback.

    The file was removed from main in commit 69cd1931, right after the
    notebooks/Task3 -> notebooks/task3 rename, so a generator run with --append now
    fails. Whether it should come back is a team decision, not this module's, but the
    person who hits the error should be told what happened rather than shown a
    FileNotFoundError from inside a read call.
    """
    p = report_path()
    if not p.is_file():
        raise SystemExit("\n".join([
            f"{p} does not exist.",
            "It was deleted from main in commit 69cd1931 (message: fixxed), right after",
            "the notebooks/Task3 to notebooks/task3 rename.",
            f"To bring it back:  git show 4bbe7956:notebooks/Task3/REPORT_TASK3.md > {p}",
            "Or drop --append and paste the section this script printed into whichever",
            "document the report now lives in."]))
    return io.open(p, encoding="utf-8").read().replace("\r\n", "\n")
