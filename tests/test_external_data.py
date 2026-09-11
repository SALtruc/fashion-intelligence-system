"""Self-test for src/external_data.py.

Run it before trusting a training run that uses the external data:

    python tests/test_external_data.py

It needs the images. Download `A2_ExternalData` from the team Drive and either put
the three folders in `Dataset/`, or point at them:

    A2_EXTERNAL_DATA=/path/to/A2_ExternalData python tests/test_external_data.py

The catalogue side is simulated with a frame of the shape `make_split()` returns,
because `preprocessed_datasets/train_manifest.csv` is gitignored and absent on a
fresh clone. That is enough: the contract under test is "external rows align onto a
training frame and nothing reaches validation", not anything about the catalogue's
contents.
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
import external_data as ed  # noqa: E402

TARGETS = ["articleType", "season", "gender", "usage"]
N_TRAIN_EXPECTED = 1899
N_EVAL_EXPECTED = 261

try:
    ed.find_external_root("ExternalCosmetics")
except FileNotFoundError as exc:
    print("SKIPPED - the external images are not on this machine.\n")
    print(exc)
    # Run directly, SystemExit is the right way to stop. Under pytest it is not:
    # raising SystemExit while pytest imports a module ends the whole session with
    # INTERNALERROR and takes every other test file down with it, so a machine
    # without the images cannot run the test suite at all. pytest is imported here
    # rather than at the top so the file still runs as a plain script without it.
    if "pytest" in sys.modules:
        import pytest

        pytest.skip("the external images are not on this machine",
                    allow_module_level=True)
    raise SystemExit(0)


def fake_split(n=100):
    """Mimic preprocessing.make_split()'s two outputs."""
    rows = [{
        "filename": f"{1000 + i}.jpg",
        "group_id": f"g{i}",
        "path": f"/fake/{1000 + i}.jpg",
        "articleType": "Tshirts" if i % 2 else "Watches",
        "season": "Summer", "gender": "Men", "usage": "Casual",
    } for i in range(n)]
    frame = pd.DataFrame(rows)
    return frame.iloc[:80].reset_index(drop=True), frame.iloc[80:].reset_index(drop=True)


ok = 0

print("1. add_to_training appends only to training, validation untouched")
training, validation = fake_split()
before_tr, before_va = len(training), len(validation)
merged = ed.add_to_training(training, "articleType")
assert len(validation) == before_va, "validation changed"
assert len(training) == before_tr, "caller's frame was mutated"
assert len(merged) == before_tr + N_TRAIN_EXPECTED, \
    f"expected +{N_TRAIN_EXPECTED}, got {len(merged) - before_tr}"
print(f"   train {before_tr} -> {len(merged)}, val stayed {before_va}   OK\n"); ok += 1

print("2. no group_id collision, and external ids are distinguishable")
assert merged["group_id"].is_unique, "group_id not unique after merge"
ext_rows = merged[merged["group_id"].str.startswith("ext:")]
assert len(ext_rows) == N_TRAIN_EXPECTED
print(f"   {len(ext_rows)} external rows all prefixed 'ext:', ids unique   OK\n"); ok += 1

print("3. make_split's uniqueness assertion would still pass on the merged frame")
assert merged["group_id"].is_unique
print("   OK\n"); ok += 1

print("4. every external image path resolves on disk")
missing = [p for p in ext_rows["path"] if not Path(p).is_file()]
assert not missing, f"{len(missing)} missing, e.g. {missing[:3]}"
print(f"   all {len(ext_rows)} files present   OK\n"); ok += 1

print("5. each target filters correctly and all four are usable")
for t in TARGETS:
    e = ed.load_external_training(target=t)
    assert len(e) == N_TRAIN_EXPECTED, f"{t}: {len(e)}"
    assert e[t].notna().all()
print("   articleType / season / gender / usage all 1899 rows   OK\n"); ok += 1

print("6. season is entirely Spring (the bias the docstring warns about)")
e = ed.load_external_training()
assert set(e["season"]) == {"Spring"}, set(e["season"])
print(f"   season values: {set(e['season'])}   OK\n"); ok += 1

print("7. a frame that is NOT from make_split is rejected")
try:
    ed.add_to_training(pd.DataFrame({"articleType": ["Tshirts"]}), "articleType")
    raise AssertionError("should have refused a frame with no path/group_id")
except ValueError as exc:
    print(f"   refused: {str(exc)[:70]}...   OK\n"); ok += 1

print("8. an unknown target is rejected")
try:
    ed.add_to_training(training, "baseColour")
    raise AssertionError("should have refused an unknown target")
except KeyError:
    print("   refused unknown target   OK\n"); ok += 1

print("9. eval set loads, is tagged, and is NOT reachable from add_to_training")
ev = ed.load_eval_set(verbose=False)
assert len(ev) == N_EVAL_EXPECTED, len(ev)
assert ev["is_eval_only"].all()
assert "eval" not in ed.TRAINING_SETS
assert not (set(ev["group_id"]) & set(merged["group_id"])), "eval images leaked into training!"
print(f"   {len(ev)} eval images, tagged is_eval_only, zero overlap with training   OK\n")
ok += 1

print("10. a bad set name is rejected rather than silently ignored")
try:
    ed.load_external_training(sets=("cosmetics9",))
    raise AssertionError("should have refused")
except KeyError:
    print("   refused unknown set   OK\n"); ok += 1

print("11. a missing data folder gives an actionable error")
import os  # noqa: E402
old_env, old_root = os.environ.get("A2_EXTERNAL_DATA"), ed._REPO_ROOT
os.environ["A2_EXTERNAL_DATA"] = str(Path("definitely_not_here").resolve())
ed._REPO_ROOT = Path("definitely_not_here").resolve()
try:
    ed.find_external_root("ExternalCosmetics")
    raise AssertionError("should have raised")
except FileNotFoundError as exc:
    assert "A2_EXTERNAL_DATA" in str(exc) and "Drive" in str(exc)
    print("   error names the env var and the Drive fix   OK\n"); ok += 1
finally:
    ed._REPO_ROOT = old_root
    if old_env is None:
        os.environ.pop("A2_EXTERNAL_DATA", None)
    else:
        os.environ["A2_EXTERNAL_DATA"] = old_env

print(f"ALL {ok} CHECKS PASSED")
