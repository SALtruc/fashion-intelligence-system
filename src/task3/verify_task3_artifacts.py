"""Check that the submitted Task 3 artifacts are the ones the metadata describes.

Two reasons this exists.

The metadata file records a sha256 for the checkpoint and one for the predictions, but
no script in the repository produces the second one: it came from a helper that was
deleted during the restructure. A hash nobody can recompute is a claim, not a check, and
this file is part of the submission.

And the predictions hash cannot be reproduced on a machine that checks out with Unix
line endings. Git stores the CSV with LF (119,328 bytes); a Windows working copy has
CRLF (125,158 bytes); the recorded hash is of the second. The content is identical, but
someone verifying on a Mac gets a different digest and reasonably concludes the file is
wrong. So both digests are computed and the answer says which normalisation matched
rather than just "no".

The checkpoint is not in git -- artifacts/** is ignored by team convention and the
weights live on the shared Drive -- so a missing .pt is reported as "not present here",
not as a failure. That distinction is the one that matters when the zip is assembled:
zipping the repository alone omits every task's model file.

    python src/task3/verify_task3_artifacts.py
"""
import hashlib
import io
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
META = ROOT / "results" / "task3" / "task3_final_metadata.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL' if ok is False else '--'}] {label}"
          + (f"   {detail}" if detail else ""))
    return ok


if not META.is_file():
    raise SystemExit(f"{META} not found")
m = json.loads(io.open(META, encoding="utf-8").read())
print(f"metadata: {META.relative_to(ROOT)}")
print(f"  model trained at commit {m.get('git_commit', '?')[:12]}, "
      f"design: {m.get('design', '?')[:40]}")
print()

results = []

# ------------------------------------------------------------------- the checkpoint
print("checkpoint")
rel = m.get("model") or m.get("artifact")
ck = (ROOT / rel) if rel else None
if ck is None or not ck.is_file():
    check(rel or "no model path recorded", None,
          "not present here; expected on the shared Drive, and it must be added to "
          "the submission zip by hand")
else:
    b = ck.read_bytes()
    results.append(check(f"{ck.relative_to(ROOT)} sha256",
                         sha(b) == m.get("sha256"),
                         f"{sha(b)[:16]}..."))
    results.append(check("size_bytes", len(b) == m.get("size_bytes"),
                         f"{len(b):,}"))

# ------------------------------------------------------------------ the predictions
print("\npredictions")
pr = ROOT / m["predictions"]
if not pr.is_file():
    results.append(check(f"{m['predictions']}", False, "not found"))
else:
    raw = pr.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    rec = m.get("predictions_sha256", "")
    which = ("this working copy" if sha(raw) == rec else
             "LF-normalised" if sha(lf) == rec else
             "CRLF-normalised" if sha(crlf) == rec else None)
    results.append(check("predictions_sha256", which is not None,
                         f"matches {which}" if which else
                         f"no normalisation matches {rec[:16]}..."))
    if which and which != "this working copy":
        print(f"       (this checkout differs from the recorded digest only in line "
              f"endings; the recorded one is of the {which} bytes)")
    rows = lf.decode("utf-8").rstrip("\n").split("\n")
    header, body = rows[0], rows[1:]
    results.append(check("row count", len(body) > 0, f"{len(body):,} data rows"))
    ids = [r.split(",")[0] for r in body]
    results.append(check("ids unique", len(set(ids)) == len(ids),
                         f"{len(set(ids)):,} distinct"))
    results.append(check("header", "," in header, header[:60]))
    # The CSV is the team's shared file: this task fills two of its four label
    # columns and leaves the other two for their owners. An empty column here is
    # correct, and a filled one would mean someone's work had been overwritten.
    cols = header.split(",")
    mine = m.get("columns_filled", [])
    theirs = m.get("columns_left_for_teammates", [])
    for c in mine:
        i = cols.index(c) if c in cols else -1
        n = sum(1 for r in body if i >= 0 and r.split(",")[i].strip())
        results.append(check(f"{c} filled", i >= 0 and n == len(body),
                             f"{n:,}/{len(body):,}"))
    for c in theirs:
        i = cols.index(c) if c in cols else -1
        n = sum(1 for r in body if i >= 0 and r.split(",")[i].strip())
        results.append(check(f"{c} left for its owner", n == 0,
                             "empty" if n == 0 else f"{n:,} rows filled -- someone "
                                                    f"else's column was written to"))

# ------------------------------------------------- the numbers the report quotes
print("\nreported metrics")
for t in ("gender", "usage"):
    v = m.get("val_macro_f1", {}).get(t)
    results.append(check(f"val_macro_f1.{t}", v is not None, f"{v}"))

print()
bad = [r for r in results if r is False]
if bad:
    raise SystemExit(f"{len(bad)} check(s) failed")
print(f"all {len(results)} checks passed")
print("note: zipping the repository does not include the checkpoint. artifacts/** is "
      "gitignored,\n      so every task's model file has to be added to the "
      "submission zip from the Drive.")
