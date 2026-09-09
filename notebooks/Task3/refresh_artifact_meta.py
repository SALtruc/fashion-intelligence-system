"""Bring the already-trained Task 3 artifact up to the team's handover schema, and
put the checkpoint where git will keep it.

Three things were wrong with what was on disk, and none needed a retrain:

  1. `task3_final_metadata.json` recorded the metrics but none of the fields
     artifacts/README.md calls essential -- git_commit, split, preprocessing, labels.
  2. The checkpoint's own `trained_on` block still said `seeds_tried`/`seed_shipped`,
     the wording that was retracted everywhere else: train_model() re-seeds from the
     module-level SEED on entry, so those three runs were repeats at seed 42, not
     three seeds. The generator script was corrected; the saved file was not.
  3. The checkpoint lived only under `artifacts/`, which is gitignored on every
     branch, so it was the one deliverable of four that had to be fetched from Drive
     separately. `models/` is ignored on no branch and is where Task 1 and Task 2
     already keep their checkpoints -- at 1.2 MB this belongs beside them.

The weights are not touched. The script proves that by comparing every tensor before
and after the round-trip, and refuses to replace the file if any differ.

    python notebooks/Task3/refresh_artifact_meta.py

Idempotent: running it twice changes nothing and produces the same sha256.
"""
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git found above {HERE}")


REPO = _repo_root()
NAME = "task3_gender_usage_C_weighted.pt"
META_NAME = "task3_final_metadata.json"

CKPT = REPO / "models" / "task3" / "checkpoints"   # tracked in git
ART = REPO / "artifacts" / "task3"                 # gitignored, mirrors to Drive
PRED = REPO / "predictions" / "task3_gender_usage_nguyen.csv"

for d in (CKPT, ART):
    d.mkdir(parents=True, exist_ok=True)

# Prefer whichever copy exists; the tracked one wins if both do.
MODEL = next((p for p in (CKPT / NAME, ART / NAME) if p.is_file()), None)
if MODEL is None:
    raise SystemExit(f"no checkpoint found at {CKPT / NAME} or {ART / NAME}")


def git(*a):
    try:
        return subprocess.run(("git", *a), cwd=str(REPO), capture_output=True,
                              text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


ck = torch.load(MODEL, map_location="cpu", weights_only=False)
before = {k: v.clone() for k, v in ck["state_dict"].items()}
print(f"loaded {MODEL.relative_to(REPO)}: {len(before)} tensors, "
      f"{sum(v.numel() for v in before.values()):,} values")

# ---------------------------------------------------------------- fix the wording
tr = ck["trained_on"]
if "seeds_tried" in tr:
    tr["repeats_tried"] = tr.pop("seeds_tried")
    tr["repeat_shipped"] = tr.pop("seed_shipped")
    tr["seeding"] = ("train_model() calls torch.manual_seed(SEED) on entry, so these "
                     "are repeats at SEED=42 differing by cuDNN nondeterminism, NOT "
                     "independent seeds. The labels 42/43/44 name the repeats only.")
    print("  renamed seeds_tried -> repeats_tried, and recorded why")
else:
    print("  wording already correct")
tr["metric"] = ("macro-F1 with labels= over all classes, including any absent from "
                "validation; sklearn's default would drop them and read higher")

tmp = (CKPT / NAME).with_suffix(".pt.tmp")
torch.save(ck, tmp)

# ------------------------------------------------- prove the weights are unchanged
back = torch.load(tmp, map_location="cpu", weights_only=False)["state_dict"]
assert set(back) == set(before), "the tensor set changed"
bad = [k for k in before if not torch.equal(back[k], before[k])]
if bad:
    tmp.unlink()
    raise SystemExit(f"REFUSING to replace: {len(bad)} tensors differ -- {bad[:3]}")
print(f"  verified: all {len(before)} tensors bit-identical after the round-trip")

MODEL = CKPT / NAME
shutil.move(str(tmp), str(MODEL))
shutil.copy2(MODEL, ART / NAME)
print(f"  checkpoint -> {MODEL.relative_to(REPO)}   (tracked)")
print(f"             -> {(ART / NAME).relative_to(REPO)}   (gitignored, for Drive)")

# ------------------------------------------------------------------- the metadata
# The commit that PRODUCED the weights, not today's HEAD. finalise_task3.py ran on
# 08/09 and its output landed in one commit; HEAD has moved several commits since,
# and recording HEAD here would claim the model came from code it never saw.
TRAIN_COMMIT = git("log", "-1", "--format=%H", "--",
                   "predictions/task3_gender_usage_nguyen.csv")
meta = {
    "artifact": f"models/task3/checkpoints/{NAME}",
    "task": "task3",
    # the date the weights were made, not the date this metadata was rewritten
    "created_at": git("log", "-1", "--format=%ad", "--date=short", TRAIN_COMMIT),
    "metadata_regenerated_on": datetime.date.today().isoformat(),
    "git_commit": TRAIN_COMMIT,
    "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
    "metadata_regenerated_at": git("rev-parse", "HEAD"),
    "dataset_version": "A2_FashionDataset as provided, images 60x80 RGB, "
                       f"{tr['n_train'] + tr['n_val']:,} labelled rows",
    "split": f"splits/{tr['split_file']}",
    "preprocessing": "embedded in the checkpoint: image_size, channel_mean, "
                     "channel_std. No separate transformer file.",
    "labels": "embedded in the checkpoint under 'classes'; head order is that list",
    "framework": f"PyTorch {torch.__version__}",
    "sha256": sha(MODEL),
    "size_bytes": MODEL.stat().st_size,
    "validation_metrics": {t: {"macro_f1": ck["val_macro_f1"][t],
                               "accuracy": ck["val_accuracy"][t]}
                           for t in ck["heads"]},
    "notes": ("python notebooks/Task3/finalise_task3.py -- design C (one shared conv "
              "body, one head per target), class-weighted loss 1/sqrt(n), mirror TTA, "
              f"{tr['epochs']} epochs, batch 256, Adam lr 0.001. Shipped checkpoint "
              f"is the MEDIAN of {len(tr['repeats_tried'])} repeats by mean macro-F1, "
              "not the best. " + tr["seeding"] + " " + tr["metric"] + "."),

    "model": f"models/task3/checkpoints/{NAME}",
    "also_on_drive": f"artifacts/task3/{NAME}",
    "in_git": True,
    "predictions": "predictions/task3_gender_usage_nguyen.csv",
    "predictions_sha256": sha(PRED),
    "design": ck["design"],
    "tta_used": ck["use_tta"],
    "classes": ck["classes"],
    "val_macro_f1": ck["val_macro_f1"],
    "val_macro_f1_over_classes_present": ck["val_macro_f1_in_val_classes"],
    "val_accuracy": ck["val_accuracy"],
    "n_test": sum(1 for _ in PRED.open(encoding="utf-8")) - 1,
    "columns_filled": ["gender", "usage"],
    "columns_left_for_teammates": ["articleType", "season"],
}

# Every field artifacts/README.md calls required must actually be filled in.
required = ["artifact", "task", "created_at", "git_commit", "dataset_version",
            "split", "preprocessing", "labels", "framework", "validation_metrics",
            "notes"]
missing = [k for k in required if not meta.get(k)]
if missing:
    raise SystemExit(f"metadata still missing required fields: {missing}")

# One dict, three files, so the copies cannot drift apart.
blob = json.dumps(meta, indent=2)
print()
for d in (CKPT, ART, HERE):
    (d / META_NAME).write_text(blob, encoding="utf-8")
    print(f"metadata -> {(d / META_NAME).relative_to(REPO)}")

print(f"\n  all {len(required)} required fields present")
print(f"  model sha256   {meta['sha256']}")
print(f"  preds sha256   {meta['predictions_sha256']}")
print(f"  trained at     {meta['git_commit'][:8]} on {meta['git_branch']}")
print(f"  meta refreshed {meta['metadata_regenerated_at'][:8]}")
print(f"\nThe checkpoint is in git now, so the Drive copy is a convenience, not the "
      f"only copy. If you keep Drive in sync, re-upload: the sha256 changed when the "
      f"seeds/repeats wording was corrected.")
