"""Bring the already-trained Task 3 artifact up to the team's handover schema.

Two things are wrong with what is on disk, and neither needs a retrain:

  1. `task3_final_metadata.json` records the metrics but none of the fields
     artifacts/README.md calls essential -- git_commit, split, preprocessing, labels.
  2. The checkpoint's own `trained_on` block still says `seeds_tried`/`seed_shipped`,
     the wording that was retracted everywhere else: train_model() re-seeds from the
     module-level SEED on entry, so those three runs were repeats at seed 42, not
     three seeds. The generator script was corrected; the saved file was not.

The weights are not touched. The script proves that by comparing every tensor
before and after the round-trip, and refuses to replace the file if any differ.
"""
import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import torch

REPO = Path("D:/g2_pr")
ART = REPO / "artifacts" / "task3"
MODEL = ART / "task3_gender_usage_C_weighted.pt"
PRED = REPO / "predictions" / "task3_gender_usage_nguyen.csv"
NB = REPO / "notebooks" / "Task3"


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
print(f"loaded {MODEL.name}: {len(before)} tensors, "
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

tmp = MODEL.with_suffix(".pt.tmp")
torch.save(ck, tmp)

# ------------------------------------------------- prove the weights are unchanged
back = torch.load(tmp, map_location="cpu", weights_only=False)["state_dict"]
assert set(back) == set(before), "the tensor set changed"
bad = [k for k in before if not torch.equal(back[k], before[k])]
if bad:
    tmp.unlink()
    raise SystemExit(f"REFUSING to replace: {len(bad)} tensors differ -- {bad[:3]}")
print(f"  verified: all {len(before)} tensors bit-identical after the round-trip")
shutil.move(str(tmp), str(MODEL))

# ------------------------------------------------------------------- the metadata
# The commit that PRODUCED the weights, not today's HEAD. finalise_task3.py ran on
# 08/09 and its output landed in one commit; HEAD has moved four commits since, and
# recording HEAD here would claim the model came from code it never saw.
TRAIN_COMMIT = git("log", "-1", "--format=%H", "--",
                   "predictions/task3_gender_usage_nguyen.csv")
meta = {
    "artifact": "artifacts/task3/task3_gender_usage_C_weighted.pt",
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

    "model": "artifacts/task3/task3_gender_usage_C_weighted.pt",
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

blob = json.dumps(meta, indent=2)
(ART / "task3_final_metadata.json").write_text(blob, encoding="utf-8")
(NB / "task3_final_metadata.json").write_text(blob, encoding="utf-8")

# Every field the README calls required must actually be filled in.
required = ["artifact", "task", "created_at", "git_commit", "dataset_version",
            "split", "preprocessing", "labels", "framework", "validation_metrics",
            "notes"]
missing = [k for k in required if not meta.get(k)]
if missing:
    raise SystemExit(f"metadata still missing required fields: {missing}")
print(f"\n  all {len(required)} required fields present")
print(f"  model sha256   {meta['sha256']}")
print(f"  preds sha256   {meta['predictions_sha256']}")
print(f"  trained at     {meta['git_commit'][:8]} on {meta['git_branch']}")
print(f"  meta refreshed {meta['metadata_regenerated_at'][:8]}")
print(f"\nwrote {ART / 'task3_final_metadata.json'}")
print(f"      {NB / 'task3_final_metadata.json'}   (tracked in git)")
print(f"\nRE-UPLOAD {MODEL.name} to the team Drive -- its sha256 changed.")
