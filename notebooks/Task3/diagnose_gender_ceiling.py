"""Where is `gender` actually losing macro-F1, and is there a ceiling like `usage` has?

The report proves `usage` sits at its annotation ceiling: an articleType -> modal-usage
oracle scores 0.3872, *below* the CNN's 0.4676, and 0.0000 on four classes. That is why
no amount of tuning moved it. `gender` is the opposite case -- the image is worth +10.6
accuracy points over the best metadata oracle -- so before declaring Task 3 final it is
worth knowing where its 0.72 is lost, and whether that loss is fixable or is the same
kind of label ceiling.

This trains nothing. It loads the shipped checkpoint and scores it per class on the
shared validation split, then measures two ceilings for comparison:

  * an `articleType -> modal gender` oracle, the same instrument used on `usage`
  * the confusion matrix, to see whether the loss is one class or spread

    python notebooks/Task3/diagnose_gender_ceiling.py
"""
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
SRC = next(p for p in (ROOT / "src" / "task3_build.py", HERE / "task3_build.py")
           if p.is_file())
CKPT = ROOT / "models" / "task3" / "checkpoints" / "task3_gender_usage_C_weighted.pt"
PREFIX_STOP = "# ## 5 "


def run_prefix():
    text = io.open(SRC, encoding="utf-8").read().replace("\r\n", "\n")
    parts = re.split(r"(?m)^# %%(.*)$", text)
    cells = [("markdown" if "markdown" in parts[k] else "code", parts[k + 1])
             for k in range(1, len(parts), 2)]
    ns = {"__name__": "__notebook__"}
    for i, (kind, body) in enumerate(cells):
        if kind == "markdown":
            if body.strip().startswith(PREFIX_STOP):
                return ns
            continue
        exec(compile(body, f"<task3_build cell {i}>", "exec"), ns)
    raise SystemExit("section 5 marker not found")


print("executing the notebook prefix (preprocessing, images to GPU) ...")
t0 = time.time()
g = run_prefix()
print(f"prefix done in {time.time() - t0:.0f}s")

TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
tr, va = g["splits"][("gender", "primary")]
print(f"shared val: {len(va):,} rows")

ck = torch.load(CKPT, map_location=g["DEVICE"], weights_only=False)
model = g["Net"](heads).to(g["DEVICE"])
model.load_state_dict(ck["state_dict"])
model.eval()
print(f"loaded the shipped checkpoint (TTA={ck['use_tta']})")


@torch.no_grad()
def predict_rows(model, fr, bs=512, tta=False):
    rows_t = torch.as_tensor(fr["_row"].values, device=g["X_t"].device)
    acc = {k: [] for k in heads}
    for i in range(0, len(rows_t), bs):
        x = g["batch_x"](rows_t[i:i + bs])
        o = model(x)
        if tta:
            f = model(torch.flip(x, dims=[-1]))
            o = {k: ((o[k].softmax(1) + f[k].softmax(1)) / 2).log() for k in heads}
        for k in heads:
            acc[k].append(o[k].argmax(1).cpu())
    return {k: np.asarray(CLASSES[k])[torch.cat(v).numpy()] for k, v in acc.items()}


pred = predict_rows(model, va, tta=bool(ck["use_tta"]))

from sklearn.metrics import classification_report, confusion_matrix, f1_score

for t in TARGETS:
    print(f"\n{'=' * 68}\n=== {t}: the shipped model, per class ===")
    rep = classification_report(va[t], pred[t], labels=CLASSES[t],
                                output_dict=True, zero_division=0)
    tbl = pd.DataFrame({
        "train n": [int((tr[t] == c).sum()) for c in CLASSES[t]],
        "val n": [int(rep[c]["support"]) for c in CLASSES[t]],
        "precision": [round(rep[c]["precision"], 3) for c in CLASSES[t]],
        "recall": [round(rep[c]["recall"], 3) for c in CLASSES[t]],
        "f1": [round(rep[c]["f1-score"], 3) for c in CLASSES[t]],
    }, index=CLASSES[t])
    print(tbl.to_string())
    macro = f1_score(va[t], pred[t], labels=CLASSES[t], average="macro",
                     zero_division=0)
    print(f"  macro-F1 {macro:.4f}   accuracy "
          f"{(np.asarray(pred[t]) == va[t].values).mean():.4f}")

    # How much macro-F1 would a perfect fix of the single worst class buy?
    f1s = {c: rep[c]["f1-score"] for c in CLASSES[t]}
    worst = min(f1s, key=f1s.get)
    lifted = (sum(f1s.values()) - f1s[worst] + 1.0) / len(f1s)
    print(f"  worst class: {worst} (F1 {f1s[worst]:.3f}). Perfecting it alone would "
          f"give macro-F1 {lifted:.4f}  ({lifted - macro:+.4f})")

print(f"\n{'=' * 68}\n=== gender confusion (rows = truth) ===")
cm = pd.DataFrame(confusion_matrix(va["gender"], pred["gender"],
                                   labels=CLASSES["gender"]),
                  index=CLASSES["gender"], columns=CLASSES["gender"])
print(cm.to_string())

# ------------------------------------------------------- the annotation ceiling
# The same instrument that showed `usage` was at its label ceiling. If articleType
# predicts gender about as well as the CNN, the labels are the limit; if the CNN is
# far ahead, the pixels carry information and capacity may still be the constraint.
print(f"\n{'=' * 68}\n=== ceiling check: articleType -> modal gender ===")
frame = g["frame"]
key = "articleType"
modal = tr.groupby(key)["gender"].agg(lambda s: s.value_counts().idxmax())
fallback = tr["gender"].value_counts().idxmax()
oracle = va[key].map(modal).fillna(fallback).values
o_macro = f1_score(va["gender"], oracle, labels=CLASSES["gender"], average="macro",
                   zero_division=0)
o_acc = (oracle == va["gender"].values).mean()
c_macro = f1_score(va["gender"], pred["gender"], labels=CLASSES["gender"],
                   average="macro", zero_division=0)
c_acc = (np.asarray(pred["gender"]) == va["gender"].values).mean()
print(f"  oracle  macro-F1 {o_macro:.4f}   accuracy {o_acc:.4f}")
print(f"  CNN     macro-F1 {c_macro:.4f}   accuracy {c_acc:.4f}")
print(f"  the image is worth {c_macro - o_macro:+.4f} macro-F1, "
      f"{c_acc - o_acc:+.4f} accuracy")
orep = classification_report(va["gender"], oracle, labels=CLASSES["gender"],
                             output_dict=True, zero_division=0)
print("\n  per class, oracle vs CNN F1:")
for c in CLASSES["gender"]:
    print(f"    {c:8} oracle {orep[c]['f1-score']:.3f}   "
          f"CNN {classification_report(va['gender'], pred['gender'], labels=CLASSES['gender'], output_dict=True, zero_division=0)[c]['f1-score']:.3f}")

out = HERE / "diagnose_gender_ceiling.csv"
rows = []
for t in TARGETS:
    rep = classification_report(va[t], pred[t], labels=CLASSES[t],
                                output_dict=True, zero_division=0)
    for c in CLASSES[t]:
        rows.append({"target": t, "class": c,
                     "train_n": int((tr[t] == c).sum()),
                     "val_n": int(rep[c]["support"]),
                     "precision": round(rep[c]["precision"], 4),
                     "recall": round(rep[c]["recall"], 4),
                     "f1": round(rep[c]["f1-score"], 4)})
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nwrote {out.relative_to(ROOT)}")
