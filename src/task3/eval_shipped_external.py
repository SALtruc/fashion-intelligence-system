"""Score the SHIPPED checkpoint on the 261 independent photographs.

The report's section 5 lists "run exact checkpoint on external set" as the last piece of
outstanding Task 3 evidence, and it is outstanding for a real reason: the 0.1292 gender
figure already in the report was measured on design C *without* class weighting, which
is not the model in the zip. Quoting a robustness number for a model other than the one
submitted is the kind of small substitution that a marker is entitled to call out.

No training. The checkpoint is loaded, run over the 261 images, and scored.

Two scoring conventions are reported because they disagree by a lot here and the
existing 0.1292 used the second one. `all` averages F1 over every class the model can
emit, so classes absent from a 261-image sample contribute a hard zero; `present`
averages only over classes that actually occur in the external labels. Section 5's
sentence should say which one it means.

THE PREDICTION, fixed before the run
    The collapse from catalogue validation to in-the-wild photographs is a property of
    the domain shift, not of the weighting, so the shipped weighted model should land
    close to the unweighted 0.1292 rather than markedly above or below it: I expect
    gender between 0.10 and 0.20 on the `present` convention. If it lands much higher,
    the honest reading is that weighting helps under domain shift, which nothing in our
    experiments has suggested and which would need its own confirmation before being
    claimed.

    python src/task3/eval_shipped_external.py
"""
import io
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _repo_root():
    for base in (HERE, *HERE.parents):
        if (base / ".git").exists():
            return base
    raise SystemExit(f"no .git above {HERE}")


ROOT = _repo_root()
RES = ROOT / "predictions" / "task3"
RES.mkdir(parents=True, exist_ok=True)
OUT = RES / "external_shipped.csv"
CKPT = ROOT / "artifacts" / "task3" / "task3_gender_usage_C_weighted.pt"
SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

# The unweighted design C on this same set, from the notebook's section 8, recorded in
# the consolidated catalog under split "independent eval".
UNWEIGHTED = {"gender": 0.1292, "usage": 0.1124}


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


print("executing the notebook prefix ...")
t0 = time.time()
g = run_prefix()
print(f"prefix done in {time.time() - t0:.0f}s")

import torch
from sklearn.metrics import classification_report

DEVICE, TARGETS, CLASSES = g["DEVICE"], g["TARGETS"], g["CLASSES"]
MEAN, STD, score = g["MEAN"], g["STD"], g["score"]

if g.get("EXTERNAL_ROOT") is None:
    raise SystemExit("external data root not found; nothing to evaluate against")
base = Path(g["EXTERNAL_ROOT"]) / "ExternalEval"
if not (base / "external_eval.csv").is_file():
    raise SystemExit(f"{base / 'external_eval.csv'} not found")

ev = pd.read_csv(base / "external_eval.csv")
ev["path"] = ev["id"].astype(str).map(lambda i: str(base / "images" / f"{i}.jpg"))
ev = ev[ev[TARGETS].notna().all(axis=1)].reset_index(drop=True)
X_ev = g["load_images"](ev["path"].tolist())
print(f"{len(ev)} independent evaluation images, {X_ev.shape[1:]} each")
for t in TARGETS:
    print(f"  {t}: {dict(ev[t].value_counts())}")

# ------------------------------------------------------------------- the exact model
if not CKPT.is_file():
    raise SystemExit(f"{CKPT} not found; it is gitignored and lives on the team Drive")
ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
heads = {k: int(v) for k, v in ck["heads"].items()}
model = g["Net"](heads).to(DEVICE)
model.load_state_dict(ck["state_dict"])
model.eval()
# Compared per target the checkpoint actually has. This notebook's CLASSES also
# carries `pair`, the 24-way joint label design B used and the shipped two-head model
# never emits, so comparing the whole dict fails for a reason that does not matter.
# What matters is that the class ORDER agrees, because the heads emit indices into
# these lists and a reordering would silently relabel every prediction.
for _t, _cls in ck["classes"].items():
    assert list(_cls) == list(CLASSES[_t]), \
        f"{_t}: checkpoint order {list(_cls)} != notebook order {list(CLASSES[_t])}"
print(f"\nloaded {CKPT.name}: {ck['design'][:44]}, use_tta={ck['use_tta']}")
print(f"  its recorded catalogue validation: {ck['val_macro_f1']}")


@torch.no_grad()
def probs(arr, mirror):
    x = torch.from_numpy(arr).to(DEVICE).permute(0, 3, 1, 2).float().div_(255.0)
    if mirror:
        x = x.flip(-1)
    x = (x - MEAN) / STD
    out = model(x)
    return {k: torch.softmax(out[k].float(), 1).cpu().numpy() for k in heads}


plain, mir = probs(X_ev, False), probs(X_ev, True)

rows = []
print("\n=== the shipped checkpoint on the independent set ===")
for tta in (False, True):
    for conv in ("all", "present"):
        line = {}
        for t in TARGETS:
            p = (plain[t] + mir[t]) / 2 if tta else plain[t]
            yp = np.array([CLASSES[t][i] for i in p.argmax(1)])
            labels = CLASSES[t] if conv == "all" else sorted(set(ev[t]))
            s = score(ev[t], yp, labels=labels)
            line[t] = s
            rows.append({"model": "C_weighted (shipped)", "split": "independent eval",
                         "tta": tta, "labels": conv, "target": t,
                         "n_images": len(ev), "n_labels": len(labels),
                         "macro_f1": round(s["macro_f1"], 4),
                         "accuracy": round(s["accuracy"], 4)})
        print(f"  tta={str(tta):5} labels={conv:8} "
              + "  ".join(f"{t} macro-F1 {line[t]['macro_f1']:.4f} "
                          f"acc {line[t]['accuracy']:.4f}" for t in TARGETS))

pd.DataFrame(rows).to_csv(OUT, index=False)
print(f"\nwrote {OUT.name}")

# ---------------------------------------------------------------------- the comparison
print("\n=== against the unweighted model already quoted in the report ===")
d = pd.DataFrame(rows)
best = d[(d.tta == bool(ck["use_tta"])) & (d.labels == "present")]
for t in TARGETS:
    got = float(best[best.target == t].macro_f1.iloc[0])
    print(f"  {t:7} unweighted {UNWEIGHTED[t]:.4f} -> shipped {got:.4f}  "
          f"({got - UNWEIGHTED[t]:+.4f})")
print("\n  Same 261 images and the same `present` convention, so this pair is")
print("  comparable. The catalogue-to-wild collapse is what it was; the point of the")
print("  run is that section 5 can now quote the model it actually submits.")
print("\nper-class, shipped model, mirror TTA:")
for t in TARGETS:
    p = (plain[t] + mir[t]) / 2
    yp = np.array([CLASSES[t][i] for i in p.argmax(1)])
    print(f"\n--- {t} ---")
    print(pd.DataFrame(classification_report(
        ev[t], yp, labels=sorted(set(ev[t])), output_dict=True, zero_division=0
    )).T.round(3).to_string())
