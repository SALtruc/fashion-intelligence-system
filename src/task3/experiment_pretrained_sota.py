"""Compare the submitted model against ImageNet-pre-trained backbones, on our split.

The spec forbids a pre-trained system as the SUBMITTED model -- "You may not use
pre-trained systems which are trained on other datasets (not given to you as part of
this assignment)" -- and in the same breath recommends one for comparison: "But using
pre-trained for comparison your results is recommened." Nothing here is a submission
candidate. The shipped model stays design C, trained from scratch.

Why this rather than quoting published numbers. The dataset is a slice of the Kaggle
"Fashion Product Images" collection, and there are published results on it, but ours
are not comparable to them: our images are 60x80 downsamples, our split is our own,
and we report macro-F1 with `labels=` over all eight `usage` classes where that work
reports accuracy over a different label set at full resolution. Quoting a number
across three simultaneous changes of definition is decoration, not evidence. Training
a strong external backbone on OUR rows and scoring it with OUR metric is the same
comparison done in a way that means something.

Three arms, each answering a different question:

  resnet18_frozen    Do generic ImageNet features, with no fine-tuning at all, already
                     beat a 289k-parameter CNN trained from scratch on this data? This
                     is the cheap upper bound on "we should have used transfer learning".
  resnet18_finetune  The same backbone with every layer trainable, which is the fair
                     form of the question -- a frozen probe understates a pre-trained
                     model and it would be convenient to stop there.
  C_weighted         Our own design, retrained in this same session, so the comparison
                     is not confounded by a different day, driver or GPU state.

All arms share the split, the loss (class-weighted on both heads), the two-head
structure and the metric, so the backbone is the only thing that differs. The
pre-trained arms see the images upscaled to 224x224 with ImageNet normalisation:
upscaling adds no information, it matches the receptive field their stem assumes. At
60x80 the ImageNet 7x7 stride-2 stem plus max-pool would leave an 8x10 feature map,
which measures the stem's unsuitability rather than the backbone's quality -- Task 1
hit the same thing and rebuilt its stem for that reason.

    python src/task3/experiment_pretrained_sota.py
"""
import argparse
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


def _results_dir():
    d = _repo_root() / "predictions" / "task3"
    d.mkdir(parents=True, exist_ok=True)
    return d


ROOT = _repo_root()
SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=8, help="fine-tuning epochs")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--out", default=str(_results_dir() / "sota_comparison.csv"))
args = ap.parse_args()
PC_OUT = args.out.replace(".csv", "_perclass.csv")


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
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

DEVICE = g["DEVICE"]
TARGETS, CLASSES, IDX = g["TARGETS"], g["CLASSES"], g["IDX"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
tr, va = g["splits"][("gender", "primary")]
X_t = g["X_t"]
print(f"split: train {len(tr):,}  val {len(va):,}")

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def imnet_batch(rows, size=224, train=False):
    """uint8 NHWC -> ImageNet-normalised NCHW at `size`, with the same mirror
    augmentation the from-scratch model uses, so that is not a difference either."""
    x = X_t[rows]
    if x.device != DEVICE:
        x = x.to(DEVICE, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if train:
        flip = torch.rand(x.shape[0], device=x.device) < 0.5
        x[flip] = torch.flip(x[flip], dims=[-1])
    return (x - IMNET_MEAN) / IMNET_STD


class TwoHead(nn.Module):
    """The same shape as our own Net: one shared body, one linear head per target."""

    def __init__(self, body, feat_dim, heads):
        super().__init__()
        self.body = body
        self.heads = nn.ModuleDict({k: nn.Linear(feat_dim, v) for k, v in heads.items()})

    def forward(self, x):
        f = self.body(x).flatten(1)
        return {k: h(f) for k, h in self.heads.items()}


def backbone():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    feat = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat


def criteria():
    """Class-weighted, exactly as the shipped model is, so the loss is not a
    difference between the arms either."""
    return {k: nn.CrossEntropyLoss(weight=g["class_weights"](tr[k], k)).to(DEVICE)
            for k in heads}


def evaluate(model, frame, size=224):
    model.eval()
    rows = torch.as_tensor(frame["_row"].values, device=X_t.device)
    acc = {k: [] for k in heads}
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            o = model(imnet_batch(rows[i:i + 256], size=size))
            for k in heads:
                acc[k].append(o[k].argmax(1).cpu())
    pred = {k: np.asarray(CLASSES[k])[torch.cat(v).numpy()] for k, v in acc.items()}
    sc = {t: g["score"](frame[t], pred[t], labels=CLASSES[t]) for t in TARGETS}
    return sc, pred


rows_out, pc_out = [], []


def record(arm, scores, params, minutes, note, pred=None, frame=None):
    r = {"arm": arm, "params": params, "minutes": round(minutes, 1), "note": note}
    for t in TARGETS:
        r[f"{t} macroF1"] = round(scores[t]["macro_f1"], 4)
        r[f"{t} acc"] = round(scores[t]["accuracy"], 4)
    rows_out.append(r)
    # Per class, because a macro average over eight classes where four hold fifteen
    # validation images between them cannot distinguish a real gain on the large
    # classes from luck on the small ones. Without this the headline is unreadable.
    if pred is not None and frame is not None:
        from sklearn.metrics import classification_report
        for t in TARGETS:
            rep = classification_report(frame[t], pred[t], labels=CLASSES[t],
                                        output_dict=True, zero_division=0)
            for c in CLASSES[t]:
                pc_out.append({"arm": arm, "target": t, "class": c,
                               "val_n": int(rep[c]["support"]),
                               "precision": round(rep[c]["precision"], 4),
                               "recall": round(rep[c]["recall"], 4),
                               "f1": round(rep[c]["f1-score"], 4)})
        pd.DataFrame(pc_out).to_csv(PC_OUT, index=False)
    print(f"  {arm:20} gender {r['gender macroF1']:.4f}  usage {r['usage macroF1']:.4f}"
          f"   {params:,} params   [{r['minutes']}m]")
    pd.DataFrame(rows_out).to_csv(args.out, index=False)


# ---------------------------------------------------------------- 1. frozen features
print("\n=== arm 1: ImageNet ResNet18, backbone FROZEN, linear heads ===")
t1 = time.time()
body, feat_dim = backbone()
body = body.to(DEVICE).eval()
for p in body.parameters():
    p.requires_grad_(False)


def features(frame):
    rows = torch.as_tensor(frame["_row"].values, device=X_t.device)
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), 256):
            out.append(body(imnet_batch(rows[i:i + 256])).flatten(1))
    return torch.cat(out)


Ftr, Fva = features(tr), features(va)
print(f"  extracted {tuple(Ftr.shape)} train and {tuple(Fva.shape)} val features")
ytr = {k: g["encode"](tr[k], k).to(DEVICE) for k in heads}
lin = nn.ModuleDict({k: nn.Linear(feat_dim, v) for k, v in heads.items()}).to(DEVICE)
opt = torch.optim.Adam(lin.parameters(), lr=1e-3, weight_decay=1e-4)
crit = criteria()
for ep in range(200):
    lin.train()
    perm = torch.randperm(len(Ftr), device=DEVICE)
    for i in range(0, len(perm), 1024):
        sel = perm[i:i + 1024]
        loss = sum(crit[k](lin[k](Ftr[sel]), ytr[k][sel]) for k in heads)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
lin.eval()
with torch.no_grad():
    pred = {k: np.asarray(CLASSES[k])[lin[k](Fva).argmax(1).cpu().numpy()]
            for k in heads}
sc = {t: g["score"](va[t], pred[t], labels=CLASSES[t]) for t in TARGETS}
_n_frozen = sum(p.numel() for p in body.parameters())
record("resnet18_frozen", sc, sum(p.numel() for p in lin.parameters()),
       (time.time() - t1) / 60,
       f"ImageNet features, no fine-tuning; only the heads train. The backbone "
       f"contributes {_n_frozen:,} FROZEN parameters on top of the trainable count.",
       pred=pred, frame=va)
del Ftr, Fva, lin
torch.cuda.empty_cache()

# --------------------------------------------------------------------- 2. fine-tuned
print(f"\n=== arm 2: ImageNet ResNet18, fine-tuned end to end, "
      f"{args.epochs} epochs ===")
t1 = time.time()
torch.manual_seed(42)
body, feat_dim = backbone()
model = TwoHead(body, feat_dim, heads).to(DEVICE)
n_par = sum(p.numel() for p in model.parameters())
opt = torch.optim.Adam(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
crit = criteria()
rows = torch.as_tensor(tr["_row"].values, device=X_t.device)
ytr = {k: g["encode"](tr[k], k).to(DEVICE) for k in heads}
best, best_state = -1.0, None
for ep in range(1, args.epochs + 1):
    model.train()
    perm = torch.randperm(len(rows), device=rows.device)
    tot = 0.0
    for i in range(0, len(perm), args.bs):
        sel = perm[i:i + args.bs]
        o = model(imnet_batch(rows[sel], train=True))
        loss = sum(crit[k](o[k], ytr[k][sel.to(DEVICE)]) for k in heads)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        tot += float(loss.detach()) * len(sel)
    sched.step()
    s, _ = evaluate(model, va)
    mean_f1 = float(np.mean([s[t]["macro_f1"] for t in TARGETS]))
    if mean_f1 > best:
        best, best_state = mean_f1, {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
    print(f"    epoch {ep:2}/{args.epochs}  loss {tot / len(rows):.4f}  "
          + "  ".join(f"{t} {s[t]['macro_f1']:.4f}" for t in TARGETS)
          + ("   <- best" if mean_f1 == best else ""))
model.load_state_dict(best_state)
sc_ft, pred_ft = evaluate(model, va)

# Saved so that a later per-class question never costs a third training run. It lives
# under artifacts/, which is gitignored, and its own note says what it is: the spec
# forbids a pre-trained system as the submitted model, so this can only ever be a
# yardstick. Keeping it out of models/ makes that hard to misread.
_ck = ROOT / "artifacts" / "task3" / "comparison_resnet18_finetune.pt"
_ck.parent.mkdir(parents=True, exist_ok=True)
torch.save({"state_dict": best_state,
            "heads": heads,
            "classes": {k: list(v) for k, v in CLASSES.items()},
            "input": "60x80 upscaled to 224x224, ImageNet normalisation",
            "val_macro_f1": {k: round(sc_ft[k]["macro_f1"], 4) for k in TARGETS},
            "note": "COMPARISON ONLY, never a submission candidate. The spec forbids "
                    "a pre-trained system as the final model and recommends one for "
                    "comparison; this is the latter."},
           _ck)
print(f"  checkpoint -> {_ck.name} ({_ck.stat().st_size / 1e6:.1f} MB, gitignored)")

record("resnet18_finetune", sc_ft, n_par, (time.time() - t1) / 60,
       f"all layers trainable, {args.epochs} epochs, best epoch by mean macro-F1",
       pred=pred_ft, frame=va)
del model
torch.cuda.empty_cache()

# ------------------------------------------------------------ 3. our own, same session
print("\n=== arm 3: our design C, class-weighted, retrained in this session ===")
t1 = time.time()
g["SEED"] = 42
mine, _ = g["train_model"]("C_weighted_reference", heads, tr, va, weighted=True,
                           verbose=False)
pred = g["predict"](mine, va, heads)
sc = {t: g["score"](va[t], pred[t], labels=CLASSES[t]) for t in TARGETS}
record("C_weighted (ours)", sc, sum(p.numel() for p in mine.parameters()),
       (time.time() - t1) / 60, "trained from scratch on the 37,745 provided rows only",
       pred=pred, frame=va)

res = pd.DataFrame(rows_out)
res.to_csv(args.out, index=False)
print(f"\nwrote {args.out}")
print()
print(res.to_string(index=False))

mine_g = res[res.arm.str.contains("ours")].iloc[0]
print("\n=== read against our own noise bands (gender 0.0114, usage 0.0640) ===")
for _, r in res.iterrows():
    if "ours" in r.arm:
        continue
    dg = r["gender macroF1"] - mine_g["gender macroF1"]
    du = r["usage macroF1"] - mine_g["usage macroF1"]
    # "trainable" matters here: the frozen arm trains 6,669 parameters but still
    # runs an 11.2M-parameter backbone, so a single ratio would misdescribe it.
    ratio = r["params"] / mine_g["params"]
    print(f"  {r.arm:20} gender {dg:+.4f} ({'beyond' if abs(dg) > 0.0114 else 'within'}"
          f" band)   usage {du:+.4f} ({'beyond' if abs(du) > 0.0640 else 'within'} band)"
          f"   {ratio:.2f}x our trainable parameters")

# The question the aggregate cannot answer: is a usage gain real, or is it luck on the
# fifteen validation images the four rare classes hold between them?
if pc_out:
    pc = pd.DataFrame(pc_out)
    u = pc[pc.target == "usage"]
    LARGE = ["Casual", "Ethnic", "Formal", "Sports"]
    print()
    print("=== usage: where each arm's score comes from ===")
    print(f"  {'arm':22} {'4 large classes':>16} {'4 rare classes':>16}"
          f" {'val images (rare)':>18}")
    for arm in u.arm.unique():
        a = u[u.arm == arm]
        big = a[a["class"].isin(LARGE)]
        rare = a[~a["class"].isin(LARGE)]
        print(f"  {arm:22} {big.f1.mean():16.4f} {rare.f1.mean():16.4f}"
              f" {int(rare.val_n.sum()):18}")
    print()
    print("  The 4 large classes hold 5,646 validation images and the 4 rare ones 15,")
    print("  yet each contributes an eighth of the macro average. A gain that lives in")
    print("  the right-hand column is a gain on fifteen images.")
    print()
    print(u.pivot_table(index="class", columns="arm", values="f1").round(4).to_string())

print("\nNone of these is a submission candidate: the spec forbids a pre-trained")
print("system as the final model and permits it only for comparison.")
