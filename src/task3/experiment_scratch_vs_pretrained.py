"""Was the pre-trained ResNet's lead its ImageNet weights, or its size and resolution?

This is the question that decides whether tonight's comparison is actionable. A
pre-trained system cannot be submitted -- the spec forbids it -- but a randomly
initialised ResNet18 trained on the provided rows is entirely legal, and Task 1
measured +0.057 on `articleType` from exactly that change. So:

  resnet18_scratch, random split   Does a bigger network at 224x224, with no ImageNet
                                   weights, reach the fine-tuned model's 0.8228? If it
                                   does, the lead was capacity and resolution, and
                                   capacity and resolution are ours to take.
  resnet18_scratch, forward split  Does any of it survive the split that mimics how
                                   the graded test set was cut? Two candidates have
                                   already won on the random split and evaporated
                                   here: epochs=30 (+0.019 -> -0.002) and D+weighted
                                   (+0.026 on one repeat -> +0.0065 mean, sign
                                   flipping). The base rate is 0 for 2.
  resnet18_finetune, forward split For the report: does the pre-trained lead itself
                                   transfer? Not submittable either way.

Four choices that keep this a fair test rather than a rigged one:

  * 20 epochs for the from-scratch arms, not the 8 the pre-trained arm needed. A
    randomly initialised network converges more slowly and giving it the shorter
    budget would measure the budget.
  * lr 1e-3 for from-scratch, the value our own model uses, rather than the 3e-4 that
    suits fine-tuning. A fine-tuning learning rate on a from-scratch network is a
    handicap disguised as consistency.
  * Both arms are scored with and without mirror TTA, so they can be read against
    whichever of our references matches the protocol.
  * The forward arms are gated on the random-split result. If the from-scratch model
    cannot beat our own CNN on the split where every effect looks largest, there is
    nothing to transfer and the script says so instead of spending another 47 minutes.

Upscaling 60x80 to 224x224 adds no information -- it is a deterministic function of
the original pixels. Any gain is architecture and compute, not data.

    THE RULE, fixed before running:
      1. SOURCE: if scratch reaches within 0.02 of the fine-tuned arm's `gender` on the
         random split, the lead was capacity and resolution, not pre-training.
      2. CANDIDATE: if scratch beats our C_weighted reference on the FORWARD split by
         more than that reference's own spread (gender 0.0103), it is a legal
         improvement candidate worth a confirmation run before any submission change.
      3. Otherwise design C ships unchanged and this is reported as comparison only.

    A recorded prediction: rule 1 FAILS and rule 2 FAILS. ImageNet pre-training on
    1.2M images is worth a great deal at 32k rows, and Task 1's from-scratch ResNet
    bought only +0.057 over its from-scratch CNN. I expect scratch to land between our
    CNN and the fine-tuned ResNet on the random split -- nearer ours, around 0.75-0.78
    -- and not to clear the forward-split bar. If that is wrong, the submitted model
    should probably change, and this file will say so.

    python src/task3/experiment_scratch_vs_pretrained.py
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
    d = _repo_root() / "results" / "task3"
    d.mkdir(parents=True, exist_ok=True)
    return d


ROOT = _repo_root()
SRC = HERE / "task3_build.py"
PREFIX_STOP = "# ## 5 "

# Our own C_weighted on the forward split: three runs, 20 epochs, class-weighted, no
# TTA, from experiment_epochs_confirm.csv. Three samples beat retraining one.
REF_FWD = {"gender": [0.5877, 0.5937, 0.5834], "usage": [0.3471, 0.3430, 0.3399]}

ap = argparse.ArgumentParser()
ap.add_argument("--epochs-scratch", type=int, default=20)
ap.add_argument("--epochs-pretrained", type=int, default=8)
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--lr-scratch", type=float, default=1e-3)
ap.add_argument("--lr-finetune", type=float, default=3e-4)
ap.add_argument("--out", default=str(_results_dir() / "scratch_vs_pretrained.csv"))
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
from sklearn.metrics import classification_report

DEVICE = g["DEVICE"]
TARGETS, CLASSES = g["TARGETS"], g["CLASSES"]
heads = {t: len(CLASSES[t]) for t in TARGETS}
X_t = g["X_t"]
SPLITS = {"random": g["splits"][("gender", "primary")],
          "forward": g["splits"][("gender", "forward")]}
for name, (a, b) in SPLITS.items():
    print(f"  {name:8} train {len(a):,}  val {len(b):,}")

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


def batch224(rows, train=False, flip_all=False):
    x = X_t[rows]
    if x.device != DEVICE:
        x = x.to(DEVICE, non_blocking=True)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
    if flip_all:
        x = torch.flip(x, dims=[-1])
    elif train:
        f = torch.rand(x.shape[0], device=x.device) < 0.5
        x[f] = torch.flip(x[f], dims=[-1])
    return (x - IMNET_MEAN) / IMNET_STD


class TwoHead(nn.Module):
    def __init__(self, body, feat, heads):
        super().__init__()
        self.body = body
        self.heads = nn.ModuleDict({k: nn.Linear(feat, v) for k, v in heads.items()})

    def forward(self, x):
        f = self.body(x).flatten(1)
        return {k: h(f) for k, h in self.heads.items()}


def make(pretrained):
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    feat = m.fc.in_features
    m.fc = nn.Identity()
    return TwoHead(m, feat, heads).to(DEVICE)


@torch.no_grad()
def infer(model, frame, tta):
    model.eval()
    rows = torch.as_tensor(frame["_row"].values, device=X_t.device)
    acc = {k: [] for k in heads}
    for i in range(0, len(rows), 256):
        sel = rows[i:i + 256]
        o = model(batch224(sel))
        if tta:
            f = model(batch224(sel, flip_all=True))
            o = {k: ((o[k].softmax(1) + f[k].softmax(1)) / 2).log() for k in heads}
        for k in heads:
            acc[k].append(o[k].argmax(1).cpu())
    return {k: np.asarray(CLASSES[k])[torch.cat(v).numpy()] for k, v in acc.items()}


def score_of(frame, pred):
    return {t: g["score"](frame[t], pred[t], labels=CLASSES[t]) for t in TARGETS}


rows_out, pc_out = [], []


def train_arm(tag, pretrained, split, epochs, lr):
    tr, va = SPLITS[split]
    t1 = time.time()
    torch.manual_seed(42)
    model = make(pretrained)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = {k: nn.CrossEntropyLoss(weight=g["class_weights"](tr[k], k)).to(DEVICE)
            for k in heads}
    rows = torch.as_tensor(tr["_row"].values, device=X_t.device)
    y = {k: g["encode"](tr[k], k).to(DEVICE) for k in heads}
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(rows), device=rows.device)
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            sel = perm[i:i + args.bs]
            o = model(batch224(rows[sel], train=True))
            loss = sum(crit[k](o[k], y[k][sel.to(DEVICE)]) for k in heads)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(sel)
        sched.step()
        s = score_of(va, infer(model, va, tta=False))
        mean_f1 = float(np.mean([s[t]["macro_f1"] for t in TARGETS]))
        if mean_f1 > best:
            best, best_state = mean_f1, {k: v.detach().cpu().clone()
                                         for k, v in model.state_dict().items()}
        if ep % 2 == 0 or ep == epochs or ep == 1:
            print(f"    epoch {ep:2}/{epochs}  loss {tot / len(rows):.4f}  "
                  + "  ".join(f"{t} {s[t]['macro_f1']:.4f}" for t in TARGETS)
                  + ("   <- best" if mean_f1 == best else ""))
    model.load_state_dict(best_state)

    out = {}
    for tta in (False, True):
        pred = infer(model, va, tta=tta)
        sc = score_of(va, pred)
        r = {"arm": tag, "split": split, "tta": tta, "params": n_par,
             "epochs": epochs, "lr": lr, "minutes": round((time.time() - t1) / 60, 1)}
        for t in TARGETS:
            r[f"{t} macroF1"] = round(sc[t]["macro_f1"], 4)
            r[f"{t} acc"] = round(sc[t]["accuracy"], 4)
        rows_out.append(r)
        out[tta] = r
        for t in TARGETS:
            rep = classification_report(va[t], pred[t], labels=CLASSES[t],
                                        output_dict=True, zero_division=0)
            for c in CLASSES[t]:
                pc_out.append({"arm": tag, "split": split, "tta": tta, "target": t,
                               "class": c, "val_n": int(rep[c]["support"]),
                               "f1": round(rep[c]["f1-score"], 4)})
        print(f"  {tag:20} {split:8} tta={str(tta):5} "
              f"gender {r['gender macroF1']:.4f}  usage {r['usage macroF1']:.4f}"
              f"   [{r['minutes']}m]")
    pd.DataFrame(rows_out).to_csv(args.out, index=False)
    pd.DataFrame(pc_out).to_csv(PC_OUT, index=False)
    del model
    torch.cuda.empty_cache()
    return out


print(f"\n=== arm 1: ResNet18 from scratch, random split, "
      f"{args.epochs_scratch} epochs, lr {args.lr_scratch} ===")
a1 = train_arm("resnet18_scratch", False, "random", args.epochs_scratch,
               args.lr_scratch)

# The published fine-tuned and our own numbers on this split, for rule 1.
FT_RANDOM_GENDER = 0.8228
OURS_RANDOM_GENDER = 0.7211
BAND_G = 0.0114
scratch_g = a1[False]["gender macroF1"]
rule1 = abs(FT_RANDOM_GENDER - scratch_g) <= 0.02
beats_ours = scratch_g - OURS_RANDOM_GENDER > BAND_G

print(f"\n=== gate before spending another hour ===")
print(f"  scratch gender {scratch_g:.4f} | fine-tuned {FT_RANDOM_GENDER:.4f} "
      f"| ours {OURS_RANDOM_GENDER:.4f}")
print(f"  rule 1, within 0.02 of fine-tuned -> {rule1}")
print(f"  beats ours by more than {BAND_G} -> {beats_ours}")

if beats_ours:
    print(f"\n=== arm 2: ResNet18 from scratch, FORWARD split, "
          f"{args.epochs_scratch} epochs ===")
    a2 = train_arm("resnet18_scratch", False, "forward", args.epochs_scratch,
                   args.lr_scratch)
    print(f"\n=== arm 3: ResNet18 fine-tuned, FORWARD split (report only) ===")
    a3 = train_arm("resnet18_finetune", True, "forward", args.epochs_pretrained,
                   args.lr_finetune)
else:
    a2 = a3 = None
    print("\n  SKIPPING the forward-split arms: a model that cannot beat our own CNN")
    print("  on the split where every effect has looked largest has nothing to")
    print("  transfer. That saves about 47 minutes and is not a null result -- it is")
    print("  the answer to rule 1.")

res = pd.DataFrame(rows_out)
res.to_csv(args.out, index=False)
print(f"\nwrote {args.out}")
print()
print(res.to_string(index=False))

print("\n=== the rule, as written before the run ===")
print(f"  1. SOURCE: scratch within 0.02 of fine-tuned on random split -> {rule1}")
if a2 is not None:
    ref_g = float(np.mean(REF_FWD["gender"]))
    ref_spread = max(REF_FWD["gender"]) - min(REF_FWD["gender"])
    d = a2[False]["gender macroF1"] - ref_g
    rule2 = d > ref_spread
    print(f"  2. CANDIDATE: forward gender {a2[False]['gender macroF1']:.4f} vs our "
          f"3-run reference {ref_g:.4f}, delta {d:+.4f} > spread {ref_spread:.4f}"
          f" -> {rule2}")
else:
    rule2 = False
    print("  2. CANDIDATE: not evaluated, the forward arms were gated out")
print(f"\n  VERDICT: {'a legal candidate worth confirming' if rule2 else 'design C ships unchanged'}")
print(f"  recorded prediction was rule 1 FAILS and rule 2 FAILS -> "
      f"{'CORRECT' if (not rule1 and not rule2) else 'WRONG in part'}")
