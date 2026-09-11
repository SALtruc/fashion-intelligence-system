#!/usr/bin/env python
"""Stage crops for the NINE cosmetics classes the first external batch did not cover.

Context
-------
Batch 1 (`Dataset/ExternalCosmetics/`, 1,200 crops) fixed Eyeshadow, Lipstick and
Nail Polish. Nine cosmetics classes are still starved, and they are the highest-value
targets left in the whole long tail:

    articleType             provided train   in test-like region   enrichment
    Kajal and Eyeliner            13                 13               6.6x
    Foundation and Primer         12                 12               6.6x
    Lip Liner                     12                 12               6.6x
    Lip Gloss                      6                  6               6.6x
    Compact                        4                  4               6.6x
    Highlighter and Blush          4                  4               6.6x
    Lip Plumper                    4                  4               6.6x
    Concealer                      2                  2               6.6x
    Body Wash and Scrub            1                  1               6.6x

6.6x is the maximum: 100% of every one of these classes sits in the id range the
graded test set is drawn from. About 122 images lifts all nine to 20.

Every one of them is also `season=Spring` in the provided data, so this batch feeds
Task 2's weakest class as a side effect - Spring is only 4.0% of train.

Where the images come from
--------------------------
The archive used for batch 1 is exhausted: it held exactly four categories
(Lipstick, Nail Polish, Eye Shadow, Makeup Brush) and Makeup Brush is not one of the
124 articleTypes.

This script targets an object-detection export instead - the same shape as batch 1,
so the proven crop-from-bounding-box approach carries over. It reads either a COCO
`_annotations.coco.json` or a YOLO export (`data.yaml` + `labels/*.txt`), because
Roboflow-style exports come in both.

Deliberately NOT Openverse. `Dataset/ExternalEval/` is built from Openverse, and
training on the same source we evaluate against would undercut the whole point of an
independent evaluation set.

Labels
------
`gender`, `season` and `usage` are propagated per articleType from the provided
training data, exactly as `add_cosmetics_season_usage.py` does for batch 1. That is
legitimate here because these labels are catalogue conventions, not visual
properties, and they are 100% consistent within each of these classes. The
`label_source` column records it so nobody mistakes them for source data.

Usage
-----
    python prepare_external_cosmetics2.py --source <download folder> --list-classes
    python prepare_external_cosmetics2.py --source <download folder> --per-class 30

Then, and this is not optional:

    python verify_external_data.py --check Dataset/ExternalCosmetics2/images
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

# Written where the loader looks, rather than at a path that happened to work on
# one laptop. Override with A2_EXTERNAL_DATA.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_data import default_data_root  # noqa: E402

OUT_ROOT = default_data_root() / "ExternalCosmetics2"
TARGET_SIZE = (60, 80)
ID_START = 910000            # batch 1 used 900000-901199; stay clear of it
SOURCE_TAG = "external_cosmetics2_v1"
MIN_BOX = 40                 # ignore boxes too small to survive the 60x80 resize

# Source class name (lowercased) -> our articleType. Several source vocabularies are
# folded in, because these exports disagree on naming. Anything not listed here is
# reported by --list-classes and skipped rather than guessed at.
CLASS_MAP = {
    "eyeliner": "Kajal and Eyeliner",
    "kajal": "Kajal and Eyeliner",
    "eye liner": "Kajal and Eyeliner",
    "foundation": "Foundation and Primer",
    "primer": "Foundation and Primer",
    "lip liner": "Lip Liner",
    "lipliner": "Lip Liner",
    "lip gloss": "Lip Gloss",
    "lipgloss": "Lip Gloss",
    "gloss": "Lip Gloss",
    "compact": "Compact",
    "powder": "Compact",
    "compact powder": "Compact",
    "face powder": "Compact",
    "blush": "Highlighter and Blush",
    "blusher": "Highlighter and Blush",
    "highlighter": "Highlighter and Blush",
    "concealer": "Concealer",
    "body wash": "Body Wash and Scrub",
    "scrub": "Body Wash and Scrub",
    "shower gel": "Body Wash and Scrub",
    "lip plumper": "Lip Plumper",
}

# Propagated per articleType from the provided train split. gender/season/usage were
# each 100% pure within these classes; Highlighter and Blush has no non-null usage of
# its own, so it takes the 'Personal Care' mode (Casual, 99.8%).
LABELS = {
    "Kajal and Eyeliner":    ("Women", "Spring", "Casual"),
    "Foundation and Primer": ("Women", "Spring", "Casual"),
    "Lip Liner":             ("Women", "Spring", "Casual"),
    "Lip Gloss":             ("Women", "Spring", "Casual"),
    "Compact":               ("Women", "Spring", "Casual"),
    "Highlighter and Blush": ("Women", "Spring", "Casual"),
    "Lip Plumper":           ("Women", "Spring", "Casual"),
    "Concealer":             ("Women", "Spring", "Casual"),
    "Body Wash and Scrub":   ("Men",   "Spring", "Casual"),
}
SUBCATEGORY = {
    "Kajal and Eyeliner": "Eyes", "Foundation and Primer": "Makeup",
    "Lip Liner": "Lips", "Lip Gloss": "Lips", "Lip Plumper": "Lips",
    "Compact": "Makeup", "Highlighter and Blush": "Makeup",
    "Concealer": "Makeup", "Body Wash and Scrub": "Bath and Body",
}


def find_coco(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def load_coco(path: Path) -> tuple[dict, list, dict]:
    d = json.loads(path.read_text(encoding="utf-8"))
    if not all(k in d for k in ("images", "annotations", "categories")):
        raise ValueError("not a COCO file")
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {i["id"]: i for i in d["images"]}
    return imgs, d["annotations"], cats


def load_yolo(root: Path) -> tuple[dict, list, dict] | None:
    """Read a YOLO export: data.yaml names + labels/*.txt with normalised xywh."""
    yamls = list(root.rglob("data.yaml"))
    if not yamls:
        return None
    names: list[str] = []
    for line in yamls[0].read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("names:"):
            rest = s.split(":", 1)[1].strip()
            if rest.startswith("["):
                names = [n.strip().strip("'\"") for n in rest.strip("[]").split(",")]
        elif names is not None and s.startswith("- ") and "names" in "".join(names[:0] or [""]):
            pass
    if not names:                                   # block-style names:
        txt = yamls[0].read_text(encoding="utf-8").splitlines()
        grab = False
        for line in txt:
            if line.strip().startswith("names:"):
                grab = True
                continue
            if grab:
                if line.strip().startswith("- "):
                    names.append(line.strip()[2:].strip().strip("'\""))
                elif line.strip() and not line.startswith(" "):
                    break
    if not names:
        return None

    cats = dict(enumerate(names))
    imgs, anns, next_img = {}, [], 0
    for lbl in sorted(root.rglob("labels/*.txt")):
        stem = lbl.stem
        img = None
        for ext in (".jpg", ".jpeg", ".png"):
            cand = list(lbl.parent.parent.rglob(f"images/{stem}{ext}"))
            if cand:
                img = cand[0]
                break
        if img is None:
            continue
        try:
            with Image.open(img) as im:
                W, H = im.size
        except Exception:
            continue
        imgs[next_img] = {"id": next_img, "file_name": str(img), "width": W, "height": H}
        for line in lbl.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            ci, xc, yc, w, h = int(p[0]), *map(float, p[1:5])
            anns.append({"image_id": next_img, "category_id": ci,
                         "bbox": [(xc - w/2)*W, (yc - h/2)*H, w*W, h*H]})
        next_img += 1
    return imgs, anns, cats


def resolve(root: Path):
    """Merge EVERY COCO json under `root`, not just the first one.

    Roboflow exports split the data into train/ valid/ test/ with a separate
    `_annotations.coco.json` in each. Their split is irrelevant here - we are mining
    this download for crops, not training on it - so reading only one folder would
    silently discard about a third of the images. For classes with four examples
    that is not an acceptable loss.

    Image ids and category ids are local to each file, so both are remapped onto a
    global numbering as the files are merged. Categories are merged BY NAME, since
    the same class can carry different ids in different splits.
    """
    files = []
    for j in find_coco(root):
        try:
            files.append((j, load_coco(j)))
        except Exception:
            continue

    if files:
        imgs, anns, cats = {}, [], {}
        name_to_id: dict[str, int] = {}
        next_img = 0
        for j, (fi, fa, fc) in files:
            local_img = {}
            for old_id, rec in fi.items():
                rec = dict(rec)
                # keep the path resolvable: file_name is relative to its own folder
                rec["file_name"] = str((j.parent / rec["file_name"]).resolve())
                local_img[old_id] = next_img
                imgs[next_img] = rec
                next_img += 1
            local_cat = {}
            for cid, name in fc.items():
                if name not in name_to_id:
                    new_id = len(name_to_id)
                    name_to_id[name] = new_id
                    cats[new_id] = name
                local_cat[cid] = name_to_id[name]
            n = 0
            for x in fa:
                if x["image_id"] not in local_img:
                    continue
                y = dict(x)
                y["image_id"] = local_img[x["image_id"]]
                y["category_id"] = local_cat[x["category_id"]]
                anns.append(y)
                n += 1
            print(f"  COCO: {j.parent.name}/{j.name}  {len(fi)} images, {n} anns")
        print(f"  merged -> {len(imgs)} images, {len(anns)} annotations")
        return imgs, anns, cats, root

    y = load_yolo(root)
    if y:
        print("  YOLO export")
        return y[0], y[1], y[2], root
    sys.exit(f"no COCO json or YOLO export found under {root}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="folder holding the download")
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--list-classes", action="store_true",
                    help="print the source's classes and how they map, then exit")
    a = ap.parse_args()

    root = Path(a.source)
    if not root.exists():
        sys.exit(f"not found: {root}")
    imgs, anns, cats, base = resolve(root)
    print(f"  {len(imgs)} images, {len(anns)} annotations, {len(cats)} classes\n")

    counts = Counter(cats.get(x["category_id"], "?") for x in anns)
    if a.list_classes:
        print(f"{'source class':30}{'anns':>7}  -> our articleType")
        for name, n in counts.most_common():
            tgt = CLASS_MAP.get(name.strip().lower(), "")
            print(f"{name:30}{n:>7}  -> {tgt or '(skipped - no mapping)'}")
        print("\nEdit CLASS_MAP at the top of this file to map anything useful "
              "that is currently skipped.")
        return

    wanted = defaultdict(list)
    for x in anns:
        tgt = CLASS_MAP.get(cats.get(x["category_id"], "").strip().lower())
        if tgt:
            wanted[tgt].append(x)
    if not wanted:
        sys.exit("no annotation matched CLASS_MAP - run --list-classes first.")

    (OUT_ROOT / "images").mkdir(parents=True, exist_ok=True)
    rows, next_id = [], ID_START
    for tgt in sorted(wanted):
        kept = 0
        for x in wanted[tgt]:
            if kept >= a.per_class:
                break
            im_rec = imgs.get(x["image_id"])
            if not im_rec:
                continue
            fp = Path(im_rec["file_name"])
            if not fp.is_absolute():
                hit = list(base.rglob(fp.name))
                if not hit:
                    continue
                fp = hit[0]
            bx, by, bw, bh = x["bbox"]
            if bw < MIN_BOX or bh < MIN_BOX:
                continue
            try:
                with Image.open(fp) as im:
                    crop = im.convert("RGB").crop(
                        (max(0, int(bx)), max(0, int(by)),
                         min(im.width, int(bx + bw)), min(im.height, int(by + bh))))
                    crop = crop.resize(TARGET_SIZE, Image.LANCZOS)
            except Exception:
                continue
            iid = next_id
            next_id += 1
            crop.save(OUT_ROOT / "images" / f"{iid}.jpg", quality=95)
            g, se, us = LABELS[tgt]
            rows.append({"id": iid, "gender": g, "masterCategory": "Personal Care",
                         "subCategory": SUBCATEGORY[tgt], "articleType": tgt,
                         "season": se, "usage": us, "source": SOURCE_TAG,
                         "label_source": "gender+season+usage propagated per "
                                         "articleType from provided train"})
            kept += 1
        print(f"  {tgt:24} {kept:>4} crops")

    if not rows:
        sys.exit("nothing produced - check --list-classes and MIN_BOX.")
    out = OUT_ROOT / "external_cosmetics2.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n  {len(rows)} crops -> {OUT_ROOT}\n  {out}")
    print("\n  NEXT, and do not skip it:")
    print(f"    python verify_external_data.py --check {OUT_ROOT / 'images'}")


if __name__ == "__main__":
    main()
