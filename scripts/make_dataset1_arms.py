"""Split dataset1 into a training portion and a held-out portion for the enrichment A/B.

The experiment this supports has two arms that differ in one thing only:

    Arm A ("supplied")  trains on the 30,278 supplied rows.
    Arm B ("enriched")  trains on those plus this script's training portion.

Both arms are then scored on the untouched supplied validation split *and* on the
held-out portion produced here, which neither arm trains on. That second set is the
only one with enough cosmetics rows to measure anything: the supplied validation
split holds 1 Eyeshadow, 3 Lipstick and 4 Nail Polish images, against roughly 155
per class here.

Two things this script is careful about.

Near-duplicates. The crops are derived from COCO photographs and nothing in
external_cosmetics.csv identifies the source image -- the ids are sequential and every
other column is constant. Two crops of the same photograph would therefore be split
independently, putting a near-twin of a held-out image into training and inflating the
held-out score for Arm B specifically. A perceptual hash is used to cluster them first,
and clusters are kept whole on one side of the split.

Stratification. The split is stratified by articleType so both portions carry all three
classes in proportion, which a global shuffle does not guarantee once clusters are kept
whole.

Usage:
    python scripts/make_dataset1_arms.py            # write the split
    python scripts/make_dataset1_arms.py --check    # verify an existing split
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET1 = REPO_ROOT / "notebooks" / "Task1" / "dataset1"
SOURCE_CSV = DATASET1 / "external_cosmetics.csv"
IMAGE_DIR = DATASET1 / "images"
OUT_DIR = REPO_ROOT / "preprocessed_datasets" / "task1_dataset1_arms"

TRAIN_SHARE = 0.6      # of dataset1 into Arm B's training set; the rest is held out
SPLIT_SEED = 42        # the split seed used everywhere else in Task 1
HASH_SIDE = 8          # dHash grid; 8 gives a 64-bit hash
HAMMING_RADIUS = 2     # the radius the external audit already used against supplied data


def dhash(path, side=HASH_SIDE):
    """64-bit difference hash: compares each pixel with its right-hand neighbour.

    Robust to the rescaling and mild colour shifts that distinguish two crops of one
    photograph, which is exactly the collision this split has to avoid.
    """
    image = Image.open(path).convert("L").resize((side + 1, side), Image.LANCZOS)
    pixels = np.asarray(image, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return np.packbits(bits.flatten()).tobytes()


def cluster_near_duplicates(hashes, radius=HAMMING_RADIUS):
    """Group ids whose hashes are within `radius` bits, by union-find.

    Transitive by construction: if a is near b and b is near c, all three land in one
    cluster even when a and c are further apart than the radius. That is the
    conservative direction -- it can only make clusters larger, never split a true
    duplicate pair across the boundary.
    """
    ids = list(hashes)
    packed = np.stack([np.frombuffer(hashes[i], dtype=np.uint8) for i in ids])
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Pairwise Hamming over 1,158 x 64 bits is trivial; the lookup table avoids a loop.
    popcount = np.array([bin(v).count("1") for v in range(256)], dtype=np.uint8)
    for index in range(len(ids)):
        distances = popcount[np.bitwise_xor(packed[index], packed[index + 1:])].sum(axis=1)
        for offset in np.flatnonzero(distances <= radius):
            union(ids[index], ids[index + 1 + offset])

    return {i: find(i) for i in ids}


def build():
    frame = pd.read_csv(SOURCE_CSV, dtype={"id": str})
    print(f"Read {len(frame):,} external rows from {SOURCE_CSV.relative_to(REPO_ROOT)}")

    missing = [i for i in frame["id"] if not (IMAGE_DIR / f"{i}.jpg").is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} rows have no image, first: {missing[:5]}")

    print(f"Hashing {len(frame):,} images...")
    hashes = {row.id: dhash(IMAGE_DIR / f"{row.id}.jpg") for row in frame.itertuples()}
    clusters = cluster_near_duplicates(hashes)
    frame["cluster"] = frame["id"].map(clusters)

    sizes = frame["cluster"].value_counts()
    print(f"Near-duplicate clusters: {len(sizes):,} for {len(frame):,} images "
          f"(largest {int(sizes.max())}, {int((sizes > 1).sum())} clusters hold more than one)")

    # Split whole clusters, stratified by class. Clusters are assigned largest-first into
    # whichever side is furthest below its target, which keeps both portions close to the
    # requested ratio without ever splitting a cluster.
    rng = np.random.default_rng(SPLIT_SEED)
    role = {}
    for article_type, group in frame.groupby("articleType", sort=True):
        cluster_sizes = group.groupby("cluster").size()
        # Shuffle first, then sort by size with a stable sort, so equal-sized clusters keep
        # the shuffled order. Every cluster here is a singleton, so without the shuffle the
        # "sort" is a no-op and the split becomes the first 60% of ids per class -- ids are
        # sequential in collection order, which is not something to stake the result on.
        order = list(rng.permutation(list(cluster_sizes.index)))
        order.sort(key=lambda cluster: -int(cluster_sizes[cluster]))
        target = TRAIN_SHARE * len(group)
        taken = 0
        for cluster in order:
            if taken < target:
                role.update({i: "train" for i in group.loc[group["cluster"] == cluster, "id"]})
                taken += int(cluster_sizes[cluster])
            else:
                role.update({i: "heldout" for i in group.loc[group["cluster"] == cluster, "id"]})
        print(f"  {article_type:<14} {len(group):>5} rows -> {taken:>4} train, "
              f"{len(group) - taken:>4} held out")

    frame["arm_role"] = frame["id"].map(role)
    frame["relative_path"] = frame["id"].map(lambda i: f"notebooks/Task1/dataset1/images/{i}.jpg")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"train_share": TRAIN_SHARE, "seed": SPLIT_SEED, "radius": HAMMING_RADIUS,
               "rows": len(frame), "clusters": int(len(sizes))}
    version = hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    target_dir = OUT_DIR / version
    target_dir.mkdir(parents=True, exist_ok=True)

    train = frame[frame["arm_role"] == "train"].drop(columns=["arm_role"])
    heldout = frame[frame["arm_role"] == "heldout"].drop(columns=["arm_role"])
    # LF explicitly, on every platform. PARALLEL_RUN.md tells the operator this split can
    # be regenerated per machine instead of copied because it is seeded; that is only true
    # byte for byte if the line endings do not follow the OS. pandas and write_text both
    # emit CRLF on Windows, and files under preprocessed_datasets/ are hashed by byte.
    train.to_csv(target_dir / "external_train.csv", index=False, lineterminator="\n")
    heldout.to_csv(target_dir / "external_heldout.csv", index=False, lineterminator="\n")

    # A cluster that appears on both sides would be the one failure that matters, so it is
    # recorded as a checked fact rather than a claim in a comment.
    straddling = set(train["cluster"]) & set(heldout["cluster"])
    metadata = {
        **payload,
        "version": version,
        "train_rows": int(len(train)),
        "heldout_rows": int(len(heldout)),
        "straddling_clusters": len(straddling),
        "per_class": {k: {"train": int((train["articleType"] == k).sum()),
                          "heldout": int((heldout["articleType"] == k).sum())}
                      for k in sorted(frame["articleType"].unique())},
        "note": ("external_train.csv is added to Arm B's training set only. "
                 "external_heldout.csv is trained on by neither arm and is scored by both."),
    }
    (target_dir / "split_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n",
                                                    encoding="utf-8", newline="\n")

    assert not straddling, f"{len(straddling)} clusters straddle the split"
    print(f"\nWrote {target_dir.relative_to(REPO_ROOT)}")
    print(f"  external_train.csv    {len(train):>5} rows  -> Arm B training only")
    print(f"  external_heldout.csv  {len(heldout):>5} rows  -> scored by both arms, trained by neither")
    print(f"  no cluster straddles the split")
    return target_dir


def check():
    versions = sorted(OUT_DIR.glob("*/split_metadata.json"))
    if not versions:
        print("No split found. Run without --check to build one.")
        return 1
    for path in versions:
        meta = json.loads(path.read_text(encoding="utf-8"))
        train = pd.read_csv(path.parent / "external_train.csv", dtype={"id": str})
        heldout = pd.read_csv(path.parent / "external_heldout.csv", dtype={"id": str})
        overlap = set(train["id"]) & set(heldout["id"])
        straddle = set(train["cluster"]) & set(heldout["cluster"])
        print(f"{path.parent.name}: {len(train)} train / {len(heldout)} held out")
        print(f"  id overlap        : {len(overlap)} (must be 0)")
        print(f"  cluster straddle  : {len(straddle)} (must be 0)")
        print(f"  per class         : {meta['per_class']}")
        if overlap or straddle:
            return 1
    print("\nSplit is sound.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify an existing split")
    args = parser.parse_args()
    sys.exit(check() if args.check else (build() and 0))
