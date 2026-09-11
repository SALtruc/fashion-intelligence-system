#!/usr/bin/env python
"""Build an INDEPENDENT EVALUATION set for COSC2753 A2 from openly-licensed photos.

Why this exists
---------------
The brief (section 3.3, "Independent Evaluation of your Ultimate Judgement") offers
exactly two ways to evaluate our own judgement independently:

    * "Using data collected completely outside of the scope of your original
       training and evaluation"
    * "Comparing your performance to other works in literature"

`Dataset/ExternalCosmetics/` does NOT satisfy the first one. Those 1,200 crops go
into `train`, so they are inside the scope of our training by construction. This
script builds the other thing: images the models have never seen and never will,
from a different source, with a different photographic style.

That style difference is the point, not a defect. The provided data is 60x80 Myntra
catalogue photography - centred product, white background, studio light. Every
number the team reports is conditional on that convention. An in-the-wild photo set
measures how much of the performance survives when the convention goes away, which
is exactly the HD requirement to "explore how the current status of the data will
affect to the result of the models".

Source: the Openverse API (api.openverse.org), which indexes ~800M openly-licensed
works and returns per-image licence and attribution metadata. No API key needed.
Provenance for every image is recorded in the output CSV so the report can cite it
and the evaluator can re-check it.

Leakage: near-zero by construction (Flickr/Wikimedia photos are not Myntra
catalogue shots), but "by construction" is not evidence. Run the images through
`verify_external_data.py --check` anyway and keep the manifest.

Honest limitations, to state in the report rather than hide:
    1. Keyword search returns false positives. Searching "lipstick" returns a fish
       called a Lipstick Tang. Human review of the contact sheet is MANDATORY.
    2. `season` and `usage` are marketing labels, not visual properties. A human
       has to assign them, and that human is guessing the same way Myntra's
       merchandiser was. Report them as lower-confidence than articleType/gender.
    3. This set is for EVALUATION ONLY. Do not train on it. In-the-wild photos in
       `train` would widen the domain gap already measured at 91.7% separable on
       the cosmetics crops.

Usage
-----
    python collect_external_eval.py --plan                   # show the class plan
    python collect_external_eval.py --collect --per-class 12
    python collect_external_eval.py --sheet                  # contact sheet to label
    python collect_external_eval.py --finalize                # after editing labels.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

sys.stdout.reconfigure(encoding="utf-8")

API = "https://api.openverse.org/v1/images/"
UA = "COSC2753-A2-student-project/1.0 (academic coursework)"

# Licences we accept. cc0/pdm carry no obligations; `by` needs attribution, which we
# record for every image anyway. `by-sa` is deliberately EXCLUDED: share-alike would
# attach copyleft obligations to the derived dataset we hand the evaluator, and that
# is a needless complication for a coursework artefact.
LICENCES = "cc0,pdm,by"

# Anonymous requests accept page_size <= 20. Asking for 21 returns 401
# Unauthorized, not 429 - measured, not guessed. Page deeper instead.
PAGE_SIZE = 20
MAX_PAGES = 8
MIN_SIDE = 200          # reject thumbnails; we need room to centre-crop
TARGET = (60, 80)       # the provided dataset's exact image size
# Overridable so the pipeline can be self-tested against a throwaway directory
# without touching the real collection.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_data import default_data_root  # noqa: E402

OUT_ROOT = Path(os.environ.get("EXTERNAL_EVAL_ROOT",
                               default_data_root() / "ExternalEval"))

# The plan. Two bands on purpose:
#   COMMON - so per-class evaluation numbers rest on more than one image, and so the
#            headline "accuracy off-catalogue" figure is not dominated by oddities.
#            These are the classes the models actually see most.
#   RARE   - the long tail that drives macro-F1, including the classes whose entire
#            support sits in the high-id (test-like) region.
# Query phrasing matters: Openverse ANDs the terms, so multi-word product names score
# zero. Prefer one strong noun plus at most one disambiguator.
PLAN_COMMON = [
    ("Tshirts", "t-shirt"), ("Shirts", "dress shirt"), ("Jeans", "blue jeans"),
    ("Casual Shoes", "sneakers"), ("Watches", "wristwatch"), ("Handbags", "handbag"),
    ("Sunglasses", "sunglasses"), ("Heels", "high heels"), ("Sandals", "sandals"),
    ("Belts", "leather belt"), ("Socks", "socks"), ("Backpacks", "backpack"),
    ("Caps", "baseball cap"), ("Dresses", "dress"), ("Trousers", "trousers"),
    ("Shorts", "shorts"), ("Sweaters", "sweater"), ("Jackets", "jacket"),
    ("Kurtas", "kurta"), ("Sarees", "sari"), ("Flip Flops", "flip flops"),
    ("Wallets", "wallet"), ("Ties", "necktie"), ("Scarves", "scarf"),
    ("Formal Shoes", "oxford shoe"), ("Sports Shoes", "running shoe"),
    ("Earrings", "earrings"), ("Perfume and Body Mist", "perfume bottle"),
]
PLAN_RARE = [
    ("Umbrellas", "umbrella"), ("Water Bottle", "water bottle"),
    ("Footballs", "soccer ball"), ("Basketballs", "basketball"),
    ("Hat", "fedora hat"), ("Key chain", "keychain"),
    ("Trolley Bag", "suitcase"), ("Shoe Laces", "shoelaces"),
    ("Wristbands", "wristband"), ("Tights", "tights"),
    ("Swimwear", "swimsuit"), ("Blazers", "blazer"),
    ("Lehenga Choli", "lehenga"), ("Nehru Jackets", "nehru jacket"),
    ("Salwar", "salwar"), ("Jewellery Set", "necklace set"),
    ("Kajal and Eyeliner", "eyeliner"), ("Lip Gloss", "lip gloss"),
    ("Compact", "powder compact"), ("Concealer", "concealer"),
    ("Nail Polish", "nail polish"), ("Lipstick", "lipstick"),
    ("Eyeshadow", "eyeshadow palette"), ("Rain Jacket", "raincoat"),
    ("Waistcoat", "waistcoat"), ("Jumpsuit", "jumpsuit"),
    ("Robe", "bathrobe"), ("Camisoles", "camisole"),
]
PLAN = PLAN_COMMON + PLAN_RARE


def _get(url: str, timeout: int = 40) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class RateLimited(Exception):
    """Anonymous Openverse quota exhausted. Not a bug - stop cleanly and resume."""


def _api_get(qs: str, attempts: int = 4) -> dict:
    """One search request, with backoff on 429.

    The anonymous tier sends no Retry-After and no X-RateLimit-* headers, so there
    is nothing to read the reset window off. Back off a few times, then give up and
    let the caller checkpoint - re-running later costs nothing thanks to the cache.
    """
    delay = 8
    for i in range(attempts):
        try:
            return json.loads(_get(API + "?" + qs).decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                # NOT a quota problem, and must not be reported as one. The usual
                # cause is a parameter the anonymous tier rejects - page_size > 20
                # is the one that bit us.
                raise RuntimeError(
                    "Openverse returned 401 Unauthorized. This is a rejected "
                    "request parameter, not a rate limit: check page_size <= 20."
                ) from exc
            if exc.code != 429:
                raise
            if i == attempts - 1:
                raise RateLimited(f"HTTP 429 after {attempts} attempts") from exc
            print(f"    . rate limited, waiting {delay}s")
            time.sleep(delay)
            delay *= 2
    raise RateLimited("unreachable")


def search(query: str, want: int, cache_dir: Path) -> list[dict]:
    """Page through Openverse until `want` usable hits or the pages run out.

    Every raw page is cached to disk keyed by (query, page). The anonymous quota is
    small and undocumented, so a run will often die partway; caching makes the next
    run free for everything already fetched, which is what makes a multi-pass
    collection practical at all.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() else "_" for c in query)
    hits, page, seen = [], 1, set()

    while len(hits) < want and page <= MAX_PAGES:
        blob = cache_dir / f"{safe}_p{page}.json"
        if blob.exists():
            payload = json.loads(blob.read_text(encoding="utf-8"))
        else:
            qs = urllib.parse.urlencode({
                "q": query, "license": LICENCES, "extension": "jpg",
                "page_size": PAGE_SIZE, "page": page, "mature": "false",
            })
            payload = _api_get(qs)                     # may raise RateLimited
            blob.write_text(json.dumps(payload), encoding="utf-8")
            time.sleep(0.5)                            # be a polite API citizen

        results = payload.get("results") or []
        if not results:
            break
        for r in results:
            u = r.get("url")
            if not u or u in seen:
                continue
            if min(r.get("width") or 0, r.get("height") or 0) < MIN_SIDE:
                continue
            seen.add(u)
            hits.append(r)
            if len(hits) >= want:
                break
        page += 1
    return hits


def to_dataset_format(raw: bytes) -> Image.Image:
    """Centre-crop to the provided data's 3:4 aspect, then resize to 60x80.

    Matching the provided geometry exactly matters: if the eval images differed in
    shape, any accuracy drop would be confounded by resize artefacts and the whole
    measurement would prove nothing.
    """
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = im.size
    tw, th = TARGET
    want = tw / th
    if w / h > want:                                   # too wide -> trim the sides
        nw = int(round(h * want))
        left = (w - nw) // 2
        im = im.crop((left, 0, left + nw, h))
    else:                                              # too tall -> trim top/bottom
        nh = int(round(w / want))
        top = (h - nh) // 2
        im = im.crop((0, top, w, top + nh))
    return im.resize(TARGET, Image.LANCZOS)


PROV_FIELDS = ["id", "articleType_proposed", "query", "title", "creator", "license",
               "license_url", "source", "foreign_landing_url", "direct_url", "sha256"]


def _load_prov() -> list[dict]:
    prov = OUT_ROOT / "provenance.csv"
    if not prov.exists():
        return []
    with prov.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write_prov(rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT_ROOT / "provenance.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=PROV_FIELDS)
        wr.writeheader()
        wr.writerows(rows)


def _sync_worksheet(rows: list[dict]) -> None:
    """Add a blank labelling line per new candidate; never touch existing ones.

    Overwriting would throw away human labelling work, and this file gets edited by
    hand between collection passes.
    """
    if not rows:
        return
    labels = OUT_ROOT / "labels.csv"
    existing: dict[str, list[str]] = {}
    if labels.exists():
        with labels.open(encoding="utf-8") as fh:
            rd = csv.reader(fh)
            next(rd, None)
            existing = {r[0]: r for r in rd if r}

    with labels.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["id", "keep", "articleType", "gender", "season", "usage", "note"])
        for r in rows:
            wr.writerow(existing.get(
                str(r["id"]),
                [r["id"], "", r["articleType_proposed"], "", "", "", ""]))
    print(f"  worksheet : {labels}  "
          f"({len(rows) - len(existing)} new, {len(existing)} preserved)")


def cmd_collect(per_class: int, bands: str) -> None:
    """Collect candidates, resumably.

    Rate limits mean this will usually take more than one run. Everything is
    checkpointed after every class, so re-running picks up where it stopped and
    already-satisfied classes cost zero API calls.
    """
    plan = {"common": PLAN_COMMON, "rare": PLAN_RARE, "all": PLAN}[bands]
    raw_dir, thumb_dir = OUT_ROOT / "images_raw", OUT_ROOT / "images"
    cache_dir = OUT_ROOT / "search_cache"
    for d in (raw_dir, thumb_dir, cache_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows = _load_prov()
    seen_sha = {r["sha256"] for r in rows}
    next_id = max((int(r["id"]) for r in rows), default=799999) + 1
    have: dict[str, int] = {}
    for r in rows:
        have[r["articleType_proposed"]] = have.get(r["articleType_proposed"], 0) + 1
    if rows:
        print(f"  resuming: {len(rows)} candidates already on disk\n")

    stopped = None
    for article_type, query in plan:
        need = per_class - have.get(article_type, 0)
        if need <= 0:
            continue
        print(f"  {article_type:24} q={query!r}  need {need}")
        try:
            found = search(query, need * 3, cache_dir)   # over-fetch; downloads fail
        except RateLimited as exc:
            stopped = f"{article_type} ({exc})"
            break

        got = 0
        for r in found:
            if got >= need:
                break
            try:
                raw = _get(r["url"])
            except Exception as exc:                     # noqa: BLE001
                print(f"    ! download: {type(exc).__name__}")
                continue
            sha = hashlib.sha256(raw).hexdigest()
            if sha in seen_sha:                          # same photo, two queries
                continue
            try:
                small = to_dataset_format(raw)
            except Exception as exc:                     # noqa: BLE001
                print(f"    ! decode: {type(exc).__name__}")
                continue
            seen_sha.add(sha)
            iid = next_id
            next_id += 1
            (raw_dir / f"{iid}.jpg").write_bytes(raw)
            small.save(thumb_dir / f"{iid}.jpg", quality=95)
            rows.append({
                "id": iid,
                "articleType_proposed": article_type,
                "query": query,
                "title": (r.get("title") or "").replace("\n", " ")[:160],
                "creator": (r.get("creator") or "")[:120],
                "license": f"{r.get('license', '')}-{r.get('license_version', '')}",
                "license_url": r.get("license_url") or "",
                "source": r.get("source") or "",
                "foreign_landing_url": r.get("foreign_landing_url") or "",
                "direct_url": r.get("url"),
                "sha256": sha,
            })
            got += 1
        print(f"    kept {got}")
        _write_prov(rows)                                # checkpoint every class

    _write_prov(rows)
    _sync_worksheet(rows)

    print(f"\n  {len(rows)} candidates total -> {OUT_ROOT}")
    if stopped:
        print(f"\n  STOPPED EARLY at {stopped}")
        print("  The anonymous Openverse quota is spent. Nothing is lost: rerun the")
        print("  same command later and it resumes from the cache for free.")


SHEET_HEAD = """<!doctype html><meta charset="utf-8">
<title>ExternalEval contact sheet</title>
<style>
 body{font:14px/1.5 system-ui,sans-serif;background:#141416;color:#e8e8ea;margin:0;
      padding:24px}
 h2{margin:28px 0 8px;font-size:15px;color:#eda100;border-bottom:1px solid #333;
    padding-bottom:4px}
 .row{display:flex;flex-wrap:wrap;gap:10px}
 figure{margin:0;width:150px}
 .pair{display:flex;gap:3px;align-items:flex-start}
 img{width:110px;height:147px;object-fit:contain;background:#fff;border-radius:3px;
     display:block}
 img.tiny{width:36px;height:48px;image-rendering:pixelated}
 figcaption{font-size:11px;color:#9a9aa2;margin-top:3px;line-height:1.35}
 .t{color:#6f6f78;display:block;max-height:2.7em;overflow:hidden}
 .hint{background:#1e2733;border-left:3px solid #2a78d6;padding:10px 14px;
       margin-bottom:8px}
 code{background:#0d1117;padding:1px 4px;border-radius:3px}
</style>
<div class="hint"><b>How to use.</b> For each image decide <code>keep</code> = 1 or 0
in <code>labels.csv</code>. Reject anything that is not the product itself: a person
wearing it is fine, a shop shelf or an unrelated homonym is not. The <b>title
under each image</b> catches most of the junk on its own. The small pixelated
thumbnail beside each photo is the 60&times;80 the model actually sees &mdash; look at
it, but <b>do not reject an image merely for being hard to read at that size</b>, or
the eval set becomes optimistically easy compared with the provided test data. Then
fill
<code>gender</code> (Men/Women/Boys/Girls/Unisex), <code>season</code>
(Summer/Fall/Winter/Spring) and <code>usage</code> (Casual/Sports/Formal/Ethnic/Smart
Casual/Travel/Party/Home). Correct <code>articleType</code> if the search phrase
mislabelled it.</div>"""


def cmd_sheet() -> None:
    """One scrollable HTML page: every candidate at 2x with its proposed label.

    A contact sheet rather than a folder of files because accept/reject on a few
    hundred images is a minutes-long job when you can see them all at once, and an
    hours-long job when you cannot.
    """
    rows = list(csv.DictReader((OUT_ROOT / "provenance.csv").open(encoding="utf-8")))
    by_class: dict[str, list[dict]] = {}
    for r in rows:
        by_class.setdefault(r["articleType_proposed"], []).append(r)

    parts = [SHEET_HEAD]
    for cls in sorted(by_class):
        parts.append(f"<h2>{cls} &middot; {len(by_class[cls])}</h2><div class='row'>")
        for r in by_class[cls]:
            title = (r["title"] or "").replace("&", "&amp;").replace("<", "&lt;")
            parts.append(
                f"<figure><div class='pair'>"
                f"<img src='images_raw/{r['id']}.jpg' loading='lazy' title='{title}'>"
                f"<img class='tiny' src='images/{r['id']}.jpg' loading='lazy'></div>"
                f"<figcaption><b>{r['id']}</b> &middot; {r['license']}<br>"
                f"<span class='t'>{title}</span></figcaption></figure>")
        parts.append("</div>")

    out = OUT_ROOT / "contact_sheet.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"  {out}  ({len(rows)} images, {len(by_class)} classes)")
    print("  open it in a browser, then edit labels.csv next to it")


def cmd_finalize() -> None:
    """Turn the human-edited worksheet into the shippable evaluation set."""
    import pandas as pd

    prov = pd.read_csv(OUT_ROOT / "provenance.csv")
    lab = pd.read_csv(OUT_ROOT / "labels.csv")
    lab = lab[lab["keep"].astype(str).str.strip().str.lower().isin({"1", "y", "yes"})]
    if lab.empty:
        sys.exit("labels.csv has no rows marked keep=1 - nothing to finalize.")

    df = lab.merge(prov[["id", "license", "license_url", "creator",
                         "foreign_landing_url", "source"]], on="id", how="left")
    blank = [c for c in ("articleType", "gender", "season", "usage")
             if df[c].isna().any()]
    if blank:
        n = int(df[blank].isna().any(axis=1).sum())
        print(f"  WARNING: {n} kept rows have blank {blank}. They stay in the CSV "
              f"but will be skipped for those targets.")

    df["source_dataset"] = "external_eval_openverse_v1"
    cols = ["id", "articleType", "gender", "season", "usage", "note",
            "license", "license_url", "creator", "foreign_landing_url",
            "source", "source_dataset"]
    out = OUT_ROOT / "external_eval.csv"
    df[cols].to_csv(out, index=False)

    # CC BY requires attribution on redistribution, and most of what Openverse
    # returns from Flickr is CC BY 2.0. A creator name buried in a CSV column does
    # not discharge that obligation in any practical sense - a plain credits file
    # shipped beside the images does. Written unconditionally: it costs nothing when
    # everything happens to be CC0, and forgetting it when it is not is the failure
    # mode worth designing out.
    lines = ["Attribution for Dataset/ExternalEval",
             "=" * 36, "",
             "Images collected via the Openverse API (api.openverse.org) and resized",
             "to 60x80 for COSC2753 Assignment 2. Each entry below is one image in",
             "images/, named by its id. Licence terms are per-image; follow the",
             "licence URL for the exact conditions.", ""]
    def _txt(value: object) -> str:
        """Empty CSV cells arrive from pandas as NaN, which is truthy - printing it
        verbatim puts the literal string 'nan' into a credits file we hand to a
        marker. Normalise to an empty string instead."""
        return "" if value is None or pd.isna(value) else str(value).strip()

    for _, r in df.sort_values("id").iterrows():
        creator = _txt(r.get("creator")) or "unknown creator"
        licence = _txt(r.get("license")) or "licence not recorded"
        lines.append(f"{r['id']}.jpg  {creator}  ({licence})")
        landing = _txt(r.get("foreign_landing_url"))
        if landing:
            lines.append(f"           {landing}")
    credits = OUT_ROOT / "ATTRIBUTION.txt"
    credits.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  kept {len(df)} of {len(prov)} candidates -> {out}")
    print(f"  credits: {credits}  (ship this next to the images)")
    for c in ("articleType", "gender", "season", "usage"):
        print(f"    {c:12} {df[c].notna().sum():>4} labelled, "
              f"{df[c].nunique()} distinct")
    print("\n  NEXT, and do not skip it:")
    print(f"    python verify_external_data.py --check {OUT_ROOT / 'images'} "
          f"--labels {out}")
    print("  Leakage here should be zero by construction. Prove it, keep the")
    print("  manifest, and cite the number in the report.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", action="store_true", help="print the class plan, exit")
    ap.add_argument("--collect", action="store_true", help="download candidates")
    ap.add_argument("--sheet", action="store_true", help="build the contact sheet")
    ap.add_argument("--finalize", action="store_true", help="apply labels.csv")
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--bands", choices=["common", "rare", "all"], default="all")
    a = ap.parse_args()

    if a.plan:
        print(f"{len(PLAN_COMMON)} common + {len(PLAN_RARE)} rare = "
              f"{len(PLAN)} classes")
        for label, band in (("COMMON", PLAN_COMMON), ("RARE", PLAN_RARE)):
            print(f"\n  -- {label} --")
            for at, q in band:
                print(f"    {at:24} {q}")
    elif a.collect:
        cmd_collect(a.per_class, a.bands)
    elif a.sheet:
        cmd_sheet()
    elif a.finalize:
        cmd_finalize()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
