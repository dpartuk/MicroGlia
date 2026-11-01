"""
Build the web bundle for expert labeling from the pipeline output (bundle.PIPE_METHOD, now pair_probe).

Phase 2 experiment. It reads pipeline slide folders and writes:

    labeling/web/
        index.html              # The labeling page (copied from labeling/src/labeling_page.html)
        slides.js               # window.LABELING = {...}: every slide file in labeling/web/slides
        slides/<slide>.js       # window.SLIDE_DATA[<slide>] = {...}: RAW image + cells

Each slide file holds the full RAW image once, as a lossless greyscale PNG, and per cell
the mask bits inside its box, the outline polygon and a mask hash (bundle.py). It also records
the slide hash, the pipeline settings and the pipeline commit. The page draws the cut and the
surroundings in the browser, so the cut pixels are the exact RAW values.
Data is in .js files (not .json) so that the page also works when opened from disk.

Published slides are locked in labeling/published_slides.json (--mark-published after a publish).
A locked slide is not rewritten when its masks differ from the lock: the script prints the
difference and keeps the published file. --force rewrites it anyway.

--keep-labels (with --force) keeps the rater's labels on a replaced slide. Each labeled cell of the
published slide is matched to one new cell (best IoU >= bundle.MIN_IOU, one to one). The new cell keeps
the old cell key, so the page shows it as labeled, and records the old mask hash as prev_mh.
merge_labels.py reports such labels as matched_previous_cut.
"""

import argparse
import base64
import csv
import datetime
import glob
import json
import os
import re
import shutil

import cv2
import numpy as np

import bundle

PAGE_SRC = os.path.join(bundle.ROOT, "labeling", "src", "labeling_page.html")
PAD = 18                    # Cut box padding (px), same as the pipeline
OUTLINE_EPSILON = 0.6       # Douglas-Peucker tolerance for the outline polygon (px)


def b64(data):
    return base64.b64encode(data).decode("ascii")


def find_raw(raw_name):
    hits = glob.glob(os.path.join(bundle.ROOT, "Data", "pair-data", "*", raw_name))
    if not hits:
        raise FileNotFoundError("RAW image not found: " + raw_name)
    return cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE)


def build_slide(slide_dir):
    slide = os.path.basename(slide_dir.rstrip("/"))
    summary = json.load(open(os.path.join(slide_dir, "summary.json")))
    raw = find_raw(summary["raw_image"])
    img_h, img_w = raw.shape
    ok, png = cv2.imencode(".png", raw, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    cells, keys = [], set()
    with open(os.path.join(slide_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            x1, y1, x2, y2 = (int(row[k]) for k in ("x1", "y1", "x2", "y2"))
            full = np.zeros((img_h, img_w), dtype=bool)
            full[y1:y2, x1:x2] = cv2.imread(os.path.join(slide_dir, row["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 0
            ys, xs = np.nonzero(full)
            cx, cy = int(round(xs.mean())), int(round(ys.mean()))
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            key = slide + "@" + str(cx) + "_" + str(cy)
            if key in keys:
                # Two cells with the same centroid (a drawn outline inside another): number the later ones
                n = 2
                while f"{key}_{n}" in keys:
                    n += 1
                print(f"      {slide}: cells share the centroid {cx}, {cy}; cell {row['cell_id']} gets key suffix _{n}")
                key = f"{key}_{n}"
            keys.add(key)

            tight = full[box[1]:box[3], box[0]:box[2]]
            contours, _ = cv2.findContours(tight.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            bits = b64(np.packbits(tight.ravel()).tobytes())
            cells.append({
                "key": key,
                "mh": bundle.mask_hash(box, bits),
                "cell_id": int(row["cell_id"]),
                "cx": cx, "cy": cy,
                "box": box,
                "area": int(row["area"]),
                "bits": bits,
                "outline": [cv2.approxPolyDP(c, OUTLINE_EPSILON, True).reshape(-1, 2).tolist() for c in contours],
            })

    return {"name": slide, "hash": bundle.slide_hash(cells), "width": img_w, "height": img_h, "pad": PAD,
            "raw_image": summary["raw_image"], "settings": summary["params"], "commit": bundle.pipeline_commit(),
            "exported": datetime.date.today().isoformat(),
            "raw": "data:image/png;base64," + b64(png.tobytes()), "cells": cells}


def read_labeled_keys(files):
    """{slide: set of cell keys with a label} from store slide documents (ArtifactData out_dir) or page exports."""
    keys = {}
    for path in files:
        doc = json.load(open(path))
        doc = doc.get("data", doc)
        if "cells" in doc:
            rows = [(doc["slide"], k, v) for part in ("cells", "repeats") for k, v in doc.get(part, {}).items()]
        else:
            rows = [(r["slide"], r["cell_key"], r) for r in doc.get("labels", [])]
        for slide, key, rec in rows:
            if rec.get("label"):
                keys.setdefault(slide, set()).add(key)
    return keys


def keep_label_keys(data, published, labeled):
    """Give each labeled published cell's key to its best new cell. Returns the number of kept keys."""
    old_cells = {c["key"]: c for c in published["cells"]}
    new_masks = [(i, bundle.decode_mask(c, data["width"], data["height"])) for i, c in enumerate(data["cells"])]
    taken, kept = {}, 0
    for key in sorted(labeled):
        old = old_cells.get(key)
        if old is None:
            print(f"      {key}: not in the published slide, label not kept")
            continue
        i, iou = bundle.best_match(bundle.decode_mask(old, published["width"], published["height"]), new_masks)
        if i is None or iou < bundle.MIN_IOU or i in taken:
            print(f"      {key}: no single new cell (IoU {iou:.2f}), label not kept")
            continue
        taken[i] = (key, old["mh"], iou)
    in_use = {c["key"] for c in data["cells"]}
    for i, (key, mh, iou) in taken.items():
        cell = data["cells"][i]
        if key != cell["key"] and key in in_use:
            raise RuntimeError(f"Kept key {key} is also the key of another new cell")
        in_use.discard(cell["key"])
        in_use.add(key)
        cell["key"], cell["prev_mh"], cell["prev_iou"] = key, mh, round(iou, 3)
        kept += 1
    data["prev_hash"] = published["hash"]
    data["hash"] = bundle.slide_hash(data["cells"])
    return kept


def write_slide(data):
    os.makedirs(bundle.SLIDES_DIR, exist_ok=True)
    with open(os.path.join(bundle.SLIDES_DIR, data["name"] + ".js"), "w") as f:
        f.write("(window.SLIDE_DATA = window.SLIDE_DATA || {})[" + json.dumps(data["name"]) + "] = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")


def export(slide_dirs, force, labeled):
    lock = bundle.load_lock()
    blocked = 0
    for slide_dir in slide_dirs:
        data = build_slide(slide_dir)
        name, path = data["name"], os.path.join(bundle.SLIDES_DIR, data["name"] + ".js")
        locked = lock.get(name)
        if locked and data["hash"] != locked["hash"]:
            print(f"  {name}: masks differ from the published slide (hash {data['hash']} vs locked {locked['hash']})")
            if os.path.exists(path):
                bundle.report(name, bundle.compare(bundle.slide_file_masks(bundle.read_slide_js(path)),
                                                   [(c["cell_id"], bundle.decode_mask(c, data["width"], data["height"])) for c in data["cells"]]))
            if not force:
                print("      kept the published file. Use --force to replace it (labels on changed cells will not match).")
                blocked += 1
                continue
            if name in labeled and os.path.exists(path):
                kept = keep_label_keys(data, bundle.read_slide_js(path), labeled[name])
                print(f"      --keep-labels: {kept} of {len(labeled[name])} labeled cells keep their key")
            print("      --force: replaced. Republish, then run --mark-published.")
        elif locked and os.path.exists(path) and bundle.read_slide_js(path).get("hash") == locked["hash"]:
            print(f"  {name}: locked, unchanged (hash {locked['hash']}), file kept")
            continue
        write_slide(data)
        size = os.path.getsize(path) / 1e6
        print(f"  {name}: {len(data['cells'])} cells, hash {data['hash']}, {size:.2f} MB" + ("" if locked else "  (not published yet)"))
    return blocked


def slide_order(d):
    """Sort key: data folder (Data/pair-data/<n>), then slide number (B2_<n>)."""
    hits = glob.glob(os.path.join(bundle.ROOT, "Data", "pair-data", "*", d["raw_image"]))
    folder = os.path.basename(os.path.dirname(hits[0])) if hits else "~"
    number = re.search(r"_B2_(\d+)_", d["name"])
    return (folder, int(number.group(1)) if number else 0, d["name"])


def write_index(repeat_fraction):
    slides = []
    for path in glob.glob(os.path.join(bundle.SLIDES_DIR, "*.js")):
        d = bundle.read_slide_js(path)
        slides.append((slide_order(d), {"name": d["name"], "hash": d["hash"], "width": d["width"], "height": d["height"], "cells": len(d["cells"])}))
    slides = [s for _, s in sorted(slides, key=lambda x: x[0])]
    with open(os.path.join(bundle.WEB_DIR, "slides.js"), "w") as f:
        f.write("window.LABELING = ")
        json.dump({"version": 3, "repeat_fraction": repeat_fraction, "slides": slides}, f, indent=1)
        f.write(";\n")
    shutil.copyfile(PAGE_SRC, os.path.join(bundle.WEB_DIR, "index.html"))
    print(f"Wrote {bundle.WEB_DIR}: {len(slides)} slides, {sum(s['cells'] for s in slides)} cells")


def mark_published():
    lock = bundle.load_lock()
    today = datetime.date.today().isoformat()
    for path in sorted(glob.glob(os.path.join(bundle.SLIDES_DIR, "*.js"))):
        d = bundle.read_slide_js(path)
        old = lock.get(d["name"])
        if old and old["hash"] == d["hash"]:
            continue
        lock[d["name"]] = {"hash": d["hash"], "cells": len(d["cells"]), "raw_image": d["raw_image"],
                           "settings": d["settings"], "commit": d["commit"], "published": today}
        print(f"  locked {d['name']}: hash {d['hash']}, {len(d['cells'])} cells" + (" (replaces " + old["hash"] + ")" if old else ""))
    bundle.save_lock(lock)
    print(f"Wrote {bundle.LOCK_PATH}: {len(lock)} slides. Commit it to git.")


def main():
    parser = argparse.ArgumentParser(description="Build the expert labeling web bundle from the pipeline output")
    parser.add_argument("slides", nargs="*", help="Pipeline slide output folders (default: all in " + bundle.PIPE_METHOD + "/output)")
    parser.add_argument("--repeat-fraction", type=float, default=0.10,
                        help="Share of each slide's cells shown a second time (intra-rater check)")
    parser.add_argument("--force", action="store_true", help="Replace locked slides whose masks changed")
    parser.add_argument("--keep-labels", nargs="*", default=[],
                        help="With --force: store documents or exports whose labeled cells keep their key (globs allowed)")
    parser.add_argument("--mark-published", action="store_true",
                        help="Only lock every slide file in labeling/web/slides as published (run after a publish)")
    args = parser.parse_args()

    if args.mark_published:
        mark_published()
        return
    slide_dirs = args.slides or sorted(d for d in glob.glob(os.path.join(bundle.PIPE_DIR, "*")) if os.path.isfile(os.path.join(d, "manifest.csv")))
    labeled = read_labeled_keys(sorted({p for pat in args.keep_labels for p in glob.glob(pat)}))
    blocked = export(slide_dirs, args.force, labeled)
    write_index(args.repeat_fraction)
    if blocked:
        print(f"{blocked} locked slide(s) kept unchanged because their masks differ. See above.")


if __name__ == "__main__":
    main()
