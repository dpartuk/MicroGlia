"""
Build the web bundle for expert labeling from pair_contour output.

Phase 2 experiment. It reads pair_contour slide folders and writes:

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
"""

import argparse
import base64
import csv
import datetime
import glob
import json
import os
import shutil

import cv2
import numpy as np

import bundle

PAGE_SRC = os.path.join(bundle.ROOT, "labeling", "src", "labeling_page.html")
PAD = 18                    # Cut box padding (px), same as pair_contour
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
                raise RuntimeError("Duplicate cell key " + key)
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


def write_slide(data):
    os.makedirs(bundle.SLIDES_DIR, exist_ok=True)
    with open(os.path.join(bundle.SLIDES_DIR, data["name"] + ".js"), "w") as f:
        f.write("(window.SLIDE_DATA = window.SLIDE_DATA || {})[" + json.dumps(data["name"]) + "] = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")


def export(slide_dirs, force):
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
            print("      --force: replaced. Republish, then run --mark-published.")
        elif locked and os.path.exists(path) and bundle.read_slide_js(path).get("hash") == locked["hash"]:
            print(f"  {name}: locked, unchanged (hash {locked['hash']}), file kept")
            continue
        write_slide(data)
        size = os.path.getsize(path) / 1e6
        print(f"  {name}: {len(data['cells'])} cells, hash {data['hash']}, {size:.2f} MB" + ("" if locked else "  (not published yet)"))
    return blocked


def write_index(repeat_fraction):
    slides = []
    for path in sorted(glob.glob(os.path.join(bundle.SLIDES_DIR, "*.js"))):
        d = bundle.read_slide_js(path)
        slides.append({"name": d["name"], "hash": d["hash"], "width": d["width"], "height": d["height"], "cells": len(d["cells"])})
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
    parser = argparse.ArgumentParser(description="Build the expert labeling web bundle from pair_contour output")
    parser.add_argument("slides", nargs="*", help="pair_contour slide output folders (default: all in pair_contour/output)")
    parser.add_argument("--repeat-fraction", type=float, default=0.10,
                        help="Share of each slide's cells shown a second time (intra-rater check)")
    parser.add_argument("--force", action="store_true", help="Replace locked slides whose masks changed")
    parser.add_argument("--mark-published", action="store_true",
                        help="Only lock every slide file in labeling/web/slides as published (run after a publish)")
    args = parser.parse_args()

    if args.mark_published:
        mark_published()
        return
    slide_dirs = args.slides or sorted(d for d in glob.glob(os.path.join(bundle.PIPE_DIR, "*")) if os.path.isfile(os.path.join(d, "manifest.csv")))
    blocked = export(slide_dirs, args.force)
    write_index(args.repeat_fraction)
    if blocked:
        print(f"{blocked} locked slide(s) kept unchanged because their masks differ. See above.")


if __name__ == "__main__":
    main()
