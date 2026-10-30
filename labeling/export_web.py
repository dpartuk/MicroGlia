"""
Build the web bundle for expert labeling from pair_contour output.

Phase 2 experiment. It reads pair_contour slide folders and writes:

    labeling/web/
        index.html              # The labeling page (copied from labeling/src/labeling_page.html)
        slides.js               # window.SLIDES = [...]: slide names, sizes, cell counts
        slides/<slide>.js       # window.SLIDE_DATA[<slide>] = {...}: RAW image + cells
    labeling/reference/<slide>/ # Local only, for merge_labels.py; never published
        cells_index.csv         # cell_key, cell_id, centroid, box, area
        masks.npz               # Full-slide boolean mask per cell_key

Each slide file holds the full RAW image once, as a lossless greyscale PNG, and per cell
the mask bits inside its box and the outline polygon. The page draws the cut and the
surroundings in the browser, so the cut pixels are the exact RAW values.
Data is in .js files (not .json) so that the page also works when opened from disk.

Labels are keyed by slide name + mask centroid (cell_key), not by cell number,
because cell numbers change when the pipeline changes.
"""

import argparse
import base64
import csv
import glob
import json
import os
import shutil

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE_DIR = os.path.join(ROOT, "pair_contour", "output")
WEB_DIR = os.path.join(ROOT, "labeling", "web")
REF_DIR = os.path.join(ROOT, "labeling", "reference")
PAGE_SRC = os.path.join(ROOT, "labeling", "src", "labeling_page.html")

PAD = 18                    # Cut box padding (px), same as pair_contour
OUTLINE_EPSILON = 0.6       # Douglas-Peucker tolerance for the outline polygon (px)


def b64(data):
    return base64.b64encode(data).decode("ascii")


def cell_key(slide, cx, cy):
    return slide + "@" + str(cx) + "_" + str(cy)


def find_raw(raw_name):
    hits = glob.glob(os.path.join(ROOT, "Data", "pair-data", "*", raw_name))
    if not hits:
        raise FileNotFoundError("RAW image not found: " + raw_name)
    return cv2.imread(hits[0], cv2.IMREAD_GRAYSCALE)


def export_slide(slide_dir):
    slide = os.path.basename(slide_dir.rstrip("/"))
    summary = json.load(open(os.path.join(slide_dir, "summary.json")))
    raw = find_raw(summary["raw_image"])
    img_h, img_w = raw.shape
    ok, png = cv2.imencode(".png", raw, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    cells, index_rows, masks, checks = [], [], {}, []
    with open(os.path.join(slide_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            x1, y1, x2, y2 = (int(row[k]) for k in ("x1", "y1", "x2", "y2"))
            full = np.zeros((img_h, img_w), dtype=bool)
            full[y1:y2, x1:x2] = cv2.imread(os.path.join(slide_dir, row["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 0
            ys, xs = np.nonzero(full)
            cx, cy = int(round(xs.mean())), int(round(ys.mean()))
            mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            key = cell_key(slide, cx, cy)
            if key in masks:
                raise RuntimeError("Duplicate cell key " + key)
            masks[key] = full

            tight = full[my1:my2, mx1:mx2]
            contours, _ = cv2.findContours(tight.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            outline = [cv2.approxPolyDP(c, OUTLINE_EPSILON, True).reshape(-1, 2).tolist() for c in contours]
            cells.append({
                "key": key,
                "cell_id": int(row["cell_id"]),
                "cx": cx, "cy": cy,
                "box": [mx1, my1, mx2, my2],
                "area": int(row["area"]),
                "bits": b64(np.packbits(tight.ravel()).tobytes()),
                "outline": outline,
            })
            index_rows.append([key, row["cell_id"], cx, cy, mx1, my1, mx2, my2, row["area"]])
            checks.append(int(raw[full].astype(np.int64).sum()))

    data = {"name": slide, "width": img_w, "height": img_h, "pad": PAD,
            "raw": "data:image/png;base64," + b64(png.tobytes()), "cells": cells}
    os.makedirs(os.path.join(WEB_DIR, "slides"), exist_ok=True)
    with open(os.path.join(WEB_DIR, "slides", slide + ".js"), "w") as f:
        f.write("(window.SLIDE_DATA = window.SLIDE_DATA || {})[" + json.dumps(slide) + "] = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")

    ref = os.path.join(REF_DIR, slide)
    os.makedirs(ref, exist_ok=True)
    with open(os.path.join(ref, "cells_index.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell_key", "cell_id", "cx", "cy", "x1", "y1", "x2", "y2", "area", "raw_sum_in_mask"])
        w.writerows(r + [c] for r, c in zip(index_rows, checks))
    np.savez_compressed(os.path.join(ref, "masks.npz"), keys=np.array(list(masks)), masks=np.stack(list(masks.values())))
    return {"name": slide, "width": img_w, "height": img_h, "cells": len(cells)}


def main():
    parser = argparse.ArgumentParser(description="Build the expert labeling web bundle from pair_contour output")
    parser.add_argument("slides", nargs="*", help="pair_contour slide output folders (default: all in pair_contour/output)")
    parser.add_argument("--repeat-fraction", type=float, default=0.10,
                        help="Share of each slide's cells shown a second time (intra-rater check)")
    args = parser.parse_args()

    slide_dirs = args.slides or sorted(d for d in glob.glob(os.path.join(PIPE_DIR, "*")) if os.path.isfile(os.path.join(d, "manifest.csv")))
    index = []
    for slide_dir in slide_dirs:
        entry = export_slide(slide_dir)
        index.append(entry)
        size = os.path.getsize(os.path.join(WEB_DIR, "slides", entry["name"] + ".js")) / 1e6
        print(f"  {entry['name']}: {entry['cells']} cells, {size:.2f} MB")

    with open(os.path.join(WEB_DIR, "slides.js"), "w") as f:
        f.write("window.LABELING = ")
        json.dump({"version": 2, "repeat_fraction": args.repeat_fraction, "slides": index}, f, indent=1)
        f.write(";\n")
    stale = os.path.join(WEB_DIR, "cells.js")
    if os.path.exists(stale):
        os.remove(stale)
    shutil.copyfile(PAGE_SRC, os.path.join(WEB_DIR, "index.html"))
    print(f"Wrote {WEB_DIR}: {len(index)} slides, {sum(e['cells'] for e in index)} cells")


if __name__ == "__main__":
    main()
