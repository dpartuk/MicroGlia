"""
Merge expert label exports into labeling/labels.csv and report agreement.

Input: one or more JSON exports from the labeling page ("Export my labels").
Each label carries a cell_key (slide name + mask centroid). The script finds the
key's mask in labeling/reference/ (written by export_web.py) and matches it to the
cells of the CURRENT pair_contour output by mask overlap (IoU). A label whose best
IoU is below --min-iou is not matched and is reported, never guessed.

Agreement:
  * inter-rater: Cohen's kappa for every pair of raters, on cells both labeled (first look).
  * intra-rater: Cohen's kappa between the first look and the second look of the same rater.
"""

import argparse
import csv
import glob
import itertools
import json
import os
from collections import Counter, defaultdict

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REF_DIR = os.path.join(ROOT, "labeling", "reference")
PIPE_DIR = os.path.join(ROOT, "pair_contour", "output")
OUT_CSV = os.path.join(ROOT, "labeling", "labels.csv")


def load_reference(slide):
    z = np.load(os.path.join(REF_DIR, slide, "masks.npz"))
    return dict(zip(z["keys"].tolist(), z["masks"]))


def load_current(slide):
    slide_dir = os.path.join(PIPE_DIR, slide)
    shape = tuple(json.load(open(os.path.join(slide_dir, "summary.json")))["image_size"])[::-1]
    cells = {}
    with open(os.path.join(slide_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            x1, y1, x2, y2 = (int(row[k]) for k in ("x1", "y1", "x2", "y2"))
            full = np.zeros(shape, dtype=bool)
            full[y1:y2, x1:x2] = cv2.imread(os.path.join(slide_dir, row["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 0
            cells[int(row["cell_id"])] = full
    return cells


def best_match(mask, current):
    best_id, best_iou = None, 0.0
    for cid, cm in current.items():
        inter = np.count_nonzero(mask & cm)
        if inter == 0:
            continue
        iou = inter / np.count_nonzero(mask | cm)
        if iou > best_iou:
            best_id, best_iou = cid, iou
    return best_id, best_iou


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def main():
    parser = argparse.ArgumentParser(description="Merge labeling page exports into labels.csv")
    parser.add_argument("exports", nargs="+", help="JSON export files (globs allowed)")
    parser.add_argument("--min-iou", type=float, default=0.5, help="Lowest mask IoU accepted as the same cell")
    parser.add_argument("--out", default=OUT_CSV)
    args = parser.parse_args()

    files = sorted({p for pat in args.exports for p in glob.glob(pat)})
    labels = []
    for path in files:
        data = json.load(open(path))
        if "cells" in data:
            # A shared-store slide document saved as <dir>/labels/<rater id>/slides/<slide>.json
            rater_id = os.path.basename(os.path.dirname(os.path.dirname(path)))
            rows = [dict(v, cell_key=k, slide=data["slide"], repeat=False) for k, v in data["cells"].items()]
            rows += [dict(v, cell_key=k, slide=data["slide"], repeat=True) for k, v in data.get("repeats", {}).items()]
            data = {"rater_id": rater_id, "rater": rater_id, "labels": rows}
        for row in data["labels"]:
            if row.get("label"):
                labels.append(dict(row, rater_id=data["rater_id"], rater=data.get("rater") or data["rater_id"], source=os.path.basename(path)))
    print(f"Read {len(labels)} labels from {len(files)} file(s)")

    ref_cache, cur_cache, match_cache = {}, {}, {}
    out_rows, unmatched = [], 0
    for row in labels:
        slide, key = row["slide"], row["cell_key"]
        if key not in match_cache:
            if slide not in ref_cache:
                ref_cache[slide], cur_cache[slide] = load_reference(slide), load_current(slide)
            match_cache[key] = best_match(ref_cache[slide][key], cur_cache[slide])
        cid, iou = match_cache[key]
        if iou < args.min_iou:
            unmatched += 1
            cid = None
        out_rows.append([slide, key, cid if cid is not None else "", round(iou, 3), row["rater_id"], row["rater"],
                         row["label"], row.get("conf") or "", row.get("note") or "", row["repeat"], row.get("t", ""), row["source"]])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide", "cell_key", "current_cell_id", "iou", "rater_id", "rater", "label", "confidence", "note", "repeat", "time", "source"])
        w.writerows(out_rows)
    print(f"Wrote {args.out}: {len(out_rows)} rows, {unmatched} not matched to a current cell (IoU < {args.min_iou})")

    first = defaultdict(dict)   # rater -> key -> label
    second = defaultdict(dict)
    for r in labels:
        (second if r["repeat"] else first)[r["rater"]][r["cell_key"]] = r["label"]

    print("\nLabels per rater (first look):")
    for rater, d in first.items():
        print(f"  {rater}: {len(d)} cells  {dict(Counter(d.values()))}")
    print("\nInter-rater Cohen's kappa (first look, shared cells):")
    for ra, rb in itertools.combinations(sorted(first), 2):
        keys = sorted(set(first[ra]) & set(first[rb]))
        k = cohen_kappa([first[ra][x] for x in keys], [first[rb][x] for x in keys])
        print(f"  {ra} vs {rb}: kappa {k:.3f} on {len(keys)} cells")
    print("\nIntra-rater Cohen's kappa (first vs second look):")
    for rater in sorted(second):
        keys = sorted(set(second[rater]) & set(first[rater]))
        k = cohen_kappa([first[rater][x] for x in keys], [second[rater][x] for x in keys])
        print(f"  {rater}: kappa {k:.3f} on {len(keys)} cells")


if __name__ == "__main__":
    main()
