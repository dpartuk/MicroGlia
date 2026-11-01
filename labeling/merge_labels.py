"""
Merge expert labels into labeling/labels.csv and report agreement.

Input: JSON exports from the labeling page ("Export my labels"), or shared-store slide documents
saved by ArtifactData with out_dir (<dir>/labels/<rater id>/slides/<slide>.json).

Reference = the published slide files (labeling/web/slides/<slide>.js, or --slides-dir with copies
read from the published page). They hold the exact masks the rater saw. For each label:
  1. The slide file must exist and its slide hash must equal the lock (labeling/published_slides.json).
  2. The cell_key must be in the slide file.
  3. If the label carries a mask hash (mh), it must equal the slide file's mask hash for that key.
     A label made on the earlier cut of a kept cell (mh = the cell's prev_mh, export_web.py --keep-labels)
     gets the status matched_previous_cut: the rater saw the earlier cut of the same cell.
  4. The reference mask is matched to the CURRENT pipeline output (bundle.PIPE_METHOD) by IoU (>= --min-iou).
A label that fails a step keeps its row with a status that names the step; it is never guessed.

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

import bundle

OUT_CSV = os.path.join(bundle.ROOT, "labeling", "labels.csv")


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def read_labels(files):
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
    return labels


class Reference:
    """Published slide files, checked against the lock, matched to the current pipeline output."""

    def __init__(self, slides_dir, min_iou):
        self.slides_dir, self.min_iou = slides_dir, min_iou
        self.lock = bundle.load_lock()
        self.slides, self.current, self.matches = {}, {}, {}

    def slide(self, name):
        if name not in self.slides:
            path = os.path.join(self.slides_dir, name + ".js")
            if not os.path.exists(path):
                self.slides[name] = None
                print(f"  {name}: no slide file in {self.slides_dir}")
            else:
                d = bundle.read_slide_js(path)
                locked = self.lock.get(name, {}).get("hash")
                if locked != d.get("hash"):
                    print(f"  {name}: slide file hash {d.get('hash')} differs from the lock ({locked}): labels not used")
                    self.slides[name] = None
                else:
                    self.slides[name] = {"data": d, "cells": {c["key"]: c for c in d["cells"]}}
        return self.slides[name]

    def match(self, name, key):
        """(status, current cell id, iou, mask hash, earlier mask hash) for one cell key."""
        s = self.slide(name)
        if s is None:
            return "no_reference", None, 0.0, "", ""
        cell = s["cells"].get(key)
        if cell is None:
            return "unknown_key", None, 0.0, "", ""
        if key not in self.matches:
            if name not in self.current:
                pipe = os.path.join(bundle.PIPE_DIR, name)
                self.current[name] = bundle.pipeline_masks(pipe) if os.path.isdir(pipe) else []
            mask = bundle.decode_mask(cell, s["data"]["width"], s["data"]["height"])
            cid, iou = bundle.best_match(mask, self.current[name])
            self.matches[key] = ("matched", cid, iou) if iou >= self.min_iou else ("no_current_cell", None, iou)
        return self.matches[key] + (cell["mh"], cell.get("prev_mh", ""))


def main():
    parser = argparse.ArgumentParser(description="Merge labeling page exports or store documents into labels.csv")
    parser.add_argument("exports", nargs="+", help="JSON files (globs allowed)")
    parser.add_argument("--slides-dir", default=bundle.SLIDES_DIR, help="Folder with the published slide files")
    parser.add_argument("--min-iou", type=float, default=bundle.MIN_IOU, help="Lowest mask IoU accepted as the same cell")
    parser.add_argument("--out", default=OUT_CSV)
    args = parser.parse_args()

    files = sorted({p for pat in args.exports for p in glob.glob(pat)})
    labels = read_labels(files)
    print(f"Read {len(labels)} labels from {len(files)} file(s)")

    ref = Reference(args.slides_dir, args.min_iou)
    out_rows, status_count = [], Counter()
    for row in labels:
        status, cid, iou, mh, prev_mh = ref.match(row["slide"], row["cell_key"])
        if status == "matched" and row.get("mh") and row["mh"] != mh:
            if prev_mh and row["mh"] == prev_mh:
                status = "matched_previous_cut"
            else:
                status, cid = "mask_hash_mismatch", None
        row["status"] = status
        status_count[status] += 1
        out_rows.append([row["slide"], row["cell_key"], mh, status, cid if cid is not None else "", round(iou, 3),
                         row["rater_id"], row["rater"], row["label"], row.get("conf") or "", row.get("note") or "",
                         row["repeat"], row.get("t", ""), row["source"]])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide", "cell_key", "mask_hash", "status", "current_cell_id", "iou", "rater_id", "rater", "label",
                    "confidence", "note", "repeat", "time", "source"])
        w.writerows(out_rows)
    print(f"Wrote {args.out}: {len(out_rows)} rows")
    for status, n in sorted(status_count.items()):
        print(f"  {status}: {n}")

    first, second = defaultdict(dict), defaultdict(dict)
    for r in labels:
        (second if r["repeat"] else first)[r["rater"]][r["cell_key"]] = r["label"]
    print("\nLabels per rater (first look):")
    for rater, d in first.items():
        print(f"  {rater}: {len(d)} cells  {dict(Counter(d.values()))}")
    pairs = list(itertools.combinations(sorted(first), 2))
    if pairs:
        print("\nInter-rater Cohen's kappa (first look, shared cells):")
        for ra, rb in pairs:
            keys = sorted(set(first[ra]) & set(first[rb]))
            print(f"  {ra} vs {rb}: kappa {cohen_kappa([first[ra][x] for x in keys], [first[rb][x] for x in keys]):.3f} on {len(keys)} cells")
    print("\nIntra-rater Cohen's kappa (first vs second look):")
    for rater in sorted(second):
        keys = sorted(set(second[rater]) & set(first[rater]))
        print(f"  {rater}: kappa {cohen_kappa([first[rater][x] for x in keys], [second[rater][x] for x in keys]):.3f} on {len(keys)} cells")


if __name__ == "__main__":
    main()
