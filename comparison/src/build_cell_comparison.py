#!/usr/bin/env python3
"""
build_cell_comparison.py

Writes the cell-by-cell comparison report at
comparison/output/cell_by_cell_comparison.csv

The contour method is the ground truth (GT). There is one row per GT cell,
with the partner cells, area, IoU and match status for every other method:

  matched  - one partner cell, and that partner covers no other GT cell
  split    - the method splits the GT cell into 2+ cells
  merged   - the method joins the GT cell with other GT cells into one cell
  missed   - the method has no cell at this location

Cells that a method found where the GT has no cell follow as "extra" rows.
Cell outlines and overlaps come from build_methods_gallery.build_slide().
"""

import csv
import os

from build_methods_gallery import METHODS, OUTPUT_DIR, build_slide, discover_slides

REPORT_PATH = os.path.join(OUTPUT_DIR, "cell_by_cell_comparison.csv")
GT_METHOD = "contour"


def classify_against_gt(gt_cell, method, cells_by_id):
    """Return (status, partner ids, IoU) of one GT cell against one method."""
    partners = gt_cell["ov"].get(method, [])
    if not partners:
        return "missed", [], 0.0
    shared = sum(px for _, px in partners)
    partner_area = sum(cells_by_id[pid]["maskArea"] for pid, _ in partners)
    iou = shared / (gt_cell["maskArea"] + partner_area - shared)
    ids = [pid for pid, _ in partners]
    if len(partners) >= 2:
        status = "split"
    elif len(cells_by_id[ids[0]]["ov"].get(GT_METHOD, [])) >= 2:
        status = "merged"
    else:
        status = "matched"
    return status, ids, iou


def compare_slide(slide_data):
    """Build the report rows and the per-cell status lookup for one slide."""
    methods = slide_data["methods"]
    others = [key for key, _, _, _ in METHODS if key in methods and key != GT_METHOD]
    by_id = {key: {c["id"]: c for c in methods[key]["cells"]} for key in methods}
    rows = []

    for gt_cell in methods[GT_METHOD]["cells"]:
        row = {
            "slide": slide_data["name"],
            "gt_cell_id": gt_cell["id"],
            "gt_area": gt_cell["area"],
        }
        for key in others:
            status, ids, iou = classify_against_gt(gt_cell, key, by_id[key])
            row[f"{key}_status"] = status
            row[f"{key}_cell_ids"] = ";".join(str(i) for i in ids)
            row[f"{key}_area"] = sum(by_id[key][i]["area"] for i in ids) if ids else ""
            row[f"{key}_iou"] = round(iou, 3) if ids else ""
        rows.append(row)

    for key in others:
        for cell in methods[key]["cells"]:
            if cell["ov"].get(GT_METHOD):
                continue
            row = {"slide": slide_data["name"], "gt_cell_id": "", "gt_area": ""}
            row[f"{key}_status"] = "extra"
            row[f"{key}_cell_ids"] = str(cell["id"])
            row[f"{key}_area"] = cell["area"]
            rows.append(row)

    return rows, others


def build_report(slides=None):
    """Write the CSV report. Returns the rows grouped by slide name."""
    if slides is None:
        slides = [s for s in (build_slide(name) for name in discover_slides()) if s]

    all_rows = []
    present = []
    rows_by_slide = {}
    for slide_data in slides:
        if GT_METHOD not in slide_data["methods"]:
            continue
        rows, others = compare_slide(slide_data)
        rows_by_slide[slide_data["name"]] = rows
        all_rows.extend(rows)
        present.extend(k for k in others if k not in present)

    fieldnames = ["slide", "gt_cell_id", "gt_area"]
    for key in present:
        fieldnames += [f"{key}_status", f"{key}_cell_ids", f"{key}_area", f"{key}_iou"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'Method':<12} | {'matched':>8} | {'split':>6} | {'merged':>7} | {'missed':>7} | {'extra':>6} | {'mean IoU':>8}")
    print("-" * 72)
    for key in present:
        tally = {s: 0 for s in ("matched", "split", "merged", "missed", "extra")}
        ious = []
        for row in all_rows:
            status = row.get(f"{key}_status")
            if status:
                tally[status] += 1
            if row.get(f"{key}_iou") not in (None, ""):
                ious.append(row[f"{key}_iou"])
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        print(f"{key:<12} | {tally['matched']:>8} | {tally['split']:>6} | {tally['merged']:>7} | "
              f"{tally['missed']:>7} | {tally['extra']:>6} | {mean_iou:>8.3f}")
    print(f"\n[SUCCESS] Cell-by-cell report written to: {REPORT_PATH} ({len(all_rows)} rows)")
    return rows_by_slide


if __name__ == "__main__":
    build_report()
