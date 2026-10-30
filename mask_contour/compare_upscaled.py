#!/usr/bin/env python3
"""
Compare two Mask Contour runs of the same field of view at different scales,
e.g. the original MASK image and an AI-upscaled (x4) copy.

The upscaled label map is reduced to the original size (nearest neighbour),
then every original cell is matched to the upscaled cells it overlaps.

Writes mask_contour/output/upscale_comparison.csv (one row per original cell,
plus "extra" rows for upscaled cells with no original partner) and prints a summary.

Example:
  python3 mask_contour/compare_upscaled.py \
      --base VID3553_MASK_B2_2_00d00h00m --upscaled VID3553_MASK_B2_2_00d00h00m_upscaled_x4 --scale 4
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
# Minimum overlap, as a fraction of the smaller cell, to count two cells as partners
MIN_OVERLAP_FRACTION = 0.10


def load_run(name):
    folder = os.path.join(OUTPUT_ROOT, name)
    labels = cv2.imread(os.path.join(folder, "whole_slide_label_map.png"), cv2.IMREAD_UNCHANGED)
    with open(os.path.join(folder, "summary.json")) as f:
        summary = json.load(f)
    return labels.astype(np.int32), summary


def main():
    parser = argparse.ArgumentParser(description="Compare a base and an upscaled Mask Contour run")
    parser.add_argument("--base", default="VID3553_MASK_B2_2_00d00h00m")
    parser.add_argument("--upscaled", default="VID3553_MASK_B2_2_00d00h00m_upscaled_x4")
    parser.add_argument("--scale", type=int, default=4)
    args = parser.parse_args()

    base, base_summary = load_run(args.base)
    up_full, up_summary = load_run(args.upscaled)
    h, w = base.shape
    up = cv2.resize(up_full, (w, h), interpolation=cv2.INTER_NEAREST)

    base_ids = [i for i in np.unique(base) if i > 0]
    up_ids = [i for i in np.unique(up_full) if i > 0]
    base_area = {i: int(c) for i, c in zip(*np.unique(base[base > 0], return_counts=True))}
    up_area_small = {i: int(c) for i, c in zip(*np.unique(up[up > 0], return_counts=True))}
    up_area_full = {i: int(c) for i, c in zip(*np.unique(up_full[up_full > 0], return_counts=True))}

    both = (base > 0) & (up > 0)
    pairs, shared = np.unique(np.stack([base[both], up[both]], axis=1), axis=0, return_counts=True)
    partners_of_base = {i: [] for i in base_ids}
    partners_of_up = {i: [] for i in up_ids}
    for (b_id, u_id), px in zip(pairs, shared):
        b_id, u_id, px = int(b_id), int(u_id), int(px)
        smaller = max(1, min(base_area[b_id], up_area_small.get(u_id, 1)))
        if px / smaller >= MIN_OVERLAP_FRACTION:
            partners_of_base[b_id].append((u_id, px))
            partners_of_up.setdefault(u_id, []).append((b_id, px))

    rows = []
    tally = {"matched": 0, "split": 0, "merged": 0, "missed": 0, "extra": 0}
    ious, area_ratios = [], []
    for b_id in base_ids:
        partners = partners_of_base[b_id]
        if not partners:
            status, iou, ratio = "missed", "", ""
        else:
            ids = [u for u, _ in partners]
            inter = sum(px for _, px in partners)
            u_area = sum(up_area_small[u] for u in ids)
            iou = inter / (base_area[b_id] + u_area - inter)
            ratio = sum(up_area_full[u] for u in ids) / (args.scale ** 2) / base_area[b_id]
            if len(ids) >= 2:
                status = "split"
            elif len(partners_of_up[ids[0]]) >= 2:
                status = "merged"
            else:
                status = "matched"
                ious.append(iou)
                area_ratios.append(ratio)
        tally[status] += 1
        rows.append({
            "base_cell_id": b_id,
            "base_area": base_area[b_id],
            "status": status,
            "upscaled_cell_ids": ";".join(str(u) for u, _ in partners),
            "upscaled_area_div_scale2": round(sum(up_area_full[u] for u, _ in partners) / args.scale ** 2, 1) if partners else "",
            "area_ratio": round(ratio, 3) if ratio != "" else "",
            "iou": round(iou, 3) if iou != "" else "",
        })
    for u_id in up_ids:
        if not partners_of_up.get(u_id):
            tally["extra"] += 1
            rows.append({"base_cell_id": "", "base_area": "", "status": "extra", "upscaled_cell_ids": str(u_id),
                         "upscaled_area_div_scale2": round(up_area_full[u_id] / args.scale ** 2, 1),
                         "area_ratio": "", "iou": ""})

    out_path = os.path.join(OUTPUT_ROOT, "upscale_comparison.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_base = sum(base_area.values())
    total_up = sum(up_area_full.values()) / args.scale ** 2
    print("=======================================================================")
    print(f" UPSCALE COMPARISON: {args.base}  vs  {args.upscaled} (x{args.scale})")
    print("=======================================================================")
    print(f" {'':<34}{'base':>12}{'upscaled':>12}")
    print(f" {'Cells':<34}{base_summary['cells']:>12}{up_summary['cells']:>12}")
    print(f" {'Open outline fragments':<34}{base_summary['open_outline_fragments']:>12}{up_summary['open_outline_fragments']:>12}")
    print(f" {'Outline px claimed (%)':<34}{base_summary['outline_px_claimed_pct']:>12}{up_summary['outline_px_claimed_pct']:>12}")
    print(f" {'Total cell area (base px)':<34}{total_base:>12}{total_up:>12.0f}")
    print(f" {'Mean cell area (base px)':<34}{base_summary['mean_cell_area']:>12}{up_summary['mean_cell_area'] / args.scale ** 2:>12.1f}")
    print("-----------------------------------------------------------------------")
    for k, v in tally.items():
        print(f" {k:<34}{v:>12}")
    if ious:
        print(f" {'Matched: mean IoU':<34}{np.mean(ious):>12.3f}")
        print(f" {'Matched: median area ratio up/base':<34}{np.median(area_ratios):>12.3f}")
    print(f"\n[SUCCESS] Per-cell comparison written to: {out_path}")


if __name__ == "__main__":
    main()
