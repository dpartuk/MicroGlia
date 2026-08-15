import os
import glob
import json
import csv
import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
COMPARISON_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "raw-data")

# Look in self-contained subfolders first, fallback to Data/
BASELINE_ROOT = os.path.join(PROJECT_ROOT, "baseline", "output")
if not os.path.exists(BASELINE_ROOT):
    BASELINE_ROOT = os.path.join(PROJECT_ROOT, "Data", "baseline-output")

SHARPENING_ROOT = os.path.join(PROJECT_ROOT, "sharpened", "output")
if not os.path.exists(SHARPENING_ROOT):
    SHARPENING_ROOT = os.path.join(PROJECT_ROOT, "Data", "boundary-sharpening-output")

OUTPUT_ROOT = os.path.join(COMPARISON_ROOT, "output")


def compare_baseline_vs_sharpening(
    raw_dir=RAW_DATA_DIR,
    base_root=BASELINE_ROOT,
    sharp_root=SHARPENING_ROOT,
    out_dir=OUTPUT_ROOT
):
    """
    Generates a side-by-side comparative analysis comparing Baseline vs Boundary Sharpening
    across all supported microscopy images.
    """
    if not os.path.exists(base_root) or not os.path.exists(sharp_root):
        raise FileNotFoundError(f"Missing input roots: baseline={base_root}, sharpened={sharp_root}")

    os.makedirs(out_dir, exist_ok=True)

    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    raw_stems = sorted([
        os.path.splitext(f)[0] for f in os.listdir(raw_dir)
        if f.lower().endswith(supported_exts)
    ])

    print("\n=======================================================================")
    print(f" COMPARING BASELINE (PRESAIZEN 2026) VS BOUNDARY SHARPENING ({len(raw_stems)} IMAGES)")
    print(f" Raw Data: {raw_dir}")
    print(f" Baseline Source: {base_root}")
    print(f" Sharpened Source: {sharp_root}")
    print(f" Output Directory: {out_dir}")
    print("=======================================================================\n")

    summary_rows = []

    for img_stem in raw_stems:
        b_dir = os.path.join(base_root, img_stem)
        s_dir = os.path.join(sharp_root, img_stem)

        b_count = len(glob.glob(os.path.join(b_dir, "*_extracted.jpg"))) if os.path.exists(b_dir) else 0
        s_count = len(glob.glob(os.path.join(s_dir, "*_original_extracted.jpg"))) if os.path.exists(s_dir) else 0

        diff = s_count - b_count
        diff_str = f"+{diff}" if diff > 0 else (f"{diff}" if diff < 0 else "=")
        pct_change = ((s_count - b_count) / b_count * 100) if b_count > 0 else 0.0

        summary_rows.append({
            "slide_name": img_stem,
            "baseline_count": b_count,
            "sharpened_count": s_count,
            "delta": diff,
            "delta_str": diff_str,
            "pct_change": round(pct_change, 2),
            "comparison_image": f"{img_stem}_comparison.jpg"
        })

        # Generate Side-by-Side Overview Comparison
        b_ov_path = os.path.join(b_dir, "overview_map.jpg")
        s_ov_path = os.path.join(s_dir, "overview_map_on_original.jpg")

        if os.path.exists(b_ov_path) and os.path.exists(s_ov_path):
            b_ov = cv2.imread(b_ov_path)
            s_ov = cv2.imread(s_ov_path)

            target_w, target_h = 720, 530
            r_b = cv2.resize(b_ov, (target_w, target_h))
            r_s = cv2.resize(s_ov, (target_w, target_h))

            # Header bars
            cv2.rectangle(r_b, (0, 0), (target_w, 40), (20, 20, 20), -1)
            cv2.putText(r_b, f"BASELINE: Presaizen 2026 ({b_count} cells)", (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2, cv2.LINE_AA)

            cv2.rectangle(r_s, (0, 0), (target_w, 40), (20, 20, 20), -1)
            cv2.putText(r_s, f"PROPOSED: Boundary Sharpening ({s_count} cells [{diff_str}])", (15, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 229, 255), 2, cv2.LINE_AA)

            comp_side = np.hstack([r_b, r_s])
            cv2.imwrite(os.path.join(out_dir, f"{img_stem}_comparison.jpg"), comp_side)

    # Print Table
    print(f"{'Image Name / Stem':<32} | {'Baseline Cells':<14} | {'Boundary Sharpening':<20} | {'Delta':<8} | {'% Gain':<8}")
    print("-" * 92)
    tot_b, tot_s = 0, 0
    for row in summary_rows:
        print(f"{row['slide_name']:<32} | {row['baseline_count']:<14d} | {row['sharpened_count']:<20d} | {row['delta_str']:<8} | {row['pct_change']:>+6.2f}%")
        tot_b += row["baseline_count"]
        tot_s += row["sharpened_count"]
    print("-" * 92)
    tot_diff = tot_s - tot_b
    tot_diff_str = f"+{tot_diff}" if tot_diff > 0 else (f"{tot_diff}" if tot_diff < 0 else "=")
    tot_pct = ((tot_s - tot_b) / tot_b * 100) if tot_b > 0 else 0.0
    print(f"{'TOTAL':<32} | {tot_b:<14d} | {tot_s:<20d} | {tot_diff_str:<8} | {tot_pct:>+6.2f}%")
    print("=======================================================================\n")

    # Save JSON summary
    summary_data = {
        "total_baseline_cells": tot_b,
        "total_sharpened_cells": tot_s,
        "net_gain": tot_diff,
        "percentage_gain": round(tot_pct, 2),
        "slides": summary_rows
    }
    with open(os.path.join(out_dir, "comparison_summary.json"), "w") as f:
        json.dump(summary_data, f, indent=2)

    # Save CSV summary
    with open(os.path.join(out_dir, "comparison_summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["slide_name", "baseline_count", "sharpened_count", "delta", "delta_str", "pct_change", "comparison_image"])
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Summary metrics and comparison grids saved to: {out_dir}\n")
    return summary_data


if __name__ == "__main__":
    compare_baseline_vs_sharpening()
