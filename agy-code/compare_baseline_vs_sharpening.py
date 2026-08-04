import os
import glob
import cv2
import numpy as np

RAW_DATA_DIR = "/Users/dpeleg/local/MicroGlia/Data/raw-data"
BASELINE_ROOT = "/Users/dpeleg/local/MicroGlia/Data/baseline-output"
SHARPENING_ROOT = "/Users/dpeleg/local/MicroGlia/Data/boundary-sharpening-output"
COMPARISON_OUT = "/Users/dpeleg/local/MicroGlia/Data/baseline_vs_sharpening_comparison"

def compare_baseline_vs_sharpening(
    raw_dir=RAW_DATA_DIR,
    base_root=BASELINE_ROOT,
    sharp_root=SHARPENING_ROOT,
    out_dir=COMPARISON_OUT
):
    """
    Generates a side-by-side comparative analysis comparing Baseline vs Boundary Sharpening
    STRICTLY for currently existing images in Data/raw-data/.
    """
    if not os.path.exists(base_root) or not os.path.exists(sharp_root):
        print("Error: Either Baseline or Boundary Sharpening output folder does not exist.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Clean old comparison files
    for f in os.listdir(out_dir):
        fp = os.path.join(out_dir, f)
        if os.path.isfile(fp):
            os.remove(fp)

    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    raw_stems = sorted([
        os.path.splitext(f)[0] for f in os.listdir(raw_dir)
        if f.lower().endswith(supported_exts)
    ])

    print("\n=======================================================================")
    print(f" COMPARING BASELINE VS NEW 'BOUNDARY SHARPENING' OPTION ({len(raw_stems)} IMAGES)")
    print(f" Raw Data Directory: {raw_dir}")
    print("=======================================================================\n")

    summary_rows = []

    for img_stem in raw_stems:
        b_dir = os.path.join(base_root, img_stem)
        s_dir = os.path.join(sharp_root, img_stem)

        b_count = len(glob.glob(os.path.join(b_dir, "*_extracted.jpg"))) if os.path.exists(b_dir) else 0
        s_count = len(glob.glob(os.path.join(s_dir, "*_original_extracted.jpg"))) if os.path.exists(s_dir) else 0

        diff = s_count - b_count
        diff_str = f"+{diff}" if diff > 0 else (f"{diff}" if diff < 0 else "=")

        summary_rows.append((img_stem, b_count, s_count, diff_str))

        # Generate Side-by-Side Overview Comparison
        b_ov_path = os.path.join(b_dir, "overview_map.jpg")
        s_ov_path = os.path.join(s_dir, "overview_map_on_original.jpg")

        if os.path.exists(b_ov_path) and os.path.exists(s_ov_path):
            b_ov = cv2.imread(b_ov_path)
            s_ov = cv2.imread(s_ov_path)

            target_w, target_h = 600, 442
            r_b = cv2.resize(b_ov, (target_w, target_h))
            r_s = cv2.resize(s_ov, (target_w, target_h))

            cv2.putText(r_b, f"Baseline ({b_count} cells)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(r_s, f"Boundary Sharpening ({s_count} cells)", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            comp_side = np.hstack([r_b, r_s])
            cv2.imwrite(os.path.join(out_dir, f"{img_stem}_comparison.jpg"), comp_side)

    print(f"{'Image Name / Stem':<32} | {'Baseline Cells':<14} | {'Boundary Sharpening':<20} | {'Difference':<10}")
    print("-" * 84)
    tot_b, tot_s = 0, 0
    for name, b_c, s_c, d_str in summary_rows:
        print(f"{name:<32} | {b_c:<14d} | {s_c:<20d} | {d_str:<10}")
        tot_b += b_c
        tot_s += s_c
    print("-" * 84)
    tot_diff = tot_s - tot_b
    tot_diff_str = f"+{tot_diff}" if tot_diff > 0 else (f"{tot_diff}" if tot_diff < 0 else "=")
    print(f"{'TOTAL':<32} | {tot_b:<14d} | {tot_s:<20d} | {tot_diff_str:<10}")
    print("=======================================================================\n")
    print(f"Comparison grids saved to: {out_dir}\n")

if __name__ == "__main__":
    compare_baseline_vs_sharpening()
