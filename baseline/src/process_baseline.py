import os
import glob
import cv2
import numpy as np
from extract_cells_baseline import extract_cells_baseline

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BASELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "raw-data")
OUTPUT_ROOT = os.path.join(BASELINE_ROOT, "output")

def process_baseline_dataset(
    raw_dir=RAW_DATA_DIR,
    output_root=OUTPUT_ROOT
):
    """
    Batch Processor for Baseline Method (Presaizen 2026).
    Discovers all supported microscopy images (.jpg, .tif, .png) in raw-data
    and executes the baseline extraction pipeline.
    Output is written to baseline/output/ (and synced to Data/baseline-output/).
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    os.makedirs(output_root, exist_ok=True)

    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    image_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(supported_exts)
    ])

    if not image_files:
        print(f"No supported microscopy images found in {raw_dir}.")
        return

    print("\n=======================================================================")
    print(f" BATCH PROCESSING BASELINE METHOD ({len(image_files)} IMAGES)")
    print(f" Raw Data Directory: {raw_dir}")
    print(f" Output Root Directory: {output_root}")
    print("=======================================================================\n")

    summary_records = []

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(raw_dir, img_name)
        stem, ext = os.path.splitext(img_name)
        img_out_dir = os.path.join(output_root, stem)

        print(f"[{idx}/{len(image_files)}] Processing Image: {img_name}")
        print(f"  -> Output Directory: {img_out_dir}")

        num_extracted = extract_cells_baseline(
            image_path=img_path,
            output_dir=img_out_dir
        )

        summary_records.append((img_name, ext.lower(), num_extracted))
        print(f"  [DONE] {img_name}: {num_extracted} baseline cells extracted.\n")

    print("\n=========================================================================")
    print(" BASELINE METHOD BATCH PROCESSING COMPLETE SUMMARY")
    print("=========================================================================")
    print(f"{'Image Name':<32} | {'Format':<6} | {'Extracted Cells':<18}")
    print("-" * 65)
    total_cells = 0
    for name, fmt, count in summary_records:
        print(f"{name:<32} | {fmt:<6} | {count:<18d}")
        total_cells += count
    print("-" * 65)
    print(f"{'GRAND TOTAL':<32} | {'--':<6} | {total_cells:<18d}")
    print("=========================================================================\n")

if __name__ == "__main__":
    process_baseline_dataset()
