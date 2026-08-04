import os
import glob
import cv2
import numpy as np
from extract_cells import extract_cells_from_image

RAW_DATA_DIR = "/Users/dpeleg/local/MicroGlia/Data/raw-data"
OUTPUT_ROOT = "/Users/dpeleg/local/MicroGlia/Data/output"

def process_entire_dataset(
    raw_dir=RAW_DATA_DIR,
    output_root=OUTPUT_ROOT,
    clahe_clip_limit=3.5,
    clahe_tile_grid=(8, 8)
):
    """
    Production Batch Dataset Processing Pipeline (Version 5 - Multi-Tile Adaptive CLAHE Calibration):
    Automatically discovers and processes ALL microscopy images (.jpg, .tif, .png) in raw-data.
    Applies local adaptive CLAHE contrast normalization and adaptive Otsu calibration to equalize
    bright centers and dark corners across all regions of all dataset images.
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    os.makedirs(output_root, exist_ok=True)

    # Discover all supported images
    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    image_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(supported_exts)
    ])

    if not image_files:
        print(f"No supported microscopy images found in {raw_dir}.")
        return

    print("\n=======================================================")
    print(f" PRODUCTION BATCH PROCESSING FOR {len(image_files)} IMAGES")
    print(f" Raw Data Directory: {raw_dir}")
    print(f" Output Root Directory: {output_root}")
    print("=======================================================\n")

    summary_records = []

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(raw_dir, img_name)
        stem, ext = os.path.splitext(img_name)
        img_out_dir = os.path.join(output_root, stem)

        print(f"[{idx}/{len(image_files)}] Processing Image: {img_name}")
        print(f"  -> Output Directory: {img_out_dir}")

        num_extracted = extract_cells_from_image(
            image_path=img_path,
            output_dir=img_out_dir,
            clahe_clip_limit=clahe_clip_limit,
            clahe_tile_grid=clahe_tile_grid
        )

        summary_records.append((img_name, ext.lower(), num_extracted))
        print(f"  [DONE] {img_name}: {num_extracted} atomic sub-cells extracted.\n")

    print("\n=========================================================================")
    print(" PRODUCTION BATCH PROCESSING COMPLETE SUMMARY (VERSION 5)")
    print("=========================================================================")
    print(f"{'Image Name':<32} | {'Format':<6} | {'Extracted Sub-Cells':<18}")
    print("-" * 65)
    total_cells = 0
    for name, fmt, count in summary_records:
        print(f"{name:<32} | {fmt:<6} | {count:<18d}")
        total_cells += count
    print("-" * 65)
    print(f"{'GRAND TOTAL':<32} | {'--':<6} | {total_cells:<18d}")
    print("=========================================================================\n")

if __name__ == "__main__":
    process_entire_dataset()
