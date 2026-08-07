import os
import glob
import cv2
import numpy as np
from cyan_sharpening_pipeline import extract_cells_with_cyan_sharpening

RAW_DATA_DIR = "/Users/dpeleg/local/MicroGlia/Data/raw-data"
OUTPUT_ROOT = "/Users/dpeleg/local/MicroGlia/Data/cyan-sharpening-output"

def process_cyan_sharpening_dataset(
    raw_dir=RAW_DATA_DIR,
    output_root=OUTPUT_ROOT
):
    """
    Batch Processor for Alternative 3: "Cyan-Sharpening"
    Executes the Cyan Chrominance + Boundary Sharpening + CLAHE Fusion pipeline.
    Output is written to Data/cyan-sharpening-output/.
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
    print(f" BATCH PROCESSING ALTERNATIVE 3: 'CYAN-SHARPENING' ({len(image_files)} IMAGES)")
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

        num_extracted = extract_cells_with_cyan_sharpening(
            image_path=img_path,
            output_dir=img_out_dir
        )

        summary_records.append((img_name, ext.lower(), num_extracted))
        print(f"  [DONE] {img_name}: {num_extracted} atomic sub-cells extracted.\n")

    print("\n=========================================================================")
    print(" ALTERNATIVE 3: 'CYAN-SHARPENING' BATCH PROCESSING COMPLETE SUMMARY")
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
    process_cyan_sharpening_dataset()
