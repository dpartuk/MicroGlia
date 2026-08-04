import os
import glob
import cv2
import argparse
from extract_cells import extract_cells
from step2_breakdown_all import process_all_step2

def process_dataset(raw_dir, output_root_dir):
    """
    Batch process all microscopy images (.jpg, .jpeg, .tif, .tiff) in raw_dir.
    For each image:
    1. Creates output directory named after the image stem: output_root_dir/<image_stem>/
    2. Runs Step 1 cell extraction into output_root_dir/<image_stem>/step-1/
    3. Runs Step 2 Version 2 sub-cell breakdown into output_root_dir/<image_stem>/step-2/
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    os.makedirs(output_root_dir, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    image_files = sorted(list(set(image_files)))
    if not image_files:
        print(f"No image files (.jpg, .tif) found in {raw_dir}")
        return

    print(f"\n=======================================================")
    print(f" STARTING BATCH PROCESSING FOR {len(image_files)} IMAGES")
    print(f" Raw Data Directory: {raw_dir}")
    print(f" Output Root Directory: {output_root_dir}")
    print(f"=======================================================\n")

    summary_results = []

    for idx, img_path in enumerate(image_files, start=1):
        filename = os.path.basename(img_path)
        img_stem, ext = os.path.splitext(filename)

        image_out_dir = os.path.join(output_root_dir, img_stem)
        step1_out_dir = os.path.join(image_out_dir, "step-1")
        step2_out_dir = os.path.join(image_out_dir, "step-2")

        print(f"[{idx}/{len(image_files)}] Processing Image: {filename}")
        print(f"  -> Output Root: {image_out_dir}")

        # Step 1: Extract outer cells
        step1_count = extract_cells(img_path, step1_out_dir)

        # Step 2: Break down outer cells into atomic sub-cells
        process_all_step2(step1_out_dir, img_path, step2_out_dir)

        # Count total sub-cells extracted in Step 2
        total_subcells = 0
        for cell_dir in glob.glob(os.path.join(step2_out_dir, "cell_*")):
            if os.path.isdir(cell_dir):
                total_subcells += len(glob.glob(os.path.join(cell_dir, "subcell_*_extracted.jpg")))

        summary_results.append({
            "filename": filename,
            "stem": img_stem,
            "format": ext.lower(),
            "step1_cells": step1_count,
            "step2_subcells": total_subcells,
            "output_dir": image_out_dir
        })
        print(f"  [DONE] {filename}: {step1_count} outer cell(s) -> {total_subcells} total sub-cell(s)\n")

    print("\n=======================================================")
    print(" BATCH PROCESSING COMPLETE SUMMARY")
    print("=======================================================")
    print(f"{'Image Name':<32} | {'Format':<6} | {'Step 1 Outer':<12} | {'Step 2 Sub-Cells':<16}")
    print("-" * 75)
    grand_total_step1 = 0
    grand_total_step2 = 0
    for res in summary_results:
        print(f"{res['filename']:<32} | {res['format']:<6} | {res['step1_cells']:<12d} | {res['step2_subcells']:<16d}")
        grand_total_step1 += res['step1_cells']
        grand_total_step2 += res['step2_subcells']
    print("-" * 75)
    print(f"{'GRAND TOTAL':<32} | {'--':<6} | {grand_total_step1:<12d} | {grand_total_step2:<16d}")
    print("=======================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process all microscopy images (.jpg, .tif) in raw-data")
    parser.add_argument("--raw-dir", default="/Users/dpeleg/local/MicroGlia/Data/raw-data",
                        help="Directory containing raw microscopy images")
    parser.add_argument("--output-root", default="/Users/dpeleg/local/MicroGlia/Data/output",
                        help="Root output directory for all image results")

    args = parser.parse_args()
    process_dataset(args.raw_dir, args.output_root)
