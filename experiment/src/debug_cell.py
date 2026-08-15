#!/usr/bin/env python3
"""
debug_cell.py

Per-Cell Debugging CLI Utility for the Sharpening Method.
Allows choosing a specific cell ID to debug and extracts all intermediate
outputs at every single algorithmic step with explicit filenames tying
directly to the code.
"""

import argparse
import glob
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SHARPENED_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from boundary_sharpening_pipeline import extract_cells_with_boundary_sharpening

RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "Data", "raw-data")
DEFAULT_OUTPUT_ROOT = os.path.join(SHARPENED_ROOT, "output")


def list_available_slides():
    print("\n=======================================================================")
    print(" AVAILABLE RAW MICROSCOPY SLIDES")
    print("=======================================================================")
    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.lower().endswith(supported_exts)])
    for idx, f in enumerate(raw_files, start=1):
        stem = os.path.splitext(f)[0]
        manifest = os.path.join(DEFAULT_OUTPUT_ROOT, stem, "manifest.csv")
        cell_count_str = "not processed yet"
        if os.path.exists(manifest):
            with open(manifest) as mf:
                cell_count_str = f"{len(mf.readlines()) - 1} cells extracted"
        print(f"  [{idx}] {stem:<32} ({cell_count_str})")
    print("=======================================================================\n")


def main():
    parser = argparse.ArgumentParser(
        description="Per-Cell Step-by-Step Debugging Utility for Sharpening Method"
    )
    parser.add_argument(
        "--cell", "-c", type=str, default="1",
        help="Cell ID(s) to debug (e.g. 7, or comma-separated 3,7,12, or 'all'). Default: 1"
    )
    parser.add_argument(
        "--slide", "-s", type=str, default="JPG_VID2724_B1_3_00d07h00m",
        help="Microscopy slide name or filename. Default: JPG_VID2724_B1_3_00d07h00m"
    )
    parser.add_argument(
        "--image", "-i", type=str, default=None,
        help="Direct path to input image file (overrides --slide)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Output root directory. Default: sharpened/output"
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List all available slides and exit"
    )

    args = parser.parse_args()

    if args.list:
        list_available_slides()
        return

    # Resolve image path
    if args.image:
        image_path = os.path.abspath(args.image)
        if not os.path.exists(image_path):
            print(f"Error: Image not found: {image_path}", file=sys.stderr)
            sys.exit(1)
        stem = os.path.splitext(os.path.basename(image_path))[0]
    else:
        # Match slide in raw-data
        slide_name = args.slide
        matched = glob.glob(os.path.join(RAW_DATA_DIR, f"{slide_name}*"))
        if not matched:
            print(f"Error: Slide '{slide_name}' not found in {RAW_DATA_DIR}", file=sys.stderr)
            list_available_slides()
            sys.exit(1)
        image_path = matched[0]
        stem = os.path.splitext(os.path.basename(image_path))[0]

    output_root = args.output or DEFAULT_OUTPUT_ROOT
    slide_out_dir = os.path.join(output_root, stem)

    # Parse cell IDs
    if args.cell.lower() == "all":
        debug_ids = "all"
    else:
        try:
            debug_ids = [int(x.strip().replace("subcell_", "").replace("cell_", "")) for x in args.cell.split(",") if x.strip()]
        except ValueError:
            print(f"Error: Invalid cell ID format: {args.cell}. Must be integer(s) or 'all'.", file=sys.stderr)
            sys.exit(1)

    print("\n=======================================================================")
    print(" SHARPENING METHOD PER-CELL STEP-BY-STEP DEBUGGER")
    print(f" Slide Image: {os.path.basename(image_path)}")
    print(f" Target Debug Cell ID(s): {debug_ids}")
    print(f" Output Directory: {slide_out_dir}")
    print("=======================================================================\n")

    num_cells = extract_cells_with_boundary_sharpening(
        image_path=image_path,
        output_dir=slide_out_dir,
        debug_cell_ids=debug_ids
    )

    print(f"\n[DONE] Processing complete. Total cells extracted: {num_cells}")
    print("\nDEBUG OUTPUT BREAKDOWN:")

    ids_to_report = range(1, num_cells + 1) if debug_ids == "all" else debug_ids
    for cid in ids_to_report:
        cell_debug_dir = os.path.join(slide_out_dir, "debug", f"cell_{cid:03d}")
        if os.path.exists(cell_debug_dir):
            files = sorted(os.listdir(cell_debug_dir))
            print(f"\n--- Cell #{cid:03d} Debug Folder: {cell_debug_dir} ---")
            for f in files:
                print(f"    • {f}")
            panel_path = os.path.join(cell_debug_dir, "16_debug_step_by_step_panel.jpg")
            print(f"  --> Visual Flowchart Panel: {panel_path}")

    print("\n=======================================================================\n")


if __name__ == "__main__":
    main()
