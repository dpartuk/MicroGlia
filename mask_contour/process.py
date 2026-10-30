#!/usr/bin/env python3
"""
Mask Contour launcher.

Uses only the MASK image (the image with cyan outlines). The RAW image is ignored.
Default: Data/pair-data/1/VID3553_MASK_B2_2_00d00h00m.tif.
Output goes to mask_contour/output/<mask stem>/.

Examples:
  python3 mask_contour/process.py
  python3 mask_contour/process.py --debug-cell 7 14
  python3 mask_contour/process.py --no-debug
  python3 mask_contour/process.py --mask Data/pair-data/1/VID3553_MASK_B2_1_00d00h00m.tif

AI-upscaled (x4) MASK image:
  python3 mask_contour/process.py --mask Data/pair-data/upscaled/VID3553_MASK_B2_2_00d00h00m.tif \\
      --name VID3553_MASK_B2_2_00d00h00m_upscaled_x4 --scale 4 --core-level 60 --close-gaps 5 \\
      --halo-level 30 --fringe-chroma 30 --fringe-radius 8
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from extract_mask_cells import extract_mask_cells

DEFAULT_MASK = os.path.join(PROJECT_ROOT, "Data", "pair-data", "1", "VID3553_MASK_B2_2_00d00h00m.tif")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")


def main():
    parser = argparse.ArgumentParser(description="Cut the area inside each cyan outline, without the outline, from the MASK image")
    parser.add_argument("--mask", default=DEFAULT_MASK, help="MASK image path")
    parser.add_argument("--debug-cell", nargs="*", default=["all"],
                        help="Cell ids for debug traces, or 'all' (default)")
    parser.add_argument("--no-debug", action="store_true", help="Skip per-cell debug traces")
    parser.add_argument("--pocket-std", type=float, default=6.0, help="Interior std below which a region is background")
    parser.add_argument("--name", default=None, help="Output folder name (default: MASK file stem)")
    parser.add_argument("--scale", type=int, default=1, help="Image scale vs. the original (area x scale^2, padding x scale)")
    parser.add_argument("--halo-level", type=int, default=1, help="Minimum cyanness of an outline pixel")
    parser.add_argument("--fringe-chroma", type=int, default=0, help="Minimum chroma of a colour-fringe pixel (0 = off)")
    parser.add_argument("--core-level", type=int, default=100, help="Minimum cyanness of a solid line pixel")
    parser.add_argument("--close-gaps", type=int, default=0, help="Closing kernel (px) that bridges gaps in the line (0 = off)")
    parser.add_argument("--fringe-radius", type=int, default=0, help="Fringe pixels count within this distance of the line")
    args = parser.parse_args()

    mask_path = os.path.abspath(args.mask)
    stem = args.name or os.path.splitext(os.path.basename(mask_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)

    debug_ids = None
    if args.debug_cell and not args.no_debug:
        debug_ids = "all" if args.debug_cell == ["all"] else [int(x) for x in args.debug_cell]

    print("=======================================================================")
    print(" MASK CONTOUR EXTRACTION")
    print(f" MASK:   {mask_path}")
    print(f" Output: {out_dir}")
    print("=======================================================================")

    summary = extract_mask_cells(
        mask_path=mask_path,
        output_dir=out_dir,
        pocket_std=args.pocket_std,
        scale=args.scale,
        halo_level=args.halo_level,
        fringe_chroma=args.fringe_chroma,
        fringe_radius=args.fringe_radius,
        core_level=args.core_level,
        close_gaps=args.close_gaps,
        debug_cell_ids=debug_ids,
    )

    print(f" Cells extracted:              {summary['cells']}")
    print(f" Background pockets rejected:  {summary['rejected_background_pockets']}")
    print(f" Outline pixels claimed:       {summary['outline_px_claimed_pct']}%")
    print(f" Open outline fragments:       {summary['open_outline_fragments']}")
    print(f" Outline pixels removed:       {summary['outline_px_removed_total']}")
    print(f" Cyan pixels left in cells:    {summary['cyan_px_inside_cells']}")
    print(f" Mean cell area:               {summary['mean_cell_area']} px2")
    print("=======================================================================")


if __name__ == "__main__":
    main()
