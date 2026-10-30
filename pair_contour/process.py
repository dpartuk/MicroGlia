#!/usr/bin/env python3
"""
Pair Contour batch launcher.

Default: the single pair Data/pair-data/1/VID3553_B2_2_00d00h00m.tif
(+ VID3553_MASK_B2_2_00d00h00m.tif). Output goes to pair_contour/output/<raw stem>/.

Examples:
  python3 pair_contour/process.py
  python3 pair_contour/process.py --debug-cell 7 14
  python3 pair_contour/process.py --no-debug
  python3 pair_contour/process.py --raw Data/pair-data/1/VID3553_B2_1_00d00h00m.tif
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from extract_pair_cells import extract_pair_cells, mask_path_for

DEFAULT_RAW = os.path.join(PROJECT_ROOT, "Data", "pair-data", "1", "VID3553_B2_2_00d00h00m.tif")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")


def main():
    parser = argparse.ArgumentParser(description="Cut cyan-outlined cells of a MASK image out of its RAW image")
    parser.add_argument("--raw", default=DEFAULT_RAW, help="RAW image path")
    parser.add_argument("--mask", default=None, help="MASK image path (default: derived from --raw)")
    parser.add_argument("--debug-cell", nargs="*", default=["all"],
                        help="Cell ids for debug traces, or 'all' (default)")
    parser.add_argument("--no-debug", action="store_true", help="Skip per-cell debug traces")
    parser.add_argument("--ring-radius", type=int, default=3, help="Outline pixels joined to a cell (px)")
    parser.add_argument("--pocket-std", type=float, default=6.0, help="RAW std below which a region is background")
    parser.add_argument("--aura-free-px", type=int, default=0, help="Aura: outward growth (px) into any non-cell pixel (0 = off)")
    parser.add_argument("--aura-px", type=int, default=0, help="Aura: further outward growth (px) through bright RAW pixels (0 = off)")
    parser.add_argument("--contact-fit-residual", type=float, default=0,
                        help="Max ellipse-fit error (px) for contact completion, 0 = off")
    parser.add_argument("--contact-max-depth", type=int, default=8, help="Contact completion depth limit (px)")
    parser.add_argument("--contact-min-free", type=float, default=0.35,
                        help="Min free share of the mask edge for contact completion (0.5 = stricter)")
    parser.add_argument("--max-gap", type=int, default=8, help="Bridge outline endpoints closer than this (px), 0 = off")
    parser.add_argument("--aura-delta", type=int, default=8, help="Bright = RAW >= background median + this value")
    args = parser.parse_args()

    raw_path = os.path.abspath(args.raw)
    mask_path = os.path.abspath(args.mask) if args.mask else mask_path_for(raw_path)
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)

    debug_ids = None
    if args.debug_cell and not args.no_debug:
        debug_ids = "all" if args.debug_cell == ["all"] else [int(x) for x in args.debug_cell]

    print("=======================================================================")
    print(" PAIR CONTOUR EXTRACTION")
    print(f" RAW:    {raw_path}")
    print(f" MASK:   {mask_path}")
    print(f" Output: {out_dir}")
    print("=======================================================================")

    summary = extract_pair_cells(
        raw_path=raw_path,
        mask_path=mask_path,
        output_dir=out_dir,
        ring_radius=args.ring_radius,
        pocket_std=args.pocket_std,
        aura_free_px=args.aura_free_px,
        aura_px=args.aura_px,
        aura_delta=args.aura_delta,
        max_gap=args.max_gap,
        contact_fit_residual=args.contact_fit_residual,
        contact_max_depth=args.contact_max_depth,
        contact_min_free=args.contact_min_free,
        debug_cell_ids=debug_ids,
    )

    print(f" Cells extracted:              {summary['cells']}")
    print(f" Excluded (cut by image edge): {summary['cells_excluded_border']}")
    print(f" Background pockets rejected:  {summary['rejected_background_pockets']}")
    print(f" Outline pixels claimed:       {summary['outline_px_claimed_pct']}%")
    print(f" Outline endpoints / bridged:  {summary['outline_endpoints']} / {summary['outline_gaps_bridged']}")
    print(f" Aura pixels added (total):    {summary['aura_px_total']}")
    print(f" Contact-completed cells / px: {summary['cells_contact_completed']} / {summary['contact_px_total']}")
    print(f" Noise sigma (local bg):       {summary['noise_sigma']}")
    print(f" Background false-alarm rate:  {summary['background_false_alarm_pct']}% of px")
    print(f" Crowded cells (QA n/a):       {summary['cells_crowded']}")
    print(f" Signal left outside (median): {summary['median_outside_signal_pct']}% of ring px")
    print(f" Cells flagged (> {summary['params']['qa_flag_pct']:.0f}%):        {summary['cells_flagged_outside_signal']}")
    print(f" Mask edge on background (med): {summary['median_edge_background_pct']}%")
    print(f" Uncovered signal blobs:       {summary['uncovered_signal_blobs']} (+ {summary['small_debris_blobs']} small debris)")
    print(f" Mean cell area:               {summary['mean_cell_area']} px2")
    print("=======================================================================")


if __name__ == "__main__":
    main()
