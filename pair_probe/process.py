#!/usr/bin/env python3
"""
Pair Probe batch launcher (copy of pair_contour/process.py with the halo band).

Default: the single pair Data/pair-data/1/VID3553_B2_2_00d00h00m.tif
(+ VID3553_MASK_B2_2_00d00h00m.tif). Output goes to pair_probe/output/<raw stem>/.

Examples:
  python3 pair_probe/process.py
  python3 pair_probe/process.py --debug-cell 7 14
  python3 pair_probe/process.py --no-debug
  python3 pair_probe/process.py --raw Data/pair-data/1/VID3553_B2_1_00d00h00m.tif
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
    parser.add_argument("--max-gap", type=int, default=8, help="Bridge outline endpoints closer than this (px), 0 = off")
    parser.add_argument("--halo-px", type=int, default=6, help="Halo band: growth steps (px) from the outline (0 = off)")
    parser.add_argument("--halo-close-px", type=int, default=2, help="Halo band: closing radius (px) for faint gaps (0 = off)")
    parser.add_argument("--halo-leak-px", type=int, default=4, help="Leak rule: drop band pixels this close to signal beyond the band (0 = off)")
    parser.add_argument("--halo-open-px", type=int, default=2, help="Neck rule: opening radius (px) that cuts attached debris (0 = off)")
    parser.add_argument("--entry-px", type=int, default=6, help="Slow entry: growth steps (px) into neighbours through bright pixels (0 = off)")
    parser.add_argument("--entry-min-ring-px", type=int, default=20, help="Slow entry: min free-side halo pixels to measure the ring")
    parser.add_argument("--entry-max-ring-width", type=float, default=5.0, help="Slow entry: max ring width (px); wider = no ring, no entry")
    parser.add_argument("--entry-band-min-px", type=int, default=2, help="Slow entry: smallest band width (px) for cells with no ring")
    parser.add_argument("--entry-strong-k", type=float, default=0.0, help="Slow entry: past the band width, strongly bright pixels (> k * sigma) up to --entry-px (0 = off)")
    parser.add_argument("--entry-shared-tol", type=float, default=0.0, help="Slow entry: shared aura tolerance (px); bright pixels about midway between two outlines (0 = off)")
    parser.add_argument("--halo-max-added", type=float, default=1.0, help="Area rule: max added share of the outline mask area")
    parser.add_argument("--probe-k", type=float, default=3.0, help="Signal: |smoothed RAW - local background| > k * sigma")
    args = parser.parse_args()

    raw_path = os.path.abspath(args.raw)
    mask_path = os.path.abspath(args.mask) if args.mask else mask_path_for(raw_path)
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)

    debug_ids = None
    if args.debug_cell and not args.no_debug:
        debug_ids = "all" if args.debug_cell == ["all"] else [int(x) for x in args.debug_cell]

    print("=======================================================================")
    print(" PAIR PROBE EXTRACTION")
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
        max_gap=args.max_gap,
        halo_px=args.halo_px,
        halo_close_px=args.halo_close_px,
        halo_max_added=args.halo_max_added,
        halo_leak_px=args.halo_leak_px,
        halo_open_px=args.halo_open_px,
        entry_px=args.entry_px,
        entry_min_ring_px=args.entry_min_ring_px,
        entry_max_ring_width=args.entry_max_ring_width,
        entry_band_min_px=args.entry_band_min_px,
        entry_strong_k=args.entry_strong_k,
        entry_shared_tol=args.entry_shared_tol,
        probe_k=args.probe_k,
        debug_cell_ids=debug_ids,
    )

    print(f" Cells extracted:              {summary['cells']}")
    print(f" Excluded (cut by image edge): {summary['cells_excluded_border']}")
    print(f" Background pockets rejected:  {summary['rejected_background_pockets']}")
    print(f" Outline pixels claimed:       {summary['outline_px_claimed_pct']}%")
    print(f" Outline endpoints / bridged:  {summary['outline_endpoints']} / {summary['outline_gaps_bridged']}")
    print(f" Probe noise sigma / false alarm: {summary['probe_sigma']} / {summary['probe_background_false_alarm_pct']}% of bg px")
    print(f" Halo band result per cell:    {summary['halo_status_counts']}")
    print(f" Slow entry: cells / pixels:   {summary['cells_with_entry']} / {summary['entry_px_total']}")
    print(f" Halo pixels added (total):    {summary['halo_px_total']} (gap-filled: {summary['halo_filled_px_total']}, contested: {summary['probe_contested_px']})")
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
