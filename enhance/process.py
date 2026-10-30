#!/usr/bin/env python3
"""
Enhance experiment launcher.

Runs the feature-preserving enhancement candidates on one RAW image, validates
each one against the original with cell masks from the paired MASK image, and
writes the images, metrics and an offline comparison report.

Default pair: Data/pair-data/1/VID3553_B2_2_00d00h00m.tif (+ VID3553_MASK_B2_2_00d00h00m.tif).
Output: enhance/output/<raw stem>/ and enhance/output/<raw stem>/report.html

Examples:
  python3 enhance/process.py
  python3 enhance/process.py --raw Data/pair-data/1/VID3553_B2_1_00d00h00m.tif
"""
import argparse
import csv
import json
import os
import sys
import time
import warnings

import cv2
import numpy as np
import tifffile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "src"))

from enhance_methods import run_candidates
from validate import PASS_LIMITS, cell_masks_from_mask_image, check_pass, validate
from build_report import build_report

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

DEFAULT_RAW = os.path.join(PROJECT_ROOT, "Data", "pair-data", "1", "VID3553_B2_2_00d00h00m.tif")
OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "output")
# Residual view: 128 + RESIDUAL_GAIN * (original - enhanced)
RESIDUAL_GAIN = 20


def mask_path_for(raw_path):
    """Return the MASK image path of a RAW image: VID3553_B2_2 -> VID3553_MASK_B2_2."""
    folder, name = os.path.split(raw_path)
    prefix, rest = name.split("_", 1)
    return os.path.join(folder, f"{prefix}_MASK_{rest}")


def save_image(out_dir, name, image):
    """Save an 8-bit PNG for viewing and a 16-bit TIFF (value x 256) that keeps the float precision."""
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), np.clip(np.round(image), 0, 255).astype(np.uint8))
    tifffile.imwrite(os.path.join(out_dir, f"{name}_16bit.tif"),
                     np.clip(np.round(image * 256.0), 0, 65535).astype(np.uint16))


def main():
    parser = argparse.ArgumentParser(description="Feature-preserving enhancement of a RAW image, with validation")
    parser.add_argument("--raw", default=DEFAULT_RAW, help="RAW image path")
    parser.add_argument("--mask", default=None, help="MASK image path (default: derived from --raw)")
    args = parser.parse_args()

    raw_path = os.path.abspath(args.raw)
    mask_path = os.path.abspath(args.mask) if args.mask else mask_path_for(raw_path)
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    out_dir = os.path.join(OUTPUT_ROOT, stem)
    os.makedirs(out_dir, exist_ok=True)

    raw = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE)
    mask_img = cv2.imread(mask_path)
    if raw is None or mask_img is None:
        raise FileNotFoundError(f"Could not read RAW ({raw_path}) or MASK ({mask_path})")

    print("=======================================================================")
    print(" ENHANCE EXPERIMENT")
    print(f" RAW:    {raw_path}")
    print(f" MASK:   {mask_path} (cell masks for validation only)")
    print(f" Output: {out_dir}")
    print("=======================================================================")

    masks = cell_masks_from_mask_image(mask_img)
    t0 = time.time()
    sigma, candidates = run_candidates(raw)
    print(f" Estimated noise sigma: {sigma:.2f} grey levels ({time.time() - t0:.1f} s for all candidates)")

    cv2.imwrite(os.path.join(out_dir, "original.png"), raw)
    raw_f = raw.astype(np.float32)
    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        baseline = validate(raw, raw_f, masks)
        for name, (image, description) in candidates.items():
            scale = image.shape[0] // raw.shape[0]
            metrics = validate(raw, image, masks, scale=scale)
            passed, failed = check_pass(metrics)
            save_image(out_dir, name, image)
            on_grid = image if scale == 1 else cv2.resize(image, raw.shape[::-1], interpolation=cv2.INTER_LANCZOS4)
            residual_view = np.clip(128 + RESIDUAL_GAIN * (raw_f - on_grid), 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(out_dir, f"{name}_residual.png"), residual_view)
            results.append({"name": name, "description": description, "scale": scale,
                            "passed": passed, "failed": failed, "metrics": metrics})

    summary = {
        "raw_image": os.path.basename(raw_path),
        "mask_image": os.path.basename(mask_path),
        "image_size": [raw.shape[1], raw.shape[0]],
        "noise_sigma": round(sigma, 3),
        "residual_gain": RESIDUAL_GAIN,
        "pass_limits": PASS_LIMITS,
        "original": baseline,
        "candidates": results,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    keys = list(baseline.keys())
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "passed", "failed_checks"] + keys)
        writer.writerow(["original", "", ""] + [baseline[k] for k in keys])
        for r in results:
            writer.writerow([r["name"], r["passed"], ";".join(r["failed"])] + [r["metrics"][k] for k in keys])

    report_path = build_report(out_dir, summary)

    print(f"\n {'Candidate':<10} {'Result':<6} {'BgStd':>6} {'ResEdge':>8} {'Texture':>8} {'Edges':>6} {'Aura':>6} {'SSIM':>7}")
    print(f" {'original':<10} {'':<6} {baseline['flat_background_std']:>6} {'':>8} {1.0:>8} {1.0:>6} {1.0:>6} {1.0:>7}")
    for r in results:
        m = r["metrics"]
        print(f" {r['name']:<10} {'PASS' if r['passed'] else 'FAIL':<6} {m['flat_background_std']:>6} "
              f"{m['residual_edge_corr']:>8} {m['cell_texture_ratio']:>8} {m['edge_strength_ratio']:>6} "
              f"{m['aura_contrast_ratio']:>6} {m['ssim']:>7}")
    print(f"\n[SUCCESS] Report written to: {report_path}")


if __name__ == "__main__":
    main()
