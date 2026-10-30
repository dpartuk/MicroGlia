#!/usr/bin/env python3
"""
validate.py

Checks that an enhanced image keeps the features of the original RAW image.
Cell masks come from the cyan outlines of the MASK image (same field of view).

Tests (all on the original pixel grid; an upscaled candidate is first reduced back
with Lanczos, which is also the round-trip test; Lanczos up + Lanczos down
returns the original within 0.1 grey level, area averaging does not):
  1. Noise:      noise sigma before / after
  2. Residual:   original - enhanced must look like noise, not like cell structure
                 (residual_edge_corr: correlation of |residual| with the edge map;
                  residual_cell_ratio: residual std inside cells / in flat background)
  3. Features:   per-cell mean intensity, texture std, edge strength on the outline band,
                 and aura contrast (bright band just outside the outline vs. background)
  4. Similarity: PSNR and SSIM against the original
"""

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from enhance_methods import estimate_noise_sigma


def ellipse(k):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def cell_masks_from_mask_image(mask_bgr, core_level=100, min_area=60, max_area=30000):
    """
    Return (labels, outline_band, aura_band, flat_background) from the cyan outlines.
    labels: int32 cell id per pixel (interior without cyan pixels), 0 elsewhere.
    """
    b, g, r = [mask_bgr[:, :, i].astype(np.int16) for i in range(3)]
    cyanness = np.clip(np.minimum(b, g) - r, 0, 255)
    core = cyanness >= core_level
    halo = cyanness > 0

    n, regions, stats, _ = cv2.connectedComponentsWithStats(np.where(core, 0, 255).astype(np.uint8), connectivity=4)
    background_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    labels = np.zeros(regions.shape, dtype=np.int32)
    next_id = 1
    for region_id in range(1, n):
        if region_id == background_id or not (min_area <= stats[region_id, cv2.CC_STAT_AREA] <= max_area):
            continue
        labels[(regions == region_id) & ~halo] = next_id
        next_id += 1

    background = regions == background_id
    cells_any = labels > 0
    outline_band = cv2.dilate(core.astype(np.uint8), ellipse(5)) > 0
    near_line = cv2.dilate(halo.astype(np.uint8), ellipse(3)) > 0
    aura_band = (cv2.dilate(cells_any.astype(np.uint8), ellipse(13)) > 0) & background & ~near_line
    flat_background = (cv2.dilate((~background).astype(np.uint8), ellipse(31)) == 0)
    return labels, outline_band, aura_band, flat_background


def edge_map(image):
    """Gradient magnitude after a light blur, so pixel noise does not dominate."""
    s = cv2.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(s, cv2.CV_32F, 1, 0), cv2.Sobel(s, cv2.CV_32F, 0, 1))


def per_cell_stats(image, labels, n_cells):
    ids = labels.ravel()
    vals = image.ravel().astype(np.float64)
    count = np.bincount(ids, minlength=n_cells + 1)[1:]
    total = np.bincount(ids, weights=vals, minlength=n_cells + 1)[1:]
    total_sq = np.bincount(ids, weights=vals * vals, minlength=n_cells + 1)[1:]
    mean = total / np.maximum(count, 1)
    std = np.sqrt(np.maximum(total_sq / np.maximum(count, 1) - mean * mean, 0))
    return mean, std


# Pass limits: an enhancement keeps the features when every check holds
PASS_LIMITS = {
    "residual_edge_corr": ("max", 0.10),    # residual must not follow the cell edges
    "cell_mean_abs_shift": ("max", 0.50),   # grey levels
    "cell_texture_ratio": ("range", (0.95, 1.05)),
    "edge_strength_ratio": ("range", (0.95, 1.05)),
    "aura_contrast_ratio": ("min", 0.95),
}


def check_pass(metrics):
    """Return (passed, list of failed check names)."""
    failed = []
    for key, (kind, limit) in PASS_LIMITS.items():
        value = metrics.get(key)
        if value is None:
            continue
        if kind == "max" and value > limit:
            failed.append(key)
        elif kind == "min" and value < limit:
            failed.append(key)
        elif kind == "range" and not (limit[0] <= value <= limit[1]):
            failed.append(key)
    return not failed, failed


def validate(original, enhanced, masks, scale=1):
    """Return a metrics dict for one enhanced image against the original."""
    labels, outline_band, aura_band, flat_background = masks
    orig = original.astype(np.float32)
    if scale > 1:
        h, w = orig.shape
        enh = cv2.resize(enhanced, (w, h), interpolation=cv2.INTER_LANCZOS4)
    else:
        enh = enhanced.astype(np.float32)

    residual = orig - enh
    edges = edge_map(orig)
    cells_any = labels > 0
    structure_zone = cells_any | outline_band | aura_band
    res_energy = cv2.GaussianBlur(np.abs(residual), (0, 0), 2.0)
    if residual.std() > 0:
        residual_edge_corr = float(np.corrcoef(res_energy[structure_zone], edges[structure_zone])[0, 1])
    else:
        residual_edge_corr = 0.0

    n_cells = int(labels.max())
    mean0, std0 = per_cell_stats(orig, labels, n_cells)
    mean1, std1 = per_cell_stats(enh, labels, n_cells)
    bg_level = float(np.median(orig[flat_background]))
    aura0 = float(orig[aura_band].mean() - bg_level)
    aura1 = float(enh[aura_band].mean() - np.median(enh[flat_background]))

    return {
        "noise_sigma": round(estimate_noise_sigma(np.clip(enh, 0, 255)), 3),
        "flat_background_std": round(float(enh[flat_background].std()), 3),
        "residual_std": round(float(residual.std()), 3),
        "residual_edge_corr": round(residual_edge_corr, 3),
        "residual_cell_ratio": round(float(residual[cells_any].std() / max(residual[flat_background].std(), 1e-6)), 3),
        "cell_mean_abs_shift": round(float(np.mean(np.abs(mean1 - mean0))), 3),
        "cell_texture_ratio": round(float(np.median(std1 / np.maximum(std0, 1e-6))), 3),
        "edge_strength_ratio": round(float(edge_map(enh)[outline_band].mean() / edges[outline_band].mean()), 3),
        "aura_contrast_ratio": round(aura1 / aura0, 3) if aura0 != 0 else None,
        "psnr_db": round(float(peak_signal_noise_ratio(orig, enh, data_range=255)), 2) if residual.std() > 0 else None,
        "ssim": round(float(structural_similarity(orig, enh, data_range=255)), 4),
        "cells_measured": n_cells,
    }
