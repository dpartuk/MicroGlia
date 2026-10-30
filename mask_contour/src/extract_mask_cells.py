#!/usr/bin/env python3
"""
extract_mask_cells.py

Mask Contour Pipeline (Phase 2b).

Input is only the MASK image, e.g. VID3553_MASK_B2_2_00d00h00m.tif
(a phase-contrast image with cyan cell outlines). The RAW image is not used.

Each cell is the area enclosed by a cyan outline, without the outline itself.
The crop is taken from the MASK image.

Steps:
1. Cyan line detection: cyanness = min(B, G) - R. Grey pixels give 0.
   The "core" is the solid line, the "halo" is every cyan-tinted pixel (core + anti-aliased edge).
2. Cell regions: connected components (4-connectivity) of the non-core pixels.
   The largest component is the background; every other enclosed region is a cell.
3. Empty pocket rejection: regions with flat texture are background
   enclosed between touching outlines, not cells.
4. Interior mask: the enclosed region minus every cyan-tinted pixel,
   so no outline pixel is part of the cell.
Options for AI-upscaled MASK images (defaults keep the original behaviour):
  - halo_level: cyanness at or above this value is outline (upscaling adds cyan noise)
  - fringe_chroma / fringe_radius: coloured pixels (max - min channel >= fringe_chroma)
    within fringe_radius px of the core line are outline too (upscaling adds colour fringes)
  - close_gaps: morphological closing (px) of the core line, to bridge gaps the upscaler made
  - scale: multiplies the area limits (scale^2), crop padding and context window (scale)

5. Crop from the MASK image onto a magenta canvas. The MASK image holds only
   grey and cyan-tinted pixels, so magenta never occurs in cell data.
"""

import csv
import glob
import json
import os
import shutil

import cv2
import numpy as np

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# Canvas colour (BGR) outside the cell mask. Pure magenta does not occur in the MASK image.
CANVAS_BGR = (255, 0, 255)


def detect_cyan_lines(mask_bgr, core_level=100, halo_level=1, fringe_chroma=0, fringe_radius=0):
    """Return (cyanness, core, outline) for the cyan outlines of the MASK image."""
    b, g, r = [mask_bgr[:, :, i].astype(np.int16) for i in range(3)]
    cyanness = np.clip(np.minimum(b, g) - r, 0, 255).astype(np.uint8)
    core = cyanness >= core_level
    outline = cyanness >= halo_level
    if fringe_chroma > 0 and fringe_radius > 0:
        chroma = np.maximum(np.maximum(b, g), r) - np.minimum(np.minimum(b, g), r)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * fringe_radius + 1, 2 * fringe_radius + 1))
        near_core = cv2.dilate(core.astype(np.uint8), kernel) > 0
        outline |= near_core & (chroma >= fringe_chroma)
    return cyanness, core, outline


def find_cell_regions(core, min_area, max_area):
    """Label the regions enclosed by the core lines. Returns (labels, background_id, kept ids)."""
    free = np.where(core, 0, 255).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(free, connectivity=4)
    if n <= 1:
        return labels, 0, []
    background_id = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    kept = []
    for region_id in range(1, n):
        if region_id == background_id:
            continue
        area = stats[region_id, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            kept.append((region_id, stats[region_id]))
    return labels, background_id, kept


def cut_on_canvas(image_bgr, mask):
    """Keep image pixels inside mask; paint the rest with CANVAS_BGR."""
    canvas = np.empty_like(image_bgr)
    canvas[:] = CANVAS_BGR
    return np.where(mask[:, :, None] > 0, image_bgr, canvas)


def save_single_cell_debug_trace(debug_dir, cell_idx, steps, marked_img, bbox_tuple, meta, context_half=250):
    """
    Saves a step-by-step debug breakdown for one cell, in the same layout as contour/:
    numbered step images, a collage panel, and cell_debug_summary.json.
    """
    os.makedirs(debug_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox_tuple
    img_h, img_w = marked_img.shape[:2]

    # Step 0: Zoomed context crop around the cell with its bounding box (500x500 window at scale 1)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    zx1, zx2 = max(0, cx - context_half), min(img_w, cx + context_half)
    zy1, zy2 = max(0, cy - context_half), min(img_h, cy + context_half)
    zoomed_location = marked_img[zy1:zy2, zx1:zx2].copy()
    if context_half > 250:
        zoomed_location = cv2.resize(zoomed_location, None, fx=250 / context_half, fy=250 / context_half,
                                     interpolation=cv2.INTER_AREA)
    cv2.putText(zoomed_location, f"Target Cell BBox: ({x1},{y1})-({x2},{y2})", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 229, 255), 1, cv2.LINE_AA)

    step_images = {"00_step0_bounding_box_location.jpg": zoomed_location}
    step_images.update(steps)
    for fname, img_data in step_images.items():
        cv2.imwrite(os.path.join(debug_dir, fname), img_data)

    # Collage of every step (Step 0 + pipeline steps), 4 tiles per row
    panel_items = list(step_images.items())
    tile_size = 280
    cols = 4
    rows = (len(panel_items) + cols - 1) // cols
    panel = np.zeros((tile_size * rows + 40, tile_size * cols, 3), dtype=np.uint8) + 20
    for idx, (fname, item_img) in enumerate(panel_items):
        if item_img.ndim == 2:
            item_img = cv2.cvtColor(item_img, cv2.COLOR_GRAY2BGR)
        px1, py1 = (idx % cols) * tile_size, (idx // cols) * tile_size + 30
        ih, iw = item_img.shape[:2]
        scale = min((tile_size - 10) / iw, (tile_size - 35) / ih)
        rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(item_img, (rw, rh), interpolation=cv2.INTER_NEAREST)
        ox = px1 + (tile_size - rw) // 2
        oy = py1 + 25 + (tile_size - 35 - rh) // 2
        panel[oy:oy + rh, ox:ox + rw] = resized
        cv2.rectangle(panel, (px1 + 2, py1 + 2), (px1 + tile_size - 2, py1 + tile_size - 2), (60, 60, 60), 1)
        title = f"{idx}. " + fname.split("_", 2)[-1].rsplit(".", 1)[0].replace("_", " ")
        cv2.putText(panel, title, (px1 + 8, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 255), 1, cv2.LINE_AA)

    header_text = f"MASK CONTOUR DEBUG TRACE: Cell #{cell_idx:03d} | BBox: ({x1},{y1})-({x2},{y2}) | Area: {meta['area_pixels']} px^2"
    cv2.putText(panel, header_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(debug_dir, "99_debug_step_by_step_panel.jpg"), panel)

    with open(os.path.join(debug_dir, "cell_debug_summary.json"), "w") as jf:
        json.dump(meta, jf, indent=2)


def extract_mask_cells(
    mask_path,
    output_dir,
    core_level=100,
    min_area=60,
    max_area=30000,
    pocket_std=6.0,
    subcell_padding=18,
    halo_level=1,
    fringe_chroma=0,
    fringe_radius=0,
    close_gaps=0,
    scale=1,
    location_max_side=1600,
    debug_cell_ids="all",
):
    """
    Cut the area inside every cyan outline of mask_path, without the outline, from mask_path.

    Args:
        mask_path: MASK image (phase-contrast image with cyan outlines)
        output_dir: Output folder for this image
        core_level: Minimum cyanness of a solid outline pixel
        min_area: Smallest enclosed region (px) kept as a cell
        max_area: Largest enclosed region (px) kept as a cell
        pocket_std: Regions with interior std below this value are background pockets
        subcell_padding: Padding (px) around each cell crop
        halo_level: Minimum cyanness of an outline pixel
        fringe_chroma: Minimum chroma of a colour-fringe pixel (0 = off)
        fringe_radius: Fringe pixels count only within this distance (px) of the core line
        close_gaps: Closing kernel size (px) applied to the core line (0 = off)
        scale: Image scale factor vs. the original (area limits x scale^2, padding x scale)
        location_max_side: The per-cell location image is resized to at most this side (px)
        debug_cell_ids: None, a list of cell ids, or "all"

    Returns:
        Summary dict (also written to summary.json)
    """
    image = cv2.imread(mask_path)
    if image is None:
        raise FileNotFoundError(f"Could not read MASK image: {mask_path}")

    min_area, max_area = min_area * scale * scale, max_area * scale * scale
    subcell_padding = subcell_padding * scale

    os.makedirs(output_dir, exist_ok=True)
    for old_file in glob.glob(os.path.join(output_dir, "subcell_*")):
        os.remove(old_file)
    shutil.rmtree(os.path.join(output_dir, "debug"), ignore_errors=True)

    if debug_cell_ids is None:
        target_debug_ids = set()
    elif str(debug_cell_ids).lower() == "all":
        target_debug_ids = "all"
    else:
        target_debug_ids = {int(x) for x in debug_cell_ids}

    img_h, img_w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.join(output_dir, "00_original_reference.jpg"), image)

    # STEP 1: CYAN LINE DETECTION
    cyanness, core, halo = detect_cyan_lines(image, core_level, halo_level, fringe_chroma, fringe_radius)
    if close_gaps > 0:
        gap_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_gaps, close_gaps))
        core = cv2.morphologyEx(core.astype(np.uint8), cv2.MORPH_CLOSE, gap_kernel) > 0
        halo |= core
    cv2.imwrite(os.path.join(output_dir, "01b_outline_removed_pixels.png"), halo.astype(np.uint8) * 255)
    core_u8 = core.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, "01_cyan_core_lines.png"), core_u8)

    # STEP 2: ENCLOSED CELL REGIONS
    labels, background_id, regions = find_cell_regions(core, min_area, max_area)

    # STEP 3-4: POCKET REJECTION & INTERIOR MASK WITHOUT OUTLINE PIXELS
    cells = []
    rejected_pockets = 0
    for region_id, stat in regions:
        rx, ry, rw, rh = (int(v) for v in stat[:4])
        region = labels[ry:ry + rh, rx:rx + rw] == region_id
        interior = region & ~halo[ry:ry + rh, rx:rx + rw]
        if not interior.any():
            continue

        interior_std = float(np.std(gray[ry:ry + rh, rx:rx + rw][interior]))
        if interior_std < pocket_std:
            rejected_pockets += 1
            continue

        cells.append({
            "region_id": region_id,
            "roi": (rx, ry, rx + rw, ry + rh),
            "region": region,
            "mask": interior,
            "region_area": int(region.sum()),
            "outline_px_removed": int((region & ~interior).sum()),
            "interior_std": interior_std,
        })

    # Sort top-to-bottom, left-to-right like the other methods
    cells.sort(key=lambda c: (c["roi"][1], c["roi"][0]))

    # STEP 5: CROP FROM MASK IMAGE, MANIFEST, OVERVIEW & QA MAPS
    whole_slide_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    region_union = np.zeros((img_h, img_w), dtype=bool)
    label_map = np.zeros((img_h, img_w), dtype=np.uint16)
    overview = image.copy()
    manifest_rows = []

    for idx, cell in enumerate(cells, start=1):
        subcell_id = f"subcell_{idx:03d}"
        ex1, ey1, ex2, ey2 = cell["roi"]
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[ey1:ey2, ex1:ex2][cell["mask"]] = 255
        region_union[ey1:ey2, ex1:ex2] |= cell["region"]

        ys, xs = np.nonzero(cell["mask"])
        x, y = ex1 + int(xs.min()), ey1 + int(ys.min())
        w, h = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        x1, y1 = max(0, x - subcell_padding), max(0, y - subcell_padding)
        x2, y2 = min(img_w, x + w + subcell_padding), min(img_h, y + h + subcell_padding)
        touches_border = x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h
        area = int(cell["mask"].sum())

        crop_mask = full_mask[y1:y2, x1:x2]
        crop_image = image[y1:y2, x1:x2]
        extracted = cut_on_canvas(crop_image, crop_mask)

        # A: interior cut, B: same box with the outline, C: binary mask, D: marked whole-image context
        orig_filename = f"{subcell_id}_original_extracted.png"
        annotated_filename = f"{subcell_id}_annotated.png"
        mask_filename = f"{subcell_id}_mask.png"
        location_filename = f"{subcell_id}_marked.jpg"
        cv2.imwrite(os.path.join(output_dir, orig_filename), extracted)
        cv2.imwrite(os.path.join(output_dir, annotated_filename), crop_image)
        cv2.imwrite(os.path.join(output_dir, mask_filename), crop_mask)

        marked_img = image.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        loc_scale = min(1.0, location_max_side / max(img_h, img_w))
        location_img = marked_img if loc_scale == 1.0 else cv2.resize(marked_img, None, fx=loc_scale, fy=loc_scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(output_dir, location_filename), location_img)

        whole_slide_mask = np.maximum(whole_slide_mask, full_mask)
        label_map[full_mask > 0] = idx

        contours, _ = cv2.findContours(full_mask[ey1:ey2, ex1:ex2], cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE, offset=(ex1, ey1))
        cv2.drawContours(overview, contours, -1, (0, 255, 0), 1)
        cv2.putText(overview, str(idx), (x, max(10, y - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (0, 255, 0), 1, cv2.LINE_AA)

        manifest_rows.append({
            "cell_id": idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "area": area,
            "region_area": cell["region_area"],
            "outline_px_removed": cell["outline_px_removed"],
            "interior_std": round(cell["interior_std"], 2),
            "touches_border": touches_border,
            "original_filename": orig_filename,
            "annotated_filename": annotated_filename,
            "mask_filename": mask_filename,
            "location_filename": location_filename,
        })

        # PER-CELL DEBUG OUTPUTS
        if target_debug_ids == "all" or idx in target_debug_ids:
            region_crop = np.where(labels[y1:y2, x1:x2] == cell["region_id"], 255, 0).astype(np.uint8)
            # Removed view: kept interior in white, removed cyan-tinted pixels in red
            removed_view = cv2.cvtColor(crop_mask, cv2.COLOR_GRAY2BGR)
            removed_view[(region_crop > 0) & (crop_mask == 0)] = (0, 0, 255)
            cyan_crop = cv2.applyColorMap(cyanness[y1:y2, x1:x2], cv2.COLORMAP_OCEAN)
            save_single_cell_debug_trace(
                debug_dir=os.path.join(output_dir, "debug", f"cell_{idx:03d}"),
                cell_idx=idx,
                steps={
                    "01_step1_mask_image_roi.png": crop_image,
                    "02_step1_cyanness_map.png": cyan_crop,
                    "03_step1_cyan_core_lines.png": core_u8[y1:y2, x1:x2],
                    "04_step2_enclosed_region.png": region_crop,
                    "05_step4_outline_removed.png": removed_view,
                    "06_step4_interior_mask.png": crop_mask,
                    "07_step5_cut_from_mask_image.png": extracted,
                },
                marked_img=marked_img,
                bbox_tuple=(x1, y1, x2, y2),
                context_half=250 * scale,
                meta={
                    "cell_id": idx,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1},
                    "area_pixels": area,
                    "region_area_pixels": cell["region_area"],
                    "outline_pixels_removed": cell["outline_px_removed"],
                    "interior_std": round(cell["interior_std"], 2),
                    "touches_border": touches_border,
                    "params": {"core_level": core_level, "halo_level": halo_level,
                               "fringe_chroma": fringe_chroma, "fringe_radius": fringe_radius,
                               "close_gaps": close_gaps, "scale": scale, "pocket_std": pocket_std,
                               "subcell_padding": subcell_padding},
                },
            )

    # QA 1: outline pixels next to no cell region belong to open (leaking) outlines
    qa_size = 4 * scale + 1
    near_cell = cv2.dilate(region_union.astype(np.uint8), np.ones((qa_size, qa_size), np.uint8)) > 0
    unclaimed = core & ~near_cell
    n_frag, _, frag_stats, _ = cv2.connectedComponentsWithStats(unclaimed.astype(np.uint8), connectivity=8)
    open_fragments = int((frag_stats[1:, cv2.CC_STAT_AREA] >= 10 * scale * scale).sum()) if n_frag > 1 else 0
    overview[unclaimed] = (0, 0, 255)
    # QA 2: no outline pixel may be inside any cell mask
    cyan_in_cells = int((halo & (whole_slide_mask > 0)).sum())

    cv2.imwrite(os.path.join(output_dir, "02_cell_outline_overview.jpg"), overview)
    cv2.imwrite(os.path.join(output_dir, "whole_slide_ground_truth_mask.png"), whole_slide_mask)
    cv2.imwrite(os.path.join(output_dir, "whole_slide_label_map.png"), label_map)

    with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["cell_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    core_total = int(core.sum())
    summary = {
        "mask_image": os.path.basename(mask_path),
        "image_size": [img_w, img_h],
        "cells": len(manifest_rows),
        "rejected_background_pockets": rejected_pockets,
        "open_outline_fragments": open_fragments,
        "outline_px_total": core_total,
        "outline_px_claimed_pct": round(100.0 * (core_total - int(unclaimed.sum())) / max(1, core_total), 2),
        "outline_px_removed_total": int(sum(r["outline_px_removed"] for r in manifest_rows)),
        "cyan_px_inside_cells": cyan_in_cells,
        "mean_cell_area": round(float(np.mean([r["area"] for r in manifest_rows])), 1) if manifest_rows else 0,
        "total_foreground_px": int((whole_slide_mask > 0).sum()),
        "canvas_bgr": list(CANVAS_BGR),
        "params": {
            "core_level": core_level, "min_area": min_area, "max_area": max_area,
            "pocket_std": pocket_std, "subcell_padding": subcell_padding,
            "halo_level": halo_level, "fringe_chroma": fringe_chroma,
            "fringe_radius": fringe_radius, "close_gaps": close_gaps, "scale": scale,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
