#!/usr/bin/env python3
"""
extract_pair_cells.py

Pair Probe Pipeline (Phase 2). Copy of pair_contour/src/extract_pair_cells.py (frozen):
the outline is the seed of the cell, and a signal probe adds the pixels around it
that belong to the cell (halo and processes). Aura growth and contact completion are removed.

Input is an image pair of the same field of view:
  - RAW  image, e.g. VID3553_B2_2_00d00h00m.tif       (the real pixel data)
  - MASK image, e.g. VID3553_MASK_B2_2_00d00h00m.tif  (the same image with cyan cell outlines)

The cyan outlines in the MASK image define the cells. The pixels are cut from
the RAW image, so the cyan line does not hide the cell edge in the crop.

Steps:
1. Cyan line detection: cyanness = min(B, G) - R. Grey pixels give 0.
   The "core" is the solid line, the "halo" includes the anti-aliased edge.
2. Cell regions: connected components (4-connectivity) of the non-core pixels.
   The largest component is the background; every other enclosed region is a cell.
3. Empty pocket rejection: regions with flat RAW texture are background
   enclosed between touching outlines, not cells.
4. Cell mask: the region plus the line pixels around it (within ring_radius),
   because the outline is drawn on top of the cell boundary.
1b. Gap bridging: skeleton endpoints of the core line that face each other within
   max_gap px are joined with a 1-px line, so an outline with a small gap still
   encloses its cell. Unpaired endpoints (short spurs) are left as they are.
4b. Halo band (problem 1: halo and body parts just outside the outline). The outline
   marks where the cell is, not exactly where it ends. A pixel is signal if the smoothed
   RAW (Gaussian, probe_smooth px) differs from the local background by more than
   probe_k * noise sigma of the unsmoothed RAW, brighter (halo) or darker (cell body).
   All cells grow together, 1 px per step (8-connected), through signal pixels only, up to
   halo_px steps. A pixel goes to the cell that reaches it first; a pixel that two cells
   reach in the same step goes to neither. Other cells' outline masks are never entered.
   Leak rule: a real halo ends inside the band. Where this cell's signal continues past
   halo_px (reached in steps halo_px+1 .. halo_px+halo_leak_px), the band touches something
   else (a neighbour without an outline, debris, a process). Band pixels within
   halo_leak_px of such pixels are removed.
   Neck rule: an opening (disk of radius halo_open_px) of outline mask + band cuts thin
   necks; band parts that no longer touch the outline mask (attached debris) are removed.
   Then a closing (disk of radius halo_close_px) fills the faint gaps of the band, only
   within halo_px of the outline, outside the leak zone, and never with pixels of another
   cell or contested pixels.
   halo_px = 6: on B2_1 the signal share outside the outline falls from 66% (1 px) to its
   plateau at 6 px (B2_2: 7-8 px). Area rule: a cell whose band adds more than
   halo_max_added x its outline mask area keeps its outline-only mask.
   Thin processes that leave the cell (problem 2) need a different rule; not handled here.
4b2. Slow entry into neighbours: the drawn boundary between two touching cells can lie inside
   one of them, so part of its halo is inside the neighbour's outline (example: B2_1 cell 549).
   After the halo band, a second growth enters other cells' outline masks, the pixels their halo
   growth took first (a shared aura between two cells), and contested pixels,
   1 px per step (8-connected), up to entry_px steps. It starts from this cell's own bright pixels
   and takes a pixel only if it is bright (smoothed RAW above the local background by more than
   probe_k * sigma: halo, not the neighbour's dark body), within entry_px of this cell's outline,
   closer to this cell's outline than to the centre of the neighbour it lies in, and on this
   cell's halo ring: its distance from the cell centre is inside the range (10th-90th percentile)
   of the halo pixels the band found on the free side. A cell with fewer than entry_min_ring_px
   such halo pixels, or with a ring wider than entry_max_ring_width px (no ring: an elongated or
   irregular cell), gets no entry.
   The entered pixels also stay in the neighbour: touching cells overlap there.
   Holes that the cell and its entered ring enclose (the cell's own dark edge inside the
   neighbour's outline, between the body and the ring), within entry_px of the outline, are filled.
4d. Border exclusion: a cell whose final mask (with outline and probe pixels)
   reaches the image edge is not cut, because part of it lies outside
   the image. It still blocks its neighbours in step 4b, and it keeps its
   cell number, so the numbering has gaps and other cells keep their numbers.
4e. Taken vs. ignored map (03_taken_vs_ignored.png): RAW image with the cut cells
   filled green and the outlined cells that are not cut filled red (cut by the image
   edge, empty pockets, and size-rejected regions of at least ignored_min_px).
5. Cut from RAW onto a magenta canvas. The RAW images are pure grey (B = G = R),
   so magenta never occurs in cell data and white cell auras stay visible.
6. QA "signal left outside": RAW pixels that differ from the local background by
   more than qa_k * noise sigma. Per cell: their share in a qa_ring_px ring outside
   the final mask (missed cell or aura), and the share of the mask edge that is
   plain background (over-growth). A cell whose ring is mostly covered by
   neighbours is "crowded": its score is kept but not used for flags or means.
   Per image: signal blobs no cell covers (>= qa_blob_px: candidate missed cells).
"""

import csv
import glob
import json
import os
import shutil

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)

# Canvas colour (BGR) outside the cell mask. Pure magenta cannot occur in a grey image.
CANVAS_BGR = (255, 0, 255)


def mask_path_for(raw_path):
    """Return the MASK image path of a RAW image: VID3553_B2_2 -> VID3553_MASK_B2_2."""
    folder, name = os.path.split(raw_path)
    prefix, rest = name.split("_", 1)
    return os.path.join(folder, f"{prefix}_MASK_{rest}")


def detect_cyan_lines(mask_bgr, core_level=100):
    """Return (cyanness, core, halo) for the cyan outlines of the MASK image."""
    b, g, r = [mask_bgr[:, :, i].astype(np.int16) for i in range(3)]
    cyanness = np.clip(np.minimum(b, g) - r, 0, 255).astype(np.uint8)
    core = cyanness >= core_level
    halo = cyanness > 0
    return cyanness, core, halo


def bridge_outline_gaps(core, max_gap, border=3):
    """
    Join skeleton endpoints of the core line that lie within max_gap px of each other.
    Returns (bridged core, endpoint count, list of bridges as ((x1, y1), (x2, y2))).
    """
    skeleton = skeletonize(core)
    neighbours = cv2.filter2D(skeleton.astype(np.uint8), -1, np.ones((3, 3), np.float32),
                              borderType=cv2.BORDER_CONSTANT) - skeleton
    h, w = core.shape
    ends = [(int(y), int(x)) for y, x in np.argwhere(skeleton & (neighbours == 1))
            if border <= y < h - border and border <= x < w - border]
    bridges = []
    if max_gap > 0 and len(ends) >= 2:
        pts = np.array(ends)
        pairs = sorted(cKDTree(pts).query_pairs(max_gap), key=lambda ab: np.hypot(*(pts[ab[0]] - pts[ab[1]])))
        used = set()
        bridged = core.astype(np.uint8)
        for a, b in pairs:
            if a in used or b in used:
                continue
            used.update((a, b))
            (ya, xa), (yb, xb) = ends[a], ends[b]
            cv2.line(bridged, (xa, ya), (xb, yb), 1, 1)
            bridges.append(((xa, ya), (xb, yb)))
        core = bridged > 0
    return core, len(ends), bridges


def local_background(gray, background, sigma=20):
    """Local background level: Gaussian-weighted mean of the background pixels only."""
    weight = cv2.GaussianBlur(background.astype(np.float32), (0, 0), sigma)
    value = cv2.GaussianBlur(np.where(background, gray, 0).astype(np.float32), (0, 0), sigma)
    return value / np.maximum(weight, 1e-3)


def robust_sigma(values):
    """Noise sigma from the median absolute deviation."""
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


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


def probe_growth(seed_owner, signal, max_steps):
    """
    Grow all cells together through signal pixels, 1 px per step (8-connected).
    seed_owner: int32 map, 0 = no cell, k = cell k. Returns (owner, steps, contested):
    the grown owner map, the step at which each pixel was reached, and the pixels
    that two cells reached in the same step (they go to neither cell).
    """
    owner = seed_owner.copy()
    steps = np.zeros(owner.shape, dtype=np.int16)
    contested = np.zeros(owner.shape, dtype=bool)
    kernel = np.ones((3, 3), np.uint8)
    none = np.iinfo(np.int32).max
    for step in range(1, max_steps + 1):
        free = signal & (owner == 0) & ~contested
        highest = cv2.dilate(owner.astype(np.float32), kernel)
        lowest = cv2.erode(np.where(owner > 0, owner, none).astype(np.float32), kernel)
        reached = free & (highest > 0)
        if not reached.any():
            break
        single = reached & (highest == lowest)
        owner[single] = highest[single].astype(np.int32)
        steps[single] = step
        contested |= reached & ~single
    return owner, steps, contested


def cut_on_canvas(image_bgr, mask):
    """Keep image pixels inside mask; paint the rest with CANVAS_BGR."""
    canvas = np.empty_like(image_bgr)
    canvas[:] = CANVAS_BGR
    return np.where(mask[:, :, None] > 0, image_bgr, canvas)


def save_single_cell_debug_trace(debug_dir, cell_idx, steps, marked_img, bbox_tuple, meta):
    """
    Saves a step-by-step debug breakdown for one cell, in the same layout as contour/:
    numbered step images, a collage panel, and cell_debug_summary.json.
    """
    os.makedirs(debug_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox_tuple
    img_h, img_w = marked_img.shape[:2]

    # Step 0: Zoomed context crop around the cell with its bounding box (500x500 window)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    zx1, zx2 = max(0, cx - 250), min(img_w, cx + 250)
    zy1, zy2 = max(0, cy - 250), min(img_h, cy + 250)
    zoomed_location = marked_img[zy1:zy2, zx1:zx2].copy()
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

    header_text = f"PAIR CONTOUR DEBUG TRACE: Cell #{cell_idx:03d} | BBox: ({x1},{y1})-({x2},{y2}) | Area: {meta['area_pixels']} px^2"
    cv2.putText(panel, header_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(debug_dir, "99_debug_step_by_step_panel.jpg"), panel)

    with open(os.path.join(debug_dir, "cell_debug_summary.json"), "w") as jf:
        json.dump(meta, jf, indent=2)


def extract_pair_cells(
    raw_path,
    mask_path,
    output_dir,
    core_level=100,
    min_area=60,
    max_area=30000,
    ring_radius=3,
    pocket_std=6.0,
    max_gap=8,
    halo_px=6,
    halo_close_px=2,
    halo_leak_px=4,
    halo_open_px=2,
    entry_px=6,
    entry_min_ring_px=20,
    entry_max_ring_width=5.0,
    entry_band_fallback=True,
    entry_band_min_px=2,
    entry_strong_k=0.0,
    entry_shared_tol=0.0,
    halo_max_added=1.0,
    probe_k=3.0,
    probe_smooth=1.0,
    qa_ring_px=4,
    qa_k=3.0,
    qa_flag_pct=25.0,
    qa_min_ring_coverage=0.5,
    qa_blob_px=150,
    subcell_padding=18,
    ignored_min_px=20,
    debug_cell_ids="all",
):
    """
    Cut every cyan-outlined cell of mask_path out of raw_path.

    Args:
        raw_path: RAW image (pixels are cut from this image)
        mask_path: MASK image (same image with cyan outlines)
        output_dir: Output folder for this pair
        core_level: Minimum cyanness of a solid outline pixel
        min_area: Smallest enclosed region (px) kept as a cell
        max_area: Largest enclosed region (px) kept as a cell
        ring_radius: Outline pixels within this distance of a region join its mask
        pocket_std: Regions with RAW std below this value are background pockets
        max_gap: Outline endpoints closer than this (px) are bridged (0 = off)
        halo_px: Halo band growth steps (px) from the outline mask (0 = off: outline-only mask)
        halo_close_px: Radius (px) of the closing that fills faint gaps in the band (0 = off)
        halo_leak_px: Leak rule: band pixels this close (px) to signal beyond the band are removed (0 = off)
        halo_open_px: Neck rule: radius (px) of the opening that cuts attached debris (0 = off)
        entry_px: Slow entry: growth steps (px) into neighbours through bright pixels (0 = off)
        entry_min_ring_px: Slow entry needs at least this many free-side halo pixels to measure the ring
        entry_max_ring_width: A ring no wider (px, 10th-90th percentile of radius) than this is a halo ring
        entry_band_fallback: Cells with no ring use a band parallel to the outline, as wide as the free-side halo
        entry_band_min_px: Smallest width (px) of that band
        entry_strong_k: Past that width, strongly bright pixels (> entry_strong_k * sigma) up to entry_px (0 = off)
        entry_shared_tol: Shared aura: bright pixels no farther (+ this tolerance, px) from this outline than from a neighbour's (0 = off)
        halo_max_added: Area rule: the band may add at most this share of the outline mask area
        probe_k: A pixel is signal if |smoothed RAW - local background| > probe_k * noise sigma
        probe_smooth: Gaussian sigma (px) of the RAW smoothing for the signal
        qa_ring_px: Width (px) of the ring outside each mask checked for missed signal
        qa_k: A pixel is signal if |RAW - local background| > qa_k * noise sigma
        qa_flag_pct: Cells with more outside signal than this (%) are flagged
        qa_min_ring_coverage: Below this share of free ring pixels a cell is "crowded"
        qa_blob_px: Uncovered signal blobs of at least this size are candidate missed cells
        subcell_padding: Padding (px) around each cell crop
        ignored_min_px: Size-rejected regions at least this large are shown as ignored cells
        debug_cell_ids: None, a list of cell ids, or "all"

    Returns:
        Summary dict (also written to summary.json)
    """
    raw = cv2.imread(raw_path)
    annotated = cv2.imread(mask_path)
    if raw is None:
        raise FileNotFoundError(f"Could not read RAW image: {raw_path}")
    if annotated is None:
        raise FileNotFoundError(f"Could not read MASK image: {mask_path}")
    if raw.shape != annotated.shape:
        raise ValueError(f"Image sizes differ: RAW {raw.shape} vs MASK {annotated.shape}")

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

    img_h, img_w = raw.shape[:2]
    raw_gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(os.path.join(output_dir, "00_original_reference.jpg"), raw)
    cv2.imwrite(os.path.join(output_dir, "00_annotated_reference.jpg"), annotated)

    # STEP 1: CYAN LINE DETECTION
    cyanness, core, halo = detect_cyan_lines(annotated, core_level)

    # STEP 1b: BRIDGE SMALL GAPS IN THE OUTLINES
    core, outline_endpoints, bridges = bridge_outline_gaps(core, max_gap)
    halo = halo | core
    core_u8 = core.astype(np.uint8) * 255
    cv2.imwrite(os.path.join(output_dir, "01_cyan_core_lines.png"), core_u8)

    # STEP 2: ENCLOSED CELL REGIONS
    labels, background_id, regions = find_cell_regions(core, min_area, max_area)

    # STEP 3-4: POCKET REJECTION & CELL MASK WITH OUTLINE PIXELS
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_radius + 1, 2 * ring_radius + 1))
    cells = []
    rejected_pockets = 0
    pocket_region_ids = []
    for region_id, stat in regions:
        rx, ry, rw, rh = (int(v) for v in stat[:4])
        # ROI around the region, large enough for the dilated ring
        ex1, ey1 = max(0, rx - ring_radius - 1), max(0, ry - ring_radius - 1)
        ex2, ey2 = min(img_w, rx + rw + ring_radius + 1), min(img_h, ry + rh + ring_radius + 1)
        interior = labels[ey1:ey2, ex1:ex2] == region_id

        interior_std = float(np.std(raw_gray[ey1:ey2, ex1:ex2][interior]))
        if interior_std < pocket_std:
            rejected_pockets += 1
            pocket_region_ids.append(region_id)
            continue

        grown = cv2.dilate(interior.astype(np.uint8), ring_kernel) > 0
        cell_roi = interior | (grown & halo[ey1:ey2, ex1:ex2])
        cells.append({
            "region_id": region_id,
            "roi": (ex1, ey1, ex2, ey2),
            "mask": cell_roi,
            "interior_area": int(interior.sum()),
            "ring_px": int((cell_roi & ~interior).sum()),
            "interior_std": interior_std,
        })

    # STEP 4b: HALO BAND (all cells grow together through RAW signal pixels, up to halo_px)
    seed_owner = np.zeros((img_h, img_w), dtype=np.int32)
    for k, cell in enumerate(cells, start=1):
        ex1, ey1, ex2, ey2 = cell["roi"]
        owner_roi = seed_owner[ey1:ey2, ex1:ex2]
        owner_roi[cell["mask"] & (owner_roi == 0)] = k
    near_line = cv2.dilate(core.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    probe_background = (labels == background_id) & ~near_line & (seed_owner == 0)
    probe_dev = raw_gray.astype(np.float32) - local_background(raw_gray, probe_background)
    # Noise sigma from the unsmoothed RAW (as in the QA step). The MAD of the smoothed image is far
    # smaller (pixel noise averages out) and gives a threshold below one grey level.
    probe_sigma = robust_sigma(probe_dev[probe_background])
    if probe_smooth > 0:
        probe_dev = cv2.GaussianBlur(probe_dev, (0, 0), probe_smooth)
    probe_signal = np.abs(probe_dev) > probe_k * probe_sigma
    # Growth is synchronous, so the first halo_px steps are the same as a growth of halo_px only
    owner, probe_steps, contested = probe_growth(seed_owner, probe_signal, halo_px + halo_leak_px)
    beyond_band = probe_steps > halo_px
    leak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_leak_px + 1, 2 * halo_leak_px + 1))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_open_px + 1, 2 * halo_open_px + 1))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * halo_close_px + 1, 2 * halo_close_px + 1))
    # Same numbering key as pair_contour (ROI grown by ring_radius + 1), so cell numbers match
    key_reach = ring_radius + 1
    reach = halo_px + halo_leak_px + 1
    for k, cell in enumerate(cells, start=1):
        ex1, ey1, ex2, ey2 = cell["roi"]
        cell["sort_key"] = (max(0, ey1 - key_reach), max(0, ex1 - key_reach))
        gx1, gy1 = max(0, ex1 - reach), max(0, ey1 - reach)
        gx2, gy2 = min(img_w, ex2 + reach), min(img_h, ey2 + reach)
        base = np.zeros((gy2 - gy1, gx2 - gx1), dtype=bool)
        base[ey1 - gy1:ey2 - gy1, ex1 - gx1:ex2 - gx1] = cell["mask"]
        roi_owner = owner[gy1:gy2, gx1:gx2]
        mine = roi_owner == k
        signal_band = mine & ~beyond_band[gy1:gy2, gx1:gx2] & ~base
        # Leak rule: no band near this cell's signal beyond the band
        leak_zone = np.zeros_like(base)
        if halo_leak_px > 0:
            leak_zone = cv2.dilate((mine & beyond_band[gy1:gy2, gx1:gx2]).astype(np.uint8), leak_kernel) > 0
        band_kept = signal_band & ~leak_zone
        # Neck rule: cut thin necks, keep only band parts that still touch the outline mask
        if halo_open_px > 0 and band_kept.any():
            opened = cv2.morphologyEx((base | band_kept).astype(np.uint8), cv2.MORPH_OPEN, open_kernel) > 0
            band_kept &= opened
            n_parts, parts = cv2.connectedComponents((base | band_kept).astype(np.uint8), connectivity=8)
            touching = np.unique(parts[base])
            band_kept &= np.isin(parts, touching[touching > 0])
        grown = base | band_kept
        filled = np.zeros_like(base)
        if halo_close_px > 0 and halo_px > 0:
            # Fill faint gaps of the band: within halo_px of the outline, outside the leak zone,
            # free and uncontested pixels only
            band = cv2.distanceTransform((~base).astype(np.uint8), cv2.DIST_L2, 5) <= halo_px
            closed = cv2.morphologyEx(grown.astype(np.uint8), cv2.MORPH_CLOSE, close_kernel) > 0
            filled = (closed & ~grown & band & ~leak_zone & ((roi_owner == 0) | mine)
                      & ~contested[gy1:gy2, gx1:gx2])
        added = (grown | filled) & ~base
        if not added.any():
            status = "no_signal"
        elif added.sum() > halo_max_added * base.sum():
            status = "reverted_area"
        else:
            status = "grown"
        cell["outline_mask"] = base
        cell["halo_signal"] = band_kept
        cell["halo_dropped"] = signal_band & ~band_kept
        cell["halo_filled"] = filled
        cell["halo_status"] = status
        cell["halo_px"] = int(added.sum()) if status == "grown" else 0
        cell["halo_filled_px"] = int(filled.sum()) if status == "grown" else 0
        cell["roi"] = (gx1, gy1, gx2, gy2)
        cell["mask"] = base | added if status == "grown" else base

    # STEP 4b2: SLOW ENTRY INTO NEIGHBOURS (bright halo pixels inside other cells' outline masks)
    bright = probe_dev > probe_k * probe_sigma
    centres = np.zeros((len(cells) + 1, 2))
    for k, cell in enumerate(cells, start=1):
        gx1, gy1, gx2, gy2 = cell["roi"]
        ys, xs = np.nonzero(cell["outline_mask"])
        centres[k] = (gy1 + ys.mean(), gx1 + xs.mean())
    step_kernel = np.ones((3, 3), np.uint8)
    for k, cell in enumerate(cells, start=1):
        cell["entry_mask"] = np.zeros_like(cell["mask"])
        cell["entry_px"] = 0
        if entry_px <= 0:
            continue
        gx1, gy1, gx2, gy2 = cell["roi"]
        base = cell["outline_mask"]
        # Neighbour pixels: other cells' outline masks and the pixels their halo growth took first
        roi_seed = owner[gy1:gy2, gx1:gx2]
        others = (roi_seed > 0) & (roi_seed != k) & ~base
        d_own = cv2.distanceTransform((~base).astype(np.uint8), cv2.DIST_L2, 5)
        # Distance from each neighbour pixel to the centre of the neighbour it lies in
        yy, xx = np.mgrid[gy1:gy2, gx1:gx2]
        cy, cx = centres[roi_seed, 0], centres[roi_seed, 1]
        d_centre = np.hypot(yy - cy, xx - cx)
        closer = ~others | (d_own < d_centre)
        # Halo ring of this cell, measured on the halo pixels the band found on the free side
        ring_src = cell["halo_signal"] & bright[gy1:gy2, gx1:gx2]
        if ring_src.sum() < entry_min_ring_px:
            continue
        r_here = np.hypot(yy - centres[k, 0], xx - centres[k, 1])
        r_lo, r_hi = np.percentile(r_here[ring_src], [10, 90])
        if r_hi - r_lo <= entry_max_ring_width:
            on_ring = (r_here >= r_lo) & (r_here <= r_hi)
        elif entry_band_fallback:
            # No ring (elongated or irregular cell): a band parallel to the outline, as wide as
            # the halo the band found on the free side
            band_width = min(float(entry_px), max(float(entry_band_min_px), float(np.percentile(d_own[ring_src], 90))))
            on_ring = d_own <= band_width
            if entry_strong_k > 0:
                # A strong, continuous aura may go on past the free-side width (up to entry_px)
                strong = probe_dev[gy1:gy2, gx1:gx2] > entry_strong_k * probe_sigma
                on_ring |= strong & (d_own <= entry_px)
        else:
            continue
        if entry_shared_tol > 0:
            # Shared aura: a bright wall about midway between this cell's outline and a neighbour's
            # outline belongs to both; a halo that hugs the neighbour's outline stays the neighbour's
            nb_outline = (seed_owner[gy1:gy2, gx1:gx2] > 0) & (seed_owner[gy1:gy2, gx1:gx2] != k) & ~base
            d_nb = cv2.distanceTransform((~nb_outline).astype(np.uint8), cv2.DIST_L2, 5)
            on_ring = on_ring | (d_own <= d_nb + entry_shared_tol)
        allowed = (bright[gy1:gy2, gx1:gx2] & (d_own <= entry_px) & closer & on_ring
                   & (others | contested[gy1:gy2, gx1:gx2]) & ~cell["mask"])
        grown = cell["mask"] & bright[gy1:gy2, gx1:gx2]
        entered = np.zeros_like(base)
        for _ in range(entry_px):
            ring = (cv2.dilate((grown | entered).astype(np.uint8), step_kernel) > 0) & allowed & ~entered
            if not ring.any():
                break
            entered |= ring
        if entered.any():
            # Fill the gap between the body and the entered ring (the cell's own dark edge)
            enclosed = binary_fill_holes(cell["mask"] | entered) & ~(cell["mask"] | entered)
            entered |= enclosed & (d_own <= entry_px)
        if cell["halo_status"] == "grown":
            cell["entry_mask"] = entered
            cell["entry_px"] = int(entered.sum())
            cell["mask"] = cell["mask"] | entered

    # Sort top-to-bottom, left-to-right like the other methods
    cells.sort(key=lambda c: c["sort_key"])

    # STEP 6 (prep): LOCAL BACKGROUND & NOISE FOR THE "SIGNAL LEFT OUTSIDE" QA
    final_union = np.zeros((img_h, img_w), dtype=bool)
    for cell in cells:
        ex1, ey1, ex2, ey2 = cell["roi"]
        final_union[ey1:ey2, ex1:ex2] |= cell["mask"]
    qa_background = (labels == background_id) & ~near_line & ~final_union
    bg_level_map = local_background(raw_gray, qa_background)
    deviation = raw_gray.astype(np.float32) - bg_level_map
    noise_sigma = robust_sigma(deviation[qa_background])
    signal = np.abs(deviation) > qa_k * noise_sigma
    qa_ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * qa_ring_px + 1, 2 * qa_ring_px + 1))
    edge_kernel = np.ones((3, 3), np.uint8)

    # STEP 5: CUT FROM RAW, MANIFEST, OVERVIEW & QA MAPS
    whole_slide_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    outline_union = np.zeros((img_h, img_w), dtype=bool)
    label_map = np.zeros((img_h, img_w), dtype=np.uint16)
    overview = raw.copy()
    manifest_rows = []
    border_excluded = []
    ignored_mask = np.zeros((img_h, img_w), dtype=bool)

    for idx, cell in enumerate(cells, start=1):
        subcell_id = f"subcell_{idx:03d}"
        ex1, ey1, ex2, ey2 = cell["roi"]
        full_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        full_mask[ey1:ey2, ex1:ex2][cell["mask"]] = 255
        outline_union[ey1:ey2, ex1:ex2] |= cell["outline_mask"]

        ys, xs = np.nonzero(cell["mask"])
        x, y = ex1 + int(xs.min()), ey1 + int(ys.min())
        w, h = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
        x1, y1 = max(0, x - subcell_padding), max(0, y - subcell_padding)
        x2, y2 = min(img_w, x + w + subcell_padding), min(img_h, y + h + subcell_padding)
        touches_border = x <= 0 or y <= 0 or x + w >= img_w or y + h >= img_h
        area = int(cell["mask"].sum())

        # STEP 4d: BORDER EXCLUSION - part of the cell is outside the image, so it is not cut
        if touches_border:
            border_excluded.append(idx)
            ignored_mask |= full_mask > 0
            contours, _ = cv2.findContours(full_mask[ey1:ey2, ex1:ex2], cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE, offset=(ex1, ey1))
            cv2.drawContours(overview, contours, -1, (160, 160, 160), 1)
            continue

        # STEP 6: QA - signal left outside the mask, and mask edge on plain background
        qx1, qy1 = max(0, x - qa_ring_px - 1), max(0, y - qa_ring_px - 1)
        qx2, qy2 = min(img_w, x + w + qa_ring_px + 1), min(img_h, y + h + qa_ring_px + 1)
        q_mask = full_mask[qy1:qy2, qx1:qx2] > 0
        q_ring_all = (cv2.dilate(q_mask.astype(np.uint8), qa_ring_kernel) > 0) & ~q_mask
        q_ring = q_ring_all & ~final_union[qy1:qy2, qx1:qx2]
        ring_coverage = float(q_ring.sum()) / max(1, int(q_ring_all.sum()))
        crowded = ring_coverage < qa_min_ring_coverage
        q_edge = q_mask & ~(cv2.erode(q_mask.astype(np.uint8), edge_kernel) > 0)
        q_signal = signal[qy1:qy2, qx1:qx2]
        outside_signal_pct = 100.0 * float(q_signal[q_ring].mean()) if q_ring.any() else 0.0
        edge_background_pct = 100.0 * float((np.abs(deviation[qy1:qy2, qx1:qx2][q_edge]) <= noise_sigma).mean()) if q_edge.any() else 0.0
        qa_flag = (not crowded) and outside_signal_pct > qa_flag_pct

        crop_mask = full_mask[y1:y2, x1:x2]
        crop_raw = raw[y1:y2, x1:x2]
        annotated_crop = annotated[y1:y2, x1:x2]
        extracted = cut_on_canvas(crop_raw, crop_mask)

        # A: RAW cut, B: MASK image crop, C: binary mask, D: marked whole-slide context
        orig_filename = f"{subcell_id}_original_extracted.png"
        annotated_filename = f"{subcell_id}_annotated.png"
        mask_filename = f"{subcell_id}_mask.png"
        location_filename = f"{subcell_id}_marked.jpg"
        cv2.imwrite(os.path.join(output_dir, orig_filename), extracted)
        cv2.imwrite(os.path.join(output_dir, annotated_filename), annotated_crop)
        cv2.imwrite(os.path.join(output_dir, mask_filename), crop_mask)

        marked_img = raw.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, location_filename), marked_img)

        whole_slide_mask = np.maximum(whole_slide_mask, full_mask)
        label_map[(full_mask > 0) & (label_map == 0)] = idx

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
            "interior_area": cell["interior_area"],
            "ring_px": cell["ring_px"],
            "halo_px": cell["halo_px"],
            "halo_filled_px": cell["halo_filled_px"],
            "halo_status": cell["halo_status"],
            "entry_px": cell["entry_px"],
            "interior_std": round(cell["interior_std"], 2),
            "outside_signal_pct": round(outside_signal_pct, 1),
            "edge_background_pct": round(edge_background_pct, 1),
            "ring_coverage_pct": round(100.0 * ring_coverage, 1),
            "crowded": crowded,
            "qa_flag": qa_flag,
            "touches_border": touches_border,
            "original_filename": orig_filename,
            "annotated_filename": annotated_filename,
            "mask_filename": mask_filename,
            "location_filename": location_filename,
        })

        # PER-CELL DEBUG OUTPUTS
        if target_debug_ids == "all" or idx in target_debug_ids:
            region_crop = np.where(labels[y1:y2, x1:x2] == cell["region_id"], 255, 0).astype(np.uint8)
            outline_full = np.zeros((img_h, img_w), dtype=np.uint8)
            outline_full[ey1:ey2, ex1:ex2][cell["outline_mask"]] = 255
            outline_crop = outline_full[y1:y2, x1:x2]
            # Halo view on RAW: outline mask edge white, signal pixels added yellow, gap-filled pixels
            # green (red for both when the cell failed the area rule), contested pixels magenta
            halo_view = crop_raw.copy()
            kept = cell["halo_status"] == "grown"
            dropped_full = np.zeros((img_h, img_w), dtype=bool)
            dropped_full[ey1:ey2, ex1:ex2] = cell["halo_dropped"]
            halo_view[dropped_full[y1:y2, x1:x2]] = (255, 120, 0)
            for part, colour in ((cell["halo_signal"], (0, 220, 255)), (cell["halo_filled"], (80, 200, 0))):
                part_full = np.zeros((img_h, img_w), dtype=bool)
                part_full[ey1:ey2, ex1:ex2] = part
                halo_view[part_full[y1:y2, x1:x2]] = colour if kept else (0, 0, 255)
            halo_view[contested[y1:y2, x1:x2]] = (255, 0, 255)
            entry_full = np.zeros((img_h, img_w), dtype=bool)
            entry_full[ey1:ey2, ex1:ex2] = cell["entry_mask"]
            halo_view[entry_full[y1:y2, x1:x2]] = (0, 140, 255)
            outline_edge = (outline_crop > 0) & ~(cv2.erode(outline_crop, edge_kernel) > 0)
            halo_view[outline_edge] = (255, 255, 255)
            cyan_crop = cv2.applyColorMap(cyanness[y1:y2, x1:x2], cv2.COLORMAP_OCEAN)
            # QA view: mask edge green, outside ring blue, signal left in the ring red
            qa_view = crop_raw.copy()
            ring_crop = (cv2.dilate(crop_mask, qa_ring_kernel) > 0) & ~final_union[y1:y2, x1:x2]
            qa_view[ring_crop] = (qa_view[ring_crop] * 0.5 + np.array([120, 40, 0]) * 0.5).astype(np.uint8)
            qa_view[ring_crop & signal[y1:y2, x1:x2]] = (0, 0, 255)
            edge_crop = (crop_mask > 0) & ~(cv2.erode(crop_mask, edge_kernel) > 0)
            qa_view[edge_crop] = (0, 255, 0)
            save_single_cell_debug_trace(
                debug_dir=os.path.join(output_dir, "debug", f"cell_{idx:03d}"),
                cell_idx=idx,
                steps={
                    "01_step1_mask_image_roi.png": annotated_crop,
                    "02_step1_cyanness_map.png": cyan_crop,
                    "03_step1_cyan_core_lines.png": core_u8[y1:y2, x1:x2],
                    "04_step2_enclosed_region.png": region_crop,
                    "05_step4a_region_plus_outline.png": outline_crop,
                    **({"06_step4b_halo_band.png": halo_view} if halo_px > 0 else {}),
                    "07_step5a_raw_roi.png": crop_raw,
                    "08_step5b_cut_from_raw.png": extracted,
                    "09_step6_qa_signal_outside.png": qa_view,
                },
                marked_img=marked_img,
                bbox_tuple=(x1, y1, x2, y2),
                meta={
                    "cell_id": idx,
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "width": x2 - x1, "height": y2 - y1},
                    "area_pixels": area,
                    "interior_area_pixels": cell["interior_area"],
                    "outline_pixels_recovered": cell["ring_px"],
                    "halo": {"status": cell["halo_status"], "pixels_added": cell["halo_px"],
                             "pixels_gap_filled": cell["halo_filled_px"],
                             "pixels_dropped_leak_or_neck": int(cell["halo_dropped"].sum()),
                             "signal_pixels_found": int(cell["halo_signal"].sum()),
                             "entry_pixels_in_neighbours": cell["entry_px"]},
                    "interior_raw_std": round(cell["interior_std"], 2),
                    "touches_border": touches_border,
                    "qa": {"outside_signal_pct": round(outside_signal_pct, 1),
                           "edge_background_pct": round(edge_background_pct, 1),
                           "ring_coverage_pct": round(100.0 * ring_coverage, 1), "crowded": crowded,
                           "flagged": qa_flag, "noise_sigma": round(noise_sigma, 3),
                           "ring_px": qa_ring_px, "k": qa_k, "flag_pct": qa_flag_pct},
                    "params": {"core_level": core_level, "ring_radius": ring_radius,
                               "pocket_std": pocket_std, "halo_px": halo_px,
                               "halo_close_px": halo_close_px, "halo_max_added": halo_max_added,
                               "halo_leak_px": halo_leak_px, "halo_open_px": halo_open_px,
                               "entry_px": entry_px,
                               "entry_min_ring_px": entry_min_ring_px,
                               "entry_max_ring_width": entry_max_ring_width, "probe_k": probe_k, "probe_sigma": round(probe_sigma, 3),
                               "subcell_padding": subcell_padding},
                },
            )

    # QA: outline pixels that no cell outline mask claims (before the probe), in red
    unclaimed = core & ~outline_union
    overview[unclaimed] = (0, 0, 255)
    # QA: bridged outline gaps, in magenta
    for (xa, ya), (xb, yb) in bridges:
        cv2.circle(overview, ((xa + xb) // 2, (ya + yb) // 2), 6, (255, 0, 255), 1)
    # QA: signal blobs that no cell covers (candidate cells without an outline), in orange
    covered = cv2.dilate(final_union.astype(np.uint8), qa_ring_kernel) > 0
    smooth_dev = cv2.GaussianBlur(deviation, (0, 0), 1.0)
    smooth_sigma = robust_sigma(smooth_dev[qa_background])
    blob_mask = (np.abs(smooth_dev) > qa_k * smooth_sigma) & ~covered & ~near_line
    n_blob, _, blob_stats, _ = cv2.connectedComponentsWithStats(blob_mask.astype(np.uint8), connectivity=8)
    blobs = [[int(v) for v in blob_stats[i, :5]] for i in range(1, n_blob)]
    uncovered_blobs = [b for b in blobs if b[4] >= qa_blob_px]
    debris_blobs = [b for b in blobs if min_area <= b[4] < qa_blob_px]
    for bx, by, bw, bh, _ in uncovered_blobs:
        cv2.rectangle(overview, (bx, by), (bx + bw, by + bh), (0, 165, 255), 1)

    cv2.imwrite(os.path.join(output_dir, "02_cell_outline_overview.jpg"), overview)

    # STEP 4e: TAKEN VS. IGNORED MAP
    region_areas = np.bincount(labels.ravel())
    kept_region_ids = {c["region_id"] for c in cells}
    size_rejected_ids = [i for i in range(1, len(region_areas))
                         if i != background_id and i not in kept_region_ids
                         and i not in pocket_region_ids and region_areas[i] >= ignored_min_px]
    ignored_mask |= np.isin(labels, pocket_region_ids + size_rejected_ids)
    taken_mask = whole_slide_mask > 0
    taken_map = raw.copy()
    for mask_, colour in ((ignored_mask & ~taken_mask, (40, 40, 230)), (taken_mask, (60, 200, 60))):
        layer = taken_map.copy()
        layer[mask_] = colour
        taken_map = cv2.addWeighted(layer, 0.45, taken_map, 0.55, 0)
        contours, _ = cv2.findContours(mask_.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(taken_map, contours, -1, colour, 1)
    legend = [(f"Taken: {len(manifest_rows)} cells", (60, 200, 60)),
              (f"Ignored: {len(border_excluded)} cut by image edge, {rejected_pockets} empty pockets, "
               f"{len(size_rejected_ids)} wrong size", (40, 40, 230))]
    # Legend in its own strip above the image, so it hides no cells
    legend_bar = np.full((24 * len(legend) + 10, img_w, 3), 25, dtype=np.uint8)
    for i, (text, colour) in enumerate(legend):
        cv2.rectangle(legend_bar, (10, 8 + 24 * i), (26, 24 + 24 * i), colour, -1)
        cv2.putText(legend_bar, text, (34, 22 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "03_taken_vs_ignored.png"), np.vstack([legend_bar, taken_map]))
    cv2.imwrite(os.path.join(output_dir, "whole_slide_ground_truth_mask.png"), whole_slide_mask)
    cv2.imwrite(os.path.join(output_dir, "whole_slide_label_map.png"), label_map)

    with open(os.path.join(output_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()) if manifest_rows else ["cell_id"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    core_total = int(core.sum())
    summary = {
        "raw_image": os.path.basename(raw_path),
        "mask_image": os.path.basename(mask_path),
        "image_size": [img_w, img_h],
        "cells": len(manifest_rows),
        "cells_excluded_border": len(border_excluded),
        "regions_ignored_size": len(size_rejected_ids),
        "excluded_border_cell_ids": border_excluded,
        "rejected_background_pockets": rejected_pockets,
        "outline_endpoints": outline_endpoints,
        "outline_gaps_bridged": len(bridges),
        "bridges": bridges,
        "noise_sigma": round(noise_sigma, 3),
        "cells_crowded": int(sum(r["crowded"] for r in manifest_rows)),
        "median_outside_signal_pct": round(float(np.median([r["outside_signal_pct"] for r in manifest_rows if not r["crowded"]] or [0])), 2),
        "background_false_alarm_pct": round(100.0 * float(signal[qa_background].mean()), 2),
        "cells_flagged_outside_signal": int(sum(r["qa_flag"] for r in manifest_rows)),
        "median_edge_background_pct": round(float(np.median([r["edge_background_pct"] for r in manifest_rows if not r["crowded"]] or [0])), 2),
        "uncovered_signal_blobs": len(uncovered_blobs),
        "uncovered_signal_blob_boxes": uncovered_blobs,
        "small_debris_blobs": len(debris_blobs),
        "outline_px_total": core_total,
        "outline_px_claimed_pct": round(100.0 * (core_total - int(unclaimed.sum())) / max(1, core_total), 2),
        "probe_sigma": round(probe_sigma, 3),
        "probe_background_false_alarm_pct": round(100.0 * float(probe_signal[probe_background].mean()), 2),
        "halo_status_counts": {st: int(sum(r["halo_status"] == st for r in manifest_rows))
                               for st in ("grown", "no_signal", "reverted_area")},
        "halo_px_total": int(sum(r["halo_px"] for r in manifest_rows)),
        "halo_filled_px_total": int(sum(r["halo_filled_px"] for r in manifest_rows)),
        "cells_with_entry": int(sum(r["entry_px"] > 0 for r in manifest_rows)),
        "entry_px_total": int(sum(r["entry_px"] for r in manifest_rows)),
        "probe_contested_px": int(contested.sum()),
        "mean_cell_area": round(float(np.mean([r["area"] for r in manifest_rows])), 1) if manifest_rows else 0,
        "total_foreground_px": int((whole_slide_mask > 0).sum()),
        "canvas_bgr": list(CANVAS_BGR),
        "params": {
            "core_level": core_level, "min_area": min_area, "max_area": max_area,
            "ring_radius": ring_radius, "pocket_std": pocket_std, "max_gap": max_gap,
            "halo_px": halo_px, "halo_close_px": halo_close_px, "halo_max_added": halo_max_added,
            "halo_leak_px": halo_leak_px, "halo_open_px": halo_open_px, "entry_px": entry_px,
            "entry_min_ring_px": entry_min_ring_px, "entry_max_ring_width": entry_max_ring_width,
            "entry_band_min_px": entry_band_min_px, "entry_strong_k": entry_strong_k,
            "entry_shared_tol": entry_shared_tol, "probe_k": probe_k, "probe_smooth": probe_smooth, "qa_ring_px": qa_ring_px, "qa_k": qa_k, "qa_flag_pct": qa_flag_pct,
            "qa_min_ring_coverage": qa_min_ring_coverage, "qa_blob_px": qa_blob_px,
            "subcell_padding": subcell_padding,
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
