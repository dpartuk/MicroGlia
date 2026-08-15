import os
import glob
import json
import csv
import cv2
import numpy as np

def save_single_cell_baseline_debug_trace(
    debug_dir,
    cell_idx,
    img,
    contrast_map,
    contrast_sharpened,
    bin_mask,
    closed_mask,
    cell_mask_full,
    extracted_orig,
    marked_img,
    bbox_tuple,
    area,
    otsu_val,
    calibrated_thresh
):
    """
    Saves a step-by-step debug breakdown for a specific baseline cell ID.
    Every intermediate step is saved with explicit naming tying directly to the baseline code.
    Also produces an annotated visual collage flowchart panel and JSON metadata.
    """
    os.makedirs(debug_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox_tuple
    img_h, img_w = img.shape[:2]

    def crop_bgr(arr):
        crop = arr[y1:y2, x1:x2]
        if len(crop.shape) == 2:
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        return crop

    step_images = {
        "01_step1_raw_rgb_roi.jpg": crop_bgr(img),
        "02_step1_contrast_map.jpg": crop_bgr(contrast_map),
        "03_step2_contrast_sharpened.jpg": crop_bgr(contrast_sharpened),
        "04_step2_binary_mask.jpg": crop_bgr(bin_mask),
        "05_step2_closed_mask.jpg": crop_bgr(closed_mask),
        "06_step3_cell_contour_mask.jpg": crop_bgr(cell_mask_full),
        "07_step4_final_extracted.jpg": extracted_orig,
        "08_step4_slide_location_marked.jpg": marked_img
    }

    for fname, img_data in step_images.items():
        cv2.imwrite(os.path.join(debug_dir, fname), img_data)

    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    zx1, zx2 = max(0, cx - 250), min(img_w, cx + 250)
    zy1, zy2 = max(0, cy - 250), min(img_h, cy + 250)
    zoomed_location = marked_img[zy1:zy2, zx1:zx2]
    cv2.imwrite(os.path.join(debug_dir, "08b_step4_zoomed_location_context.jpg"), zoomed_location)

    # 3x3 Visual Debug Collage Panel
    panel_items = [
        ("1. Raw RGB Crop", step_images["01_step1_raw_rgb_roi.jpg"]),
        ("2. Contrast CLAHE Map", step_images["02_step1_contrast_map.jpg"]),
        ("3. Contrast Sharpened", step_images["03_step2_contrast_sharpened.jpg"]),
        ("4. Otsu Binary Mask", step_images["04_step2_binary_mask.jpg"]),
        ("5. Morphology Closed", step_images["05_step2_closed_mask.jpg"]),
        ("6. Contour Mask", step_images["06_step3_cell_contour_mask.jpg"]),
        ("7. Final Extracted Cell", extracted_orig),
        ("8. Zoomed Slide Context", zoomed_location),
        ("9. Whole Slide Marked", cv2.resize(marked_img, (280, 280)))
    ]

    tile_size = 280
    panel = np.zeros((tile_size * 3 + 40, tile_size * 3, 3), dtype=np.uint8) + 20

    for idx, (title, item_img) in enumerate(panel_items):
        row = idx // 3
        col = idx % 3
        px1 = col * tile_size
        py1 = row * tile_size + 30

        ih, iw = item_img.shape[:2]
        scale = min((tile_size - 10) / iw, (tile_size - 35) / ih)
        rw, rh = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(item_img, (rw, rh))

        ox = px1 + (tile_size - rw) // 2
        oy = py1 + 25 + (tile_size - 35 - rh) // 2
        panel[oy:oy+rh, ox:ox+rw] = resized

        cv2.rectangle(panel, (px1 + 2, py1 + 2), (px1 + tile_size - 2, py1 + tile_size - 2), (60, 60, 60), 1)
        cv2.putText(panel, title, (px1 + 8, py1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 229, 255), 1, cv2.LINE_AA)

    header_text = f"BASELINE DEBUG TRACE: Cell #{cell_idx:03d} | BBox: ({x1},{y1})-({x2},{y2}) | Area: {int(area)} px2 | Otsu: {int(otsu_val)} (Calib: {calibrated_thresh})"
    cv2.putText(panel, header_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(debug_dir, "09_debug_flowchart_panel.jpg"), panel)

    meta = {
        "cell_id": cell_idx,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "width": x2 - x1,
        "height": y2 - y1,
        "area": int(area),
        "otsu_threshold": int(otsu_val),
        "calibrated_threshold": int(calibrated_thresh),
        "flagged_possible_merge": bool(area > 3500),
        "step_files": list(step_images.keys())
    }
    with open(os.path.join(debug_dir, "cell_debug_summary.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"    [DEBUG] Saved baseline step-by-step debug outputs for Cell #{cell_idx:03d} -> {debug_dir}")


def extract_cells_baseline(
    image_path,
    output_dir,
    subcell_padding=5,
    clahe_clip_limit=3.5,
    clahe_tile_grid=(8, 8),
    debug_cell_ids=None
):
    """
    Baseline Cell Extraction Core (Presaizen 2026 / Version 5):
    1. Multi-Tile Local Adaptive CLAHE Contrast Mapping
    2. Adaptive Otsu Threshold Calibration
    3. Unsharp Masking
    4. Deduplicated Non-Composite Filter
    5. Duplicate Subset Filter & Cell Process Attachment Merger
    6. Homogeneity Filter
    7. Dual Crop, Overview, Manifest & Per-Cell Debug Trace
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    target_debug_ids = set()
    if debug_cell_ids is not None:
        if isinstance(debug_cell_ids, (int, str)) and str(debug_cell_ids).lower() == "all":
            target_debug_ids = "all"
        elif isinstance(debug_cell_ids, int):
            target_debug_ids = {debug_cell_ids}
        elif isinstance(debug_cell_ids, (list, tuple, set)):
            target_debug_ids = {int(x) for x in debug_cell_ids}

    # Clean existing cell crops in output_dir
    for old_file in glob.glob(os.path.join(output_dir, "subcell_*.jpg")):
        try: os.remove(old_file)
        except OSError: pass

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    cv2.imwrite(os.path.join(output_dir, "00_original_reference.jpg"), img)

    # 1. CONTRAST SIGNAL MAPPING & LOCAL ADAPTIVE CLAHE
    b, g, r = cv2.split(img.astype(np.float32))
    cyan_diff = (b + g) / 2.0 - r
    cyan_diff = np.clip(cyan_diff, 0, 255).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)

    if np.max(cyan_diff) == 0:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        contrast_map = clahe.apply(gray)
    else:
        contrast_map = clahe.apply(cyan_diff)

    # 2. UNSHARP MASKING & ADAPTIVE OTSU THRESHOLD CALIBRATION
    blur = cv2.GaussianBlur(contrast_map, (5, 5), 1.0)
    contrast_sharpened = cv2.addWeighted(contrast_map, 1.8, blur, -0.8, 0)
    cv2.imwrite(os.path.join(output_dir, "01_contrast_sharpened_map.jpg"), contrast_sharpened)

    otsu_val, _ = cv2.threshold(contrast_sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    calibrated_thresh = max(20, int(0.65 * otsu_val))

    _, bin_mask = cv2.threshold(contrast_sharpened, calibrated_thresh, 255, cv2.THRESH_BINARY)
    bin_mask[:, :3] = 0; bin_mask[:, -3:] = 0; bin_mask[:3, :] = 0; bin_mask[-3:, :] = 0
    cv2.imwrite(os.path.join(output_dir, "02_binary_mask.jpg"), bin_mask)

    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            if area >= 80 and not (area > 0.70 * total_img_area or (w > 0.85 * img_w and h > 0.85 * img_h)):
                candidates.append((x, y, w, h, area, c))

    # 3. DEDUPLICATED NON-COMPOSITE FILTER
    non_composite = []
    for i, (cx1, cy1, cw1, ch1, ca1, cc1) in enumerate(candidates):
        contained_raw = []
        for j, (cx2, cy2, cw2, ch2, ca2, cc2) in enumerate(candidates):
            if i == j: continue
            inter_x1 = max(cx1, cx2)
            inter_y1 = max(cy1, cy2)
            inter_x2 = min(cx1 + cw1, cx2 + cw2)
            inter_y2 = min(cy1 + ch1, cy2 + ch2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            box2_area = cw2 * ch2
            if inter_area >= 0.80 * box2_area and ca1 > ca2 and ca2 >= 200:
                contained_raw.append((j, cx2, cy2, cw2, ch2, ca2))

        unique_contained = []
        for item in contained_raw:
            j, x2, y2, w2, h2, a2 = item
            is_dup_contained = False
            for u in unique_contained:
                uj, ux2, uy2, uw2, uh2, ua2 = u
                inter = max(0, min(x2 + w2, ux2 + uw2) - max(x2, ux2)) * max(0, min(y2 + h2, uy2 + uh2) - max(y2, uy2))
                union = w2 * h2 + uw2 * uh2 - inter
                iou = inter / union if union > 0 else 0
                if iou > 0.60:
                    is_dup_contained = True
                    break
            if not is_dup_contained:
                unique_contained.append(item)

        if len(unique_contained) < 2:
            non_composite.append((cx1, cy1, cw1, ch1, ca1, cc1))

    # 4. DUPLICATE SUBSET FILTER & CELL PROCESS ATTACHMENT MERGER
    subcell_tuples = []
    merged_leg_indices = set()

    for i, (cx1, cy1, cw1, ch1, ca1, cc1) in enumerate(non_composite):
        if i in merged_leg_indices: continue

        is_dup_or_leg = False
        for j, (cx2, cy2, cw2, ch2, ca2, cc2) in enumerate(non_composite):
            if i == j or j in merged_leg_indices: continue

            inter_x1 = max(cx1, cx2)
            inter_y1 = max(cy1, cy2)
            inter_x2 = min(cx1 + cw1, cx2 + cw2)
            inter_y2 = min(cy1 + ch1, cy2 + ch2)
            inter_w = max(0, inter_x2 - inter_x1)
            inter_h = max(0, inter_y2 - inter_y1)
            inter_area = inter_w * inter_h
            box1_area = cw1 * ch1

            if inter_area >= 0.75 * box1_area and ca2 > ca1:
                is_dup_or_leg = True
                break

            if ca1 < 200 and ca2 >= 250:
                dist_x = max(0, max(cx1, cx2) - min(cx1 + cw1, cx2 + cw2))
                dist_y = max(0, max(cy1, cy2) - min(cy1 + ch1, cy2 + ch2))
                if inter_area >= 0.08 * box1_area or (dist_x <= 3 and dist_y <= 3):
                    is_dup_or_leg = True
                    combined_pts = np.vstack((cc2, cc1))
                    combined_hull = cv2.convexHull(combined_pts)
                    cbx, cby, cbw, cbh = cv2.boundingRect(combined_hull)
                    non_composite[j] = (cbx, cby, cbw, cbh, cv2.contourArea(combined_hull), combined_hull)
                    break

        if not is_dup_or_leg:
            subcell_tuples.append(non_composite[i])

    # 5. HOMOGENEITY FILTER
    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    final_subcells = []

    for (cx1, cy1, cw1, ch1, ca1, cc1) in subcell_tuples:
        mask_sub = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask_sub, [cc1], -1, 255, cv2.FILLED)

        crop_sub = img[cy1:cy1+ch1, cx1:cx1+cw1]
        crop_mask = mask_sub[cy1:cy1+ch1, cx1:cx1+cw1]

        if np.max(cyan_diff) > 0:
            crop_hsv = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2HSV)
            crop_cyan = cv2.inRange(crop_hsv, lower_cyan, upper_cyan)
            non_white = np.any(crop_sub < 240, axis=2) & (crop_mask > 0)
            interior = non_white & (crop_cyan == 0)
            crop_gray = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2GRAY)
            interior_pixels = crop_gray[interior] if np.any(interior) else crop_gray[non_white]
        else:
            crop_gray = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2GRAY) if len(crop_sub.shape) == 2 else crop_sub
            interior_pixels = crop_gray[crop_mask > 0]

        if len(interior_pixels) > 0:
            std_val = np.std(interior_pixels)
            min_val = np.min(interior_pixels)
            if std_val < 10 and min_val > 220:
                continue

        final_subcells.append((cx1, cy1, cw1, ch1, ca1, cc1))

    final_subcells = sorted(final_subcells, key=lambda s: (s[1], s[0]))

    overview_img = img.copy()
    manifest_rows = []

    for idx, (x, y, w, h, area, contour) in enumerate(final_subcells, start=1):
        subcell_id = f"subcell_{idx:03d}"

        x1 = max(0, x - subcell_padding)
        y1 = max(0, y - subcell_padding)
        x2 = min(img_w, x + w + subcell_padding)
        y2 = min(img_h, y + h + subcell_padding)

        touches_border = (x <= 4 or y <= 4 or (x + w) >= img_w - 4 or (y + h) >= img_h - 4)
        cell_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        if touches_border:
            hull_c = cv2.convexHull(contour)
            cv2.drawContours(cell_mask_full, [hull_c], -1, 255, cv2.FILLED)
        else:
            cv2.drawContours(cell_mask_full, [contour], -1, 255, cv2.FILLED)

        crop_img = img[y1:y2, x1:x2]
        crop_mask = cell_mask_full[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR) if len(crop_img.shape) == 2 else crop_img
        white_canvas = np.full_like(crop_bgr, 255)
        mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
        extracted_orig = np.where(mask_3ch > 0, crop_bgr, white_canvas)
        cell_filename = f"{subcell_id}_extracted.jpg"
        cv2.imwrite(os.path.join(output_dir, cell_filename), extracted_orig)

        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        location_filename = f"{subcell_id}_marked.jpg"
        cv2.imwrite(os.path.join(output_dir, location_filename), marked_img)

        cv2.rectangle(overview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_img, str(idx), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        flagged = bool(area > 3500)
        manifest_rows.append({
            "cell_id": idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "area": int(area),
            "filename": cell_filename,
            "location_filename": location_filename,
            "flagged_possible_merge": flagged
        })

        if target_debug_ids == "all" or idx in target_debug_ids:
            cell_debug_dir = os.path.join(output_dir, "debug", f"cell_{idx:03d}")
            save_single_cell_baseline_debug_trace(
                debug_dir=cell_debug_dir,
                cell_idx=idx,
                img=img,
                contrast_map=contrast_map,
                contrast_sharpened=contrast_sharpened,
                bin_mask=bin_mask,
                closed_mask=closed_mask,
                cell_mask_full=cell_mask_full,
                extracted_orig=extracted_orig,
                marked_img=marked_img,
                bbox_tuple=(x1, y1, x2, y2),
                area=area,
                otsu_val=otsu_val,
                calibrated_thresh=calibrated_thresh
            )

    cv2.imwrite(os.path.join(output_dir, "overview_map.jpg"), overview_img)

    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        fieldnames = [
            "cell_id", "x1", "y1", "x2", "y2", "width", "height", "area",
            "filename", "location_filename", "flagged_possible_merge"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(final_subcells)
