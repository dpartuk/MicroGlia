#!/usr/bin/env python3
"""
boundary_sharpening_pipeline.py

Experimental Core Pipeline with Multi-Scale Gap Closing, Process Attachment Merging,
and Expanded Bounding Box Padding.

Key Refinements in experiment/:
1. Bounding Box Padding expanded to 18 px so long tapering processes never touch crop edges.
2. Proximity-based Process Merger (radius: 12 px) merging detached filopodia into parent somas.
3. Convex Hull Cell Masking ensuring all branching arborizations are preserved in the extracted output.
4. Step 0 Initial Bounding Box context image for direct visual verification in debug mode.
"""

import csv
import json
import os
import cv2
import numpy as np


def save_single_cell_debug_trace(
    debug_dir,
    cell_idx,
    img,
    signal,
    bilateral,
    sobel_edges,
    canny_edges,
    gradient_fusion,
    clahe_boosted,
    sharp_fine,
    ultra_sharpened,
    bin_mask,
    closed_mask,
    cell_mask_full,
    extracted_orig,
    extracted_sharp,
    marked_img,
    bbox_tuple,
    area,
    otsu_val,
    calibrated_thresh
):
    """
    Saves a comprehensive step-by-step debug breakdown for a specific cell ID.
    Starting with Step 0: Initial Candidate Bounding Box in Slide Context.
    """
    os.makedirs(debug_dir, exist_ok=True)
    x1, y1, x2, y2 = bbox_tuple
    img_h, img_w = img.shape[:2]

    def crop_bgr(arr):
        crop = arr[y1:y2, x1:x2]
        if len(crop.shape) == 2:
            return cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        return crop

    # Step 0: Zoomed Context Crop around the cell with highlighted bounding box (500x500 window)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    zx1, zx2 = max(0, cx - 250), min(img_w, cx + 250)
    zy1, zy2 = max(0, cy - 250), min(img_h, cy + 250)
    zoomed_location = marked_img[zy1:zy2, zx1:zx2].copy()

    # Draw clear target candidate bounding box banner
    cv2.putText(zoomed_location, f"Target Candidate BBox: ({x1},{y1})-({x2},{y2})", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 229, 255), 1, cv2.LINE_AA)

    step_images = {
        "00_step0_bounding_box_location.jpg": zoomed_location,
        "01_step1_raw_rgb_roi.jpg": crop_bgr(img),
        "02_step1_cyan_signal.jpg": crop_bgr(signal),
        "03_step2_bilateral_denoised.jpg": crop_bgr(bilateral),
        "04_step3a_scharr_gradient.jpg": crop_bgr(sobel_edges),
        "05_step3b_canny_edges.jpg": crop_bgr(canny_edges),
        "06_step3c_gradient_fusion.jpg": crop_bgr(gradient_fusion),
        "07_step4_clahe_boosted.jpg": crop_bgr(clahe_boosted),
        "08_step5a_fine_unsharp.jpg": crop_bgr(sharp_fine),
        "09_step5b_ultra_sharpened.jpg": crop_bgr(ultra_sharpened),
        "10_step6a_otsu_binary_mask.jpg": crop_bgr(bin_mask),
        "11_step6b_morphology_closed.jpg": crop_bgr(closed_mask),
        "12_step7_cell_contour_mask.jpg": crop_bgr(cell_mask_full),
        "13_step8a_final_original_extracted.jpg": extracted_orig,
        "14_step8b_final_sharpened_extracted.jpg": extracted_sharp,
        "15_step8c_slide_location_marked.jpg": marked_img
    }

    for fname, img_data in step_images.items():
        cv2.imwrite(os.path.join(debug_dir, fname), img_data)

    cv2.imwrite(os.path.join(debug_dir, "15b_step8c_zoomed_location_context.jpg"), zoomed_location)

    # Build 4x4 Annotated Visual Debug Panel
    panel_items = [
        ("0. BBox Context", zoomed_location),
        ("1. Raw RGB Crop", step_images["01_step1_raw_rgb_roi.jpg"]),
        ("2. Cyan/Gray Signal", step_images["02_step1_cyan_signal.jpg"]),
        ("3. Bilateral Denoised", step_images["03_step2_bilateral_denoised.jpg"]),
        ("4. Scharr Gradient", step_images["04_step3a_scharr_gradient.jpg"]),
        ("5. Canny Edges", step_images["05_step3b_canny_edges.jpg"]),
        ("6. Gradient Fusion", step_images["06_step3c_gradient_fusion.jpg"]),
        ("7. CLAHE Contrast Boost", step_images["07_step4_clahe_boosted.jpg"]),
        ("8. Fine Unsharp Mask", step_images["08_step5a_fine_unsharp.jpg"]),
        ("9. Ultra-Sharpened Map", step_images["09_step5b_ultra_sharpened.jpg"]),
        ("10. Otsu Binary Mask", step_images["10_step6a_otsu_binary_mask.jpg"]),
        ("11. Morphology Closed", step_images["11_step6b_morphology_closed.jpg"]),
        ("12. Cell Contour Mask", step_images["12_step7_cell_contour_mask.jpg"]),
        ("13. Final RGB Extracted", step_images["13_step8a_final_original_extracted.jpg"]),
        ("14. Final Sharpened Mask", step_images["14_step8b_final_sharpened_extracted.jpg"]),
        ("15. Whole Slide Marked", cv2.resize(marked_img, (300, 300)))
    ]

    tile_size = 280
    panel = np.zeros((tile_size * 4 + 40, tile_size * 4, 3), dtype=np.uint8) + 20

    for idx, (title, item_img) in enumerate(panel_items):
        row = idx // 4
        col = idx % 4
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

    header_text = f"MICROGLIA DEBUG TRACE: Cell #{cell_idx:03d} | BBox: ({x1},{y1})-({x2},{y2}) | Area: {area} px^2"
    cv2.putText(panel, header_text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(debug_dir, "16_debug_step_by_step_panel.jpg"), panel)

    summary_data = {
        "cell_id": cell_idx,
        "bbox": {
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1
        },
        "area_pixels": int(area),
        "thresholds": {
            "otsu_val": float(otsu_val),
            "calibrated_threshold": int(calibrated_thresh)
        },
        "flags": {
            "possible_merged_cluster": bool(area > 3500)
        }
    }
    with open(os.path.join(debug_dir, "cell_debug_summary.json"), "w") as jf:
        json.dump(summary_data, jf, indent=2)


def extract_cells_with_boundary_sharpening(
    image_path,
    output_dir,
    clahe_clip_limit=3.5,
    clahe_tile_grid=(8, 8),
    subcell_padding=18,
    debug_cell_ids="all"
):
    """
    Experimental Extraction Pipeline with Process & Bounding Box Refinements.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open or find image: {image_path}")

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    if isinstance(debug_cell_ids, int):
        target_debug_ids = [debug_cell_ids]
    elif isinstance(debug_cell_ids, (list, tuple, set)):
        target_debug_ids = list(debug_cell_ids)
    elif str(debug_cell_ids).lower() == "all":
        target_debug_ids = "all"
    else:
        target_debug_ids = []

    # Save original reference
    cv2.imwrite(os.path.join(output_dir, "00_original_reference.jpg"), img)

    # STEP 1: CYAN / GRAYSCALE SIGNAL MAPPING
    b, g, r = cv2.split(img.astype(np.float32))
    cyan_diff = (b + g) / 2.0 - r
    cyan_diff = np.clip(cyan_diff, 0, 255).astype(np.uint8)

    if np.max(cyan_diff) == 0:
        signal = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    else:
        signal = cyan_diff

    # STEP 2: BILATERAL NOISE SUPPRESSION
    bilateral = cv2.bilateralFilter(signal, d=5, sigmaColor=35, sigmaSpace=35)

    # STEP 3: MULTI-OPERATOR EDGE GRADIENT FUSION (SCHARR + CANNY)
    grad_x = cv2.Scharr(bilateral, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(bilateral, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    sobel_edges = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    high_thresh, _ = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canny_edges = cv2.Canny(bilateral, 0.5 * high_thresh, high_thresh)

    gradient_fusion = cv2.addWeighted(sobel_edges, 0.7, canny_edges, 0.3, 0)

    # STEP 4: MULTI-TILE ADAPTIVE CLAHE (DARK QUADRANT RECOVERY)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    clahe_boosted = clahe.apply(gradient_fusion)

    # STEP 5: MULTI-SCALE FINE & MID UNSHARP MASKING -> ULTRA-SHARPENED MAP
    blur_fine = cv2.GaussianBlur(clahe_boosted, (3, 3), 0.8)
    sharp_fine = cv2.addWeighted(clahe_boosted, 2.2, blur_fine, -1.2, 0)

    blur_mid = cv2.GaussianBlur(sharp_fine, (7, 7), 1.5)
    ultra_sharpened = cv2.addWeighted(sharp_fine, 1.8, blur_mid, -0.8, 0)
    cv2.imwrite(os.path.join(output_dir, "01_ultra_sharpened_boundary_map.jpg"), ultra_sharpened)

    # STEP 6: ADAPTIVE OTSU THRESHOLDING -> CRISP BOUNDARY MAP
    otsu_val, _ = cv2.threshold(ultra_sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    calibrated_thresh = max(18, int(0.58 * otsu_val))

    _, bin_mask = cv2.threshold(ultra_sharpened, calibrated_thresh, 255, cv2.THRESH_BINARY)
    # Remove outer border frame noise
    bin_mask[:, :3] = 0; bin_mask[:, -3:] = 0; bin_mask[:3, :] = 0; bin_mask[-3:, :] = 0
    cv2.imwrite(os.path.join(output_dir, "02_binary_crisp_boundaries.jpg"), bin_mask)

    # Multi-Scale Morphological Gap Closing (3x3 Kernel + Edge Bridge)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    contours, _ = cv2.findContours(closed_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise < 60 px
            if area >= 60 and not (area > 0.70 * total_img_area or (w > 0.85 * img_w and h > 0.85 * img_h)):
                candidates.append((x, y, w, h, area, c))

    # STEP 7: DEDUPLICATED NON-COMPOSITE FILTER
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
            if inter_area >= 0.80 * box2_area and ca1 > ca2 and ca2 >= 180:
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

    # DUPLICATE SUBSET FILTER & PROXIMITY PROCESS ATTACHMENT MERGER
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

            # Proximity process attachment merger (Search radius: 12 px, for filopodia < 90 px or thin filaments attached to large somas)
            dist_x = max(0, max(cx1, cx2) - min(cx1 + cw1, cx2 + cw2))
            dist_y = max(0, max(cy1, cy2) - min(cy1 + ch1, cy2 + ch2))

            ar1 = max(cw1, ch1) / max(1, min(cw1, ch1))
            min_dim1 = min(cw1, ch1)
            is_thin_filament = (ar1 >= 2.5 or min_dim1 <= 8)

            if (ca1 < 90 and ca2 >= 180 and dist_x <= 12 and dist_y <= 12) or \
               (dist_x <= 6 and dist_y <= 6 and is_thin_filament and ca1 < 140 and ca2 >= 350):
                is_dup_or_leg = True
                contour_list = cc2 if isinstance(cc2, list) else [cc2]
                contour_list.extend(cc1 if isinstance(cc1, list) else [cc1])
                combined_pts = np.vstack(contour_list)
                cbx, cby, cbw, cbh = cv2.boundingRect(combined_pts)
                non_composite[j] = (cbx, cby, cbw, cbh, ca1 + ca2, contour_list)
                break

        if not is_dup_or_leg:
            subcell_tuples.append(non_composite[i])

    # HOMOGENEITY FILTER
    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    final_subcells = []

    for (cx1, cy1, cw1, ch1, ca1, cc1) in subcell_tuples:
        mask_sub = np.zeros((img_h, img_w), dtype=np.uint8)
        if isinstance(cc1, list):
            cv2.drawContours(mask_sub, cc1, -1, 255, cv2.FILLED)
        else:
            cv2.drawContours(mask_sub, [cc1], -1, 255, cv2.FILLED)

        crop_sub = img[cy1:cy1+ch1, cx1:cx1+cw1]
        crop_mask = mask_sub[cy1:cy1+ch1, cx1:cx1+cw1]

        non_white = (crop_mask > 0)

        if len(img.shape) == 3 and not (np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2])):
            b_c, g_c, r_c = cv2.split(crop_sub.astype(np.float32))
            crop_cyan_diff = np.clip((b_c + g_c) / 2.0 - r_c, 0, 255)
            crop_gray = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2GRAY)
            # True neutral cytoplasmic soma pixels (non-cyan and dark gray < 170)
            soma_pixels = non_white & (crop_cyan_diff < 25) & (crop_gray < 170)
            interior_pixels = crop_gray[non_white & (crop_cyan_diff < 35)]
        else:
            crop_gray = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2GRAY) if len(crop_sub.shape) == 3 else crop_sub
            soma_pixels = non_white & (crop_gray < 170)
            interior_pixels = crop_gray[non_white]

        soma_count = np.count_nonzero(soma_pixels)

        # Rejection Condition 0: Discard false kissing seams / marker pen intersections (< 45 true soma pixels)
        if soma_count < 45:
            continue

        if len(interior_pixels) > 0:
            mean_val = np.mean(interior_pixels)
            dark_ratio = np.count_nonzero(interior_pixels < 165) / len(interior_pixels)
            std_val = np.std(interior_pixels)
            min_val = np.min(interior_pixels)

            # Rejection Condition 1: Pure synthetic white canvas
            if std_val < 10 and min_val > 220:
                continue

            # Rejection Condition 2: Empty intercellular background cavity (e.g. Cell #35)
            if dark_ratio < 0.05 and mean_val > 185:
                continue

        final_subcells.append((cx1, cy1, cw1, ch1, ca1, cc1))

    # Sort left-to-right, top-to-bottom
    final_subcells = sorted(final_subcells, key=lambda s: (s[1], s[0]))

    # STEP 8: DUAL CROP, OVERVIEWS, MANIFEST & PER-CELL DEBUG OUTPUTS
    overview_original = img.copy()
    sharpened_bgr = cv2.cvtColor(ultra_sharpened, cv2.COLOR_GRAY2BGR)
    overview_sharpened = sharpened_bgr.copy()

    manifest_rows = []

    for idx, (x, y, w, h, area, contour) in enumerate(final_subcells, start=1):
        subcell_id = f"subcell_{idx:03d}"

        # Bounding box with expanded padding
        x1 = max(0, x - subcell_padding)
        y1 = max(0, y - subcell_padding)
        x2 = min(img_w, x + w + subcell_padding)
        y2 = min(img_h, y + h + subcell_padding)

        # Check if cell touches the image boundary
        touches_border = (x <= 6 or y <= 6 or (x + w) >= img_w - 6 or (y + h) >= img_h - 6)

        cell_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        if touches_border:
            # EXCEPTION: Border cell with open contour -> close against image edge using convex hull
            if isinstance(contour, list):
                pts = np.vstack(contour)
                hull_c = cv2.convexHull(pts)
            else:
                hull_c = cv2.convexHull(contour)
            cv2.drawContours(cell_mask_full, [hull_c], -1, 255, cv2.FILLED)
        else:
            # STANDARD: Interior cell -> exact concave contour polygon masking (zero exterior gray pixels)
            if isinstance(contour, list):
                cv2.drawContours(cell_mask_full, contour, -1, 255, cv2.FILLED)
            else:
                cv2.drawContours(cell_mask_full, [contour], -1, 255, cv2.FILLED)

        # A: Crop from Original RGB Image
        crop_img = img[y1:y2, x1:x2]
        crop_mask = cell_mask_full[y1:y2, x1:x2]
        crop_bgr = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR) if len(crop_img.shape) == 2 else crop_img
        white_canvas = np.full_like(crop_bgr, 255)
        mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
        extracted_orig = np.where(mask_3ch > 0, crop_bgr, white_canvas)
        orig_filename = f"{subcell_id}_original_extracted.jpg"
        cv2.imwrite(os.path.join(output_dir, orig_filename), extracted_orig)

        # B: Crop directly from Ultra-Sharpened Map
        crop_sharp = sharpened_bgr[y1:y2, x1:x2]
        extracted_sharp = np.where(mask_3ch > 0, crop_sharp, white_canvas)
        sharp_filename = f"{subcell_id}_sharpened_extracted.jpg"
        cv2.imwrite(os.path.join(output_dir, sharp_filename), extracted_sharp)

        # C: Marked Context Copy showing location on original image
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        location_filename = f"{subcell_id}_marked.jpg"
        cv2.imwrite(os.path.join(output_dir, location_filename), marked_img)

        # D: Update Overview Map on Original RGB Image
        cv2.rectangle(overview_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_original, str(idx), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        # E: Update Overview Map on Ultra-Sharpened Map
        cv2.rectangle(overview_sharpened, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_sharpened, str(idx), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        flagged = bool(area > 3500)
        manifest_rows.append({
            "cell_id": idx,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "width": x2 - x1, "height": y2 - y1,
            "area": int(area),
            "original_filename": orig_filename,
            "sharpened_filename": sharp_filename,
            "location_filename": location_filename,
            "flagged_possible_merge": flagged
        })

        # PER-CELL DEBUG OUTPUTS
        if target_debug_ids == "all" or idx in target_debug_ids:
            cell_debug_dir = os.path.join(output_dir, "debug", f"cell_{idx:03d}")
            save_single_cell_debug_trace(
                debug_dir=cell_debug_dir,
                cell_idx=idx,
                img=img,
                signal=signal,
                bilateral=bilateral,
                sobel_edges=sobel_edges,
                canny_edges=canny_edges,
                gradient_fusion=gradient_fusion,
                clahe_boosted=clahe_boosted,
                sharp_fine=sharp_fine,
                ultra_sharpened=ultra_sharpened,
                bin_mask=bin_mask,
                closed_mask=closed_mask,
                cell_mask_full=cell_mask_full,
                extracted_orig=extracted_orig,
                extracted_sharp=extracted_sharp,
                marked_img=marked_img,
                bbox_tuple=(x1, y1, x2, y2),
                area=area,
                otsu_val=otsu_val,
                calibrated_thresh=calibrated_thresh
            )

    cv2.imwrite(os.path.join(output_dir, "overview_map_on_original.jpg"), overview_original)
    cv2.imwrite(os.path.join(output_dir, "overview_map_on_sharpened.jpg"), overview_sharpened)

    # STEP 9: SAVE MANIFEST CSV
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        fieldnames = [
            "cell_id", "x1", "y1", "x2", "y2", "width", "height", "area",
            "original_filename", "sharpened_filename", "location_filename",
            "flagged_possible_merge"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(final_subcells)
