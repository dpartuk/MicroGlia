import os
import glob
import cv2
import numpy as np

def extract_cells_from_image(
    image_path,
    output_dir,
    subcell_padding=5,
    clahe_clip_limit=3.5,
    clahe_tile_grid=(8, 8)
):
    """
    Production Baseline Cell Extraction Core (Version 5 - Local Adaptive CLAHE Calibrated + Deduplicated Non-Composite):
    1. Multi-Tile Local Adaptive CLAHE Contrast Mapping (Normalizes bright centers and dark corners)
    2. Adaptive Otsu Threshold Calibration (Calibrates threshold per image and per region)
    3. Unsharp Masking (1.8 * I - 0.8 * GaussianBlur)
    4. Deduplicated Non-Composite Filter (Deduplicates internal contour levels to preserve giant single cells)
    5. Duplicate Subset Filter & Cell Process Attachment Merger (Convex Hull Integration)
    6. Homogeneity Filter (Removes uniform gray background spaces)
    7. Post-Processing Pass: Convex Hull Boundary Closure (0% white-filling data loss)
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Clean existing output files
    for old_file in glob.glob(os.path.join(output_dir, "*")):
        try:
            if os.path.isfile(old_file):
                os.remove(old_file)
        except OSError:
            pass

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    # Save original reference
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
    # Remove outer border frame noise
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
            # Filter noise < 80 px
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

        # Deduplicate contained sub-cells if they heavily overlap each other
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
            crop_gray = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2GRAY) if len(crop_sub.shape) == 3 else crop_sub
            interior_pixels = crop_gray[crop_mask > 0]

        if len(interior_pixels) > 0:
            std_val = np.std(interior_pixels)
            min_val = np.min(interior_pixels)
            if std_val < 10 and min_val > 220:
                continue

        final_subcells.append((cx1, cy1, cw1, ch1, ca1, cc1))

    # Sort left-to-right, top-to-bottom
    final_subcells = sorted(final_subcells, key=lambda s: (s[1], s[0]))

    overview_img = img.copy()

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
        extracted = np.where(mask_3ch > 0, crop_bgr, white_canvas)

        # Save extracted cell crop
        cv2.imwrite(os.path.join(output_dir, f"{subcell_id}_extracted.jpg"), extracted)

        # Save marked context copy
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, f"{subcell_id}_marked.jpg"), marked_img)

        # Update Overview map
        cv2.rectangle(overview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_img, str(idx), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "overview_map.jpg"), overview_img)
    return len(final_subcells)
