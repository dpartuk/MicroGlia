import os
import glob
import shutil
import cv2
import numpy as np
import argparse

def get_step1_cell_bboxes(original_image_path, min_area=100, padding=10):
    """
    Recompute Step 1 bounding boxes for all cells in the original image.
    This provides exact global coordinates (gx1, gy1, gx2, gy2) for each cell.
    """
    img = cv2.imread(original_image_path)
    if img is None:
        raise ValueError(f"Could not read original image: {original_image_path}")

    img_h, img_w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    cell_bboxes = {}
    for idx, c in enumerate(valid_contours, start=1):
        x, y, w, h = cv2.boundingRect(c)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)
        cell_key = f"cell_{idx:03d}"
        cell_bboxes[cell_key] = (x1, y1, x2, y2)

    return cell_bboxes

def process_single_cell_step2(cell_file_path, orig_img, global_bbox, cell_output_dir, subcell_padding=5):
    """
    Process one cell and break it down into its true atomic sub-cells using Version 2 Pipeline:
    1. Cyan Contrast Mapping ( (B+G)/2 - R )
    2. Unsharp Masking (1.8 * I_cyan - 0.8 * GaussianBlur)
    3. Unmasked Raw Microscopy Crop Extraction (preserves 100% of cell pixels)
    4. Non-Composite Filter (remove outer container wrappers)
    5. Duplicate Subset Filter & Cell Process Attachment Merger (merge attached process legs)
    6. Homogeneity Filter (remove empty gray background spaces)
    """
    os.makedirs(cell_output_dir, exist_ok=True)

    # Clean out any old subcell files from previous runs to prevent stale duplicates
    for old_file in glob.glob(os.path.join(cell_output_dir, "subcell_*")):
        try:
            os.remove(old_file)
        except OSError:
            pass

    cell_basename = os.path.splitext(os.path.basename(cell_file_path))[0]
    cell_name = cell_basename.replace("_extracted", "")

    # Copy Step 1 extracted cell into subfolder as input reference
    shutil.copy(cell_file_path, os.path.join(cell_output_dir, "input_extracted.jpg"))

    # Copy Step 1 marked image if available
    step1_dir = os.path.dirname(cell_file_path)
    step1_marked_path = os.path.join(step1_dir, f"{cell_name}_marked.jpg")
    if os.path.exists(step1_marked_path):
        shutil.copy(step1_marked_path, os.path.join(cell_output_dir, "input_marked.jpg"))

    # Extract unmasked raw original image crop if available, otherwise read cell_file_path
    if orig_img is not None and global_bbox is not None:
        gx1, gy1, gx2, gy2 = global_bbox
        img = orig_img[gy1:gy2, gx1:gx2]
        cv2.imwrite(os.path.join(cell_output_dir, "input_unmasked_raw.jpg"), img)
    else:
        img = cv2.imread(cell_file_path)
        gx1, gy1 = (0, 0)

    if img is None:
        print(f"Error reading image for {cell_name}")
        return 0

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    # 1. Cyan Contrast Map: (B + G) / 2 - R
    b, g, r = cv2.split(img.astype(np.float32))
    cyan_diff = (b + g) / 2.0 - r
    cyan_diff = np.clip(cyan_diff, 0, 255).astype(np.uint8)

    # 2. Unsharp Masking to sharpen faint cyan edges
    blur = cv2.GaussianBlur(cyan_diff, (5, 5), 1.0)
    cyan_sharpened = cv2.addWeighted(cyan_diff, 1.8, blur, -0.8, 0)
    _, cyan_bin = cv2.threshold(cyan_sharpened, 35, 255, cv2.THRESH_BINARY)

    # Thin morphological closing
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cyan_closed = cv2.morphologyEx(cyan_bin, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    contours, _ = cv2.findContours(cyan_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            # Filter noise < 100 and full-image container (> 70% area or > 85% width & height)
            if area >= 100 and not (area > 0.7 * total_img_area or (w > 0.85 * img_w and h > 0.85 * img_h)):
                candidates.append((x, y, w, h, area, c))

    # 3. Non-Composite Filter: only split containers if they enclose >= 2 significant sub-cells (area >= 300)
    non_composite = []
    for i, (cx1, cy1, cw1, ch1, ca1, cc1) in enumerate(candidates):
        num_contained_cells = 0
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
            if inter_area >= 0.8 * box2_area and ca1 > ca2 and ca2 >= 300:
                num_contained_cells += 1

        if num_contained_cells < 2:
            non_composite.append((cx1, cy1, cw1, ch1, ca1, cc1))

    # 4. Duplicate Subset Filter & Cell Process Attachment Merger
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

            # A: Strict internal duplicate check
            if inter_area >= 0.75 * box1_area and ca2 > ca1:
                is_dup_or_leg = True
                break

            # B: Cell Process / Leg Attachment check (small extension touching/overlapping larger cell body)
            if ca1 < 250 and ca2 >= 350:
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

    # 5. Homogeneity Filter: Filter out sub-cells that lack internal cell structure (empty gray space)
    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    final_subcell_tuples = []
    for (cx1, cy1, cw1, ch1, ca1, cc1) in subcell_tuples:
        mask_sub = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(mask_sub, [cc1], -1, 255, cv2.FILLED)

        crop_sub = img[cy1:cy1+ch1, cx1:cx1+cw1]
        crop_mask = mask_sub[cy1:cy1+ch1, cx1:cx1+cw1]

        crop_hsv = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2HSV)
        crop_cyan = cv2.inRange(crop_hsv, lower_cyan, upper_cyan)

        non_white = np.any(crop_sub < 240, axis=2) & (crop_mask > 0)
        interior = non_white & (crop_cyan == 0)

        crop_gray = cv2.cvtColor(crop_sub, cv2.COLOR_BGR2GRAY)
        interior_pixels = crop_gray[interior] if np.any(interior) else crop_gray[non_white]

        if len(interior_pixels) > 0:
            std_val = np.std(interior_pixels)
            min_val = np.min(interior_pixels)
            if std_val < 25 and min_val > 130:
                print(f"[{cell_name}] Filtered out homogeneous empty sub-cell: bbox=({cx1},{cy1},{cw1},{ch1}), Std={std_val:.1f}, Min={min_val}")
                continue

        final_subcell_tuples.append((cx1, cy1, cw1, ch1, ca1, cc1))

    # Fallback if empty: use primary contour or non-white region
    if not final_subcell_tuples:
        if candidates:
            final_subcell_tuples = candidates
        else:
            non_white = np.any(img < 240, axis=2).astype(np.uint8) * 255
            nw_contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if nw_contours:
                best_c = max(nw_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(best_c)
                final_subcell_tuples = [(x, y, w, h, cv2.contourArea(best_c), best_c)]

    # Sort sub-cells top-to-bottom, left-to-right
    final_subcell_tuples = sorted(final_subcell_tuples, key=lambda s: (s[1], s[0]))

    overview_img = img.copy()

    for idx, (x, y, w, h, area, contour) in enumerate(final_subcell_tuples, start=1):
        x1 = max(0, x - subcell_padding)
        y1 = max(0, y - subcell_padding)
        x2 = min(img_w, x + w + subcell_padding)
        y2 = min(img_h, y + h + subcell_padding)

        cell_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(cell_mask_full, [contour], -1, 255, cv2.FILLED)

        crop_img = img[y1:y2, x1:x2]
        crop_mask = cell_mask_full[y1:y2, x1:x2]

        white_canvas = np.full_like(crop_img, 255)
        mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
        white_canvas = np.where(mask_3ch > 0, crop_img, white_canvas)

        # 1. Extracted sub-cell on white canvas
        subcell_filename = f"subcell_{idx:02d}_extracted.jpg"
        cv2.imwrite(os.path.join(cell_output_dir, subcell_filename), white_canvas)

        # 2. Marked on cell crop
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(cell_output_dir, f"subcell_{idx:02d}_marked.jpg"), marked_img)

        # 3. Marked on full original JPG
        if orig_img is not None and global_bbox is not None:
            orig_marked = orig_img.copy()
            glob_x1 = gx1 + x1
            glob_y1 = gy1 + y1
            glob_x2 = gx1 + x2
            glob_y2 = gy1 + y2
            cv2.rectangle(orig_marked, (glob_x1, glob_y1), (glob_x2, glob_y2), (0, 0, 255), 3)
            cv2.putText(orig_marked, f"{cell_name} Subcell #{idx}", (glob_x1, max(15, glob_y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(cell_output_dir, f"subcell_{idx:02d}_marked_original.jpg"), orig_marked)

        # 4. Overview update
        cv2.rectangle(overview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_img, str(idx), (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(cell_output_dir, "subcells_overview.jpg"), overview_img)
    return len(final_subcell_tuples)

def process_all_step2(step1_dir, original_image_path, output_dir):
    """
    Process all cells extracted in Step 1 and create a separate folder for each cell under step-2.
    """
    os.makedirs(output_dir, exist_ok=True)

    cell_files = sorted(glob.glob(os.path.join(step1_dir, "cell_*_extracted.jpg")))
    if not cell_files:
        raise FileNotFoundError(f"No Step 1 extracted cell files found in {step1_dir}")

    print(f"Found {len(cell_files)} extracted cells from Step 1.")

    orig_img = cv2.imread(original_image_path) if os.path.exists(original_image_path) else None
    cell_bboxes = get_step1_cell_bboxes(original_image_path) if orig_img is not None else {}

    summary = {}
    for cell_file in cell_files:
        cell_basename = os.path.splitext(os.path.basename(cell_file))[0]
        cell_name = cell_basename.replace("_extracted", "")  # e.g., cell_001

        cell_output_dir = os.path.join(output_dir, cell_name)
        global_bbox = cell_bboxes.get(cell_name, None)

        count = process_single_cell_step2(cell_file, orig_img, global_bbox, cell_output_dir)
        summary[cell_name] = count
        print(f"[{cell_name}] Extracted {count} atomic sub-cell(s) into {cell_output_dir}")

    print(f"\nStep 2 Version 2 pipeline completed for all {len(cell_files)} cells. Output root: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Version 2 Step 2 pipeline for all cells")
    parser.add_argument("--step1-dir", default="/Users/dpeleg/local/MicroGlia/Data/agy-extracted",
                        help="Directory containing Step 1 extracted cells")
    parser.add_argument("--original-image", default="/Users/dpeleg/local/MicroGlia/Data/raw-data/JPG_VID2724_B1_3_00d07h00m.jpg",
                        help="Path to original full JPG image")
    parser.add_argument("--output-dir", default="/Users/dpeleg/local/MicroGlia/Data/step-2",
                        help="Output directory for Step 2 folders")

    args = parser.parse_args()
    process_all_step2(args.step1_dir, args.original_image, args.output_dir)
