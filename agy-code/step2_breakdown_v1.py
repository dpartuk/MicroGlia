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
    Process one Step 1 extracted cell image and break it down into its sub-cells.
    """
    os.makedirs(cell_output_dir, exist_ok=True)

    cell_basename = os.path.splitext(os.path.basename(cell_file_path))[0]
    cell_name = cell_basename.replace("_extracted", "")

    # Copy Step 1 extracted cell into subfolder as input reference
    shutil.copy(cell_file_path, os.path.join(cell_output_dir, "input_extracted.jpg"))

    # Copy Step 1 marked image if available
    step1_dir = os.path.dirname(cell_file_path)
    step1_marked_path = os.path.join(step1_dir, f"{cell_name}_marked.jpg")
    if os.path.exists(step1_marked_path):
        shutil.copy(step1_marked_path, os.path.join(cell_output_dir, "input_marked.jpg"))

    img = cv2.imread(cell_file_path)
    if img is None:
        print(f"Error reading {cell_file_path}")
        return 0

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # Thin morphological closing to separate internal sub-cell cyan boundaries
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    contours, _ = cv2.findContours(cyan_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    subcell_contours = []
    if contours:
        for c in contours:
            area = cv2.contourArea(c)
            x, y, w, h = cv2.boundingRect(c)
            # Ignore noise < 120 area, and ignore full-image container (> 70% total image area or > 85% width & height)
            if area >= 120 and not (area > 0.7 * total_img_area or (w > 0.85 * img_w and h > 0.85 * img_h)):
                subcell_contours.append(c)

    # Fallback if no internal sub-cells detected: use non-white region or primary cyan contour
    if not subcell_contours:
        non_white = np.any(img < 240, axis=2).astype(np.uint8) * 255
        nw_contours, _ = cv2.findContours(non_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if nw_contours:
            subcell_contours = [max(nw_contours, key=cv2.contourArea)]

    # Sort sub-cells top-to-bottom, left-to-right
    subcell_contours = sorted(subcell_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    overview_img = img.copy()
    gx1, gy1, _, _ = global_bbox if global_bbox else (0, 0, img_w, img_h)

    for idx, contour in enumerate(subcell_contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)

        # Sub-cell padding
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

        # 2. Marked on Step 1 cell crop
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
    return len(subcell_contours)

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
        print(f"[{cell_name}] Extracted {count} sub-cell(s) into {cell_output_dir}")

    print(f"\nStep 2 completed for all {len(cell_files)} cells. Output root: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Expand breakdown to all cells from Step 1")
    parser.add_argument("--step1-dir", default="/Users/dpeleg/local/MicroGlia/Data/agy-extracted",
                        help="Directory containing Step 1 extracted cells")
    parser.add_argument("--original-image", default="/Users/dpeleg/local/MicroGlia/Data/raw-data/JPG_VID2724_B1_3_00d07h00m.jpg",
                        help="Path to original full JPG image")
    parser.add_argument("--output-dir", default="/Users/dpeleg/local/MicroGlia/Data/step-2",
                        help="Output directory for Step 2 folders")

    args = parser.parse_args()
    process_all_step2(args.step1_dir, args.original_image, args.output_dir)
