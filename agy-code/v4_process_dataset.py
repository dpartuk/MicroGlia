import os
import glob
import cv2
import numpy as np
import argparse

def extract_direct_1step_single_image(image_path, output_dir, subcell_padding=5):
    """
    Direct 1-Step Sub-Cell Extraction:
    Skips Step 1 completely and extracts atomic sub-cells directly from the full image (.jpg or .tif).
    Uses:
    1. Cyan Difference Contrast Map: I_cyan = (B+G)/2 - R
    2. Unsharp Masking: I_sharp = 1.8 * I_cyan - 0.8 * GaussianBlur(I_cyan)
    3. Non-Composite Filter
    4. Duplicate Subset Filter & Cell Process Attachment Merger
    5. Homogeneity Filter
    6. Post-Processing Pass: Convex Hull Boundary Closure Reconstruction for border cells
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Clean old files
    for old_file in glob.glob(os.path.join(output_dir, "*")):
        try:
            if os.path.isfile(old_file):
                os.remove(old_file)
        except OSError:
            pass

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    img_h, img_w = img.shape[:2]
    total_img_area = img_h * img_w

    # Save original reference copy
    cv2.imwrite(os.path.join(output_dir, "input_original.jpg"), img)

    # 1. Cyan Contrast Map: (B + G) / 2 - R
    b, g, r = cv2.split(img.astype(np.float32))
    cyan_diff = (b + g) / 2.0 - r
    cyan_diff = np.clip(cyan_diff, 0, 255).astype(np.uint8)

    # 2. Unsharp Masking to sharpen faint cyan edges
    blur = cv2.GaussianBlur(cyan_diff, (5, 5), 1.0)
    cyan_sharpened = cv2.addWeighted(cyan_diff, 1.8, blur, -0.8, 0)
    cv2.imwrite(os.path.join(output_dir, "cyan_sharpened_map.jpg"), cyan_sharpened)

    # Threshold sharpened cyan map
    _, cyan_bin = cv2.threshold(cyan_sharpened, 35, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, "cyan_binary_mask.jpg"), cyan_bin)

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

            # B: Cell Process / Leg Attachment check
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

    # 5. Homogeneity Filter
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
                continue

        final_subcell_tuples.append((cx1, cy1, cw1, ch1, ca1, cc1))

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
        subcell_filename = f"subcell_{idx:03d}_extracted.jpg"
        cv2.imwrite(os.path.join(output_dir, subcell_filename), white_canvas)

        # 2. Marked copy on full image
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, f"subcell_{idx:03d}_marked.jpg"), marked_img)

        # 3. Overview update
        cv2.rectangle(overview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_img, str(idx), (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "subcells_overview.jpg"), overview_img)

    # 6. POST-PROCESSING PASS: Convex Hull Boundary Closure Reconstruction for border cells
    reconstructed_count = 0
    sub_files = sorted(glob.glob(os.path.join(output_dir, "subcell_*_extracted.jpg")))

    for sub_file in sub_files:
        sub_img = cv2.imread(sub_file)
        if sub_img is None: continue

        for c in contours:
            area = cv2.contourArea(c)
            if area < 100: continue
            x, y, w, h = cv2.boundingRect(c)
            touches_border = (x <= 5 or y <= 5 or (x + w) >= img_w - 5 or (y + h) >= img_h - 5)

            if touches_border:
                p = subcell_padding
                x1, y1 = max(0, x - p), max(0, y - p)
                x2, y2 = min(img_w, x + w + p), min(img_h, y + h + p)
                crop_img = img[y1:y2, x1:x2]

                if crop_img.shape == sub_img.shape:
                    hull_c = cv2.convexHull(c)
                    cell_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
                    cv2.drawContours(cell_mask_full, [hull_c], -1, 255, cv2.FILLED)
                    crop_mask = cell_mask_full[y1:y2, x1:x2]

                    white_canvas = np.full_like(crop_img, 255)
                    mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
                    reconstructed = np.where(mask_3ch > 0, crop_img, white_canvas)

                    cv2.imwrite(sub_file, reconstructed)
                    reconstructed_count += 1
                    break

    return len(final_subcell_tuples), reconstructed_count

def process_dataset(raw_dir, output_root_dir):
    """
    Batch process all microscopy images (.jpg, .jpeg, .tif, .tiff) using Direct 1-Step Extraction.
    Creates: output_root_dir/<image_stem>/
    """
    if not os.path.exists(raw_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    os.makedirs(output_root_dir, exist_ok=True)

    extensions = ["*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.JPG", "*.JPEG", "*.TIF", "*.TIFF"]
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(raw_dir, ext)))

    image_files = sorted(list(set(image_files)))
    if not image_files:
        print(f"No image files (.jpg, .tif) found in {raw_dir}")
        return

    print(f"\n=======================================================")
    print(f" DIRECT 1-STEP BATCH PROCESSING FOR {len(image_files)} IMAGES")
    print(f" Raw Data Directory: {raw_dir}")
    print(f" Output Root Directory: {output_root_dir}")
    print(f"=======================================================\n")

    summary_results = []

    for idx, img_path in enumerate(image_files, start=1):
        filename = os.path.basename(img_path)
        img_stem, ext = os.path.splitext(filename)

        image_out_dir = os.path.join(output_root_dir, img_stem)

        print(f"[{idx}/{len(image_files)}] Processing Image: {filename}")
        print(f"  -> Output Directory: {image_out_dir}")

        total_subcells, border_reconstructed = extract_direct_1step_single_image(img_path, image_out_dir)

        summary_results.append({
            "filename": filename,
            "stem": img_stem,
            "format": ext.lower(),
            "subcells": total_subcells,
            "border_reconstructed": border_reconstructed,
            "output_dir": image_out_dir
        })
        print(f"  [DONE] {filename}: {total_subcells} atomic sub-cell(s) ({border_reconstructed} border cells reconstructed)\n")

    print("\n=========================================================================")
    print(" DIRECT 1-STEP BATCH PROCESSING COMPLETE SUMMARY")
    print("=========================================================================")
    print(f"{'Image Name':<32} | {'Format':<6} | {'Atomic Sub-Cells':<18} | {'Border Reconstructed':<20}")
    print("-" * 85)
    grand_total_subcells = 0
    grand_total_reconstructed = 0
    for res in summary_results:
        print(f"{res['filename']:<32} | {res['format']:<6} | {res['subcells']:<18d} | {res['border_reconstructed']:<20d}")
        grand_total_subcells += res['subcells']
        grand_total_reconstructed += res['border_reconstructed']
    print("-" * 85)
    print(f"{'GRAND TOTAL':<32} | {'--':<6} | {grand_total_subcells:<18d} | {grand_total_reconstructed:<20d}")
    print("=========================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Direct 1-Step Batch Processor for all microscopy images")
    parser.add_argument("--raw-dir", default="/Users/dpeleg/local/MicroGlia/Data/raw-data",
                        help="Directory containing raw microscopy images")
    parser.add_argument("--output-root", default="/Users/dpeleg/local/MicroGlia/Data/output",
                        help="Root output directory for all image results")

    args = parser.parse_args()
    process_dataset(args.raw_dir, args.output_root)
