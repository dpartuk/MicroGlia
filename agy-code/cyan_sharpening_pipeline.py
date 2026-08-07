import os
import glob
import cv2
import numpy as np

def extract_cells_with_cyan_sharpening(
    image_path,
    output_dir,
    min_cell_area=60,
    max_cell_area=500000,
    subcell_padding=5,
    min_subcell_area=35
):
    """
    Alternative 3: "Cyan-Sharpening" Pipeline
    Combines lab-annotated Cyan HSV Color Space chrominance masks (H in [75, 115])
    with Scharr/Canny Edge Gradient Fusion and CLAHE contrast enhancement.
    Uses the cyan contour ring as an explicit topological anchor guiding single-cell
    extraction and boundary silhouette masking.
    """
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    img_h, img_w = img.shape[:2]

    # STEP 1: CONVERT TO HSV & EXTRACT WHOLE-SLIDE BINARY CYAN BORDER MAP
    # Pre-Processing on Original Image: Cyan pixels = 255 (WHITE), All other pixels = 0 (BLACK)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # Clean cyan mask via morphological ellipse closing
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel_med)

    # Save Whole-Slide Binary Cyan Border Map (Cyan=White, Other=Black)
    cv2.imwrite(os.path.join(output_dir, "whole_slide_cyan_borders_white.jpg"), cyan_closed)
    cv2.imwrite(os.path.join(output_dir, "whole_slide_cyan_borders_black_inv.jpg"), cv2.bitwise_not(cyan_closed))

    # STEP 2: GRAYSCALE MULTI-TILE CLAHE & SCHARR GRADIENT
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)

    scharr_x = cv2.Scharr(clahe_gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(clahe_gray, cv2.CV_64F, 0, 1)
    scharr_mag = cv2.magnitude(scharr_x, scharr_y)
    scharr_norm = np.uint8(np.clip(scharr_mag / np.max(scharr_mag) * 255.0, 0, 255))

    canny_edges = cv2.Canny(clahe_gray, 40, 120)
    edge_fused = cv2.addWeighted(scharr_norm, 0.7, canny_edges, 0.3, 0)

    # STEP 3: CYAN-GUIDED TOPOLOGICAL EDGE FUSION
    # Multiply intensity edge map by cyan mask boost to anchor cell boundaries
    cyan_boost = np.where(cyan_closed > 0, 1.5, 1.0).astype(np.float32)
    cyan_fused_float = cv2.addWeighted(clahe_gray, 0.65, edge_fused, 0.35, 0).astype(np.float32) * cyan_boost
    cyan_sharpened = np.uint8(np.clip(cyan_fused_float, 0, 255))

    # STEP 4: CONTOUR EXTRACTION & CANDIDATE DEDUPLICATION
    contours, _ = cv2.findContours(cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)
        if min_cell_area <= area <= max_cell_area:
            x, y, w, h = cv2.boundingRect(c)
            candidates.append((x, y, w, h, area, c))

    # SUB-CELL IoMin DEDUPLICATION
    def io_min(b1, b2):
        x1 = max(b1[0], b2[0])
        y1 = max(b1[1], b2[1])
        x2 = min(b1[0]+b1[2], b2[0]+b2[2])
        y2 = min(b1[1]+b1[3], b2[1]+b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        a1 = b1[2] * b1[3]
        a2 = b2[2] * b2[3]
        return inter / float(min(a1, a2) + 1e-6)

    candidates = sorted(candidates, key=lambda s: s[4], reverse=True)
    final_subcells = []

    for cand in candidates:
        duplicate = False
        for k in final_subcells:
            if io_min(cand, k) > 0.50:
                duplicate = True
                break
        if not duplicate:
            final_subcells.append(cand)

    # Sort left-to-right, top-to-bottom
    final_subcells = sorted(final_subcells, key=lambda s: (s[1], s[0]))

    # STEP 5: GENERATE SINGLE-CELL CROPS & OUTPUT MAPS
    overview_original = img.copy()
    cyan_sharpened_bgr = cv2.cvtColor(cyan_sharpened, cv2.COLOR_GRAY2BGR)
    overview_cyan_sharpened = cyan_sharpened_bgr.copy()

    for idx, (x, y, w, h, area, contour) in enumerate(final_subcells, start=1):
        subcell_id = f"subcell_{idx:03d}"

        x1 = max(0, x - subcell_padding)
        y1 = max(0, y - subcell_padding)
        x2 = min(img_w, x + w + subcell_padding)
        y2 = min(img_h, y + h + subcell_padding)

        cell_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(cell_mask_full, [contour], -1, 255, cv2.FILLED)

        # A: Crop from Original RGB Image
        crop_img = img[y1:y2, x1:x2]
        crop_mask = cell_mask_full[y1:y2, x1:x2]
        white_canvas = np.full_like(crop_img, 255)
        mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
        extracted_orig = np.where(mask_3ch > 0, crop_img, white_canvas)
        cv2.imwrite(os.path.join(output_dir, f"{subcell_id}_original_extracted.jpg"), extracted_orig)

        # B: Crop from Cyan-Sharpened Map
        crop_cyan_sharp = cyan_sharpened_bgr[y1:y2, x1:x2]
        extracted_cyan_sharp = np.where(mask_3ch > 0, crop_cyan_sharp, white_canvas)
        cv2.imwrite(os.path.join(output_dir, f"{subcell_id}_cyan_sharpened_extracted.jpg"), extracted_cyan_sharp)

        # C: Save Marked Context Location Image
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.drawContours(marked_img, [contour], -1, (255, 255, 0), 2) # Bright cyan contour highlight
        cv2.putText(marked_img, f"Subcell #{idx:03d}", (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(os.path.join(output_dir, f"{subcell_id}_marked.jpg"), marked_img)

        # D: Update Overview Maps
        cv2.rectangle(overview_original, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_original, str(idx), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

        cv2.rectangle(overview_cyan_sharpened, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_cyan_sharpened, str(idx), (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    cv2.imwrite(os.path.join(output_dir, "overview_map_on_original.jpg"), overview_original)
    cv2.imwrite(os.path.join(output_dir, "overview_map_on_cyan_sharpened.jpg"), overview_cyan_sharpened)

    # STEP 6: GENERATE INTERACTIVE WEB GALLERY
    generate_cyan_html_gallery(output_dir)

    return len(final_subcells)

def generate_cyan_html_gallery(output_dir):
    """
    Generates a responsive Safari/Chrome web gallery for cyan-sharpening extracted cells.
    """
    jpg_files = sorted(glob.glob(os.path.join(output_dir, "subcell_*.jpg")))
    if not jpg_files: return

    rel_files = [os.path.basename(f) for f in jpg_files]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Cyan-Sharpening Gallery - {os.path.basename(output_dir)}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #121212;
            color: #e0e0e0;
            margin: 0;
            padding: 20px;
        }}
        h1 {{
            font-size: 22px;
            margin-bottom: 5px;
            color: #00e5ff;
        }}
        p {{
            color: #888;
            font-size: 14px;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
            gap: 15px;
        }}
        .card {{
            background: #1e1e1e;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            transition: transform 0.15s ease, border-color 0.15s ease;
        }}
        .card:hover {{
            transform: translateY(-3px);
            border-color: #00e5ff;
            box-shadow: 0 4px 15px rgba(0, 229, 255, 0.2);
        }}
        .card img {{
            max-width: 100%;
            max-height: 140px;
            object-fit: contain;
            background: #ffffff;
            border-radius: 4px;
        }}
        .card .title {{
            margin-top: 8px;
            font-size: 11px;
            word-break: break-all;
            color: #ccc;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <h1>MicroGlia Cyan-Sharpening Extracted Cells</h1>
    <p>Directory: <code>{output_dir}</code> | Total Sub-Cell Crops: {len(rel_files)}</p>
    <div class="grid">
"""
    for fname in rel_files:
        html_content += f"""        <div class="card">
            <img src="{fname}" alt="{fname}" loading="lazy">
            <div class="title">{fname}</div>
        </div>
"""
    html_content += """    </div>
</body>
</html>
"""
    with open(os.path.join(output_dir, "view_extracted_cells.html"), "w") as f:
        f.write(html_content)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        extract_cells_with_cyan_sharpening(sys.argv[1], sys.argv[2])
