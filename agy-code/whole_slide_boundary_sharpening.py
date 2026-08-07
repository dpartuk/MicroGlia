import os
import glob
import cv2
import numpy as np

def sharpen_whole_slide_boundaries(
    image_path,
    output_dir,
    lower_cyan=(75, 40, 140),
    upper_cyan=(115, 255, 255)
):
    """
    Dedicated Whole-Slide Boundary Sharpening Module (Cell Extraction Disabled).
    Operates directly on full-resolution whole-slide images to generate high-precision
    whole-slide border maps and sharpened topological edge composites.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {image_path}")

    stem = os.path.splitext(os.path.basename(image_path))[0]
    img_h, img_w = img_bgr.shape[:2]

    # STEP 1: CONVERT TO HSV & GENERATE BINARY CYAN BORDER MAP (Cyan=255/White, Other=0/Black)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(lower_cyan, dtype=np.uint8)
    upper = np.array(upper_cyan, dtype=np.uint8)
    cyan_mask = cv2.inRange(hsv, lower, upper)

    # Ellipse closing to bridge tiny line breaks in cyan borders
    kernel_med = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel_med)

    # STEP 2: STREAM A - GRAYSCALE MULTI-TILE CLAHE & SCHARR TISSUE GRADIENT
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)

    scharr_x = cv2.Scharr(clahe_gray, cv2.CV_64F, 1, 0)
    scharr_y = cv2.Scharr(clahe_gray, cv2.CV_64F, 0, 1)
    scharr_mag = cv2.magnitude(scharr_x, scharr_y)
    scharr_norm = np.uint8(np.clip(scharr_mag / np.max(scharr_mag) * 255.0, 0, 255))

    # STEP 3: STREAM B - CANNY 1-PIXEL EDGE TRACING ON BINARY BORDER MAP
    canny_cyan_border = cv2.Canny(cyan_closed, 50, 150)

    # STEP 4: DUAL-STREAM TOPOLOGICAL EDGE FUSION
    cyan_mask_norm = (cyan_closed > 0).astype(np.float32)
    tissue_arbors_inside_cyan = (scharr_norm.astype(np.float32) * cyan_mask_norm)

    fused_float = (0.50 * clahe_gray.astype(np.float32) +
                   0.30 * tissue_arbors_inside_cyan +
                   0.20 * canny_cyan_border.astype(np.float32))
    fused_composite = np.uint8(np.clip(fused_float, 0, 255))

    # STEP 5: SAVE WHOLE-SLIDE BOUNDARY MAP ARTIFACTS
    path_orig = os.path.join(output_dir, f"{stem}_whole_slide_original.jpg")
    path_binary = os.path.join(output_dir, f"{stem}_whole_slide_binary_borders.jpg")
    path_scharr = os.path.join(output_dir, f"{stem}_whole_slide_scharr_edges.jpg")
    path_canny = os.path.join(output_dir, f"{stem}_whole_slide_canny_cyan.jpg")
    path_fused = os.path.join(output_dir, f"{stem}_whole_slide_fused_composite.jpg")

    cv2.imwrite(path_orig, img_bgr)
    cv2.imwrite(path_binary, cyan_closed)
    cv2.imwrite(path_scharr, scharr_norm)
    cv2.imwrite(path_canny, canny_cyan_border)
    cv2.imwrite(path_fused, fused_composite)

    # STEP 6: BUILD MULTI-PANEL QA COMPARISON MAP
    panel = np.zeros((1280, 1280, 3), dtype=np.uint8) + 255
    s = 580

    p1 = cv2.resize(img_bgr, (s, s))
    p2 = cv2.resize(cv2.cvtColor(cyan_closed, cv2.COLOR_GRAY2BGR), (s, s))
    p3 = cv2.resize(cv2.cvtColor(canny_cyan_border, cv2.COLOR_GRAY2BGR), (s, s))
    p4 = cv2.resize(cv2.cvtColor(fused_composite, cv2.COLOR_GRAY2BGR), (s, s))

    panel[50:50+s, 40:40+s] = p1
    panel[50:50+s, 660:660+s] = p2
    panel[680:680+s, 40:40+s] = p3
    panel[680:680+s, 660:660+s] = p4

    cv2.putText(panel, f"(A) Whole-Slide Raw Input: {stem}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
    cv2.putText(panel, "(B) Whole-Slide Binary Cyan Border Map (Cyan=255)", (660, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
    cv2.putText(panel, "(C) Whole-Slide Canny 1-Pixel Border Outline", (40, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
    cv2.putText(panel, "(D) Whole-Slide Sharpened Fused Composite Map", (660, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)

    path_panel = os.path.join(output_dir, f"{stem}_whole_slide_panel.jpg")
    cv2.imwrite(path_panel, panel)

    print(f"  [DONE] Processed Whole-Slide Boundaries for: {stem}")
    print(f"         - Binary Border Map: {path_binary}")
    print(f"         - Fused Composite Map: {path_fused}")
    print(f"         - Multi-Panel Visual QA: {path_panel}")

    return path_fused

def process_all_raw_slides_whole_boundary_sharpening(
    raw_dir="/Users/dpeleg/local/MicroGlia/Data/raw-data",
    output_dir="/Users/dpeleg/local/MicroGlia/Data/whole-slide-sharpened-borders"
):
    """
    Batch Processor for Whole-Slide Boundary Sharpening (Cell Extraction Disabled).
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    image_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(supported_exts)
    ])

    print("\n=======================================================================")
    print(f" WHOLE-SLIDE BOUNDARY SHARPENING ENGINE (CELL EXTRACTION DISABLED)")
    print(f" Processing {len(image_files)} Whole-Slide Images in: {raw_dir}")
    print(f" Output Root Directory: {output_dir}")
    print("=======================================================================\n")

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(raw_dir, img_name)
        print(f"[{idx}/{len(image_files)}] Processing Whole-Slide: {img_name}")
        sharpen_whole_slide_boundaries(img_path, output_dir)

    print("\n=======================================================================")
    print(" WHOLE-SLIDE BOUNDARY SHARPENING COMPLETE")
    print("=======================================================================\n")

if __name__ == "__main__":
    process_all_raw_slides_whole_boundary_sharpening()
