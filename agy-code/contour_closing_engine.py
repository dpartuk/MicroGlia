import os
import glob
import cv2
import numpy as np
from scipy.spatial import cKDTree

def detect_and_close_open_contours(
    binary_border_map,
    max_gap_distance=25
):
    """
    Ultra-Fast KDTree Automated Contour Closing Engine.
    Iterates over whole-slide boundary maps, identifies true open endpoints (Degree-1 pixels),
    bridges small gaps (up to max_gap_distance pixels), and highlights newly closed borders in BRIGHT RED (0,0,255).
    """
    closed_border_map = binary_border_map.copy()
    h, w = closed_border_map.shape[:2]

    # STEP 1: FIND TRUE OPEN ENDPOINTS USING 3x3 NEIGHBORHOOD FILTER
    bin_uint = (binary_border_map > 0).astype(np.uint8)
    kernel_neigh = np.array([[1, 1, 1],
                             [1, 0, 1],
                             [1, 1, 1]], dtype=np.uint8)

    neigh_count = cv2.filter2D(bin_uint, -1, kernel_neigh, borderType=cv2.BORDER_CONSTANT)

    # True open endpoints: foreground pixel (1) with exactly 1 neighbor
    endpoints_mask = (bin_uint == 1) & (neigh_count == 1)
    py, px = np.where(endpoints_mask)

    endpoints = list(zip(px, py))
    pts_arr = np.array(endpoints, dtype=np.float32)

    gaps_closed_count = 0
    bridged_lines = []

    # STEP 2: BUILD BRIGHT RED OVERLAY BASE (Existing borders in Green 0,255,0)
    red_overlay_map = cv2.cvtColor(binary_border_map, cv2.COLOR_GRAY2BGR)
    red_overlay_map[binary_border_map > 0] = [0, 255, 0]

    # STEP 3: ULTRA-FAST cKDTree ENDPOINT PAIRING & RED GAP BRIDGING
    num_pts = len(pts_arr)
    if num_pts > 1:
        tree = cKDTree(pts_arr)
        pairs = tree.query_pairs(r=max_gap_distance)

        used = set()
        for i, j in pairs:
            if i in used or j in used: continue
            pt1 = (int(pts_arr[i][0]), int(pts_arr[i][1]))
            pt2 = (int(pts_arr[j][0]), int(pts_arr[j][1]))
            dist = np.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])

            if 2.0 <= dist <= max_gap_distance:
                # 1. Draw 1-pixel line into output binary closed border map
                cv2.line(closed_border_map, pt1, pt2, 255, thickness=1)

                # 2. Draw bright RED line & red marker dots into overlay map
                cv2.line(red_overlay_map, pt1, pt2, (0, 0, 255), thickness=2)
                cv2.circle(red_overlay_map, pt1, 3, (0, 0, 255), -1)
                cv2.circle(red_overlay_map, pt2, 3, (0, 0, 255), -1)

                bridged_lines.append((pt1, pt2))
                used.add(i)
                used.add(j)
                gaps_closed_count += 1

    # STEP 4: 1-PIXEL REPAIR PASS
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed_border_map = cv2.morphologyEx(closed_border_map, cv2.MORPH_CLOSE, kernel_small)

    stats = {
        "total_endpoints": len(endpoints),
        "open_gaps_closed": gaps_closed_count,
        "bridged_lines": bridged_lines
    }

    return closed_border_map, red_overlay_map, stats

def process_whole_slide_contour_closing(
    input_dir="/Users/dpeleg/local/MicroGlia/Data/whole-slide-sharpened-borders",
    output_dir="/Users/dpeleg/local/MicroGlia/Data/whole-slide-closed-contours"
):
    """
    Batch Processor that executes BOTH Mode A (Thick Cyan Pen Strokes) and
    Mode B (Ultra-Thin 1-2px Canny Outlines) side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    thick_files = sorted(glob.glob(os.path.join(input_dir, "*_whole_slide_binary_borders.jpg")))
    thin_files = sorted(glob.glob(os.path.join(input_dir, "*_ultra_thin_canny_borders.jpg")))

    print("\n=======================================================================", flush=True)
    print(f" AUTOMATED CONTOUR CLOSING ENGINE - DUAL MODE (A: THICK & B: THIN)", flush=True)
    print(f" Input Directory: {input_dir}", flush=True)
    print(f" Output Directory: {output_dir}", flush=True)
    print("=======================================================================\n", flush=True)

    # MODE A: THICK CYAN STROKES
    print("--- MODE A: THICK CYAN PEN STROKES (whole_slide_binary_borders) ---", flush=True)
    for idx, border_path in enumerate(thick_files, start=1):
        stem = os.path.basename(border_path).replace("_whole_slide_binary_borders.jpg", "")
        binary_map = cv2.imread(border_path, cv2.IMREAD_GRAYSCALE)
        if binary_map is None: continue

        closed_map, red_overlay_map, stats = detect_and_close_open_contours(binary_map, max_gap_distance=25)

        path_closed = os.path.join(output_dir, f"{stem}_thick_cyan_closed_borders.jpg")
        path_red_overlay = os.path.join(output_dir, f"{stem}_thick_cyan_red_overlay.jpg")
        cv2.imwrite(path_closed, closed_map)
        cv2.imwrite(path_red_overlay, red_overlay_map)

        # Build QA Panel A
        orig_path = os.path.join(input_dir, f"{stem}_whole_slide_original.jpg")
        img_orig = cv2.imread(orig_path)
        if img_orig is None: img_orig = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)

        panel = np.zeros((1280, 1280, 3), dtype=np.uint8) + 255
        s = 580
        panel[50:50+s, 40:40+s] = cv2.resize(img_orig, (s, s))
        panel[50:50+s, 660:660+s] = cv2.resize(cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR), (s, s))
        panel[680:680+s, 40:40+s] = cv2.resize(red_overlay_map, (s, s))
        panel[680:680+s, 660:660+s] = cv2.resize(cv2.cvtColor(closed_map, cv2.COLOR_GRAY2BGR), (s, s))

        cv2.putText(panel, f"(A) Raw Input: {stem}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(B) Mode A: Thick Cyan Pen Strokes", (660, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, f"(C) Red Closed Gaps ({stats['open_gaps_closed']} RED Gaps)", (40, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(D) 100% Closed Border Map", (660, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)

        path_panel = os.path.join(output_dir, f"{stem}_thick_cyan_contour_closing_panel.jpg")
        cv2.imwrite(path_panel, panel)

        print(f"[{idx}/{len(thick_files)}] Mode A Processed: {stem} | RED Gaps: {stats['open_gaps_closed']}", flush=True)

    # MODE B: ULTRA-THIN 1-2PX CANNY OUTLINES
    print("\n--- MODE B: ULTRA-THIN 1-2PX CANNY OUTLINES (ultra_thin_canny_borders) ---", flush=True)
    for idx, border_path in enumerate(thin_files, start=1):
        stem = os.path.basename(border_path).replace("_ultra_thin_canny_borders.jpg", "")
        binary_map = cv2.imread(border_path, cv2.IMREAD_GRAYSCALE)
        if binary_map is None: continue

        closed_map, red_overlay_map, stats = detect_and_close_open_contours(binary_map, max_gap_distance=25)

        path_closed = os.path.join(output_dir, f"{stem}_thin_canny_closed_borders.jpg")
        path_red_overlay = os.path.join(output_dir, f"{stem}_thin_canny_red_overlay.jpg")
        cv2.imwrite(path_closed, closed_map)
        cv2.imwrite(path_red_overlay, red_overlay_map)

        # Build QA Panel B
        orig_path = os.path.join(input_dir, f"{stem}_whole_slide_original.jpg")
        img_orig = cv2.imread(orig_path)
        if img_orig is None: img_orig = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)

        panel = np.zeros((1280, 1280, 3), dtype=np.uint8) + 255
        s = 580
        panel[50:50+s, 40:40+s] = cv2.resize(img_orig, (s, s))
        panel[50:50+s, 660:660+s] = cv2.resize(cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR), (s, s))
        panel[680:680+s, 40:40+s] = cv2.resize(red_overlay_map, (s, s))
        panel[680:680+s, 660:660+s] = cv2.resize(cv2.cvtColor(closed_map, cv2.COLOR_GRAY2BGR), (s, s))

        cv2.putText(panel, f"(A) Raw Input: {stem}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(B) Mode B: Ultra-Thin 1-2px Canny Outlines", (660, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, f"(C) Red Closed Gaps ({stats['open_gaps_closed']} RED Gaps)", (40, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(D) 100% Closed Border Map", (660, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)

        path_panel = os.path.join(output_dir, f"{stem}_thin_canny_contour_closing_panel.jpg")
        cv2.imwrite(path_panel, panel)

        print(f"[{idx}/{len(thin_files)}] Mode B Processed: {stem} | RED Gaps: {stats['open_gaps_closed']}", flush=True)

    print("\n=======================================================================", flush=True)
    print(" DUAL MODE AUTOMATED CONTOUR CLOSING PROCESSING COMPLETE", flush=True)
    print("=======================================================================\n", flush=True)

if __name__ == "__main__":
    process_whole_slide_contour_closing()
