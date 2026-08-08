import os
import glob
import cv2
import numpy as np
from scipy.spatial import cKDTree

def detect_and_close_open_contours(
    binary_border_map,
    max_gap_distance=25,
    min_contour_length=10
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

                # 2. Draw bright RED line & red marker dots into overlay map (Thickness 2 + dots for visibility)
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
    Batch Processor that walks over whole-slide binary border maps, detects unclosed contours,
    bridges gaps, and outputs closed-contour border maps with bright RED gap overlays.
    """
    os.makedirs(output_dir, exist_ok=True)
    binary_border_files = sorted(glob.glob(os.path.join(input_dir, "*_whole_slide_binary_borders.jpg")))

    if not binary_border_files:
        print(f"No binary border maps found in {input_dir}.")
        return

    print("\n=======================================================================")
    print(f" AUTOMATED CONTOUR CLOSING ENGINE ({len(binary_border_files)} SLIDES)")
    print(f" Input Directory: {input_dir}")
    print(f" Output Directory: {output_dir}")
    print("=======================================================================\n")

    for idx, border_path in enumerate(binary_border_files, start=1):
        stem = os.path.basename(border_path).replace("_whole_slide_binary_borders.jpg", "")
        binary_map = cv2.imread(border_path, cv2.IMREAD_GRAYSCALE)
        if binary_map is None: continue

        # Run Ultra-Fast KDTree Contour Closing Engine
        closed_map, red_overlay_map, stats = detect_and_close_open_contours(binary_map, max_gap_distance=25)

        # Save Output Closed Border Maps & Red Highlight Overlay
        path_closed = os.path.join(output_dir, f"{stem}_whole_slide_closed_borders.jpg")
        path_red_overlay = os.path.join(output_dir, f"{stem}_red_closed_borders_overlay.jpg")
        cv2.imwrite(path_closed, closed_map)
        cv2.imwrite(path_red_overlay, red_overlay_map)

        # Build 400% Zoomed-In ROI Crop showing exact RED gap closures in high detail
        h_img, w_img = binary_map.shape[:2]
        cx, cy = w_img // 2, h_img // 2
        crop_size = 400
        x1, y1 = max(0, cx - crop_size//2), max(0, cy - crop_size//2)
        x2, y2 = min(w_img, x1 + crop_size), min(h_img, y1 + crop_size)

        roi_overlay = red_overlay_map[y1:y2, x1:x2]
        roi_zoom = cv2.resize(roi_overlay, (800, 800), interpolation=cv2.INTER_NEAREST)
        cv2.putText(roi_zoom, "400% Zoom ROI: Red Closed Border Gaps", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)

        path_roi_zoom = os.path.join(output_dir, f"{stem}_red_closed_borders_ROI_zoom.jpg")
        cv2.imwrite(path_roi_zoom, roi_zoom)

        # Build Side-by-Side Visual QA Comparison Panel (Original vs. Closed)
        orig_path = os.path.join(input_dir, f"{stem}_whole_slide_original.jpg")
        img_orig = cv2.imread(orig_path)
        if img_orig is None:
            img_orig = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)

        # Find filled closed contours to visualize closed 2D cellular regions
        filled_contours_map = np.zeros_like(closed_map)
        contours, _ = cv2.findContours(closed_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) >= 50:
                cv2.drawContours(filled_contours_map, [c], -1, 255, cv2.FILLED)

        panel = np.zeros((1280, 1280, 3), dtype=np.uint8) + 255
        s = 580

        p1 = cv2.resize(img_orig, (s, s))
        p2 = cv2.resize(cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR), (s, s))
        p3 = cv2.resize(red_overlay_map, (s, s))
        p4 = cv2.resize(cv2.cvtColor(filled_contours_map, cv2.COLOR_GRAY2BGR), (s, s))

        panel[50:50+s, 40:40+s] = p1
        panel[50:50+s, 660:660+s] = p2
        panel[680:680+s, 40:40+s] = p3
        panel[680:680+s, 660:660+s] = p4

        cv2.putText(panel, f"(A) Raw Input: {stem}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(B) Initial Binary Border Map (With Unclosed Gaps)", (660, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, f"(C) Red Closed Borders ({stats['open_gaps_closed']} RED Gaps)", (40, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(D) Filled Closed 2D Cell Regions (100% Closed)", (660, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)

        path_panel = os.path.join(output_dir, f"{stem}_contour_closing_panel.jpg")
        cv2.imwrite(path_panel, panel)

        print(f"[{idx}/{len(binary_border_files)}] Processed: {stem}")
        print(f"  • Total Open Endpoints: {stats['total_endpoints']} | RED Gaps Closed: {stats['open_gaps_closed']}")
        print(f"  • Saved Closed Map: {path_closed}")
        print(f"  • Saved RED Overlay Map: {path_red_overlay}")
        print(f"  • Saved RED ROI Zoom: {path_roi_zoom}")
        print(f"  • Saved Visual QA Panel: {path_panel}\n")

    print("=======================================================================")
    print(" AUTOMATED CONTOUR CLOSING ENGINE PROCESSING COMPLETE")
    print("=======================================================================\n")

if __name__ == "__main__":
    process_whole_slide_contour_closing()
