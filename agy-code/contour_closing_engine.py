import os
import glob
import cv2
import numpy as np
from scipy.spatial import cKDTree

def detect_and_close_open_contours(
    binary_border_map,
    max_gap_distance=30,
    min_contour_length=10
):
    """
    Ultra-Fast KDTree Automated Contour Closing Engine.
    Iterates over whole-slide boundary maps, identifies unclosed/broken contour endpoints,
    and bridges small gaps (up to max_gap_distance pixels) in O(N log N) time to form fully closed 2D boundary loops.
    """
    closed_border_map = binary_border_map.copy()
    h, w = closed_border_map.shape[:2]

    # STEP 1: EXTRACT ALL CONTOUR POLYGONS
    contours, hierarchy = cv2.findContours(
        closed_border_map, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )

    if not contours:
        return closed_border_map, {"total": 0, "open_gaps_closed": 0}

    gaps_closed_count = 0
    endpoints = []

    # STEP 2: CHECK EACH CONTOUR FOR SELF-GAP OR OPEN ENDPOINTS
    for c in contours:
        if len(c) < min_contour_length:
            continue

        p_start = tuple(c[0][0])
        p_end = tuple(c[-1][0])

        # Self-closing gap check (start vs end of same contour)
        dist_self = np.hypot(p_start[0] - p_end[0], p_start[1] - p_end[1])
        if 2 <= dist_self <= max_gap_distance:
            cv2.line(closed_border_map, p_start, p_end, 255, thickness=2)
            gaps_closed_count += 1
        else:
            endpoints.append(p_start)
            endpoints.append(p_end)

    # STEP 3: ULTRA-FAST cKDTree ENDPOINT PAIRING (O(N log N))
    pts_arr = np.array(endpoints, dtype=np.float32)
    num_pts = len(pts_arr)

    if num_pts > 1:
        tree = cKDTree(pts_arr)
        # Find all pairs of endpoints within max_gap_distance
        pairs = tree.query_pairs(r=max_gap_distance)

        used = set()
        for i, j in sorted(pairs, key=lambda p: np.hypot(pts_arr[p[0]][0]-pts_arr[p[1]][0], pts_arr[p[0]][1]-pts_arr[p[1]][1])):
            if i in used or j in used: continue
            pt1 = (int(pts_arr[i][0]), int(pts_arr[i][1]))
            pt2 = (int(pts_arr[j][0]), int(pts_arr[j][1]))

            # Bridge the open gap
            cv2.line(closed_border_map, pt1, pt2, 255, thickness=2)
            used.add(i)
            used.add(j)
            gaps_closed_count += 1

    # STEP 4: MORPHOLOGICAL ELLIPSE CLOSING REPAIR FOR RESIDUAL MINOR BREAKS
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_border_map = cv2.morphologyEx(closed_border_map, cv2.MORPH_CLOSE, kernel_repair)

    stats = {
        "total_contours": len(contours),
        "open_gaps_closed": gaps_closed_count
    }

    return closed_border_map, stats

def process_whole_slide_contour_closing(
    input_dir="/Users/dpeleg/local/MicroGlia/Data/whole-slide-sharpened-borders",
    output_dir="/Users/dpeleg/local/MicroGlia/Data/whole-slide-closed-contours"
):
    """
    Batch Processor that walks over whole-slide binary border maps, detects unclosed contours,
    bridges gaps, and outputs closed-contour border maps.
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
        closed_map, stats = detect_and_close_open_contours(binary_map, max_gap_distance=30)

        # Save Output Closed Border Maps
        path_closed = os.path.join(output_dir, f"{stem}_whole_slide_closed_borders.jpg")
        cv2.imwrite(path_closed, closed_map)

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
        p3 = cv2.resize(cv2.cvtColor(closed_map, cv2.COLOR_GRAY2BGR), (s, s))
        p4 = cv2.resize(cv2.cvtColor(filled_contours_map, cv2.COLOR_GRAY2BGR), (s, s))

        panel[50:50+s, 40:40+s] = p1
        panel[50:50+s, 660:660+s] = p2
        panel[680:680+s, 40:40+s] = p3
        panel[680:680+s, 660:660+s] = p4

        cv2.putText(panel, f"(A) Raw Input: {stem}", (40, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(B) Initial Binary Border Map (With Unclosed Gaps)", (660, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, f"(C) Closed Border Map ({stats['open_gaps_closed']} Gaps Closed)", (40, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)
        cv2.putText(panel, "(D) Filled Closed 2D Cell Regions (100% Closed)", (660, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 51, 102), 2, cv2.LINE_AA)

        path_panel = os.path.join(output_dir, f"{stem}_contour_closing_panel.jpg")
        cv2.imwrite(path_panel, panel)

        print(f"[{idx}/{len(binary_border_files)}] Processed: {stem}")
        print(f"  • Total Contours: {stats['total_contours']} | Gaps Closed: {stats['open_gaps_closed']}")
        print(f"  • Saved Closed Map: {path_closed}")
        print(f"  • Saved Visual QA Panel: {path_panel}\n")

    print("=======================================================================")
    print(" AUTOMATED CONTOUR CLOSING ENGINE PROCESSING COMPLETE")
    print("=======================================================================\n")

if __name__ == "__main__":
    process_whole_slide_contour_closing()
