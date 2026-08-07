import os
import glob
import cv2
import numpy as np

def generate_whole_slide_cyan_border_map(
    image_path_or_mat,
    lower_cyan=(75, 40, 140),
    upper_cyan=(115, 255, 255),
    invert=False
):
    """
    Pre-Processing Step on Original Image:
    Converts the entire original image into HSV color space and isolates cyan pixels.
    - If invert=False: Cyan pixels = 255 (WHITE), All other pixels = 0 (BLACK).
    - If invert=True: Cyan pixels = 0 (BLACK), All other pixels = 255 (WHITE).
    Goal: Identify and isolate pure cellular border rings across the full original image.
    """
    if isinstance(image_path_or_mat, str):
        img_bgr = cv2.imread(image_path_or_mat)
        if img_bgr is None:
            raise ValueError(f"Could not load image: {image_path_or_mat}")
    else:
        img_bgr = image_path_or_mat

    # Convert to HSV color space
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # Threshold for cyan color range
    lower = np.array(lower_cyan, dtype=np.uint8)
    upper = np.array(upper_cyan, dtype=np.uint8)
    binary_cyan_border = cv2.inRange(hsv, lower, upper)

    # Apply light morphological ellipse closing to bridge tiny line gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_border_map = cv2.morphologyEx(binary_cyan_border, cv2.MORPH_CLOSE, kernel)

    if invert:
        cleaned_border_map = cv2.bitwise_not(cleaned_border_map)

    return cleaned_border_map

def process_raw_dataset_cyan_border_maps(
    raw_dir="/Users/dpeleg/local/MicroGlia/Data/raw-data",
    output_dir="/Users/dpeleg/local/MicroGlia/Data/cyan-border-maps"
):
    """
    Processes all raw microscopy images in raw-data and outputs whole-slide binary border maps.
    """
    os.makedirs(output_dir, exist_ok=True)
    supported_exts = (".jpg", ".jpeg", ".tif", ".tiff", ".png")
    image_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(supported_exts)
    ])

    print(f"\n=======================================================================")
    print(f" GENERATING WHOLE-SLIDE BINARY CYAN BORDER MAPS ({len(image_files)} IMAGES)")
    print(f" Output Directory: {output_dir}")
    print(f"=======================================================================\n")

    for idx, img_name in enumerate(image_files, start=1):
        img_path = os.path.join(raw_dir, img_name)
        stem, ext = os.path.splitext(img_name)

        img_bgr = cv2.imread(img_path)
        if img_bgr is None: continue

        # Generate White Borders (Cyan=255, Other=0)
        border_white = generate_whole_slide_cyan_border_map(img_bgr, invert=False)
        # Generate Inverted Black Borders (Cyan=0, Other=255)
        border_black = generate_whole_slide_cyan_border_map(img_bgr, invert=True)

        path_white = os.path.join(output_dir, f"{stem}_cyan_borders_white.jpg")
        path_black = os.path.join(output_dir, f"{stem}_cyan_borders_black_inv.jpg")

        cv2.imwrite(path_white, border_white)
        cv2.imwrite(path_black, border_black)

        print(f"[{idx}/{len(image_files)}] Generated Border Map for {img_name} -> {stem}_cyan_borders_white.jpg")

    print("\n=======================================================================")
    print(" WHOLE-SLIDE BINARY CYAN BORDER MAP GENERATION COMPLETE")
    print("=======================================================================\n")

if __name__ == "__main__":
    process_raw_dataset_cyan_border_maps()
