import os
import cv2
import numpy as np
import argparse

def extract_cells(image_path, output_dir, padding=10, min_area=100):
    """
    Extract cells surrounded by cyan outlines from an input image.

    For each cell:
    1. Extract the cell and all internal pixels onto a white canvas.
    2. Save a copy of the original JPG with a bounding box (square) drawn over the cell.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    img_h, img_w = img.shape[:2]

    # Convert to HSV color space for cyan detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Cyan color bounds in OpenCV HSV format (H: 0-180, S: 0-255, V: 0-255)
    lower_cyan = np.array([75, 40, 140], dtype=np.uint8)
    upper_cyan = np.array([115, 255, 255], dtype=np.uint8)
    cyan_mask = cv2.inRange(hsv, lower_cyan, upper_cyan)

    # Close small gaps in the cyan outlines to ensure continuous closed contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find external contours (outer cyan outlines)
    contours, _ = cv2.findContours(cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Sort contours top-to-bottom, left-to-right for consistent ordering
    valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    print(f"Found {len(valid_contours)} cells surrounded by cyan color.")

    overview_img = img.copy()

    for idx, contour in enumerate(valid_contours, start=1):
        x, y, w, h = cv2.boundingRect(contour)

        # Apply padding
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(img_w, x + w + padding)
        y2 = min(img_h, y + h + padding)

        # Create full-size binary mask for current cell polygon
        cell_mask_full = np.zeros((img_h, img_w), dtype=np.uint8)
        cv2.drawContours(cell_mask_full, [contour], -1, 255, cv2.FILLED)

        # Crop image and cell mask
        crop_img = img[y1:y2, x1:x2]
        crop_mask = cell_mask_full[y1:y2, x1:x2]

        # Paste cell on a white canvas
        white_canvas = np.full_like(crop_img, 255)
        mask_3ch = cv2.merge([crop_mask, crop_mask, crop_mask])
        white_canvas = np.where(mask_3ch > 0, crop_img, white_canvas)

        # Save extracted cell on white canvas
        cell_filename = f"cell_{idx:03d}_extracted.jpg"
        cell_path = os.path.join(output_dir, cell_filename)
        cv2.imwrite(cell_path, white_canvas)

        # Save copy of original JPG with a square (bounding box) over the extracted cell
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red box
        cv2.putText(marked_img, f"Cell #{idx}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        marked_filename = f"cell_{idx:03d}_marked.jpg"
        marked_path = os.path.join(output_dir, marked_filename)
        cv2.imwrite(marked_path, marked_img)

        # Draw on overview image as well
        cv2.rectangle(overview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_img, str(idx), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        print(f"Cell {idx:02d}: BBox=({x1}, {y1}, {x2}, {y2}), Saved: {cell_filename}, {marked_filename}")

    # Save overview image
    overview_path = os.path.join(output_dir, "all_cells_overview.jpg")
    cv2.imwrite(overview_path, overview_img)
    print(f"All {len(valid_contours)} cells successfully extracted to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cells surrounded by cyan outline.")
    parser.add_argument("--image", default="/Users/dpeleg/local/MicroGlia/Data/raw-data/JPG_VID2724_B1_3_00d07h00m.jpg",
                        help="Path to input JPG image")
    parser.add_argument("--output", default="/Users/dpeleg/local/MicroGlia/Data/agy-extracted-baseline",
                        help="Output directory for extracted baseline cells")
    parser.add_argument("--padding", type=int, default=10, help="Padding in pixels around bounding box")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum contour area threshold")

    args = parser.parse_args()
    extract_cells(args.image, args.output, args.padding, args.min_area)
