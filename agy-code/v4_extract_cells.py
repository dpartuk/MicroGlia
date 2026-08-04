import os
import cv2
import numpy as np
import argparse

def extract_cells(image_path, output_dir, padding=10, min_area=100):
    """
    Extract cells surrounded by cyan outlines from an input image (.jpg or .tif).

    For each cell:
    1. Extract the cell and all internal pixels onto a white canvas.
    2. Save a copy of the original image with a bounding box drawn over the cell.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Read image (.jpg or .tif)
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Failed to read image at {image_path}")

    img_h, img_w = img.shape[:2]

    # Cyan Difference Contrast Map: (B + G) / 2 - R
    b, g, r = cv2.split(img.astype(np.float32))
    cyan_diff = (b + g) / 2.0 - r
    cyan_diff = np.clip(cyan_diff, 0, 255).astype(np.uint8)

    # Unsharp Masking to sharpen faint cyan edges
    blur = cv2.GaussianBlur(cyan_diff, (5, 5), 1.0)
    cyan_sharpened = cv2.addWeighted(cyan_diff, 1.8, blur, -0.8, 0)
    _, cyan_mask = cv2.threshold(cyan_sharpened, 35, 255, cv2.THRESH_BINARY)

    # Morphological closing to ensure continuous closed outer cell boundaries
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cyan_closed = cv2.morphologyEx(cyan_mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Find external contours (outer cyan outlines)
    contours, _ = cv2.findContours(cyan_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_area]

    # Sort contours top-to-bottom, left-to-right for consistent ordering
    valid_contours = sorted(valid_contours, key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    print(f"[{os.path.basename(image_path)}] Found {len(valid_contours)} outer cells surrounded by cyan outlines.")

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

        # Save copy of original image with red bounding box
        marked_img = img.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(marked_img, f"Cell #{idx}", (x1, max(15, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        marked_filename = f"cell_{idx:03d}_marked.jpg"
        marked_path = os.path.join(output_dir, marked_filename)
        cv2.imwrite(marked_path, marked_img)

        # Draw on overview image as well
        cv2.rectangle(overview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overview_img, str(idx), (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save overview image
    overview_path = os.path.join(output_dir, "all_cells_overview.jpg")
    cv2.imwrite(overview_path, overview_img)
    return len(valid_contours)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract cells surrounded by cyan outline from .jpg or .tif.")
    parser.add_argument("--image", default="/Users/dpeleg/local/MicroGlia/Data/raw-data/JPG_VID2724_B1_3_00d07h00m.jpg",
                        help="Path to input image (.jpg or .tif)")
    parser.add_argument("--output", default="/Users/dpeleg/local/MicroGlia/Data/agy-extracted",
                        help="Output directory for extracted cells")
    parser.add_argument("--padding", type=int, default=10, help="Padding in pixels around bounding box")
    parser.add_argument("--min-area", type=int, default=100, help="Minimum contour area threshold")

    args = parser.parse_args()
    extract_cells(args.image, args.output, args.padding, args.min_area)
