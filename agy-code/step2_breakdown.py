import os
import argparse
from step2_breakdown_all import process_single_cell_step2, get_step1_cell_bboxes, cv2

def process_step2_single_cell(
    cell_extracted_path,
    original_image_path,
    output_dir
):
    cell_basename = os.path.splitext(os.path.basename(cell_extracted_path))[0]
    cell_name = cell_basename.replace("_extracted", "")

    cell_bboxes = get_step1_cell_bboxes(original_image_path) if os.path.exists(original_image_path) else {}
    global_bbox = cell_bboxes.get(cell_name, None)
    orig_img = cv2.imread(original_image_path) if os.path.exists(original_image_path) else None

    count = process_single_cell_step2(cell_extracted_path, orig_img, global_bbox, output_dir)
    print(f"[{cell_name}] Extracted {count} atomic sub-cell(s) into {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merged Step 2: Breakdown a single cell image into atomic sub-cells")
    parser.add_argument("--input", default="/Users/dpeleg/local/MicroGlia/Data/agy-extracted/cell_004_extracted.jpg",
                        help="Path to Step 1 extracted cell image")
    parser.add_argument("--original", default="/Users/dpeleg/local/MicroGlia/Data/raw-data/JPG_VID2724_B1_3_00d07h00m.jpg",
                        help="Path to original full JPG image")
    parser.add_argument("--output", default="/Users/dpeleg/local/MicroGlia/Data/step-2/cell_004",
                        help="Output directory for breakdown")

    args = parser.parse_args()
    process_step2_single_cell(args.input, args.original, args.output)
