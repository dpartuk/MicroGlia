import os
import glob
import csv
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import DPGNet


def create_blend_window(size=352):
    """
    Generate 2D Hanning window to smoothly blend overlapping sliding window tiles.
    Suppresses edge discontinuities and stitching seams.
    """
    hann_1d = np.hanning(size)
    window_2d = np.outer(hann_1d, hann_1d).astype(np.float32)
    # Clamp minimum value to prevent divide-by-zero
    window_2d = np.maximum(window_2d, 1e-4)
    return window_2d


def run_sliding_window_inference(
    model,
    img_bgr,
    device,
    patch_size=352,
    stride=160,
    batch_size=8,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    """
    Execute dense overlapping sliding-window inference on a whole-slide microscopy image.

    Args:
        model: Trained DPGNet neural network
        img_bgr: Whole-slide BGR image array [H, W, 3]
        device: PyTorch device ('mps', 'cuda', 'cpu')
        patch_size: Square patch size (352)
        stride: Step size between sliding windows (160)
        batch_size: Mini-batch size for patch inference

    Returns:
        (prob_map, uncertainty_map) as 2D float32 numpy arrays [H, W] in [0, 1]
    """
    h, w = img_bgr.shape[:2]
    window = create_blend_window(patch_size)

    # Accumulator buffers
    prob_accum = np.zeros((h, w), dtype=np.float32)
    unc_accum = np.zeros((h, w), dtype=np.float32)
    weight_accum = np.zeros((h, w), dtype=np.float32)

    # Convert image to RGB normalized float32 tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    norm_img = (img_rgb - mean_arr) / std_arr

    # Compute sliding window coordinates
    y_steps = list(range(0, h - patch_size + 1, stride))
    if y_steps[-1] != h - patch_size:
        y_steps.append(h - patch_size)

    x_steps = list(range(0, w - patch_size + 1, stride))
    if x_steps[-1] != w - patch_size:
        x_steps.append(w - patch_size)

    patches = []
    coords = []

    for y in y_steps:
        for x in x_steps:
            crop = norm_img[y:y + patch_size, x:x + patch_size]
            patches.append(crop)
            coords.append((y, x))

    # Process in batches
    model.eval()
    with torch.no_grad():
        for b_start in range(0, len(patches), batch_size):
            b_end = min(b_start + batch_size, len(patches))
            b_patches = patches[b_start:b_end]
            b_coords = coords[b_start:b_end]

            # Batch tensor: [B, C, H, W]
            batch_tensor = (
                torch.from_numpy(np.stack(b_patches))
                .permute(0, 3, 1, 2)
                .float()
                .to(device)
            )

            preds, uncertainty = model(batch_tensor, return_uncertainty=True)

            # Ensemble prediction probabilities across 5 heads
            # P = 1/5 sum(sigmoid(P_i))
            probs_sum = torch.zeros_like(preds[0])
            for p in preds:
                probs_sum += torch.sigmoid(p)
            avg_prob = (probs_sum / len(preds)).squeeze(1).cpu().numpy()

            # Uncertainty map from EDA module
            unc_map = uncertainty.squeeze(1).cpu().numpy()

            # Accumulate with blend window
            for i, (y, x) in enumerate(b_coords):
                p_patch = avg_prob[i]
                u_patch = unc_map[i]

                prob_accum[y:y + patch_size, x:x + patch_size] += p_patch * window
                unc_accum[y:y + patch_size, x:x + patch_size] += u_patch * window
                weight_accum[y:y + patch_size, x:x + patch_size] += window

    # Normalize by accumulated blend weights
    final_prob = prob_accum / np.maximum(weight_accum, 1e-5)
    final_unc = unc_accum / np.maximum(weight_accum, 1e-5)

    # Scale uncertainty map to [0, 1] range
    unc_min, unc_max = final_unc.min(), final_unc.max()
    if unc_max > unc_min:
        final_unc = (final_unc - unc_min) / (unc_max - unc_min)
    else:
        final_unc = np.zeros_like(final_unc)

    return final_prob, final_unc


def process_slide_inference(
    model,
    img_path,
    output_dir,
    device,
    prob_threshold=0.5,
    min_cell_area=60,
    subcell_padding=18,
):
    """
    Execute full inference on one raw microscopy slide:
    1. Runs sliding-window inference with blend weights.
    2. Writes whole-slide segmentation and uncertainty maps.
    3. Extracts individual single-cell crops and builds manifest.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img_bgr.shape[:2]

    # 1. Run sliding window inference
    prob_map, unc_map = run_sliding_window_inference(model, img_bgr, device)

    # 2. Binary segmentation mask
    bin_mask = (prob_map >= prob_threshold).astype(np.uint8) * 255

    # 3. Whole-slide boundary uncertainty heatmap (ColorMap Turbo)
    unc_uint8 = (unc_map * 255).astype(np.uint8)
    unc_heatmap = cv2.applyColorMap(unc_uint8, cv2.COLORMAP_TURBO)

    # 4. Composite Overlay: original image with colored cell contours and uncertainty
    overlay = img_bgr.copy()
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw cell outlines in bright green
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

    # Save whole-slide output images
    cv2.imwrite(os.path.join(output_dir, "00_original_reference.jpg"), img_bgr)
    cv2.imwrite(os.path.join(output_dir, "dpgnet_binary_segmentation.png"), bin_mask)
    cv2.imwrite(os.path.join(output_dir, "dpgnet_uncertainty_heatmap.jpg"), unc_heatmap)
    cv2.imwrite(os.path.join(output_dir, "dpgnet_prediction_overlay.jpg"), overlay)

    # 5. Extract individual atomic sub-cells
    manifest_rows = []
    cell_idx = 0

    # Filter contours by minimum area
    valid_contours = [c for c in contours if cv2.contourArea(c) >= min_cell_area]
    # Sort left-to-right, top-to-bottom
    valid_contours.sort(key=lambda c: (cv2.boundingRect(c)[1], cv2.boundingRect(c)[0]))

    for c in valid_contours:
        cell_idx += 1
        x, y, cw, ch = cv2.boundingRect(c)
        area = cv2.contourArea(c)

        # Expanded bounding box
        x1 = max(0, x - subcell_padding)
        y1 = max(0, y - subcell_padding)
        x2 = min(w, x + cw + subcell_padding)
        y2 = min(h, y + ch + subcell_padding)

        # Single cell mask
        single_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(single_mask, [c], -1, 255, cv2.FILLED)

        # A. Original RGB crop with white background outside cell mask
        crop_rgb = img_bgr[y1:y2, x1:x2]
        crop_m = single_mask[y1:y2, x1:x2]
        white_canvas = np.full_like(crop_rgb, 255)
        mask_3ch = cv2.merge([crop_m, crop_m, crop_m])
        extracted_orig = np.where(mask_3ch > 0, crop_rgb, white_canvas)
        orig_filename = f"cell_{cell_idx:03d}_original_extracted.jpg"
        cv2.imwrite(os.path.join(output_dir, orig_filename), extracted_orig)

        # B. DPGNet Predicted Mask Crop
        crop_bin = bin_mask[y1:y2, x1:x2]
        dpgnet_mask_filename = f"cell_{cell_idx:03d}_dpgnet_mask.png"
        cv2.imwrite(os.path.join(output_dir, dpgnet_mask_filename), crop_bin)

        # C. Uncertainty Heatmap Crop
        crop_unc = unc_heatmap[y1:y2, x1:x2]
        uncertainty_filename = f"cell_{cell_idx:03d}_uncertainty.jpg"
        cv2.imwrite(os.path.join(output_dir, uncertainty_filename), crop_unc)

        # Mean uncertainty value inside cell boundary
        mean_cell_unc = float(np.mean(unc_map[y1:y2, x1:x2][crop_m > 0])) if np.any(crop_m > 0) else 0.0

        # D. Marked Context Image
        marked_img = img_bgr.copy()
        cv2.rectangle(marked_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            marked_img,
            f"Cell #{cell_idx:03d}",
            (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
        )
        location_filename = f"cell_{cell_idx:03d}_marked.jpg"
        cv2.imwrite(os.path.join(output_dir, location_filename), marked_img)

        manifest_rows.append({
            "cell_id": cell_idx,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width": x2 - x1,
            "height": y2 - y1,
            "area": int(area),
            "mean_uncertainty": round(mean_cell_unc, 4),
            "original_filename": orig_filename,
            "dpgnet_mask_filename": dpgnet_mask_filename,
            "uncertainty_filename": uncertainty_filename,
            "location_filename": location_filename,
        })

    # Save manifest.csv
    manifest_path = os.path.join(output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        fieldnames = [
            "cell_id",
            "x1",
            "y1",
            "x2",
            "y2",
            "width",
            "height",
            "area",
            "mean_uncertainty",
            "original_filename",
            "dpgnet_mask_filename",
            "uncertainty_filename",
            "location_filename",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    return len(manifest_rows)


def run_batch_inference(
    checkpoint_path="dpgnet/output/checkpoints/best_model.pth",
    data_root="Data/raw-data",
    output_root="dpgnet/output",
    device_name=None,
):
    """
    Run DPGNet inference across all raw microscopy slides in Data/raw-data.
    """
    if device_name is not None:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("===========================================================")
    print(" DPGNET BATCH INFERENCE PIPELINE")
    print(f" Checkpoint: {checkpoint_path}")
    print(f" Raw Data Directory: {data_root}")
    print(f" Output Root: {output_root}")
    print(f" Device: {device}")
    print("===========================================================")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load model
    model = DPGNet(pth_path=None).to(device)
    chk = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(chk["model_state"])
    model.eval()

    # Discover raw slide images (.jpg, .tif, .tiff)
    supported_exts = ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png")
    image_paths = []
    for ext in supported_exts:
        image_paths.extend(glob.glob(os.path.join(data_root, ext)))
    image_paths = sorted(image_paths)

    total_cells = 0
    results = []

    for idx, img_path in enumerate(image_paths, start=1):
        filename = os.path.basename(img_path)
        stem = os.path.splitext(filename)[0]
        slide_out_dir = os.path.join(output_root, stem)

        print(f"\n[{idx}/{len(image_paths)}] Processing: {filename} ...")
        n_cells = process_slide_inference(
            model=model,
            img_path=img_path,
            output_dir=slide_out_dir,
            device=device,
        )
        total_cells += n_cells
        results.append((filename, n_cells))
        print(f"  -> Extracted {n_cells} segmented cells into {slide_out_dir}")

    print("\n===========================================================")
    print(" INFERENCE COMPLETE SUMMARY")
    print("===========================================================")
    for fname, count in results:
        print(f"  {fname:<35} | {count:>5} cells")
    print("-----------------------------------------------------------")
    print(f"  GRAND TOTAL                         | {total_cells:>5} cells")
    print("===========================================================")


if __name__ == "__main__":
    run_batch_inference()
