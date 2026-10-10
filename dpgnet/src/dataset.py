import os
import glob
import random
import csv
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MicrogliaPatchDataset(Dataset):
    """
    PyTorch Dataset for Microglia Boundary-Aware Segmentation in DPGNet.
    Extracts 352x352 patches from whole-slide microscopy images and ground truth masks.
    Supports cell-centered sampling, grid sampling, and comprehensive data augmentations.
    """

    def __init__(
        self,
        slide_dirs,
        patch_size=352,
        grid_stride=96,
        is_train=True,
        include_cell_centered=True,
        normalize=True,
        seed=42
    ):
        """
        Args:
            slide_dirs: List of paths to slide output folders in contour/output/
            patch_size: Target square crop size (default 352 for DPGNet)
            grid_stride: Stride for regular grid patches across whole slide
            is_train: Boolean, enables data augmentation if True
            include_cell_centered: Sample patches centered on each detected cell
            normalize: Standardize image channels with ImageNet mean and std
            seed: Random seed for reproducibility
        """
        self.patch_size = patch_size
        self.is_train = is_train
        self.normalize = normalize
        self.samples = []  # List of tuples: (slide_idx, top, left)

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Preload whole slides and masks into memory (all 5 slides take < 25 MB RAM)
        self.slides = []
        self.masks = []

        for s_idx, s_dir in enumerate(slide_dirs):
            img_path = os.path.join(s_dir, "00_original_reference.jpg")
            mask_path = os.path.join(s_dir, "whole_slide_ground_truth_mask.png")

            if not os.path.isfile(img_path) or not os.path.isfile(mask_path):
                continue

            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            self.slides.append(img)
            self.masks.append(mask)

            h, w = mask.shape

            # 1. Grid sampling across whole slide
            y_steps = list(range(0, h - patch_size + 1, grid_stride))
            if y_steps[-1] != h - patch_size:
                y_steps.append(h - patch_size)

            x_steps = list(range(0, w - patch_size + 1, grid_stride))
            if x_steps[-1] != w - patch_size:
                x_steps.append(w - patch_size)

            for y in y_steps:
                for x in x_steps:
                    patch_mask = mask[y:y + patch_size, x:x + patch_size]
                    # Keep patches that have cellular content or context
                    if np.max(patch_mask) > 0 or random.random() < 0.15:
                        self.samples.append((len(self.slides) - 1, y, x))

            # 2. Cell-centered sampling from manifest.csv
            manifest_file = os.path.join(s_dir, "manifest.csv")
            if include_cell_centered and os.path.isfile(manifest_file):
                with open(manifest_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        cx = (int(row["x1"]) + int(row["x2"])) // 2
                        cy = (int(row["y1"]) + int(row["y2"])) // 2

                        top = max(0, min(h - patch_size, cy - patch_size // 2))
                        left = max(0, min(w - patch_size, cx - patch_size // 2))
                        self.samples.append((len(self.slides) - 1, top, left))

        # Shuffle sample index deterministically
        rng = random.Random(seed)
        rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def _apply_augmentations(self, img, mask):
        """Apply data augmentations described in DPGNet paper."""
        # 1. Random Horizontal Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 1)
            mask = cv2.flip(mask, 1)

        # 2. Random Vertical Flip
        if random.random() > 0.5:
            img = cv2.flip(img, 0)
            mask = cv2.flip(mask, 0)

        # 3. Random 90-degree rotations
        k = random.randint(0, 3)
        if k > 0:
            img = np.rot90(img, k).copy()
            mask = np.rot90(mask, k).copy()

        # 4. Random Scaling (0.75x, 1.0x, 1.25x)
        if random.random() > 0.5:
            scale = random.choice([0.75, 1.25])
            h, w = img.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

            if scale > 1.0:
                # Center crop back to patch_size
                start_y = (new_h - h) // 2
                start_x = (new_w - w) // 2
                img = scaled_img[start_y:start_y + h, start_x:start_x + w]
                mask = scaled_mask[start_y:start_y + h, start_x:start_x + w]
            else:
                # Pad back to patch_size
                pad_y = (h - new_h) // 2
                pad_x = (w - new_w) // 2
                padded_img = np.full((h, w, 3), 255, dtype=img.dtype)
                padded_mask = np.zeros((h, w), dtype=mask.dtype)
                padded_img[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled_img
                padded_mask[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = scaled_mask
                img = padded_img
                mask = padded_mask

        # 5. Color Jitter (Brightness / Contrast adjustments)
        if random.random() > 0.5:
            alpha = random.uniform(0.9, 1.1)  # contrast
            beta = random.uniform(-10, 10)    # brightness
            img = np.clip(alpha * img + beta, 0, 255).astype(np.uint8)

        return img, mask

    def __getitem__(self, idx):
        slide_idx, top, left = self.samples[idx]
        img = self.slides[slide_idx][top:top + self.patch_size, left:left + self.patch_size].copy()
        mask = self.masks[slide_idx][top:top + self.patch_size, left:left + self.patch_size].copy()

        if self.is_train:
            img, mask = self._apply_augmentations(img, mask)

        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        if self.normalize:
            img_rgb = (img_rgb - self.mean) / self.std

        # Convert to PyTorch tensors
        # Image: [C, H, W]
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float()
        # Binary Mask: [1, H, W] with {0.0, 1.0}
        mask_binary = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        return img_tensor, mask_tensor


def get_dataloaders(
    data_root="contour/output",
    batch_size=4,
    val_ratio=0.2,
    patch_size=352,
    grid_stride=96,
    num_workers=0,
    seed=42
):
    """
    Factory function to create train and validation DataLoaders.

    Args:
        data_root: Directory containing slide output folders
        batch_size: Mini-batch size for DataLoader
        val_ratio: Proportion of samples reserved for validation
        patch_size: Square patch dimension
        grid_stride: Grid stride for patch extraction
        num_workers: DataLoader background workers
        seed: Random seed

    Returns:
        (train_loader, val_loader, full_dataset)
    """
    slide_dirs = sorted([
        d for d in glob.glob(os.path.join(data_root, "*"))
        if os.path.isdir(d) and os.path.isfile(os.path.join(d, "whole_slide_ground_truth_mask.png"))
    ])

    if not slide_dirs:
        raise FileNotFoundError(f"No valid slide directories found in {data_root}")

    # Build entire sample pool
    full_dataset = MicrogliaPatchDataset(
        slide_dirs=slide_dirs,
        patch_size=patch_size,
        grid_stride=grid_stride,
        is_train=False,
        seed=seed
    )

    total_samples = len(full_dataset)
    val_count = int(total_samples * val_ratio)
    train_count = total_samples - val_count

    # Deterministic split indices
    indices = list(range(total_samples))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_indices = indices[:train_count]
    val_indices = indices[train_count:]

    # Create distinct training and validation dataset instances
    train_dataset = MicrogliaPatchDataset(
        slide_dirs=slide_dirs,
        patch_size=patch_size,
        grid_stride=grid_stride,
        is_train=True,
        seed=seed
    )
    val_dataset = MicrogliaPatchDataset(
        slide_dirs=slide_dirs,
        patch_size=patch_size,
        grid_stride=grid_stride,
        is_train=False,
        seed=seed
    )

    train_sub = torch.utils.data.Subset(train_dataset, train_indices)
    val_sub = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_sub,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_sub,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, full_dataset


if __name__ == "__main__":
    train_loader, val_loader, dataset = get_dataloaders(batch_size=4)
    print(f"Total patch samples extracted: {len(dataset)}")
    print(f"Training batches: {len(train_loader)} ({len(train_loader.dataset)} samples)")
    print(f"Validation batches: {len(val_loader)} ({len(val_loader.dataset)} samples)")

    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}, dtype: {images.dtype}")
    print(f"Batch mask shape:  {masks.shape}, dtype: {masks.dtype}")
    print(f"Mask values: {torch.unique(masks)}")
