# DPGNet: Boundary-Aware Medical Image Segmentation Framework

This package implements **DPGNet** (*IEEE Journal of Biomedical and Health Informatics*, 2025/2026) for microglial boundary-aware cell segmentation and uncertainty perception.

---

## 1. Architecture Overview

DPGNet integrates edge difference attention and multi-scale feature communication to segment delicate cellular structures without requiring explicit boundary annotations:

```
Input Image (3 x 352 x 352)
     │
     ▼
PVTv2-b2 Backbone (Multi-Scale Features X1, X2, X3, X4)
     │
     ▼
Neck (1x1 Conv -> 64 Channels)
     │
     ▼
Stage 1: Transformer Multi-Scale Fusion (TMF) -> Coarse Localization Map P1
     │
     ▼
Stage 2: Down-Up Communication (DUC 1..4) -> Multi-Scale Predictions P2..P5
     │
     ▲ (Iterative Feedback Loop)
     │
Stage 3: Edge Difference Attention (EDA 1..3) -> Boundary Uncertainty Perception Map
```

* **Backbone**: Pyramid Vision Transformer v2 (PVTv2-B2) with overlapping patch embedding and linear spatial reduction attention.
* **Model Parameters**: 25.51 Million parameters.
* **Loss Function**: Weighted Structural Loss ($\mathcal{L}_{\text{BCE}}^\omega + \mathcal{L}_{\text{IoU}}^\omega$) evaluated across 5 deeply supervised prediction heads.
* **Uncertainty Perception**: Edge Difference Attention (EDA) explicitly measures discrepancies between adjacent feature scales to identify boundary ambiguity.

---

## 2. Directory Layout

```
dpgnet/
├── README.md                      # Complete framework documentation
├── requirements.txt               # Dependencies (torch, timm, albumentations)
├── train.py                       # Training launcher CLI
├── process.py                     # Batch whole-slide inference CLI
├── gallery.py                     # Review gallery builder CLI
│
├── src/
│   ├── model/
│   │   ├── pvtv2.py               # Pyramid Vision Transformer v2 backbone
│   │   ├── dpgnet.py              # DPGNet network (TMF, DUC, EDA modules)
│   │   └── loss.py                # Weighted BCE + Weighted IoU structural loss
│   ├── dataset.py                 # Overlapping patch dataset loader and augmentations
│   ├── train.py                   # PyTorch training loop with cosine schedule
│   ├── infer.py                   # Dense sliding-window whole-slide inference
│   └── build_gallery.py           # Interactive HTML review gallery generator
│
└── output/                        # Checkpoints, predictions, and web gallery
    ├── checkpoints/
    │   └── best_model.pth         # Saved weights (Validation Dice: 0.7802)
    ├── gallery.html               # Interactive browser review gallery
    └── <dataset_name>/            # Output directory per slide:
        ├── manifest.csv           # Cellular morphology and uncertainty catalog
        ├── 00_original_reference.jpg
        ├── dpgnet_binary_segmentation.png
        ├── dpgnet_uncertainty_heatmap.jpg
        ├── dpgnet_prediction_overlay.jpg
        ├── cell_XXX_original_extracted.jpg
        ├── cell_XXX_dpgnet_mask.png
        ├── cell_XXX_uncertainty.jpg
        └── cell_XXX_marked.jpg
```

---

## 3. Workflow & CLI Commands

### Step 1: Model Training
Train DPGNet on extracted microglia image and mask pairs:
```bash
.venv/bin/python3 dpgnet/train.py --epochs 10 --batch-size 8 --lr 1e-4 --device mps
```

Key Training Hyperparameters:
* **Optimizer**: AdamW (`lr=1e-4`, weight decay = `1e-4`).
* **Learning Rate Schedule**: Cosine annealing scheduler.
* **Patch Size**: $352 \times 352$ pixels.
* **Supervision**: 5-level deep supervision sum ($\sum_{i=1}^5 \mathcal{L}_{\text{STR}}(P_i, G)$).

### Step 2: Batch Whole-Slide Inference
Run dense sliding-window inference across all raw gigapixel microscopy slides:
```bash
.venv/bin/python3 dpgnet/process.py
```

Sliding-Window Inference Highlights:
* Dense 2D Hanning blend window weighting suppresses tile boundary discontinuities and stitching artifacts.
* Sigmoid ensemble across all 5 prediction heads generates robust whole-slide probability maps.
* Extracts single-cell bounding boxes, binary masks, and boundary uncertainty heatmaps into `manifest.csv`.

### Step 3: Interactive Review Gallery
Compile extracted cellular units and whole-slide representations into an interactive HTML gallery:
```bash
.venv/bin/python3 dpgnet/gallery.py
```
Open `dpgnet/output/gallery.html` directly in any modern web browser.

---

## 4. Empirical Performance & Results

### Quantitative Training Metrics (10 Epochs)

| Metric | Epoch 1 | Epoch 5 | Epoch 10 (Best) |
| :--- | :---: | :---: | :---: |
| **Train Loss** | 0.9996 | 0.4485 | **0.3929** |
| **Validation Loss** | 0.8143 | 0.5401 | **0.4913** |
| **Validation Dice** | 0.6974 | 0.7681 | **0.7802** |
| **Validation IoU** | 0.5416 | 0.6273 | **0.6420** |

### Inference Benchmark across Whole Slides

| Slide Dataset | Extracted Cells | Mean Cell Area | Mean Uncertainty |
| :--- | :---: | :---: | :---: |
| `JPG_VID2724_B1_3_00d07h00m` | 77 | 1,420 px² | 0.5369 |
| `VID2724_A1_9_00d07h00m` | 148 | 1,732 px² | 0.5284 |
| `VID2724_A3_4_00d00h00m` | 183 | 1,498 px² | 0.5391 |
| `VID2724_A3_4_00d07h00m` | 203 | 1,514 px² | 0.5358 |
| `VID2724_B1_3_00d00h00m` | 59 | 1,605 px² | 0.5310 |
| **Total / Global Average** | **670 Cells** | **1,554 px²** | **0.5342** |

* **Hardware**: Apple Silicon GPU (`mps`).
* **Inference Speed**: ~15 seconds total across all 5 gigapixel slides (~3 seconds per slide).

---

## 5. Review Gallery Features

The standalone HTML gallery (`dpgnet/output/gallery.html`) provides comprehensive diagnostic controls:

1. **Cell Inspection Mode**:
   * **Original RGB**: Clean isolated microglial unit on white canvas.
   * **DPGNet Binary Mask**: Predicted cellular silhouette capturing fine podia arborizations.
   * **Boundary Uncertainty Heatmap**: Color-coded visualization of edge transition ambiguity.
   * **Slide Location Context**: Whole-slide map with bounding box localization.
2. **Whole-Slide Overview Mode**:
   * Toggle between Raw Slide, DPGNet Mask, Uncertainty Heatmap, Contour Overlay, and Ground Truth.
   * 2x2 multi-representation grid comparison view.
3. **Filtering & Sorting**:
   * Sort by Cell ID, Area (Ascending / Descending), or Uncertainty (Highest / Lowest).
   * Filter cells by uncertainty risk tiers (High $\ge 0.58$, Medium $0.50 - 0.58$, Low $< 0.50$).
4. **Keyboard Shortcuts**:
   * `←` / `A` / `[` : Previous cell
   * `→` / `D` / `]` : Next cell
   * `Home` / `End` : First / Last cell
   * `M` : Toggle View Mode (Cell Inspection vs Whole Slide)
   * `Escape` : Close high-resolution zoom modal
