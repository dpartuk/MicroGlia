# MicroGlia Cell Segmentation & Extraction Framework 🔬

Automated atomic sub-cell extraction, boundary sharpening, and deep learning segmentation framework for complex microglial cell microscopy data.

---

## 📌 Framework Alternatives

This repository implements four comparative pipelines for microglial cell segmentation across gigapixel microscopy slides:

1. **[`baseline/`](baseline/) (Presaizen 2026 Baseline)**:
   * Color-space differential signal isolation (`(B+G)/2 - R`).
   * Gaussian spatial smoothing and global Otsu thresholding.
   * Multi-filter containment cleaning (Non-Composite, Duplicate Subset, Homogeneity).
   * Convex hull boundary closure.

2. **[`sharpened/`](sharpened/) (Boundary Sharpening Pipeline)**:
   * First-order Scharr spatial gradients fused with Canny hysteresis edge tracing.
   * Multi-Tile Adaptive CLAHE contrast normalization ($8\times8$ grid) across dark and bright tissue regions.
   * Fine ($3\times3$) and mid-scale ($7\times7$) unsharp masking producing crisp 1-2px boundary walls.
   * Dual-crop extraction (original RGB crops and sharpened boundary crops).

3. **[`contour/`](contour/) (Contour Filtering & Convex Hull Target)**:
   * Builds upon the sharpened boundary map.
   * Applies connected-component contour filtering and convex hull isolation.
   * Generates validated whole-slide binary ground-truth masks.

4. **[`dpgnet/`](dpgnet/) (DPGNet Boundary-Aware Deep Learning Framework)**:
   * Pyramid Vision Transformer v2 (PVTv2-B2) backbone with 25.51M parameters.
   * Transformer Multi-Scale Fusion (TMF) and Down-Up Communication (DUC) with 5 deeply supervised heads.
   * Edge Difference Attention (EDA) module for boundary uncertainty perception.
   * Dense sliding-window inference with 2D Hanning window blend weighting.

---

## 📊 Quantitative Comparison Across All 5 Slides

### 1. Global Segmentation Accuracy & Cell Metrics

| Alternative | Paradigm | Parameters | Identified Cells | Mean Cell Area | Total Foreground | Dice vs GT | IoU vs GT | Recall |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **[`baseline/`](baseline/)** | Color Thresholding | 0 | 991 | 755 px² | 748,485 px² | 0.1574 | 0.0867 | 0.1247 |
| **[`sharpened/`](sharpened/)** | Multi-Scale Edge Fusion | 0 | 1,036 | 698 px² | 723,320 px² | 0.2677 | 0.1547 | 0.2118 |
| **[`contour/`](contour/)** | Morphological Hull Target | 0 | 1,071 | 656 px² | 703,481 px² | 1.0000 | 1.0000 | 1.0000 |
| **[`dpgnet/`](dpgnet/)** | Vision Transformer (PVTv2) | 25.51 M | **670** | **1,403 px²** | **939,778 px²** | **0.7785** | **0.6380** | **0.8609** |

### 2. Identified Cells by Slide

| Microscopy Slide ID | Baseline | Sharpened | Contour | DPGNet | DPGNet vs Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `JPG_VID2724_B1_3_00d07h00m` | 97 | 95 | 95 | **77** | $-20$ ($-20.6\%$) |
| `VID2724_A1_9_00d07h00m` | 241 | 261 | 270 | **148** | $-93$ ($-38.6\%$) |
| `VID2724_A3_4_00d00h00m` | 269 | 287 | 304 | **183** | $-86$ ($-32.0\%$) |
| `VID2724_A3_4_00d07h00m` | 282 | 290 | 297 | **203** | $-79$ ($-28.0\%$) |
| `VID2724_B1_3_00d00h00m` | 102 | 103 | 105 | **59** | $-43$ ($-42.2\%$) |
| **TOTAL IDENTIFIED CELLS** | **991** | **1,036** | **1,071** | **670** | **$-321$ ($-32.4\%$)** |

---

## 🔬 Biological Insights

* **Over-Segmentation in Classical Pipelines**:
  Classical pipelines evaluate pixels locally without semantic context. Delicate podia branches break apart into disjoint fragments during morphological thresholding. This fragmentation causes high cell counts (991 to 1,071) and small mean cell areas (~656 to 755 px²).

* **Branch Unification in DPGNet**:
  DPGNet uses global receptive fields from its vision transformer backbone. The network connects ramified arborizations back to the central cell body. DPGNet produces 670 intact cellular units with an 85.8% larger mean area (1,403 px²), while increasing total foreground coverage by 25.5% (939,778 px²).

---

## 📁 Repository Structure

```
.
├── README.md                      # Complete project documentation & comparative analysis
├── baseline/                      # Baseline classical pipeline (Presaizen 2026)
│   ├── process.py                 # Batch extraction launcher
│   ├── gallery.py                 # Gallery builder
│   └── src/                       # Extraction algorithms
├── sharpened/                     # Boundary sharpening pipeline
│   ├── process.py                 # Batch extraction launcher
│   ├── gallery.py                 # Gallery builder
│   └── src/                       # Gradient fusion & unsharp masking modules
├── contour/                       # Contour filtering & convex hull isolation
│   ├── process.py                 # Batch extraction launcher
│   ├── gallery.py                 # Gallery builder
│   └── src/                       # Connected component & hull pipeline
├── dpgnet/                        # DPGNet deep learning framework
│   ├── train.py                   # PyTorch training launcher CLI
│   ├── process.py                 # Dense sliding-window inference CLI
│   ├── gallery.py                 # Interactive review gallery builder CLI
│   └── src/                       # PVTv2 backbone, TMF, DUC, EDA, and loss modules
├── comparison/                    # Comparative evaluator and side-by-side gallery
│   ├── run.py                     # Generate comparative metrics
│   └── gallery.py                 # Comparison gallery builder
└── Data/
    └── raw-data/                  # Raw microscopy slide inputs (.jpg, .tif)
```

---

## 🚀 Quick Start Commands

### 1. Run DPGNet Pipeline
```bash
# Train DPGNet (10 epochs on Apple Silicon GPU)
.venv/bin/python3 dpgnet/train.py --epochs 10 --batch-size 8 --lr 1e-4

# Execute batch sliding-window inference across all 5 slides
.venv/bin/python3 dpgnet/process.py

# Rebuild interactive web review gallery
.venv/bin/python3 dpgnet/gallery.py
```

### 2. Open the Interactive Web Galleries
* **DPGNet Review Gallery**: [`dpgnet/output/gallery.html`](dpgnet/output/gallery.html)
* **Comparative Gallery**: [`comparison/output/gallery.html`](comparison/output/gallery.html)
* **Contour Pipeline Gallery**: [`contour/output/gallery.html`](contour/output/gallery.html)
* **Sharpened Pipeline Gallery**: [`sharpened/output/gallery.html`](sharpened/output/gallery.html)
* **Baseline Pipeline Gallery**: [`baseline/output/gallery.html`](baseline/output/gallery.html)
