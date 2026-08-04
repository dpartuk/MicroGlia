# MicroGlia Cell Extraction Pipeline 🔬

Automated atomic sub-cell extraction, process leg merging, and boundary sharpening pipeline for complex microglial cell microscopy data.

---

## 📌 Project Overview

This repository contains two production cell extraction approaches designed for microglial cell microscopy data (supporting both Color Cyan `.jpg`/`.tif` images and Single-Channel Monochrome `.png` images):

1. **Baseline Pipeline** (`agy-code/process_dataset.py`):
   * Direct 1-step atomic sub-cell extraction.
   * Multi-filter containment cleaning (Non-Composite, Duplicate Subset, Homogeneity).
   * Cell process leg merger (re-attaching broken cellular processes via convex hull integration).
   * Post-processing convex hull boundary closure (0% border-cell data loss).

2. **Boundary Sharpening Pipeline** (`agy-code/process_boundary_sharpening.py`):
   * Fuses Scharr gradient magnitude with Canny hysteresis edge responses.
   * Applies Multi-Tile Adaptive CLAHE contrast normalization ($8\times8$ grid) to automatically equalize bright centers and dark corners/quadrants across all images.
   * Applies fine ($3\times3$) and mid-scale ($7\times7$) unsharp masking to generate crisp 1-2px boundary walls.
   * Dual-crop output architecture (saves BOTH original RGB crops and ultra-sharpened map crops).
   * One-click Safari/Chrome web gallery generation.

---

## 📊 Pipeline Comparison Summary

| Metric / Dimension | Baseline Pipeline (`baseline-output`) | Boundary Sharpening Pipeline (`boundary-sharpening-output`) |
| :--- | :---: | :---: |
| **Total Dataset Yield (5 Images)** | **991 Sub-Cells** | **1,036 Sub-Cells (+45 Sub-Cells)** |
| **Boundary Line Width** | Standard (3-5px) | Crisp (1-2px) |
| **Faint Cyan Wall Recovery** | Standard | High (Multi-Scale Unsharp Masking) |
| **Dark Corner / Quadrant Handling** | Standard | Multi-Tile Adaptive CLAHE Normalization |
| **Crops Saved** | Original RGB | Dual Crops (Original RGB + Sharpened Map) |

---

## 📁 Repository Structure

```
.
├── README.md                             # Project Documentation
├── microglia_two_approaches_summary.md  # Detailed Technical & Empirical Summary Report
├── agy-code/                             # Core Python Codebase
│   ├── process_dataset.py                # Baseline batch processing script
│   ├── extract_cells.py                  # Baseline core extraction module
│   ├── process_boundary_sharpening.py    # Boundary Sharpening batch processing script
│   ├── boundary_sharpening_pipeline.py   # Boundary Sharpening core extraction module
│   └── compare_baseline_vs_sharpening.py # Side-by-side comparative evaluator
└── Data/
    └── raw-data/                         # Input microscopy images (.jpg, .tif)
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.9+
* OpenCV (`opencv-python`)
* NumPy (`numpy`)

### Installation

```bash
git clone git@github.com:dpartuk/MicroGlia.git
cd MicroGlia
python3 -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy
```

### Running the Pipelines

1. **Run Production Baseline Pipeline**:
   ```bash
   python agy-code/process_dataset.py
   ```

2. **Run New Boundary Sharpening Pipeline**:
   ```bash
   python agy-code/process_boundary_sharpening.py
   ```

3. **Run Side-by-Side Comparative Evaluator**:
   ```bash
   python agy-code/compare_baseline_vs_sharpening.py
   ```

---

## 📄 Documentation

For full mathematical formulations, step-by-step technical workflows, and empirical comparison matrices, read [`microglia_two_approaches_summary.md`](microglia_two_approaches_summary.md).
