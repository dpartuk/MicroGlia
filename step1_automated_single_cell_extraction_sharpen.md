# STEP 1 DESIGN SPECIFICATION: AUTOMATED SINGLE-CELL EXTRACTION & BOUNDARY SHARPENING (THE SHARPEN ALTERNATIVE)

**Project**: Topological AI Pipeline for Microglial Morphological Classification and Activation Scoring  
**Module**: Step 1 - Automated Single-Cell Extraction & Boundary Sharpening  
**Author**: Doron Peleg  
**Supervisor**: Dr. Hadas Lapid | **Advisors**: Dr. Lilach Gavish (PhD, MPH), Reut Zinger  
**Institution**: Afeka Academic College of Engineering / Hebrew University of Jerusalem  

---

## 1. Overview & System Motivation

Traditional microglial detection frameworks (e.g., Presaizen, 2026 baseline) rely on bounding-box object detectors (YOLOv11) that crop $64\times64$ or $128\times128$ square RGB regions around the central soma. Bounding-box detectors suffer from three major biological limitations:
1. **Soma-Centric Restriction**: Up to 70% of distal process arborization is cropped out or obscured by parenchymal background.
2. **Resting vs. Resolution Ambiguity**: Quiescent / Resting and Senescent / Resolution cells share similar soma sizes, causing severe misclassification when analyzed by soma bounding boxes alone.
3. **Parenchymal Noise Inclusion**: Raw bounding-box crops contain non-relevant background tissue, red blood cells, and partial fragments from neighboring cells.

To solve these limitations, **Step 1 (The Sharpen Alternative)** extracts microglial cells directly from laboratory **cyan contours** (`extract_cells.py`) and processes each crop through our **Boundary Sharpening Pipeline** (`boundary_sharpening_pipeline.py`). This produces clean, high-contrast **binary silhouette masks** ($1 = \text{cell body/arbor}$, $0 = \text{background}$) alongside raw RGB crops.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                      STEP 1: AUTOMATED SINGLE-CELL EXTRACTION & BOUNDARY SHARPENING                   │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ SUB-STEP 1.1.1: Multi-Tile CLAHE & Edge Gradient Fusion                                             │
 │ • Enhances local contrast across 8x8 tiles (clipLimit=3.0)                                          │
 │ • Computes Scharr & Canny edge gradient magnitude                                                   │
 │ • Blends intensity & edges into Fused Composite Output (α=0.7)                                      │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ SUB-STEP 1.1.2: Cyan Contour Isolation & Sub-Cell IoU Deduplication                                │
 │ • Stage 1: HSV Color Segmentation (H ∈ [85, 105]) locates cyan contour rings                        │
 │ • Stage 2: Sub-Cell IoMin Filtering (IoMin > 0.50) discards duplicate nested sub-cell boxes         │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ SUB-STEP 1.1.3: Boundary Sharpening & Binary Silhouette Mask Extraction (`boundary_sharpening.py`)   │
 │ • Single-crop CLAHE & Scharr high-pass gradient filtering                                           │
 │ • Adaptive Otsu thresholding & morphological closing (3x3 ellipse kernel)                           │
 │ • Connected component analysis strips unattached background debris                                  │
 │ • Outputs high-contrast binary silhouette mask [128x128]                                            │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Specification of Sub-Steps

---

### Sub-step 1.1.1: Multi-Tile CLAHE & Edge Gradient Fusion

#### **1. Purpose**
Raw bright-field microscopy slides suffer from non-uniform illumination (lamp vignetting, shadows) and low optical contrast between thin 1–2 pixel process branches and the parenchymal background. Multi-Tile CLAHE balances illumination across an $8\times8$ grid while Scharr and Canny directional gradient operators sharpen fine micro-arbors.

#### **2. Mathematical Formulation**
* **Local Histogram Clipping**: For local tile histogram $h(i)$ and clip limit $\beta = 3.0$:
  $$\tilde{h}(i) = \min(h(i), \beta) + \frac{1}{L} \sum_{k=0}^{L-1} \max(0, h(k) - \beta)$$
* **Scharr First-Order Directional Gradient**:
  $$K_x = \begin{bmatrix} -3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{bmatrix}, \quad K_y = \begin{bmatrix} -3 & -10 & -3 \\ 0 & 0 & 0 \\ 3 & 10 & 3 \end{bmatrix}$$
  $$G_{\text{Scharr}} = \sqrt{G_x^2 + G_y^2}$$
* **Fused Composite Output**:
  $$I_{\text{Fused}} = \alpha \cdot I_{\text{CLAHE}} + (1 - \alpha) \cdot G_{\text{Scharr}} \quad (\text{with } \alpha = 0.7)$$

#### **3. Explicit Inputs & Outputs**
* **Input**:
  - Raw whole-slide microscopy images (`.jpg` or `.tiff`, $1920\times1440$ or $4096\times4096$ pixels).
  - Processing parameters: CLAHE `clipLimit = 3.0`, `tileGridSize = (8, 8)`.
* **Output**:
  - Contrast-balanced grayscale composite image matrix $I_{\text{Fused}}$ (`uint8 [H, W]`).

![Sub-step 1.1.1 4-Stage Transformation: Multi-Tile CLAHE & Edge Gradient Fusion](/Users/dpeleg/local/MicroGlia/Data/step1_1_1_clahe_fusion_io.jpg)
*Figure 1.1: 4-Stage visual transformation pipeline for Sub-step 1.1.1. Top-Left (A): Stage 1 Raw microscopy input. Top-Right (B): Stage 2 Multi-Tile CLAHE contrast enhanced image. Bottom-Left (C): Stage 3 Scharr & Canny edge gradient map. Bottom-Right (D): Stage 4 Fused combined composite output image.*

---

### Sub-step 1.1.2: Sub-Cell IoU Deduplication & Cyan Contour Isolation

Sub-step 1.1.2 executes in two internal sequential stages: **Stage 1 (Cyan Contour Isolation)** and **Stage 2 (Sub-Cell IoU Deduplication)**.

---

#### **Sub-step 1.1.2 - Stage 1: Cyan Contour Isolation (HSV Segmentation)**
* **Purpose**: Microglial cell bodies are marked in the laboratory using hand-drawn cyan contour rings. Converting RGB to **HSV (Hue, Saturation, Value)** space decouples color identity (Hue) from lighting variations, isolating cyan rings ($H \in [85, 105]$) to center bounding boxes around each soma.
* **Input**: Fused tissue slide $I_{\text{Fused}}$ from Sub-step 1.1.1.
* **Output**: HSV Thresholded Cyan Mask Overlay highlighting every annotated cyan ring in bright cyan.

![Sub-step 1.1.2 Stage 1 Input: Fused Tissue Slide](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage1_input_fused_slide.jpg)
*Figure 1.2a (Stage 1 Input): Fused contrast-balanced tissue slide ready for HSV cyan color thresholding.*

![Sub-step 1.1.2 Stage 1 Output: Cyan HSV Contour Overlay Mask](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage1_output_cyan_overlay.jpg)
*Figure 1.2b (Stage 1 Output): HSV thresholded cyan mask overlay ($H \in [85, 105]$), locating all cell body contour rings.*

---

#### **Sub-step 1.1.2 - Stage 2: Sub-Cell IoU Deduplication (IoMin Candidate Filtering)**
* **Purpose**: Contour extraction produces duplicate candidate bounding boxes (e.g., a tight box around just the soma vs. a larger box around the soma + process). Standard NMS (Intersection over Union) fails when a small box $B$ is completely nested inside a larger box $A$. We compute **Sub-Cell IoMin (Intersection over Minimum Area)**:
  $$\text{IoMin}(A, B) = \frac{\text{Area}(A \cap B)}{\min(\text{Area}(A), \text{Area}(B))}$$
  If $\text{IoMin}(A, B) > 0.50$, the duplicate nested sub-cell box is discarded, retaining only the single optimal $128\times128$ crop anchor.
* **Input**: Candidate bounding boxes showing duplicate, overlapping nested boxes (drawn in Red).
* **Output**: Clean, deduplicated bounding box anchors (drawn in Green) centered on verified cell bodies.

![Sub-step 1.1.2 Stage 2 Input: Candidate Overlapping & Nested Bounding Boxes](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage2_input_overlapping_bboxes.jpg)
*Figure 1.2c (Stage 2 Input): Initial candidate bounding boxes containing duplicate and nested sub-cell boxes (shown in Red).*

![Sub-step 1.1.2 Stage 2 Output: Clean Deduplicated Bounding Box Anchors](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage2_output_dedup_grid.jpg)
*Figure 1.2d (Stage 2 Output): Clean, non-redundant bounding box anchors after Sub-Cell IoMin deduplication ($\text{IoMin} > 0.50$, shown in Green).*

---

### Sub-step 1.1.3: Boundary Sharpening & Binary Silhouette Mask Extraction (`boundary_sharpening_pipeline.py`)

#### **1. Purpose**
To strip away unattached parenchymal background debris, red blood cells, and partial neighbor cell fragments from extracted RGB crops ($128\times128\times3$), isolating **only** the target cell body and its attached process arbor into a clean binary silhouette mask ($1 = \text{cell body/arbor}$, $0 = \text{background}$).

#### **2. 4-Stage Internal Processing Steps (`boundary_sharpening_pipeline.py`)**:
1. **Stage 1 (Crop Input)**: Accepts extracted $128\times128\times3$ single-cell RGB crop (`subcell_XXX_original_extracted.jpg`).
2. **Stage 2 (Scharr Edge Enhancement)**: Applies local CLAHE (`clipLimit = 4.0`, `tileGridSize = (4,4)`) and Scharr first-order gradient magnitude calculation to sharpen fine process membrane edges.
3. **Stage 3 (Otsu Adaptive Binarization & Morphological Closing)**: Computes an optimal global threshold using Otsu's method (`cv2.THRESH_OTSU`) and applies morphological closing ($3\times3$ ellipse kernel) to bridge tiny gaps in thin process branches.
4. **Stage 4 (Connected Component Debris Filtering & Silhouette Mask Output)**: Evaluates connected component contours $C_k$. The central contour containing the target soma anchor is retained, while unattached floating background debris and partial neighbor fragments are stripped away, yielding the final binary silhouette mask (`subcell_XXX_sharpened_extracted.jpg`, `uint8 [128, 128]`).

#### **3. Explicit Inputs & Outputs**
* **Input**: Single-cell RGB crop (`subcell_224_original_extracted.jpg`, $128\times128\times3$).
* **Output**: Sharpened binary silhouette mask (`subcell_224_sharpened_extracted.jpg`, $128\times128$).

![Sub-step 1.1.3 4-Stage Visual Transformation: Boundary Sharpening & Binary Silhouette Mask Extraction](/Users/dpeleg/local/MicroGlia/Data/step1_1_3_boundary_sharpening_4stage.jpg)
*Figure 1.3: 4-Stage internal visual transformation pipeline for Sub-step 1.1.3 (`boundary_sharpening_pipeline.py`). Top-Left (A): Stage 1 Single-cell RGB crop input. Top-Right (B): Stage 2 Scharr high-pass edge contrast map. Bottom-Left (C): Stage 3 Otsu adaptive thresholding and morphological closing. Bottom-Right (D): Stage 4 Final sharpened binary silhouette mask output.*

---

## 3. Summary Matrix of Step 1 Sub-Steps, Inputs & Outputs

| Sub-step # | Sub-step Name | Purpose | Exact Input | Exact Output | Visual Panel Artifact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1.1.1** | Multi-Tile CLAHE & Edge Fusion | Contrast enhancement & edge gradient fusion | Raw whole-slide image (`.jpg`/`.tiff`) | 4-Stage Fused Grayscale Slide | 🖼️ [`Data/step1_1_1_clahe_fusion_io.jpg`](file:///Users/dpeleg/local/MicroGlia/Data/step1_1_1_clahe_fusion_io.jpg) |
| **1.1.2 (Stage 1)** | Cyan Contour Isolation | Isolate cyan-contoured cells via HSV | Fused slide (`step1_1_2_stage1_input_fused_slide.jpg`) | Cyan mask overlay | 🖼️ [`Data/step1_1_2_stage1_output_cyan_overlay.jpg`](file:///Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage1_output_cyan_overlay.jpg) |
| **1.1.2 (Stage 2)** | Sub-Cell IoU Deduplication | Discard duplicate nested sub-cell boxes | Candidate overlapping BBoxes | Deduplicated BBoxes | 🖼️ [`Data/step1_1_2_stage2_output_dedup_grid.jpg`](file:///Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage2_output_dedup_grid.jpg) |
| **1.1.3** | Boundary Sharpening | Extract clean binary silhouette masks | Single-cell RGB crop `[128,128,3]` | Sharpened binary silhouette mask `[128,128]` | 🖼️ [`Data/step1_1_3_boundary_sharpening_4stage.jpg`](file:///Users/dpeleg/local/MicroGlia/Data/step1_1_3_boundary_sharpening_4stage.jpg) |
