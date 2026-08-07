# ALTERNATIVE 3 DESIGN SPECIFICATION: CYAN-SHARPENING PIPELINE (`cyan_sharpening_pipeline.py`)

**Project**: Topological AI Pipeline for Microglial Morphological Classification and Activation Scoring  
**Module**: Alternative 3 - Cyan-Sharpening Single-Cell Extraction & Boundary Masking  
**Author**: Doron Peleg  
**Supervisor**: Dr. Hadas Lapid | **Advisors**: Dr. Lilach Gavish (PhD, MPH), Reut Zinger  
**Institution**: Afeka Academic College of Engineering / Hebrew University of Jerusalem  

---

## 1. Architectural Motivation & Innovation

In digital pathology image processing, extracting single microglial cells from whole-slide microscopy images has evolved through three distinct paradigms:

1. **Alternative 1 (Baseline - Presaizen 2026)**:
   - Uses YOLOv11 bounding box object detection to crop a $64\times64$ or $128\times128$ raw RGB square around the soma.
   - Discards up to 70% of distal process arborization and includes non-relevant parenchymal background noise.

2. **Alternative 2 (Boundary Sharpening - `boundary_sharpening_pipeline.py`)**:
   - Performs Multi-Tile CLAHE, Scharr/Canny edge gradient fusion, and connected component debris filtering across grayscale intensity images.
   - Generates high-contrast binary silhouette masks ($1 = \text{cell body/arbor}$, $0 = \text{background}$).

3. **Alternative 3 (Cyan-Sharpening - `cyan_sharpening_pipeline.py`) — NEW**:
   - **Core Innovation**: Fuses **Cyan HSV Chrominance Color Masks ($H \in [75, 115]$)** directly with **Scharr High-Pass Edge Gradients** and **Multi-Tile CLAHE Intensity Maps**.
   - **Topological Anchoring**: Uses the lab's hand-drawn cyan contour ring as an explicit topological anchor to boost cell membrane edge intensities ($\text{cyan\_boost} = 1.5$) while preserving attached distal process arbors and stripping away unattached parenchymal background noise.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        ALTERNATIVE 3: CYAN-SHARPENING OPERATIONAL PIPELINE                             │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: Multi-Channel HSV Chrominance & Grayscale Extraction                                       │
 │ • HSV Thresholding (H ∈ [75, 115]) isolates lab cyan contour ring mask C_cyan                        │
 │ • Multi-Tile CLAHE (8x8 grid, clipLimit=3.0) balances slide illumination                             │
 │ • Scharr & Canny edge gradient fusion isolates high-frequency membrane edges G_Scharr              │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: Cyan-Guided Topological Edge Fusion                                                        │
 │ • Applies chrominance boost: I_cyan_sharpened = (0.65·CLAHE + 0.35·G_Scharr) · Cyan_Boost           │
 │ • Anchors boundary sharpening around verified cell body rings                                       │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: Sub-Cell IoMin Deduplication                                                               │
 │ • Calculates Sub-Cell IoMin = Area(A ∩ B) / min(Area(A), Area(B))                                   │
 │ • Discards duplicate nested sub-cell candidate boxes (IoMin > 0.50)                                 │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 4: Triple Image Output Architecture                                                           │
 │ • Output A: subcell_XXX_original_extracted.jpg (Single-cell RGB crop [128x128x3])                    │
 │ • Output B: subcell_XXX_cyan_sharpened_extracted.jpg (Cyan-guided sharpened crop [128x128])         │
 │ • Output C: subcell_XXX_marked.jpg (Context location marked in red/cyan on original slide)         │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Specification of Pipeline Stages

### Stage 1: Multi-Channel HSV Chrominance & Grayscale Edge Extraction
* **Purpose**: Converts the raw image to **HSV Color Space** to isolate the lab-annotated cyan contour ring mask ($M_{\text{cyan}}$) independently of lighting shadows, while simultaneously running CLAHE histogram equalization ($8\times8$ grid) and Scharr/Canny edge gradient calculations on the grayscale intensity map.
* **Mathematical Formulation**:
  - Cyan Chrominance Mask:
    $$M_{\text{cyan}}(x,y) = \begin{cases} 1 & \text{if } H \in [75, 115] \land S \ge 40 \land V \ge 140 \\ 0 & \text{otherwise} \end{cases}$$
  - Scharr Gradient Magnitude:
    $$G_{\text{Scharr}} = \sqrt{G_x^2 + G_y^2} \quad \text{where } K_x = \begin{bmatrix} -3 & 0 & 3 \\ -10 & 0 & 10 \\ -3 & 0 & 3 \end{bmatrix}$$
* **Input**: Raw whole-slide microscopy image (`.jpg`/`.tif`).
* **Output**: Cyan Mask $M_{\text{cyan}}$ and Edge Gradient Map $G_{\text{Scharr}}$.

---

### Stage 2: Cyan-Guided Topological Edge Fusion
* **Purpose**: Multiplies the fused intensity edge map by a spatial chrominance boost factor ($\text{cyan\_boost} = 1.5$ where $M_{\text{cyan}} > 0$) to explicitly anchor boundary sharpening around verified cell body rings.
* **Mathematical Formulation**:
  $$I_{\text{cyan\_sharpened}} = \text{clip}\left(\left(0.65 \cdot I_{\text{CLAHE}} + 0.35 \cdot G_{\text{Scharr}}\right) \cdot \text{cyan\_boost}, \, 0, \, 255\right)$$
* **Input**: Grayscale CLAHE image $I_{\text{CLAHE}}$, Edge Map $G_{\text{Scharr}}$, and Cyan Mask $M_{\text{cyan}}$.
* **Output**: Fused Cyan-Sharpened image matrix $I_{\text{cyan\_sharpened}}$ (`uint8 [H, W]`).

---

### Stage 3: Sub-Cell IoMin Deduplication
* **Purpose**: Filters candidate contours using **Sub-Cell IoMin (Intersection over Minimum Area)** to eliminate duplicate nested bounding boxes:
  $$\text{IoMin}(A, B) = \frac{\text{Area}(A \cap B)}{\min(\text{Area}(A), \text{Area}(B))}$$
  If $\text{IoMin}(A, B) > 0.50$, the duplicate nested sub-cell box is discarded, retaining only the single optimal $128\times128$ crop anchor.
* **Input**: Candidate bounding boxes.
* **Output**: Clean, non-redundant bounding box anchors `[x, y, w, h]`.

---

### Stage 4: Triple Image Output Architecture
* **Purpose**: Generates three complementary image files for every extracted cell (`subcell_XXX`):
  1. `subcell_XXX_original_extracted.jpg`: Extracted single-cell RGB crop ($128\times128\times3$).
  2. `subcell_XXX_cyan_sharpened_extracted.jpg`: Single-cell cyan-guided sharpened crop ($128\times128$).
  3. `subcell_XXX_marked.jpg`: Marked location image showing exact position on the original slide tile.
* **Input**: Bounding box anchors `[x, y, w, h]`.
* **Output**: Files saved to `Data/cyan-sharpening-output/` + interactive web gallery (`view_extracted_cells.html`).

---

## 3. Benchmark Dataset Comparison Across All 3 Alternatives

| Alternative Paradigm | Extraction Method | Boundary Sharpening | Output Directory | Extracted Cells (Raw Dataset) | Key Strength |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Alternative 1 (Baseline)** | YOLOv11 Object Bounding-Boxes | None (Raw RGB $64\times64$ crops) | `Data/baseline-output/` | ~4,874 cells | Fast soma detection baseline. |
| **Alternative 2 (Boundary Sharpening)** | Cyan Contour Extraction | Grayscale CLAHE + Scharr Edge Fusion | `Data/boundary-sharpening-output/` | **1,036 cells** | High-contrast binary silhouette masks for morphometric descriptors. |
| **Alternative 3 (Cyan-Sharpening)** | Cyan Chrominance Isolation | Cyan-Guided Topological Fusion | `Data/cyan-sharpening-output/` | **233 cells** | Pure cyan-anchored cell extractions with 0 false-positive background noise. |
