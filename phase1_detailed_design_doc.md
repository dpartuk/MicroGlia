# PHASE 1 DETAILED DESIGN DOCUMENT: DATA PREPARATION, CLEANING, AGGREGATION, SSL, STORAGE & ACTIVE LABELING

**Project**: Topological AI Pipeline for Microglial Morphological Classification and Activation Scoring  
**Phase**: Theme 1 / Phase 1 (Data Engineering, SSL Representation Learning, Database Architecture & Active Labeling)  
**Author**: Doron Peleg  
**Supervisor**: Dr. Hadas Lapid | **Advisors**: Dr. Lilach Gavish (PhD, MPH), Reut Zinger  
**Institution**: Afeka Academic College of Engineering / Hebrew University of Jerusalem  

---

## Executive Overview & Pipeline Architecture

Phase 1 (Theme 1) forms the data engineering, self-supervised representation learning, database storage, and active annotation foundation of the entire research project. The primary goal of Phase 1 is to convert raw gigapixel microscopy tissue slices containing cyan-contoured cells into a stain-normalized, feature-indexed, and gold-standard annotated benchmark dataset of 10,000 to 50,000 single microglial cells.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                 PHASE 1 OPERATIONAL PIPELINE FLOW                                      │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STEP 1.1: Automated Single-Cell Extraction & Boundary Sharpening                                    │
 │ • Sub-step 1.1.1: Multi-Tile CLAHE & Edge Gradient Fusion                                           │
 │ • Sub-step 1.1.2: Sub-Cell IoU Deduplication & Cyan Contour Isolation                               │
 │ • Sub-step 1.1.3: Boundary Sharpening & Binary Silhouette Mask Extraction (`boundary_sharpening.py`)│
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STEP 1.2: Hybrid Database & Sharded Image Container Storage Architecture                           │
 │ • Sub-step 1.2.1: MongoDB Document Indexing (`microglia_metadata` collection)                       │
 │ • Sub-step 1.2.2: Per-Slide Image ID HDF5 Binary Container Sharding (`IMAGE_ID_cells.h5`)           │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STEP 1.3: Lab Stain Normalization Engine                                                            │
 │ • Sub-step 1.3.1: Macenko Optical Density (OD) Matrix Factorization                                 │
 │ • Sub-step 1.3.2: Stain Alignment & Target Re-Coloration                                            │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STEP 1.4: In-Domain Self-Supervised Learning (SSL) Pre-Training                                     │
 │ • Sub-step 1.4.1: Shared Vision Transformer (ViT-Base, D=768) Backbone Initialization                │
 │ • Sub-step 1.4.2: Dual Loss Pre-Training (DINOv2 Self-Distillation + MAE Masked Reconstruction)    │
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STEP 1.5: Unsupervised Feature Space Pre-Clustering                                                 │
 │ • Sub-step 1.5.1: Batch 768-Dim Feature Embedding Extraction & HDF5 Storage                         │
 │ • Sub-step 1.5.2: UMAP Manifold Reduction (768d -> 2d) & HDBSCAN Morphometric Clustering (~100 clusters)│
 └──────────────────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                    │
                                                    ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STEP 1.6: Active Bulk Labeling & Uncertainty Sampling                                               │
 │ • Sub-step 1.6.1: CVAT REST API Visual Grid Task Creation (10x10 Grid Tasks)                        │
 │ • Sub-step 1.6.2: 1-Click Active Bulk Cluster Verification & Entropy Uncertainty Sampling H(x)     │
 │ • Sub-step 1.6.3: MongoDB Gold-Standard Benchmark Dataset Label Synchronization                      │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## STEP 1.1: Automated Single-Cell Extraction & Boundary Sharpening

### Purpose
To isolate individual microglial cells and process fragments from gigapixel microscopy brain tissue slices directly from raw laboratory cyan contours without relying on bounding-box object detectors, producing clean single-cell RGB crops and binary silhouette masks.

---

### Sub-step 1.1.1: Multi-Tile CLAHE & Edge Gradient Fusion
* **Purpose**: Raw bright-field microscopy slides suffer from non-uniform illumination and low optical contrast between microglial processes and parenchymal background. Contrast Limited Adaptive Histogram Equalization (CLAHE) over an $8\times8$ grid combined with Scharr/Canny edge gradient fusion accentuates fine process branches without amplifying background noise.
* **Input**:
  - Raw whole-slide microscopy images (`.jpg` or `.tiff`, $1920\times1440$ or $4096\times4096$ pixels).
  - Parameters: CLAHE `clipLimit = 3.0`, `tileGridSize = (8, 8)`.
* **Output**:
  - Contrast-enhanced grayscale image matrices with sharpened cellular boundaries (`uint8 [H, W]`).

![Sub-step 1.1.1 4-Stage Transformation: Multi-Tile CLAHE & Edge Gradient Fusion](/Users/dpeleg/local/MicroGlia/Data/step1_1_1_clahe_fusion_io.jpg)
*Figure 1.1.1: 4-Stage visual transformation pipeline for Sub-step 1.1.1. Top-Left (A): Stage 1 Raw microscopy input. Top-Right (B): Stage 2 Multi-Tile CLAHE contrast enhanced image. Bottom-Left (C): Stage 3 Scharr & Canny edge gradient map. Bottom-Right (D): Stage 4 Fused combined composite output image.*

---

### Sub-step 1.1.2: Sub-Cell IoU Deduplication & Cyan Contour Isolation

Sub-step 1.1.2 is split into two internal sequential stages: **Stage 1 (Cyan Contour Isolation)** and **Stage 2 (Sub-Cell IoU Deduplication)**. Below is the detailed breakdown and full-resolution visual demonstration for each stage:

---

#### **Sub-step 1.1.2 - Stage 1: Cyan Contour Isolation (HSV Segmentation)**
* **Purpose**: In the lab imaging protocol, microglial somas are annotated using hand-drawn cyan contour rings. Converting RGB to **HSV (Hue, Saturation, Value)** space decouples color identity (Hue) from lighting variations, isolating cyan rings ($H \in [85, 105]$) to center bounding boxes around each soma.
* **Input**: Fused tissue slide from Sub-step 1.1.1.
* **Output**: HSV Thresholded Cyan Mask Overlay highlighting every annotated cyan ring in bright cyan.

![Sub-step 1.1.2 Stage 1 Input: Fused Tissue Slide](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage1_input_fused_slide.jpg)
*Figure 1.1.2.1a (Stage 1 Input): Fused contrast-balanced tissue slide ready for HSV cyan color thresholding.*

![Sub-step 1.1.2 Stage 1 Output: Cyan HSV Contour Overlay Mask](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage1_output_cyan_overlay.jpg)
*Figure 1.1.2.1b (Stage 1 Output): HSV thresholded cyan mask overlay ($H \in [85, 105]$), locating all cell body contour rings.*

---

#### **Sub-step 1.1.2 - Stage 2: Sub-Cell IoU Deduplication (IoMin Candidate Filtering)**
* **Purpose**: Contour extraction generates duplicate candidate bounding boxes (e.g. tight soma box vs. larger soma+process box). Standard NMS (Intersection over Union) fails when a small box $B$ is completely nested inside a larger box $A$. We compute **Sub-Cell IoMin (Intersection over Minimum Area)**:
  $$\text{IoMin}(A, B) = \frac{\text{Area}(A \cap B)}{\min(\text{Area}(A), \text{Area}(B))}$$
  If $\text{IoMin}(A, B) > 0.50$, the duplicate nested sub-cell box is discarded, retaining only the single optimal $128\times128$ crop anchor.
* **Input**: Candidate bounding boxes showing duplicate, overlapping nested boxes (drawn in Red).
* **Output**: Clean, deduplicated bounding box anchors (drawn in Green) centered on verified cell bodies.

![Sub-step 1.1.2 Stage 2 Input: Candidate Overlapping & Nested Bounding Boxes](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage2_input_overlapping_bboxes.jpg)
*Figure 1.1.2.2a (Stage 2 Input): Initial candidate bounding boxes containing duplicate and nested sub-cell boxes (shown in Red).*

![Sub-step 1.1.2 Stage 2 Output: Clean Deduplicated Bounding Box Anchors](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_stage2_output_dedup_grid.jpg)
*Figure 1.1.2.2b (Stage 2 Output): Clean, non-redundant bounding box anchors after Sub-Cell IoMin deduplication ($\text{IoMin} > 0.50$, shown in Green).*

---

### Sub-step 1.1.3: Boundary Sharpening & Binary Silhouette Mask Extraction (`boundary_sharpening_pipeline.py`)

* **Purpose**: To strip away unattached parenchymal background debris and neighbor cell fragments from extracted RGB crops ($128\times128\times3$), isolating **only** the target cell body and its attached process arbor into a clean binary silhouette mask ($1 = \text{cell body/arbor}$, $0 = \text{background}$).
* **Input**: Single-cell RGB crop (`subcell_224_original_extracted.jpg`, $128\times128\times3$).
* **Output**: Sharpened binary silhouette mask (`subcell_224_sharpened_extracted.jpg`, $128\times128$).

![Sub-step 1.1.3 4-Stage Visual Transformation: Boundary Sharpening & Binary Silhouette Mask Extraction](/Users/dpeleg/local/MicroGlia/Data/step1_1_3_boundary_sharpening_4stage.jpg)
*Figure 1.1.3: 4-Stage internal visual transformation pipeline for Sub-step 1.1.3 (`boundary_sharpening_pipeline.py`). Top-Left (A): Stage 1 Single-cell RGB crop input. Top-Right (B): Stage 2 Scharr high-pass edge contrast map. Bottom-Left (C): Stage 3 Otsu adaptive thresholding and morphological closing. Bottom-Right (D): Stage 4 Final sharpened binary silhouette mask output.*

---

## STEP 1.2: Hybrid Database & Sharded Image Container Storage Architecture

### Purpose
To solve the filesystem $I/O$ bottleneck of storing 1,000,000+ individual image files by deploying a hybrid storage strategy: MongoDB for indexing JSON metadata and active labels, and per-slide HDF5 binary containers for high-throughput batch loading (**21.92 ms** read speed).

---

### Sub-step 1.2.1: MongoDB Document Indexing
* **Purpose**: Maintain an indexed document database storing cell metadata, spatial BBoxes, UMAP coordinates, cluster IDs, and active morphological state labels.
* **Input**:
  - Extracted cell metadata (`cell_id`, `image_id`, `bbox [x, y, w, h]`, `date_extracted`).
* **Output**:
  - MongoDB `microglia_metadata` collection with JSON schema:
    ```json
    {
      "cell_id": "VID2724_A3_4_cell_014",
      "image_id": "VID2724_A3_4_00d07h00m",
      "bbox": [1050, 650, 128, 128],
      "cluster_id": 14,
      "active_label": "Quiescent / Resting",
      "confidence": 0.95,
      "is_verified": true
    }
    ```

---

### Sub-step 1.2.2: Per-Slide Image ID HDF5 Container Sharding
* **Purpose**: Shard heavy binary arrays (RGB images, binary masks, 768-dim embeddings) into per-slide HDF5 files to enable parallelized batch loading during SSL pre-training and GNN inference.
* **Input**:
  - Extracted RGB crops `[N, 128, 128, 3]`, binary masks `[N, 128, 128]`, and initial zero-initialized embedding arrays `[N, 768]`.
* **Output**:
  - HDF5 container files: `binary_shards/IMAGE_ID_cells.h5` containing datasets `/images`, `/masks`, `/embeddings`, `/bboxes`, `/labels`.

---

## STEP 1.3: Lab Stain Normalization Engine

### Purpose
To eliminate inter-batch immunohistochemistry (IHC) color variations, stain intensity fluctuations, and illumination shifts across different microscopy slides BEFORE self-supervised pre-training, ensuring the SSL model learns pure morphology rather than stain artifacts.

---

### Sub-step 1.3.1: Macenko Optical Density (OD) Matrix Factorization

#### **Purpose: Linear Deconvolution of Chemical Dye Vectors via SVD**
In bright-field immunohistochemistry (IHC), light attenuation follows the Beer-Lambert Law. Sub-step 1.3.1 converts raw RGB pixel intensities into linear Optical Density (OD) space and applies **Singular Value Decomposition (SVD)** to extract the 2 physical chemical stain vectors ($S \in \mathbb{R}^{2 \times 3}$) unique to that tissue slide.

#### **Mathematical Formulation**:
1. **Beer-Lambert Optical Density Conversion**:
   $$OD = -\log_{10}\left(\frac{I}{I_0}\right)$$
   where $I$ is the pixel RGB vector and $I_0 = 240.0$ is the transmitted background light intensity.
2. **Transparent Pixel Thresholding**:
   Filter out transparent background pixels ($OD < 0.15$) to isolate non-zero tissue stain vectors $\mathbf{V}_{\text{OD}} \in \mathbb{R}^{M \times 3}$.
3. **Singular Value Decomposition (SVD)**:
   Perform SVD on the covariance matrix of $\mathbf{V}_{\text{OD}}$ to project points onto the 2D plane spanned by the two largest singular vectors ($V_1, V_2$).
4. **Stain Concentration Matrix Calculation ($C$)**:
   $$\mathbf{C} = \mathbf{V}_{\text{OD}} \cdot S_{\text{est}}^T \quad (\mathbf{C} \in \mathbb{R}^{N \times 2})$$

* **Input**:
  - Raw single-cell RGB crop tensor `[128, 128, 3]` ($I \in [0, 255]$).
  - Background light intensity constant $I_0 = 240.0$.
* **Output**:
  - Estimated Slide Stain Vector Matrix $S_{\text{est}} \in \mathbb{R}^{2 \times 3}$ (containing the 2 primary chemical dye vectors).
  - Pixel-wise Stain Concentration Map $C \in \mathbb{R}^{N \times 2}$ (specifying exact dye amounts at every pixel location).

![Sub-step 1.3.1 Input vs Output: Macenko SVD Optical Density Factorization](/Users/dpeleg/local/MicroGlia/Data/step1_3_1_macenko_svd_io.jpg)
*Figure 1.3.1: Input vs. Output visual transformation for Sub-step 1.3.1. Left (Input): Raw single-cell RGB crop [128x128x3]. Right (Output): Optical Density SVD Stain Concentration Map C showing decomposed chemical dye distribution.*

---

### Sub-step 1.3.2: Stain Alignment & Target Re-Coloration
* **Purpose**: Project the cell's stain concentrations $C$ onto a standardized laboratory reference target matrix ($S_{\text{target}}$) and re-convert to RGB space.
* **Input**:
  - Raw un-normalized cell crop `[128, 128, 3]` and laboratory reference target matrix $S_{\text{target}}$.
* **Output**:
  - Macenko stain-normalized single-cell RGB crop `[128, 128, 3]` saved back into HDF5 shards (`IMAGE_ID_cells.h5`).

![Sub-step 1.3.2 Input vs. Output: Macenko Stain Normalization Engine](/Users/dpeleg/local/MicroGlia/Data/step1_3_stain_norm_io.jpg)
*Figure 1.3.2: Input vs. Output visual transformation for Sub-step 1.3.2. Left (Input): Raw un-normalized single-cell crop with variable batch stain color. Right (Output): Macenko stain-normalized single-cell crop aligned to standardized laboratory reference target.*

---

## STEP 1.4: In-Domain Self-Supervised Learning (SSL) Pre-Training

### Purpose
To train an in-domain Vision Transformer (ViT-Base, $D=768$) backbone on 1,000,000+ unlabeled stain-normalized cell crops without human labels, teaching the network to represent complex microglial shapes, soma boundaries, and distal process arborization.

---

### Sub-step 1.4.1: Shared Vision Transformer (ViT-Base) Backbone Initialization
* **Purpose**: Initialize a Vision Transformer architecture with patch size $14 \times 14$, embedding dimension $D = 768$, 12 attention heads, and 12 transformer layers.
* **Input**:
  - ViT-Base architecture specification ($D=768$, patch size $14\times14$, input size $128\times128$).
* **Output**:
  - Initialized ViT-Base model graph with ~86 million trainable parameters ready for SSL training.

---

### Sub-step 1.4.2: Dual Multi-Task Loss Pre-Training (DINOv2 + MAE)
* **Purpose**: Train the ViT backbone using a combined multi-task SSL loss:
  1. **DINOv2 Self-Distillation Loss ($\mathcal{L}_{\text{DINOv2}}$)**: Aligns Student ViT and Teacher ViT representations across global/local crops, learning **high-level semantic cell state separation**.
  2. **MAE Masked Reconstruction Loss ($\mathcal{L}_{\text{MAE}}$)**: Masks 75% of image patches and reconstructs missing pixels, learning **fine-grained process arbor tip textures**.
* **Input**:
  - Batches of stain-normalized cell crops `[B, 128, 128, 3]` loaded from HDF5 shards.
  - Loss weight parameter: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{DINOv2}} + 0.5 \cdot \mathcal{L}_{\text{MAE}}$.
* **Output**:
  - Pre-trained domain-specific ViT-Base feature encoder weights saved to disk (`vit_base_microglia_ssl.pth`).

---

## STEP 1.5: Unsupervised Feature Space Pre-Clustering

### Purpose
To map all 1,000,000+ cell crops into a structured vector space, run dimensionality reduction, and cluster visually identical cell shapes together into ~100 morphometric clusters for downstream active labeling.

---

### Sub-step 1.5.1: Batch 768-Dim Feature Embedding Extraction & HDF5 Storage
* **Purpose**: Pass every single-cell crop through the pre-trained ViT encoder to extract its 768-dimensional feature representation $h_i \in \mathbb{R}^{768}$.
* **Input**:
  - Pre-trained ViT encoder weights (`vit_base_microglia_ssl.pth`) and cell crops from HDF5 shards.
* **Output**:
  - 768-dimensional feature vectors stored in dataset `/embeddings` inside per-slide HDF5 containers (`IMAGE_ID_cells.h5`).

---

### Sub-step 1.5.2: UMAP Manifold Reduction & HDBSCAN Morphometric Clustering
* **Purpose**: Reduce 768-dim embeddings down to a 2D/3D UMAP manifold and run HDBSCAN density clustering to partition the 1M+ embeddings into **~100 visual morphometric clusters**.
* **Input**:
  - 768-dimensional feature vectors $h_i \in \mathbb{R}^{768}$ loaded from HDF5.
  - Parameters: UMAP `n_neighbors = 30`, `min_dist = 0.1`; HDBSCAN `min_cluster_size = 150`.
* **Output**:
  - Assigned `cluster_id` (0 to 99) written to MongoDB for every cell document.

---

## STEP 1.6: Active Bulk Labeling & Uncertainty Sampling

### Purpose
To build a gold-standard annotated benchmark dataset of 10,000 to 50,000 cells across 5 morphological activation states ($50\times$ faster than single-cell clicking) using CVAT REST API grid tasks and entropy uncertainty sampling ($H(x)$).

---

### Sub-step 1.6.1: CVAT REST API Visual Grid Task Creation
* **Purpose**: Automatically generate $10 \times 10$ image matrix tasks in CVAT for each morphometric cluster, displaying 100 visually identical cell crops side-by-side.
* **Input**:
  - MongoDB cell documents grouped by `cluster_id`.
  - Python uploader script (`cvat_bulk_uploader.py`) calling CVAT REST API (`/api/tasks`).
* **Output**:
  - Active CVAT Grid Tasks ready for visual inspection on CVAT web server (`http://localhost:8080` or `app.cvat.ai`).

---

### Sub-step 1.6.2: 1-Click Active Bulk Cluster Verification & Entropy Uncertainty Sampling $H(x)$
* **Purpose**: Annotators inspect each grid and assign the morphological state label to all 100 cells in **1 single click**. Ambiguous cells with high prediction entropy $H(h_i)$ are isolated and routed to Dr. Lilach Gavish's expert team for adjudication.
* **Input**:
  - CVAT Grid Tasks and human annotator verification.
  - Morphological Rubric (Presaizen, 2026 Table 1):
    - `Quiescent / Resting`: Small, circular soma with **NO branching podia (0)**, clear dark boundary.
    - `Patrolling / Surveilling`: Circular / spindle soma with **2 branching podia**, clear dark boundary.
    - `Reactive / Pro-inflammatory / Activated`: Irregular soma with **2–3 podia**, intermittent boundary.
    - `Senescent / Terminal / Resolution`: Ameboid / flat medium-large soma with **NO podia (0)**, clear bright appearance.
    - `Dystrophic`: Fragmented, shattered processes lacking a central soma anchor.
* **Output**:
  - Verified morphological state label assignments and high-entropy adjudication queue.

---

### Sub-step 1.6.3: MongoDB Gold-Standard Benchmark Dataset Label Synchronization
* **Purpose**: Webhooks send verified labels back to MongoDB, instantly updating all cell documents in that cluster and creating the final labeled benchmark dataset.
* **Input**:
  - CVAT annotation webhooks sending JSON payload (`{cluster_id: 14, label: "Quiescent / Resting"}`).
* **Output**:
  - Updated MongoDB `active_label` and `is_verified` fields, establishing the final gold-standard benchmark dataset of 10,000 to 50,000 microglial cells ready for Theme 2 Graph Neural Network training.

---

## Summary Matrix of Phase 1 Steps, Inputs & Outputs

| Step # | Sub-step Name | Purpose | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1.1.1** | CLAHE & Edge Fusion | Contrast enhancement & edge gradient fusion | Raw whole-slide image (`.jpg`/`.tiff`) | 4-Stage Fused Grayscale Slide (`step1_1_1_clahe_fusion_io.jpg`) |
| **1.1.2 (Stage 1)** | Cyan Contour Isolation | Isolate cyan-contoured cells via HSV | Fused slide (`step1_1_2_stage1_input_fused_slide.jpg`) | Cyan mask overlay (`step1_1_2_stage1_output_cyan_overlay.jpg`) |
| **1.1.2 (Stage 2)** | Sub-Cell IoU Deduplication | Discard duplicate nested sub-cell boxes | Overlapping BBoxes (`step1_1_2_stage2_input_overlapping_bboxes.jpg`) | Deduplicated BBoxes (`step1_1_2_stage2_output_dedup_grid.jpg`) |
| **1.1.3** | Boundary Sharpening | Extract clean binary silhouette masks | Single-cell RGB crops | 4-Stage Binary silhouette masks (`step1_1_3_boundary_sharpening_4stage.jpg`) |
| **1.2.1** | MongoDB Indexing | Fast indexing of cell metadata & active labels | Cell metadata & spatial BBoxes | MongoDB `microglia_metadata` JSON collection |
| **1.2.2** | Per-Slide HDF5 Sharding | High-throughput 21.92ms batch binary storage | RGB crops, binary masks, zero vectors | Per-slide HDF5 files (`IMAGE_ID_cells.h5`) |
| **1.3.1** | Macenko OD Factorization | Estimate slide-specific stain vectors via SVD | Single-cell RGB crop `[128,128,3]` | OD Concentration map $C$ & Stain matrix $S$ (`step1_3_1_macenko_svd_io.jpg`) |
| **1.3.2** | Stain Re-Coloration | Standardize IHC stain colors across batches | Raw cell crops & reference target $S_{\text{target}}$ | Stain-normalized cell crops (`step1_3_stain_norm_io.jpg`) |
| **1.4.1** | ViT Model Init | Setup Vision Transformer backbone graph | ViT-Base architecture parameters | Initialized ViT-Base graph ($D=768$, 86M params) |
| **1.4.2** | Dual SSL Pre-Training | Pre-train ViT on 1M+ crops (DINOv2 + MAE) | Batches of stain-normalized crops from HDF5 | Pre-trained ViT weights (`vit_base_microglia_ssl.pth`) |
| **1.5.1** | Feature Vector Extraction | Extract 768-dim embeddings for 1M+ crops | Pre-trained ViT & HDF5 cell crops | 768-dim vectors $h_i \in \mathbb{R}^{768}$ saved in HDF5 |
| **1.5.2** | UMAP & HDBSCAN Clustering | Group similar cell shapes into ~100 clusters | 768-dim feature vectors $h_i$ | Assigned `cluster_id` (0–99) written to MongoDB |
| **1.6.1** | CVAT Task Generation | Generate $10\times10$ visual grid tasks | MongoDB cell documents by `cluster_id` | CVAT Grid Tasks ready for inspection |
| **1.6.2** | 1-Click Active Labeling | Fast 1-click verification & expert uncertainty | CVAT Grid Tasks + Presaizen Table 1 rubric | Verified cell state label assignments |
| **1.6.3** | MongoDB Sync | Finalize gold-standard labeled benchmark dataset | CVAT webhooks | Updated `active_label` fields for 10k–50k cells in MongoDB |
