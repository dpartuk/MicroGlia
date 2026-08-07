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
* **4-Stage Pipeline Transformation**:
  1. **Stage 1 (Raw Input)**: Raw whole-slide bright-field tissue slice showing shading gradients and low optical contrast between distal processes and parenchymal background.
  2. **Stage 2 (Multi-Tile CLAHE)**: Local $8\times8$ grid histogram equalization with contrast limit (`clipLimit = 3.0`) unifies slide-wide illumination.
  3. **Stage 3 (Scharr & Canny Edge Gradient Map)**: Scharr directional gradient filters ($K_x, K_y$) combined with Canny hysteresis tracing isolate high-frequency 1–2 pixel process tip edges.
  4. **Stage 4 (Fused Combined Output)**: Blends 70% CLAHE intensity with 30% Scharr/Canny edge gradient map ($\alpha=0.7$), yielding the sharpened composite input for cell contour extraction.
* **Input**:
  - Raw whole-slide microscopy images (`.jpg` or `.tiff`, $1920\times1440$ or $4096\times4096$ pixels).
  - Processing parameters: CLAHE `clipLimit = 3.0`, `tileGridSize = (8, 8)`.
* **Output**:
  - Contrast-enhanced grayscale image matrices with sharpened cellular boundaries (`uint8 [H, W]`).

![Sub-step 1.1.1 4-Stage Transformation: Multi-Tile CLAHE & Edge Gradient Fusion](/Users/dpeleg/local/MicroGlia/Data/step1_1_1_clahe_fusion_io.jpg)
*Figure 1.1.1: 4-Stage visual transformation pipeline for Sub-step 1.1.1. Top-Left (A): Raw microscopy input. Top-Right (B): Multi-Tile CLAHE contrast enhanced image. Bottom-Left (C): Scharr & Canny edge gradient map. Bottom-Right (D): Fused combined composite output image.*

---

### Sub-step 1.1.2: Sub-Cell IoU Deduplication & Cyan Contour Isolation
* **Purpose**: Microscopy protocol marks cells using cyan contours. This sub-step detects cyan contours, crops single-cell bounding boxes, and applies contained sub-cell Intersection over Union (IoU) deduplication ($IoU > 0.5$) to eliminate duplicate overlapping crops.
* **Input**:
  - CLAHE-enhanced tissue slides and raw RGB images.
  - Cyan HSV color mask thresholding parameters: `lower_cyan = [85, 100, 100]`, `upper_cyan = [95, 255, 255]`.
* **Output**:
  - Isolated single-cell RGB crops (`uint8 [128, 128, 3]`).
  - Spatial BBox coordinates `[x, y, w, h]` relative to the original whole slide.

![Sub-step 1.1.2 Input vs. Output: Cyan Contour Isolation & Sub-Cell IoU Deduplication](/Users/dpeleg/local/MicroGlia/Data/step1_1_2_cyan_dedup_io.jpg)
*Figure 1.1.2: Input vs. Output visual transformation for Sub-step 1.1.2. Left (Input): Fused tissue slide overlaid with cyan contour thresholding. Right (Output): Deduplicated 128x128 single-cell RGB crop grid.*

---

### Sub-step 1.1.3: Boundary Sharpening & Binary Silhouette Mask Extraction (`boundary_sharpening_pipeline.py`)
* **Purpose**: To isolate clean binary cellular silhouettes ($1 = \text{cell body/process}$, $0 = \text{background}$) from extracted RGB crops, eliminating surrounding debris and background artifacts.
* **Input**:
  - Single-cell RGB crops (`subcell_XXX_original_extracted.jpg`).
* **Output**:
  - High-fidelity binary silhouette masks (`subcell_XXX_sharpened_extracted.jpg`, `uint8 [128, 128]`).

![Sub-step 1.1.3 Input vs. Output: Boundary Sharpening & Binary Silhouette Mask Extraction](/Users/dpeleg/local/MicroGlia/Data/step1_1_3_boundary_sharpening_io.jpg)
*Figure 1.1.3: Input vs. Output visual transformation for Sub-step 1.1.3 (`boundary_sharpening_pipeline.py`). Left (Input): Raw single-cell RGB crop (`subcell_224_original_extracted.jpg`). Right (Output): Sharpened binary silhouette mask (`subcell_224_sharpened_extracted.jpg`).*

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
* **Purpose**: Convert RGB pixel intensities into Optical Density (OD) space via Beer-Lambert law ($OD = -\log_{10}(I / I_0)$) and perform Singular Value Decomposition (SVD) to estimate the slide's unique Stain Vector Matrix ($S \in \mathbb{R}^{2 \times 3}$).
* **Input**:
  - Raw single-cell RGB crops (`uint8 [128, 128, 3]`).
* **Output**:
  - Stain matrix vectors $S_{\text{est}}$ and stain concentration maps $C \in \mathbb{R}^{N \times 2}$.

---

### Sub-step 1.3.2: Stain Alignment & Target Re-Coloration
* **Purpose**: Project the cell's stain concentrations onto a standardized laboratory reference target matrix ($S_{\text{target}}$) and re-convert to RGB space.
* **Input**:
  - Estimated stain concentrations $C$ and target stain matrix $S_{\text{target}}$.
* **Output**:
  - Stain-normalized single-cell RGB crops (`uint8 [128, 128, 3]`) stored back into HDF5 containers.

---

## STEP 1.4: In-Domain Self-Supervised Learning (SSL) Pre-Training

### Purpose
To train an in-domain Vision Transformer (ViT-Base, $D=768$) backbone on 1,000,000+ unlabeled stain-normalized cell crops without human labels, teaching the network to represent complex microglial shapes, soma boundaries, and distal process arborization.

---

### Sub-step 1.4.1: Shared Vision Transformer (ViT-Base) Backbone Initialization
* **Purpose**: Initialize a Vision Transformer architecture with patch size $14 \times 14$, embedding dimension $D = 768$, 12 attention heads, and 12 transformer layers.
* **Input**:
  - ViT-Base architecture parameters ($D=768$, patch size $14\times14$, input size $128\times128$).
* **Output**:
  - Initialized ViT-Base model graph with ~86 million trainable parameters.

---

### Sub-step 1.4.2: Dual Multi-Task Loss Pre-Training (DINOv2 + MAE)
* **Purpose**: Train the ViT backbone using a combined multi-task SSL loss:
  1. **DINOv2 Self-Distillation Loss ($\mathcal{L}_{\text{DINOv2}}$)**: Aligns Student ViT and Teacher ViT representations across global/local crops, learning **high-level semantic cell state separation**.
  2. **MAE Masked Reconstruction Loss ($\mathcal{L}_{\text{MAE}}$)**: Masks 75% of image patches and reconstructs missing pixels, learning **fine-grained process arbor tip textures**.
* **Input**:
  - Batches of stain-normalized cell crops `[B, 128, 128, 3]`.
  - Loss weight parameter: $\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{DINOv2}} + 0.5 \cdot \mathcal{L}_{\text{MAE}}$.
* **Output**:
  - Pre-trained domain-specific ViT-Base feature encoder weights (`vit_base_microglia_ssl.pth`).

---

## STEP 1.5: Unsupervised Feature Space Pre-Clustering

### Purpose
To map all 1,000,000+ cell crops into a structured vector space, run dimensionality reduction, and cluster visually identical cell shapes together into ~100 morphometric clusters for downstream active labeling.

---

### Sub-step 1.5.1: Batch 768-Dim Feature Embedding Extraction & HDF5 Storage
* **Purpose**: Pass every single-cell crop through the pre-trained ViT encoder to extract its 768-dimensional feature representation $h_i \in \mathbb{R}^{768}$.
* **Input**:
  - Pre-trained ViT encoder weights and all cell crops from HDF5 shards.
* **Output**:
  - 768-dimensional embedding vectors stored in dataset `/embeddings` inside `IMAGE_ID_cells.h5`.

---

### Sub-step 1.5.2: UMAP Manifold Reduction & HDBSCAN Morphometric Clustering
* **Purpose**: Reduce 768-dim embeddings down to a 2D/3D UMAP manifold and run HDBSCAN density clustering to partition the 1M+ embeddings into **~100 visual morphometric clusters**.
* **Input**:
  - 768-dimensional feature vectors $h_i \in \mathbb{R}^{768}$.
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
  - Active CVAT Grid Tasks ready for visual inspection.

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
  - Verified morphological state label assignments.

---

### Sub-step 1.6.3: MongoDB Gold-Standard Benchmark Dataset Label Synchronization
* **Purpose**: Webhooks send verified labels back to MongoDB, instantly updating all cell documents in that cluster and creating the final labeled benchmark dataset.
* **Input**:
  - CVAT annotation webhooks.
* **Output**:
  - Updated MongoDB `active_label` fields, establishing the gold-standard labeled benchmark dataset of 10,000 to 50,000 microglial cells ready for Theme 2 Graph Neural Network training.

---

## Summary Matrix of Phase 1 Steps, Inputs & Outputs

| Step # | Sub-step Name | Purpose | Input | Output |
| :--- | :--- | :--- | :--- | :--- |
| **1.1.1** | CLAHE & Edge Fusion | Contrast enhancement & edge gradient fusion | Raw whole-slide image (`.jpg`/`.tiff`) | 4-Stage Fused Grayscale Slide (`step1_1_1_clahe_fusion_io.jpg`) |
| **1.1.2** | Cyan Contour Extraction | Isolate cyan-contoured cells & deduplicate | CLAHE slide + HSV color thresholds | Single-cell RGB crops `[128,128,3]` (`step1_1_2_cyan_dedup_io.jpg`) |
| **1.1.3** | Boundary Sharpening | Extract clean binary silhouette masks | Single-cell RGB crops | Binary silhouette masks `[128,128]` (`step1_1_3_boundary_sharpening_io.jpg`) |
| **1.2.1** | MongoDB Indexing | Fast indexing of cell metadata & active labels | Cell metadata & spatial BBoxes | MongoDB `microglia_metadata` JSON collection |
| **1.2.2** | Per-Slide HDF5 Sharding | High-throughput 21.92ms batch binary storage | RGB crops, binary masks, zero vectors | Per-slide HDF5 files (`IMAGE_ID_cells.h5`) |
| **1.3.1** | Macenko OD Factorization | Estimate slide-specific stain vectors via SVD | RGB cell crops | Stain concentration matrix $C$ & stain vectors $S$ |
| **1.3.2** | Stain Re-Coloration | Standardize IHC stain colors across batches | Stain matrix $C$ & target reference $S_{\text{target}}$ | Stain-normalized RGB cell crops `[128,128,3]` |
| **1.4.1** | ViT Model Init | Setup Vision Transformer backbone graph | ViT-Base architecture parameters | Initialized ViT-Base graph ($D=768$, 86M params) |
| **1.4.2** | Dual SSL Pre-Training | Pre-train ViT on 1M+ crops (DINOv2 + MAE) | Batches of stain-normalized crops | Pre-trained ViT weights (`vit_base_microglia_ssl.pth`) |
| **1.5.1** | Feature Vector Extraction | Extract 768-dim embeddings for 1M+ crops | Pre-trained ViT & HDF5 cell crops | 768-dim vectors $h_i \in \mathbb{R}^{768}$ saved in HDF5 |
| **1.5.2** | UMAP & HDBSCAN Clustering | Group similar cell shapes into ~100 clusters | 768-dim feature vectors $h_i$ | Assigned `cluster_id` (0–99) written to MongoDB |
| **1.6.1** | CVAT Task Generation | Generate $10\times10$ visual grid tasks | MongoDB cell documents by `cluster_id` | CVAT Grid Tasks ready for inspection |
| **1.6.2** | 1-Click Active Labeling | Fast 1-click verification & expert uncertainty | CVAT Grid Tasks + Presaizen Table 1 rubric | Verified cell state label assignments |
| **1.6.3** | MongoDB Sync | Finalize gold-standard labeled benchmark dataset | CVAT webhooks | Updated `active_label` fields for 10k–50k cells in MongoDB |
