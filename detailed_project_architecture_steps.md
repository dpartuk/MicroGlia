# Detailed 10-Step Microglial AI Project Architecture
## Tailored for Controlled Laboratory Imaging Protocols

---

## Architectural Overview & Lab Protocol Context

Because all dataset images are generated in-house under a standardized laboratory imaging setup (identical to `VID2724_A3_4_00d07h00m.tif`), the imaging hardware, physical pixel resolution ($\mu\text{m/pixel}$), and cyan contour annotations remain consistent. Image variation is strictly bounded to:
1. Minor IHC staining intensity and color hue shifts.
2. Cell density variations (hundreds to thousands of cells per tissue section).

This controlled protocol simplifies the architecture, speeds up Self-Supervised Learning (SSL) convergence, and eliminates the need for out-of-domain generalization.

```
[ Step 1: Automated Cell Extraction & Dual Crop Storage ]
                           │
                           ▼
[ Step 2: In-Domain SSL Pre-Training (DINOv2 + MAE) ]
                           │
                           ▼
[ Step 3: Unsupervised Feature Clustering (HDBSCAN / UMAP) ]
                           │
                           ▼
[ Step 4: Active Bulk Labeling & Uncertainty Sampling (10k-50k Cells) ]
                           │
                           ▼
[ Step 5: Lab-Calibrated Color & Density Augmentation Engine ]
                           │
                           ▼
[ Step 6: Deterministic Physical Spatial Graph Construction ]
                           │
                           ▼
[ Step 7: Multi-Task Joint Model Training (SSL ViT + GATv2) ]
                           │
                           ▼
[ Step 8: Whole-Slide High-Throughput Inference & Seam NMS ]
                           │
                           ▼
[ Step 9: Continuous Activation Index Computation (0.00–1.00) ]
                           │
                           ▼
[ Step 10: Pharmacological Drug & PBM Analytics Platform ]
```

---

## Step 1: Automated Whole-Slide Single-Cell Extraction & Dual-Crop Storage

### Technical Details
- Execute our automated extraction pipelines (`extract_cells.py` / `boundary_sharpening_pipeline.py`) across lab microscopy tissue slice images.
- Implements Multi-Tile Local Adaptive CLAHE ($8\times8$ grid) + Scharr/Canny edge gradient fusion + contained sub-cell IoU deduplication tuned for cyan contours.
- For every detected cell contour, save a dual-representation pair:
  - **Crop A (Raw RGB)**: Original color image crop ($128\times128$).
  - **Crop B (Sharpened Mask)**: Binary silhouette mask isolating cell morphology.

### Motivation
Standardizes single-cell input shapes, removes extraneous parenchymal background, and isolates individual cellular targets for scalable batch processing.

### Expected Output
A standardized repository of **1,000,000+ extracted single-cell crops** saved in paired RGB and binary mask formats.

---

## Step 2: In-Domain Self-Supervised Pre-Training (SSL via DINOv2 + MAE)

### Technical Details
- Pre-train a Vision Transformer (ViT-Base/16) backbone on all 1,000,000+ unlabeled lab single-cell crops using two complementary SSL objectives:
  1. **DINOv2 Self-Distillation**: Student and teacher networks trained with multi-crop cross-entropy loss.
  2. **Masked Autoencoder (MAE)**: Randomly mask out **75% to 85%** of image patches and train a decoder to reconstruct missing cellular processes.

### Motivation & Lab Calibration Advantage
Because all images come from your lab's microscope setup, the SSL backbone converges **$2\times$ faster** and learns razor-sharp, domain-specific representations of microglial membrane textures, process branching, and soma geometries without learning irrelevant out-of-domain noise.

### Expected Output
A specialized **Lab-Calibrated Microglial Feature Encoder** capable of converting any cell image crop into a rich, low-dimensional 768-vector embedding.

---

## Step 3: Unsupervised Feature Space Pre-Clustering (HDBSCAN / UMAP)

### Technical Details
- Pass all 1,000,000+ unlabeled cell crops through the pre-trained SSL encoder to extract 768-dimensional feature vectors.
- Reduce embedding dimensions using UMAP and run density-based clustering (**HDBSCAN / k-Means**) to partition cells into ~100 distinct visual morphometric clusters.

### Motivation
Organizing cells into homogeneous visual clusters (e.g. 500 identical resting cells in 1 cluster) prepares the dataset for ultra-fast annotation. Annotators can inspect and label entire clusters at once, rather than annotating random isolated crops.

### Expected Output
An organized **Morphometric Cluster Map** where >80% of cell crops belong to dense, visually homogenous clusters.

---

## Step 4: Active Bulk Labeling & Uncertainty Sampling (10k–50k Cells)

### Technical Details
- Deploy a web-based annotation interface (CVAT / Streamlit) equipped with dual active labeling modes:
  1. **Bulk Cluster Verification**: Annotators review and verify pre-clustered cell groups in 1 click ($10\times$ faster).
  2. **Active Learning Uncertainty Sampling**: Compute prediction entropy $H(x) = -\sum p_i \log p_i$ for ambiguous border clusters, routing only the top 5% highest-uncertainty samples to human experts (Dr. Lilach Gavish's team).

### Motivation
Maximizes human expert annotation efficiency, ensuring expert effort is focused strictly on difficult biological boundaries (*Resting* vs. *Resolution*, overlapping cells) while automating routine cell labeling.

### Expected Output
A gold-standard **Annotated Benchmark Dataset** of 10,000 to 50,000 microglial cell crops labeled across 5 morphological states (*Resting*, *Surveilling*, *Activated*, *Resolution*, *Dystrophic*).

---

## Step 5: Lab-Calibrated Color Normalization & Density Augmentation Engine

### Technical Details
- Build a lightweight, lab-tuned augmentation pipeline applied during model training:
  - **Macenko / Vahadane Stain Normalization**: Standardizes mild IHC hue variations across lab staining runs.
  - **Color Jitter & Brightness Normalization**: Accommodates minor illumination differences.
  - **Density Sampling & Random $360^\circ$ Rotation**: Simulates sparse vs. dense cell regions.

### Motivation
Because lab optics and physical resolution are fixed, we do not waste model capacity on aggressive synthetic spatial distortions. Focus is kept strictly on color/staining normalization and cell density invariance.

### Expected Output
A calibrated augmentation engine ensuring robust performance across all future lab staining batches.

---

## Step 6: Deterministic Physical Spatial Graph Construction ($G = (V, E)$)

### Technical Details
- For each whole-slide tissue section:
  - Define primary soma bodies as Soma Nodes ($V_{\text{soma}}$).
  - Define distal process branches and beaded fragments as Fragment Nodes ($V_{\text{fragment}}$).
  - Connect nodes using Delaunay triangulation and exact physical spatial distance thresholds ($d_{ij} \le 35\,\mu\text{m}$) mapped to lab physical pixel dimensions.

### Motivation
Because physical spatial scale ($\mu\text{m/pixel}$) is fixed across your lab's microscope, graph distance thresholds are exact and deterministic—capturing process neighborhood topology perfectly without scale ambiguity.

### Expected Output
Whole-image **Spatial Cellular Graphs** ($G=(V,E)$) representing single cells and their distal process fragments as connected topological networks.

---

## Step 7: Multi-Task Joint Model Training (SSL ViT Backbone + GATv2 Graph Encoder)

### Technical Details
- Train a multi-task neural network architecture combining:
  1. **Visual Branch**: Pre-trained DINOv2 SSL ViT encoder backbone.
  2. **Topological Branch**: Graph Attention Network (GATv2) message-passing encoder.
- Combined loss function:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal (Classification)}} + \lambda_1 \mathcal{L}_{\text{MSE (Activation Index)}} + \lambda_2 \mathcal{L}_{\text{Reconstruction (Dystrophic IoU)}}$$

### Motivation
Fuses rich visual feature representations with spatial topological neighborhood context, allowing the network to classify individual cells, reconstruct shattered dystrophic fragments, and resolve *Resting* vs. *Resolution* ambiguity.

### Expected Output
A trained, high-accuracy **Microglial Multi-Task Classification Model** achieving state-of-the-art macro-F1 score (>0.94).

---

## Step 8: Whole-Slide High-Throughput Inference & Seam Deduplication

### Technical Details
- Deploy model for whole-slide inference:
  1. Slice ultra-high-resolution microscopy images into overlapping $1024\times1024$ tiles.
  2. Run cell extraction and GNN inference in parallel across tiles.
  3. Apply Non-Maximum Suppression (NMS, $IoU > 0.5$) across tile overlap seams to eliminate duplicate counts.

### Motivation
Microscopy tissue sections are massive gigapixel images. Tiling enables fast parallel execution, while seam deduplication prevents double-counting cells cut by tile boundaries.

### Expected Output
Automated, reliable **Per-Slice Cell Activation Counts** (`Resting: 420, Surveilling: 180, Activated: 95, Resolution: 210, Dystrophic: 45`) generated in seconds per slide.

---

## Step 9: Image-Level Continuous Activation Index Computation

### Technical Details
- Calculate a continuous Activation Index ($0.00$ to $1.00$) for every analyzed tissue image:
  $$\text{Activation Index} = \frac{N_{\text{Activated}} + 0.6 \cdot N_{\text{Surveilling}} + 0.3 \cdot N_{\text{Resolution}} + 1.0 \cdot N_{\text{Dystrophic}}}{N_{\text{Total}}}$$

### Motivation
Discrete cell counts alone do not provide a single continuous bio-score to measure subtle functional shifts across tissue sections. A continuous index enables precise quantitative comparison across experimental conditions.

### Expected Output
A standardized **Continuous Microglial Activation Score (0.00 to 1.00)** for every analyzed tissue section.

---

## Step 10: Pharmacological Drug & PBM Response Analytics Platform

### Technical Details
- Compute statistical sensitivity metrics (Pearson $r$, Spearman $\rho$, ANOVA) correlating activation scores against candidate drug dosage levels and Photobiomodulation (PBM) light fluence ($J/\text{cm}^2$).
- Generate interactive HTML visual QC galleries displaying extracted cells categorized by state.

### Motivation
Directly fulfills the core translational objective of **Dr. Lilach Gavish, PhD, MPH**: evaluating whether candidate anti-inflammatory drugs or PBM light therapies successfully modulate microglial activation, reduce dystrophic degeneration, and accelerate morphological resolution.

### Expected Output
Automated **Pharmacological Drug Screening Reports & Interactive QC Web Galleries** enabling researchers to evaluate drug efficacy effortlessly.

---

## Summary Matrix: All 10 Steps

| Step # | Action Item | Core Motivation | Expected Output / Deliverable |
| :---: | :--- | :--- | :--- |
| **1** | **Automated Cell Extraction** | Isolate single cells from lab microscopy images. | **1,000,000+ Dual Cell Crops** (RGB + Mask) |
| **2** | **In-Domain SSL Pre-Training** | Learn lab-specific features zero-shot ($2\times$ faster). | **Lab-Calibrated DINOv2/MAE Encoder** |
| **3** | **Unsupervised Feature Clustering** | Group similar cells together for fast annotation. | **Morphometric Cluster Map** (~100 clusters) |
| **4** | **Active Bulk Labeling** | Label 10k–50k cells $10\times$ faster with expert QC. | **Gold-Standard Labeled Dataset** (10k–50k cells) |
| **5** | **Lab Stain Normalization** | Handle mild staining/color & cell density variations. | **Stain- & Density-Calibrated Engine** |
| **6** | **Deterministic Graph Construction** | Connect somas & fragments using fixed physical scale. | **Exact Spatial Cellular Graphs** ($G=(V,E)$) |
| **7** | **Multi-Task Joint Training** | Fuse visual features + spatial graph topology. | **Trained Multi-Task Classification Model** (F1>0.94) |
| **8** | **Whole-Slide High-Throughput Inference** | Process gigapixel slides in parallel without duplicates. | **Automated Per-Slice State Cell Counts** |
| **9** | **Continuous Activation Index** | Provide a single quantitative bio-score per slide. | **Continuous Activation Index (0.00–1.00)** |
| **10** | **Pharmacological Drug Analytics** | Evaluate drug & PBM therapeutic efficacy. | **Pharmacological Reports & QC Web Galleries** |
