# High-Throughput Microglial Activation State Counting Engine
## Project Architecture & Technical Roadmap Strategy

---

## Two Core Project Themes

The project architecture is organized into two primary operational themes:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ THEME 1: DATA PREPARATION, CLEANING, AGGREGATION, SSL, STORAGE & LABELING                   │
│   1.1 Automated Whole-Slide Cell Extraction & Cleaning (CLAHE + Edge Fusion + Deduplication)│
│   1.2 Dual-Crop Storage & Dataset Aggregation (Raw RGB + Binary Silhouette Mask)            │
│   1.3 In-Domain Self-Supervised Pre-Training (SSL via DINOv2 + MAE on 1M+ Unlabeled Crops)   │
│   1.4 Unsupervised Feature Space Pre-Clustering (UMAP + HDBSCAN into ~100 Clusters)         │
│   1.5 Active Bulk Labeling & Uncertainty Sampling (10k–50k Labeled Cells via CVAT)         │
│   1.6 Lab Stain Normalization Engine (Macenko Stain Normalization for IHC Shifts)           │
└──────────────────────────────────────────────┬──────────────────────────────────────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ THEME 2: TRAINING, CLASSIFICATION, COUNTING & EVALUATION                                    │
│   2.1 Deterministic Spatial Graph Construction (G = (V, E) connecting Somas & Fragments)     │
│   2.2 Multi-Task Joint Model Training (Pre-trained SSL ViT + GATv2 Graph Encoder)           │
│   2.3 Whole-Slide High-Throughput Inference & Seam NMS Deduplication                        │
│   2.4 Per-State Cell Counting & Continuous Activation Index Computation (0.00–1.00)        │
│   2.5 Pharmacological Drug & PBM Response Analytics Platform (Dr. Lilach Gavish Screening)  │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# THEME 1: Data Preparation, Cleaning, Aggregation, SSL, Storage & Labeling

### 1.1 Automated Whole-Slide Cell Extraction & Cleaning
- **Technical Details**: Execute `extract_cells.py` / `boundary_sharpening_pipeline.py` using Multi-Tile CLAHE ($8\times8$ grid), Scharr/Canny edge fusion, and contained sub-cell IoU deduplication.
- **Motivation**: Removes parenchymal background noise and isolates single-cell targets from massive whole-slide images.
- **Expected Output**: Standardized single-cell bounding regions across all lab microscopy slices.

### 1.2 Dual-Crop Storage & Dataset Aggregation
- **Technical Details**: Save every extracted cell as a dual-representation pair: Crop A (Raw RGB, $128\times128$) and Crop B (Sharpened Binary Mask).
- **Motivation**: Preserves both intracellular staining texture and pure morphological silhouettes.
- **Expected Output**: Paired dataset repository of **1,000,000+ extracted single-cell crops**.

### 1.3 In-Domain Self-Supervised Pre-Training (SSL via DINOv2 + MAE)
- **Technical Details**: Pre-train a ViT-Base backbone on 1M+ unlabeled crops using DINOv2 self-distillation and MAE (75%–85% patch masking).
- **Motivation**: Learns deep microglial membrane semantics zero-shot without human labels ($2\times$ faster on lab setup).
- **Expected Output**: A **Lab-Calibrated Microglial Feature Encoder** generating 768-dim embedding vectors.

### 1.4 Unsupervised Feature Space Pre-Clustering
- **Technical Details**: Reduce SSL embeddings using UMAP and cluster with HDBSCAN/k-Means into ~100 morphometric clusters.
- **Motivation**: Groups visually identical cells together to enable rapid, high-volume bulk annotation.
- **Expected Output**: An organized **Morphometric Cluster Map** with >80% cells in dense clusters.

### 1.5 Active Bulk Labeling & Uncertainty Sampling
- **Technical Details**: Annotate 10k–50k cells using 1-click bulk cluster verification + prediction entropy $H(x)$ sampling for ambiguous border cells.
- **Motivation**: Speeds up annotation by **$10\times$** while focusing human expert effort (Dr. Lilach Gavish's team) strictly on hard edge cases.
- **Expected Output**: Gold-standard **Annotated Benchmark Dataset** of 10,000 to 50,000 cells across 5 states (*Resting*, *Surveilling*, *Activated*, *Resolution*, *Dystrophic*).

### 1.6 Lab Stain Normalization Engine
- **Technical Details**: Apply Macenko / Vahadane stain normalization + color jitter online during training.
- **Motivation**: Standardizes mild IHC stain variations across lab staining runs.
- **Expected Output**: A stain-invariant preprocessing pipeline.

---

# THEME 2: Training, Classification, Counting & Evaluation

### 2.1 Deterministic Spatial Graph Construction ($G = (V, E)$)
- **Technical Details**: Define Soma Nodes ($V_{\text{soma}}$) and Process Fragment Nodes ($V_{\text{fragment}}$); connect via Delaunay/k-NN edges using fixed lab pixel scale ($\mu\text{m/pixel}$, $d_{ij} \le 35\,\mu\text{m}$).
- **Motivation**: Captures process arborization topology to resolve *Resting* vs. *Resolution* ambiguity and reconstruct shattered *Dystrophic* cells.
- **Expected Output**: Whole-image **Spatial Cellular Graphs** ($G=(V,E)$).

### 2.2 Multi-Task Joint Model Training
- **Technical Details**: Train a joint network combining pre-trained DINOv2 SSL ViT backbone + GATv2 Graph Encoder using Focal + Contrastive + Reconstruction loss.
- **Motivation**: Fuses rich visual features with spatial topological context for state classification and dystrophic reconstruction.
- **Expected Output**: Trained **Multi-Task Classification Model** (Macro-F1 > 0.94).

### 2.3 Whole-Slide High-Throughput Inference & Seam NMS
- **Technical Details**: Tile gigapixel images into overlapping $1024\times1024$ regions, run parallel inference, and apply Non-Maximum Suppression (NMS, $IoU>0.5$) across tile seams.
- **Motivation**: Enables fast whole-slide processing while preventing duplicate cell counts at tile boundaries.
- **Expected Output**: Automated per-slide cell extraction and classification in seconds.

### 2.4 Per-State Cell Counting & Continuous Activation Index Computation
- **Technical Details**: Output discrete counts $\{N_{\text{Resting}}, N_{\text{Surveilling}}, N_{\text{Activated}}, N_{\text{Resolution}}, N_{\text{Dystrophic}}\}$ and compute Activation Index ($0.00$–$1.00$):
  $$\text{Activation Index} = \frac{N_{\text{Activated}} + 0.6 \cdot N_{\text{Surveilling}} + 0.3 \cdot N_{\text{Resolution}} + 1.0 \cdot N_{\text{Dystrophic}}}{N_{\text{Total}}}$$
- **Motivation**: Provides a continuous, standardized bio-score to compare tissue activation across experimental conditions.
- **Expected Output**: Per-slice cell counts & continuous Activation Score ($0.00$–$1.00$).

### 2.5 Pharmacological Drug & PBM Response Analytics Platform
- **Technical Details**: Calculate Pearson $r$, Spearman $\rho$, and ANOVA correlating activation scores against candidate drug dosage levels and PBM light fluence ($J/\text{cm}^2$); generate interactive HTML QC galleries.
- **Motivation**: Evaluates pharmacological drug efficacy and PBM neuroprotection for Dr. Lilach Gavish's translational research.
- **Expected Output**: Automated **Pharmacological Screening Reports & Interactive QC Web Galleries**.
