# High-Throughput Microglial Activation State Counting Engine
## Project Architecture & Technical Roadmap Strategy

---

## Executive Architectural Summary

**Goal**: Automatically segment, classify, and count microglial cells across discrete morphological activation states (*Resting*, *Surveilling*, *Activated*, *Resolution*, *Dystrophic*) in large microscopy tissue images, computing both per-state cell counts and an image-level continuous Activation Index.

**Input Scale**: Large-scale tissue section images yielding **1,000,000+ extracted unlabeled single-cell crops**.

**Core Learning Strategy**: **Self-Supervised Pre-Training (SSL via DINOv2 / MAE)** on millions of unlabeled cells, followed by downstream fine-tuning on $10,000+$ to $50,000+$ labeled cells.

---

## 6-Phase Architectural Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: SSL Pre-Training on 1,000,000+ Unlabeled Cell Crops (DINOv2 / MAE) │
│   • Masked Autoencoder (MAE): Reconstruct 75%-85% masked cell patches      │
│   • DINOv2 Self-Distillation: Learn stain- & rotation-invariant features   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Semi-Supervised Active Annotation (10k–50k Labeled Cells)          │
│   • DINOv2 Feature Space Clustering (k-Means/HDBSCAN)                      │
│   • Cluster-Wise Bulk Annotations + Active Learning Uncertainty Sampling    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: Spatial Graph Neural Network Topology (G = (V, E))                  │
│   • Primary Soma Nodes + Distal Fragment Nodes                              │
│   • Delaunay Triangulation & Spatial Proximity Edges (d <= 35 µm)            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: Multi-Task Model Fine-Tuning                                       │
│   • Pre-trained SSL Encoder Backbone + GATv2 Graph Attention Head           │
│   • Focal Classification Loss + Dystrophic Fragment Reconstruction Loss     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 5: Whole-Slide Inference & Per-State Counting Engine                   │
│   • Tile-Based Inference with Border Non-Maximum Suppression (NMS)           │
│   • Automated Per-State Cell Counts & Continuous Activation Index (0–1)     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 6: Pharmacological & PBM Drug Response Benchmarking Platform          │
│   • Dose-Response Analytics (Pearson r, Spearman rho) for Drug Screening    │
│   • Automated Excel / CSV Reports & Interactive Visual QC Galleries        │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Self-Supervised Learning (SSL) Pre-Training on Millions of Unlabeled Cells

Because our automated extraction pipelines (`extract_cells.py` / `boundary_sharpening_pipeline.py`) can generate **millions of unlabeled cell crops** across whole microscopy datasets, **Self-Supervised Learning (SSL)** is the core foundational engine of our project architecture.

### 1. DINOv2 Self-Distillation (Oquab et al., 2023)
- Uses a Student and Teacher Vision Transformer (ViT) network with multi-crop self-distillation.
- Forces the model to recognize that a zoomed-in process branch crop, a rotated cell mask, and an entire soma crop belong to the same underlying cellular structural entity.
- Learns invariant feature representations resistant to IHC staining variations, background tissue noise, and optical blur—without needing a single human label!

### 2. Masked Autoencoder (MAE) Patch Reconstruction (He et al., 2022)
- Randomly masks out **75% to 85%** of visual patches in each single-cell crop.
- Trains a ViT decoder to reconstruct missing cellular processes from the remaining 15% visible patches.
- Forces the neural network to learn deep spatial semantics and continuity of microglial process arborization zero-shot.

### 3. Massive Efficiency Advantage
- Pre-training on **1,000,000+ unlabeled cells** builds a domain-specific microglial foundation backbone.
- Downstream fine-tuning requires only **10,000 to 50,000 labeled cells** to achieve state-of-the-art classification accuracy, reducing manual annotation effort by **90%**.

---

## Phase 2: High-Throughput Semi-Supervised Active Annotation Strategy

Using the pre-trained SSL feature embedding space:

1. **Unsupervised Pre-Clustering (DINOv2 + HDBSCAN)**:
   - Pass all extracted cell crops through the SSL DINOv2 backbone to obtain 768-dimensional feature vectors.
   - Run **HDBSCAN / k-Means** clustering to group extracted crops into ~100 distinct visual morphometric clusters.
2. **Bulk Cluster Labeling**:
   - Annotators inspect clusters in bulk. Homogeneous clusters (e.g., 500 clear resting cells) can be verified and annotated in a single click, speeding up labeling by **$10\times$**.
3. **Active Learning & Uncertainty Sampling**:
   - For ambiguous border clusters, compute model prediction entropy $H(x) = -\sum p_i \log p_i$.
   - Route only the top 5% highest-uncertainty samples to human expert annotators (Dr. Lilach Gavish's team), maximizing annotation impact per unit effort.

---

## Phase 3: Spatial Graph Neural Network (GNN) Topology

To solve the critical bottleneck where distal processes are severed or dystrophic cells are fragmented:

1. **Graph Node Definition ($V$)**:
   - $V_{\text{soma}}$: Nodes representing central soma bodies.
   - $V_{\text{fragment}}$: Nodes representing distal process branches and beaded fragments.
2. **Graph Edge Definition ($E$)**:
   - Spatial proximity edges established via Delaunay triangulation and Euclidean distance thresholds ($d_{ij} \le 35\,\mu\text{m}$).
3. **Node Features ($h_i$)**:
   - Morphometric descriptors (soma area, perimeter, circularity, fractal dimension $D_f$) combined with SSL DINOv2 embedding vectors.
4. **Message Passing (GATv2 / MPNN)**:
   - Graph Attention Networks aggregate process fragment topology into soma representations, resolving *Resting* vs. *Resolution* ambiguity and grouping shattered *Dystrophic* fragments into unified cellular entities.

---

## Phase 4: Model Training Architecture

### Multi-Task Joint Model Architecture
```
                         ┌───────────────────────────┐
                         │   Input Extracted Crop    │
                         └─────────────┬─────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
         ┌─────────────────────┐               ┌─────────────────────┐
         │ Raw RGB Cell Crop   │               │ Binary Mask / Graph │
         └──────────┬──────────┘               └──────────┬──────────┘
                    │                                     │
                    ▼                                     ▼
         ┌─────────────────────┐               ┌─────────────────────┐
         │ Pre-trained DINOv2  │               │ GATv2 Graph Encoder │
         │    SSL Backbone     │               │                     │
         └──────────┬──────────┘               └──────────┬──────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       │ Concatenate Feature Vectors
                                       ▼
                         ┌───────────────────────────┐
                         │ Multi-Class Classification│
                         │      & Scoring Head       │
                         └─────────────┬─────────────┘
                                       │
       ┌───────────────────────────────┼───────────────────────────────┐
       ▼                               ▼                               ▼
┌──────────────┐                ┌──────────────┐                ┌──────────────┐
│  State Class │                │  Activation  │                │  Dystrophic  │
│(5 Categories)│                │ Index (0-1)  │                │Reconstruction│
└──────────────┘                └──────────────┘                └──────────────┘
```

---

## Phase 5: Whole-Slide Inference & Per-State Counting Engine

1. **Tile Decomposition**: Slice ultra-high-resolution tissue images into overlapping $1024\times1024$ tiles.
2. **Parallel Extraction**: Run `extract_cells.py` / `boundary_sharpening_pipeline.py` to segment cell contours across tiles.
3. **Cross-Tile Seam Deduplication**: Apply Non-Maximum Suppression (NMS) with $IoU > 0.5$ on overlapping tile margins.
4. **Automated Counting Output**:
   $$\text{Per-Image Output} = \{N_{\text{Resting}}, N_{\text{Surveilling}}, N_{\text{Activated}}, N_{\text{Resolution}}, N_{\text{Dystrophic}}\}$$
5. **Continuous Activation Index**:
   $$\text{Activation Index (0–1)} = \frac{N_{\text{Activated}} + 0.6 \cdot N_{\text{Surveilling}} + 0.3 \cdot N_{\text{Resolution}} + 1.0 \cdot N_{\text{Dystrophic}}}{N_{\text{Total}}}$$

---

## Phase 6: Pharmacological Drug & PBM Response Benchmarking Platform

1. **Automated Dose-Response Analytics**:
   - Plot cell state distribution stacked bar charts across drug concentrations and PBM light fluences ($\text{J/cm}^2$).
   - Compute Pearson correlation ($r$) and Spearman rank ($\rho$) to measure drug efficacy in shifting microglia towards neurorepair (*Resolution*) phenotypes.
2. **Exportable Reports & HTML QC Galleries**:
   - Generate automated Excel/CSV spreadsheets detailing per-slice cell counts.
   - Render interactive HTML web galleries allowing researchers (Dr. Lilach Gavish's team) to click and inspect extracted cells per category.

---

## Immediate Recommended Action Plan

| Step | Action Item | Tool / Technology | Timeline |
| :---: | :--- | :--- | :---: |
| **1** | Run SSL pre-training (DINOv2 / MAE) on 100,000+ extracted unlabeled cell crops. | PyTorch, DINOv2 / MAE | **Weeks 1–2** |
| **2** | Perform HDBSCAN clustering on SSL embedding space to organize crops into visual clusters. | scikit-learn, HDBSCAN | **Week 2** |
| **3** | Setup cluster-assisted bulk labeling interface in CVAT or custom web tool. | CVAT / Streamlit | **Week 3** |
| **4** | Annotate initial 10,000–20,000 cell sample using active cluster labeling. | Human Annotators | **Weeks 4–5** |
| **5** | Fine-tune SSL Backbone + GNN multi-class classifier on labeled dataset. | PyTorch Geometric | **Weeks 6–7** |
