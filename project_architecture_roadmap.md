# High-Throughput Microglial Activation State Counting Engine
## Project Architecture & Technical Roadmap Strategy

---

## Executive Architectural Summary

**Goal**: Automatically segment, classify, and count microglial cells across discrete morphological activation states (*Resting*, *Surveilling*, *Activated*, *Resolution*, *Dystrophic*) in large microscopy tissue images, computing both per-state cell counts and an image-level continuous Activation Index.

**Input Scale**: Large-scale tissue section images containing hundreds of cyan-contoured cells per slice, yielding $100,000+$ extracted single-cell crops.

**Annotation Plan**: Annotating $10,000+$ to $50,000+$ extracted single cells prior to model training.

---

## 6-Phase Architectural Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Semi-Supervised Active Annotation (10k–50k Cells)                 │
│   • DINOv2 Unsupervised Pre-Clustering (k-Means/HDBSCAN)                     │
│   • Cluster-Wise Bulk Annotations + Active Learning Uncertainty Sampling    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: Dual-Crop Preprocessing & Feature Embedding Space                   │
│   • Crop Pair Storage (Original RGB + Binary Silhouette Mask)              │
│   • Contrastive DINOv2 / Masked Autoencoder (MAE) Fine-Tuning               │
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
│ PHASE 4: Multi-Task Model Training                                          │
│   • Joint DINOv2 Vision Transformer + GATv2 Graph Attention Architecture    │
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

## Phase 1: High-Throughput Semi-Supervised Active Annotation Strategy

To annotate $10,000+$ to $50,000+$ extracted cell crops efficiently without annotator fatigue:

1. **Unsupervised Pre-Clustering (DINOv2 + HDBSCAN)**:
   - Pass all extracted unlabelled cell crops through a zero-shot DINOv2 Vision Transformer to obtain 768-dimensional feature vectors.
   - Run **HDBSCAN / k-Means** clustering to group extracted crops into ~100 distinct visual morphometric clusters (e.g., highly ramified resting cells, swollen ameboid activated cells, isolated process fragments).
2. **Bulk Cluster Labeling**:
   - Annotators inspect clusters in bulk. Clear clusters (e.g., 500 homogenous resting cells) can be verified and annotated in a single click, speeding up labeling by **$10\times$**.
3. **Active Learning & Uncertainty Sampling**:
   - For ambiguous border clusters, compute model prediction entropy $H(x) = -\sum p_i \log p_i$.
   - Route only the top 5% highest-uncertainty samples to human expert annotators (Dr. Lilach Gavish's team), maximizing annotation impact per unit effort.

---

## Phase 2: Dual-Crop Preprocessing & Feature Embedding Space

Store and preprocess every extracted cell as a dual-representation pair:
- **Representation A (Raw RGB Crop)**: Preserves original staining intensity, texture, and intracellular details.
- **Representation B (Sharpened Contour / Binary Mask)**: Pure morphological silhouette removing background tissue noise and neighboring cell artifacts.

### Data Augmentation Pipeline
To ensure model robustness across microscopy staining variations:
- Stain intensity jittering & contrast adjustment.
- Random $360^\circ$ rotation and horizontal/vertical flips.
- Elastic morphological warping to simulate histological sectioning distortions.

---

## Phase 3: Spatial Graph Neural Network (GNN) Topology

To solve the critical bottleneck where distal processes are severed or dystrophic cells are fragmented:

1. **Graph Node Definition ($V$)**:
   - $V_{\text{soma}}$: Nodes representing central soma bodies.
   - $V_{\text{fragment}}$: Nodes representing distal process branches and beaded fragments.
2. **Graph Edge Definition ($E$)**:
   - Spatial proximity edges established via Delaunay triangulation and Euclidean distance thresholds ($d_{ij} \le 35\,\mu\text{m}$).
3. **Node Features ($h_i$)**:
   - Morphometric descriptors (soma area, perimeter, circularity, fractal dimension $D_f$) combined with DINOv2 embedding vectors.
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
         │ DINOv2 ViT Backbone │               │ GATv2 Graph Encoder │
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

### Combined Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal (Classification)}} + \lambda_1 \mathcal{L}_{\text{MSE (Activation Index)}} + \lambda_2 \mathcal{L}_{\text{Contrastive (InfoNCE)}}$$

- **Focal Loss**: Handles class imbalance (e.g., higher frequency of surveilling cells vs. rare dystrophic cells).
- **Contrastive Loss**: Enforces clear embedding separation between morphologically adjacent states (*Resting* vs. *Resolution*).

---

## Phase 5: Whole-Slide Inference & Per-State Counting Engine

### Whole-Slide Processing Pipeline
1. **Tile Decomposition**: Slice ultra-high-resolution microscopy section into overlapping $1024\times1024$ tiles.
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
| **1** | Run zero-shot DINOv2 embedding extraction on existing 4,874 dataset crops. | PyTorch, DINOv2 | **Week 1** |
| **2** | Perform HDBSCAN clustering to organize crops into 50–100 visual clusters. | scikit-learn, HDBSCAN | **Week 1** |
| **3** | Setup cluster-assisted bulk labeling interface in CVAT or custom web tool. | CVAT / Streamlit | **Week 2** |
| **4** | Annotate initial 10,000 cell sample using active cluster labeling. | Human Annotators | **Weeks 3–4** |
| **5** | Train baseline DINOv2 + GNN multi-class classifier on 10k dataset. | PyTorch Geometric | **Weeks 5–6** |
