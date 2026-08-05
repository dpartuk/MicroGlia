# SCHOOL OF DATA SCIENCE: INTELLIGENT SYSTEMS
## AFEKA ACADEMIC COLLEGE OF ENGINEERING / HEBREW UNIVERSITY OF JERUSALEM

# Topological AI Pipeline for Microglial Morphological Classification and Activation Scoring
### A Research Proposal Submitted toward the Degree of Master of Science (M.Sc.) in Intelligent Systems

* **Student Name**: Doron Peleg
* **Academic Program**: M.Sc. in Intelligent Systems (Machine Learning)
* **Supervisor**: Dr. Hadas Lapid
* **Advisors**: Dr. Lilach Gavish (PhD, MPH), Reut Zinger
* **Submission Date**: August 2026

---

## Table of Abbreviations

| Abbreviation | Definition |
| :--- | :--- |
| **AI** | Artificial Intelligence |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve |
| **CLAHE** | Contrast Limited Adaptive Histogram Equalization |
| **CNS** | Central Nervous System |
| **CNN** | Convolutional Neural Network |
| **CV** | Computer Vision |
| **CVAT** | Computer Vision Annotation Tool |
| **DINOv2** | Self-Supervised Vision Transformer for Representation Learning |
| **GATv2** | Graph Attention Network (Version 2) |
| **GNN** | Graph Neural Network |
| **IHC** | Immunohistochemistry |
| **IoU** | Intersection over Union |
| **MAE** | Masked Autoencoder |
| **mAP** | Mean Average Precision |
| **ML** | Machine Learning |
| **MongoDB** | Document-Oriented Database for JSON Metadata & Label Indexing |
| **MPNN** | Message Passing Neural Network |
| **PBM** | Photobiomodulation Therapy |
| **ROS** | Reactive Oxygen Species |
| **SOTA** | State-of-the-Art |
| **SSL** | Self-Supervised Learning |
| **TBI** | Traumatic Brain Injury |
| **ViT** | Vision Transformer |
| **YOLO** | You Only Look Once (Object Detection Framework) |

---

## 1. Motivation

Traumatic Brain Injury (TBI) and secondary neurodegenerative disorders represent a major global health crisis, affecting an estimated 69 million individuals annually and standing as a leading cause of long-term neurocognitive impairment (Dewan et al., 2018; Maas et al., 2017). Following primary neurotrauma, a progressive secondary injury cascade ensues, driven by chronic neuroinflammation, oxidative stress, and blood-brain barrier breakdown. In military populations, mild TBI (mTBI) resulting from blast acceleration is especially prevalent, impacting 15%–22% of deployed service members and predisposing patients to persistent neurological deficits and post-traumatic stress disorder (PTSD) (Hoge et al., 2008; Okie, 2005).

Microglia—the primary resident immune sentinels of the Central Nervous System (CNS)—are the central cellular orchestrators of this neuroinflammatory cascade (Salter & Beggs, 2014; Wolf et al., 2017). Under physiological homeostasis, surveilling microglia display a highly ramified morphology with small somas and delicate, dynamic processes that continuously scan the parenchymal microenvironment. Upon encountering mechanical or biochemical stress, microglia undergo rapid morphological metamorphosis, retracting process arbors, enlarging soma volumes, and transitioning into ameboid, hypertrophic, or dystrophic states. Because microglial structural shifts directly mirror their underlying functional polarization (pro-inflammatory M1-like vs. pro-resolving M2-like states), high-throughput quantitative morphometry provides a critical, non-destructive window into post-injury tissue pathophysiology (Leyh et al., 2021).

![Figure 1: Original Whole-Slide Brain Tissue Microscopy Image](/Users/dpeleg/local/MicroGlia/scratch/figures/figure1_original_tissue.jpg)
*Figure 1: Example of an original lab microscopy brain tissue slice image (`JPG_VID2724_B1_3_00d07h00m.jpg`) containing hundreds of microglial cells marked with cyan contours.*

In translational pharmacology and neuro-therapeutics, evaluating the impact of candidate therapeutic drugs and physical modalities—such as Photobiomodulation (PBM) light therapy—on microglial activation and morphological recovery is paramount (Gavish & Houreld, 2019; Hamblin, 2018). In ongoing collaborative research advised by **Dr. Lilach Gavish, PhD, MPH**, quantifying how pharmacological drug candidates modulate microglial state transitions (e.g., accelerating resolution or suppressing dystrophic degeneration) is key to discovering novel neuroprotective treatments. However, screening pharmacological drug effects across large-scale tissue slices requires an objective, scalable, and spatially sensitive microglial quantification pipeline—a capability severely bottlenecked by current manual and parametric stereology methods.

Recent baseline research at Afeka Academic College of Engineering by Presaizen (2026) established a 4-class soma-centric classification pipeline using YOLOv11 object detection on microscopy brain tissue slices. While providing an initial proof-of-concept, Presaizen (2026) highlighted key computational challenges: bounding-box object detectors restrict analysis to the local soma region. Consequently, soma-centric models encounter difficulties in distinguishing between Resting and Resolution states and in detecting shattered dystrophic microglia lacking a central soma anchor. Addressing these challenges through a topological, foundation-model-guided AI framework is essential for establishing unbiased morphometric biomarkers to evaluate drug and PBM therapeutic efficacy in translational neurobiology.

---

## 2. Research Question and Hypothesis

### 2.1 Research Questions
* **RQ1**: Can automated cyan contour cell extraction combined with Graph Neural Networks (GNNs) overcome soma-centric bounding box limitations to accurately capture fragmented microglial arborization?
* **RQ2**: Does incorporating spatial process topology resolve the persistent biological confusion between Resting and Resolution microglial states?
* **RQ3**: Can a continuous, multi-parametric activation index derived from graph topological representations provide superior sensitivity in quantifying pharmacological drug treatments and Photobiomodulation (PBM) therapeutic responses compared to standard discrete classification?

### 2.2 Research Hypothesis
It is hypothesized that transitioning from soma-centric bounding boxes to automated cyan contour cell extraction and modeling the spatial neighborhood of fragmented distal processes using Graph Neural Networks (GNNs) will resolve the confusion between Resting and Resolution states by capturing the full cellular silhouette. Furthermore, by representing process fragments as nodes in a spatial proximity graph, the framework will significantly increase the recall of dystrophic/shattered cells lacking a unified soma anchor, yielding a sensitive, continuous activation index for evaluating drug and PBM therapeutic efficacy.

---

## 3. Research Objectives

* **Objective 1 (Active Bulk Dataset Labeling & Benchmark Construction)**: Establish a gold-standard benchmark dataset of 10,000 to 50,000 single-cell crops labeled across 5 morphological activation states (*Resting*, *Surveilling*, *Activated*, *Resolution*, *Dystrophic*) using SSL feature space pre-clustering (HDBSCAN), 1-click bulk cluster verification, and entropy-driven uncertainty sampling $H(x)$ indexed in MongoDB.
* **Objective 2 (Boundary Sharpening & Silhouette Extraction)**: Process cyan-contoured single-cell crops using automated boundary sharpening (`boundary_sharpening_pipeline.py`) and CLAHE edge fusion to extract clean, high-fidelity binary silhouette masks.
* **Objective 3 (Graph Topology Construction)**: Construct spatial proximity graphs connecting soma nodes and process fragment nodes, training a Graph Neural Network (GNN) to reconstruct shattered dystrophic cells into unified biological entities.
* **Objective 4 (Self-Supervised Feature Space)**: Implement a contrastive self-supervised representation learning space (DINOv2 / Masked Autoencoders) fine-tuned on stain-normalized cell crops and masks to separate morphologically subtle activation states.
* **Objective 5 (Pharmacological & Experimental Validation)**: Formulate a continuous Multi-Parametric Activation Index (0–1 scale) and validate its sensitivity against experimental pharmacological drug-treated and PBM-irradiated rodent brain slices in collaboration with Dr. Lilach Gavish.

---

## 4. Literature Review

### 4.1 Microglia and Neuroinflammation
Microglia constitute 10%–15% of all glial cells in the CNS and serve as the frontline defense against traumatic, ischemic, and neurodegenerative insults. Under homeostatic conditions, surveilling microglia display an arborized morphology with delicate processes extending tens of micrometers from a small soma. Following TBI, inflammatory signaling cascades induce rapid structural metamorphosis: retracting processes, swelling somas, and transitioning into activated ameboid macrophages capable of phagocytosis.

### 4.2 Pharmacological & Photobiomodulation (PBM) Therapeutics
Quantifying microglial morphological response to candidate anti-inflammatory drugs and Photobiomodulation (PBM) therapy is central to translational neuro-therapeutics (Gavish & Houreld, 2019; Hamblin, 2018). In rodent injury models, therapeutic drug compounds and PBM light irradiation stimulate mitochondrial bioenergetics, attenuate reactive oxygen species (ROS), and accelerate microglial polarization from pro-inflammatory (M1-like) to pro-resolving (M2-like) phenotypes. High-throughput quantitative morphometry allows drug discovery researchers to measure dose-dependent morphological shifts across entire brain slices.

### 4.3 Morphometric Analysis of Microglia
Traditional microglial morphometry relies on manual thresholding and skeletonization (e.g., ImageJ FracLac or HALO modules). Key quantitative metrics include fractal dimension ($D_f$), lacunarity, soma-to-cell area ratio, total process length, and number of branch endpoints. While informative, manual and parametric morphometry suffers from severe inter-rater variability, labor intensity, and failure in dense tissue slices with overlapping processes.

### 4.4 Automated Detection and Deep Learning
Recent computational advances have applied Convolutional Neural Networks (CNNs) and object detectors (YOLOv8, YOLOv11) to microglial quantification (Anwer et al., 2023; Morera et al., 2024; Hsu et al., 2025 - StainAI). While high-throughput bounding-box detectors excel at counting central somas, they cropped out distal process arborization ($64\times64$ crops), discarding up to 70% of the morphological information required to distinguish subtle functional states.

### 4.5 Cyan Contour Extraction & Image Processing
Microscopy imaging protocols in our laboratory generate whole-slide tissue images where cells are marked with cyan contours. By applying multi-tile CLAHE ($8\times8$ grid), Scharr/Canny edge gradient fusion, and contained sub-cell IoU deduplication, our extraction pipeline isolates individual cell bodies and arbors directly from the raw slide without relying on bounding-box object detectors.

### 4.6 Graph Neural Networks (GNNs) in Cellular Topology
Graph Neural Networks (GNNs) model complex non-Euclidean spatial relationships. In neurobiology, representing segmented cellular somas and process fragments as nodes in a spatial proximity graph ($G = (V, E)$) enables Message Passing Neural Networks (MPNNs) or Graph Attention Networks (GATs) to learn topological connectivity. GNNs enable the reconstruction of "shattered" dystrophic microglia—a critical bottleneck in neurodegeneration research.

### 4.7 Literature Gap & Summary
Despite rapid progress in deep learning for digital pathology, existing microglial pipelines remain strictly soma-centric and bounding-box constrained. No current framework integrates cyan contour cell extraction with spatial GNN topological modeling to resolve Resting vs. Resolution state confusion or reconstruct fragmented dystrophic cells. This project addresses this critical gap.

---

## 5. Preliminary Work and Study Rationale

This project builds directly upon the baseline M.Sc. thesis project completed at Afeka Academic College of Engineering by Tali Presaizen (Jan 2026), supervised by Sharon Yalov-Handzel, PhD, and co-advised by Dr. Lilach Gavish, PhD, MPH.

The baseline study established a soma-centric annotated dataset of 4,874 microglial cells extracted from phase-contrast and fluorescence microscopy images of PBM-treated rat brain slices. The dataset categorized cells into four discrete morphological states: Resting, Surveilling, Activated, and Resolution.

![Figure 2: 4 Microglial Morphological Activation Cell States with Side-by-Side Sharpened Silhouette Crops](/Users/dpeleg/local/MicroGlia/scratch/figures/figure3_cell_states_panel.jpg)
*Figure 2: Dual-crop representation panel across the 4 microglial morphological activation states: (A) Resting (Ramified), (B) Surveilling, (C) Activated (Ameboid), and (D) Resolution. Each state shows the raw RGB cell crop alongside its isolated, boundary-sharpened single-cell silhouette crop (`subcell_XXX_sharpened_extracted.jpg`).*

The baseline architecture utilized a YOLOv11 object detector for 4-class classification. While achieving respectable baseline metrics (mAP@0.5 ≈ 0.76, macro-F1 ≈ 0.69), the study revealed key performance challenges:

1. **Soma-Centric Restriction**: Bounding boxes crop out distal process branches, restricting analysis to the central soma.
2. **Resting vs. Resolution Ambiguity**: Misclassification occurred between Resting and Resolution states, as both states share similar soma sizes but differ in distal process topology.
3. **Challenges on Dystrophic Microglia**: Dystrophic/shattered microglia in injured brain tissue lack a central soma anchor, causing object detectors to miss fragmented cellular entities.

These baseline findings provide the direct rationale for our proposed Topological AI Pipeline.

---

## 6. Proposed Methodology & Two-Theme Project Architecture

The project architecture is structured into two core operational themes, ensuring clean separation between data engineering / self-supervised representation learning and downstream model training / pharmacological evaluation:

### THEME 1: Data Preparation, Cleaning, Aggregation, SSL, Storage & Labeling
* **Stage 1.1 (Automated Cell Extraction & Cleaning)**: Execute `extract_cells.py` / `boundary_sharpening_pipeline.py` using Multi-Tile CLAHE ($8\times8$ grid), Scharr/Canny edge fusion, and contained sub-cell IoU deduplication.

![Figure 3: Whole-Slide Cell Contour Extraction Grid Map](/Users/dpeleg/local/MicroGlia/scratch/figures/figure2_extraction_map.jpg)
*Figure 3: Example of a whole-slide cell contour extraction grid map (`VID2724_A3_4_00d07h00m`), illustrating automated single-cell extraction and side-by-side pipeline evaluation.*

* **Stage 1.2 (Hybrid Database & Image Container Storage)**: Deploy **MongoDB** as a document store indexing JSON metadata, spatial BBoxes (`[x, y, w, h]`), cluster IDs, and active labels. Store binary cell crops, masks, and DINOv2 embeddings in HDF5 containers sharded directly by original whole-slide Image ID (`IMAGE_ID_cells.h5`).
* **Stage 1.3 (Lab Stain Normalization Engine)**: Apply Macenko optical density matrix factorization to all 1M+ cell crops BEFORE SSL pre-training to eliminate IHC color shifts.
* **Stage 1.4 (In-Domain Self-Supervised Pre-Training)**: Pre-train ViT-Base on 1M+ stain-normalized crops using DINOv2 self-distillation and MAE patch reconstruction.
* **Stage 1.5 (Unsupervised Feature Space Pre-Clustering)**: Reduce SSL embeddings using UMAP and cluster with HDBSCAN/k-Means into ~100 morphometric clusters.
* **Stage 1.6 (Active Bulk Labeling & Uncertainty Sampling)**: Annotate 10k–50k cells using MongoDB-driven 1-click bulk cluster verification + top 5% entropy expert sampling.

### THEME 2: Training, Classification, Counting & Evaluation
* **Stage 2.1 (Deterministic Spatial Graph Construction)**: Connect Soma Nodes ($V_{\text{soma}}$) and Fragment Nodes ($V_{\text{fragment}}$) via Delaunay/k-NN edges using spatial BBox coordinates (`[x, y, w, h]`) and fixed lab pixel scale ($\mu\text{m/pixel}$, $d_{ij} \le 35\,\mu\text{m}$).
* **Stage 2.2 (Multi-Task Joint Model Training)**: Train a joint network combining pre-trained DINOv2 SSL ViT backbone + GATv2 Graph Encoder trained with Focal + Contrastive + Reconstruction loss.
* **Stage 2.3 (Whole-Slide High-Throughput Inference & Seam NMS)**: Tile gigapixel images into overlapping $1024\times1024$ regions, run parallel inference, and apply BBox-driven Non-Maximum Suppression (NMS, $IoU>0.5$) across tile seams.
* **Stage 2.4 (Per-State Cell Counting & Continuous Activation Index)**: Output discrete 5-state counts and compute continuous Activation Index ($0.00$–$1.00$).
* **Stage 2.5 (Pharmacological Drug & PBM Response Analytics Platform)**: Calculate Pearson $r$, Spearman $\rho$, and ANOVA correlating activation scores against candidate drug dosage levels and PBM light fluence ($J/\text{cm}^2$) for Dr. Lilach Gavish's screening pipeline.

---

## 7. Work Plan and Project Stages

The proposed research will be executed across six structured project stages over a total estimated duration of **26 weeks (~6.5 months)**, adhering to the departmental proposal guidelines:

### Stage 1 – Theme 1: Dataset Extraction, MongoDB Setup & Stain Normalization
* **Task**: Run automated cell extraction (`extract_cells.py`). Set up MongoDB document store for JSON metadata, spatial BBoxes (`[x, y, w, h]`), and active labels. Perform Macenko stain normalization on all 1,000,000+ cell crops sharded by original Image ID (`IMAGE_ID_cells.h5`).
* **Goal**: Establish a clean, stain-normalized single-cell crop repository indexed in MongoDB.
* **Estimated Duration**: 4 weeks

### Stage 2 – Theme 1: SSL Pre-Training & Active Cluster Labeling
* **Task**: Pre-train DINOv2 and MAE backbones on 1M+ stain-normalized crops. Run HDBSCAN clustering and annotate 10,000–50,000 cells using MongoDB-driven active bulk cluster verification in CVAT.
* **Goal**: Deploy a domain-specific SSL feature encoder and establish a gold-standard labeled benchmark dataset.
* **Estimated Duration**: 4 weeks

### Stage 3 – Theme 2: Spatial GNN Graph Construction & Topology
* **Task**: Construct physical spatial proximity graphs ($G=(V,E)$) connecting soma nodes and process fragment nodes using BBox spatial coordinates (`[x, y, w, h]`). Implement and train GATv2/MPNN GNN architectures.
* **Goal**: Reconstruct shattered dystrophic microglia into single biological entities and capture full arborization topology.
* **Estimated Duration**: 5 weeks

### Stage 4 – Theme 2: Multi-Task Joint Model Fine-Tuning
* **Task**: Fine-tune the joint DINOv2 ViT + GATv2 GNN architecture on labeled data using combined Focal, Contrastive, and Dystrophic Reconstruction losses.
* **Goal**: Achieve state-of-the-art per-class classification accuracy (Macro-F1 > 0.94) and resolve Resting vs. Resolution state ambiguity.
* **Estimated Duration**: 4 weeks

### Stage 5 – Theme 2: High-Throughput Whole-Slide Inference Engine
* **Task**: Build the whole-slide inference pipeline with overlapping $1024\times1024$ tile processing, BBox-driven Non-Maximum Suppression (NMS), per-state counting, and continuous Activation Index ($0.00$–$1.00$) computation.
* **Goal**: Deliver a fast, automated whole-slide cell counting engine.
* **Estimated Duration**: 5 weeks

### Stage 6 – Theme 2: Pharmacological Validation, Thesis Writing & Defense
* **Task**: Perform statistical sensitivity validation (Pearson $r$, Spearman $\rho$) on drug-treated and PBM-irradiated rat brain slices with Dr. Lilach Gavish. Write final M.Sc. thesis and defend before academic committee.
* **Goal**: Complete and submit the final M.Sc. thesis document and defend the research.
* **Estimated Duration**: 4 weeks

---

## 8. Evaluation Plan and Benchmarking

The framework will be evaluated across three complementary quantitative tiers:
1. **Segmentation Metrics**: Dice Coefficient, Intersection over Union (IoU), and Boundary-F1 score compared against manual polygonal ground truth.
2. **Morphological Classification**: Macro-F1 score, Per-class Precision/Recall, and Confusion Matrix analysis across Resting, Surveilling, Activated, Resolution, and Dystrophic states (benchmarked against YOLOv11 baseline F1=0.69).
3. **Biological & Clinical Sensitivity**: Pearson correlation ($r$) and Spearman rank ($\rho$) between the computed Activation Index (0–1) and biological pharmacological drug dosage / PBM light fluence ($\text{J/cm}^2$) across tissue slices.

---

## 9. Expected Scientific and Technological Contribution

1. **Novel Methodological Paradigm**: First AI framework combining foundation-model segmentation with Graph Neural Networks for microglial morphometry.
2. **Dystrophic Microglia Reconstruction**: Solves the critical bottleneck of detecting shattered dystrophic microglia lacking a central soma anchor.
3. **Open-Source Benchmark Dataset**: Provides an open-source, fragment-first polygonal dataset of 4,874 microglial cells for the research community.
4. **Translational Pharmacology Impact**: Delivers a scalable, reproducible tool for quantifying pharmacological drug impact and Photobiomodulation (PBM) neuroprotective efficacy in translational neurobiology under the direction of Dr. Lilach Gavish.

---

## 10. References

1. Anwer, D. M., Gubinelli, F., Kurt, Y. A., et al. (2023). A comparison of machine learning approaches for the quantification of microglial cells in the brain of mice, rats and non-human primates. *PLOS ONE*, 18(4), e0284480.
2. Dewan, M. C., Rattani, A., Gupta, S., et al. (2018). Estimating the global incidence of traumatic brain injury. *Journal of Neurosurgery*, 130(4), 1080-1097.
3. Gavish, L., & Houreld, N. N. (2019). Therapeutic Efficacy of Photobiomodulation (PBM) in Wound Healing and Neuroinflammation. *Photomedicine and Laser Surgery*, 37(3), 150-162.
4. Hamblin, M. R. (2018). Photobiomodulation for traumatic brain injury and neurodegenerative diseases. *Photonics & Lasers in Medicine*, 7(3), 231-244.
5. He, K., Chen, X., Xie, S., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *IEEE/CVF CVPR*, 16000-16009.
6. Hoge, C. W., McGurk, D., Thomas, J. L., et al. (2008). Mild traumatic brain injury in U.S. Soldiers returning from Iraq. *New England Journal of Medicine*, 358(5), 453-463.
7. Hsu, C.-H., Hsu, Y.-Y., Chang, B.-M., et al. (2025). StainAI: quantitative mapping of stained microglia and insights into brain-wide neuroinflammation and therapeutic effects in cardiac arrest. *Communications Biology*, 8, 7926.
8. Kim, J., Pavlidis, P., & Vogel Ciernia, A. (2024). Development of a High-Throughput Pipeline to Characterize Microglia Morphological States at a Single-Cell Resolution. *eNeuro*, 11(6), ENEURO.0010-24.2024.
9. Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015-4026.
10. Leyh, J., Schafer, M. K., et al. (2021). Microglial morphodynamics in traumatic brain injury and recovery. *Glia*, 69(8), 1950-1965.
11. Maas, A. I., Menon, D. K., Adelson, P. D., et al. (2017). Traumatic brain injury: integrated approaches to improve prevention, clinical care, and research. *The Lancet Neurology*, 16(12), 987-1048.
12. Macenko, M., Niethammer, M., Marron, J. S., et al. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE ISBI*, 1107-1110.
13. Morera, H., Dave, P., Kolinko, Y., et al. (2024). A novel deep learning-based method for automatic stereology of microglia cells from low magnification images. *Neurotoxicology and Teratology*, 102, 107336.
14. Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning Robust Visual Features Without Supervision. *arXiv preprint arXiv:2304.07193*.
15. Pachitariu, M., & Stringer, C. (2024). Cellpose 3.0: accurate segmentation of biological images using foundation models. *Nature Methods*, 21(4), 701-710.
16. Presaizen, T. (2026). *AI-Powered Microglial Classification for Activation Scoring*. Master's Thesis, School of Data Science: Intelligent Systems, Afeka Academic College of Engineering & Hebrew University of Jerusalem.
17. Salter, M. W., & Beggs, S. (2014). Sublime microglia: expanding roles for the guardians of the CNS. *Cell*, 158(1), 15-24.
18. Veličković, P., Cucurull, G., Casanova, A., et al. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.
19. Wolf, S. A., Boddeke, H. W., & Kettenmann, H. (2017). Microglia in Physiology and Pathology. *Physiological Reviews*, 97(4), 1339-1393.
20. Xiong, H., Zheng, S., Qi, X., Liu, J. (2025). μGlia-Flow, an automatic workflow for microglia segmentation and classification. *Journal of Neuroscience Methods*, 402, 110022.
21. Zähringer, A., Vinnakota, J. M., Wertheimer, T., et al. (2025). AIstain: Enhancing microglial phagocytosis analysis through deep learning. *Cell Reports Methods*, 5(11), 101207.
