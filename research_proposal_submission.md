# SCHOOL OF DATA SCIENCE: INTELLIGENT SYSTEMS
## AFEKA ACADEMIC COLLEGE OF ENGINEERING / HEBREW UNIVERSITY OF JERUSALEM

# Topological AI Pipeline for Microglial Morphological Classification and Activation Scoring
### A Research Proposal Submitted toward the Degree of Master of Science (M.Sc.) in Intelligent Systems

* **Student Name**: Doron Peleg
* **Academic Program**: M.Sc. in Intelligent Systems (Machine Learning)
* **Supervisors**: Dr. Hadas Lapid, Dr. Lilach Gavish (PhD, MPH), Reut Zinger
* **Submission Date**: August 2026

---

## Table of Abbreviations

| Abbreviation | Definition |
| :--- | :--- |
| **AI** | Artificial Intelligence |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic Curve |
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
| **MPNN** | Message Passing Neural Network |
| **PBM** | Photobiomodulation Therapy |
| **ROS** | Reactive Oxygen Species |
| **SAM** | Segment Anything Model |
| **SOTA** | State-of-the-Art |
| **TBI** | Traumatic Brain Injury |
| **U-Net** | Convolutional Network Architecture for Biological Segmentation |
| **ViT** | Vision Transformer |
| **YOLO** | You Only Look Once (Object Detection Framework) |

---

## 1. Motivation

Traumatic Brain Injury (TBI) represents a major global health crisis, affecting an estimated 69 million individuals annually and standing as a leading cause of long-term disability and mortality worldwide (Dewan et al., 2018; Maas et al., 2017). Beyond acute primary mechanical tissue disruption, TBI initiates a progressive secondary injury cascade characterized by sustained neuroinflammation, blood-brain barrier breakdown, and metabolic crisis. In military populations, mild TBI (mTBI) resulting from blast acceleration is especially prevalent, impacting 15%–22% of deployed service members and frequently predisposing patients to chronic neurodegenerative sequelae, mood disturbances, and post-traumatic stress disorder (PTSD) (Hoge et al., 2008; Okie, 2005).

Microglia—the primary resident immune macrophages of the Central Nervous System (CNS)—are the central cellular orchestrators of this neuroinflammatory response (Salter & Beggs, 2014; Wolf et al., 2017). Under homeostatic conditions, resting microglia display a highly arborized morphology with small somas and delicate, dynamic processes that continuously scan the parenchymal microenvironment. Upon encountering mechanical or biochemical stress, microglia undergo rapid morphological metamorphosis, retracting process arbors, enlarging soma volumes, and transitioning into ameboid, hypertrophic, or dystrophic states. Because microglial structural shifts directly mirror their functional polarization (pro-inflammatory M1-like vs. pro-resolving M2-like states), quantitative morphometric profiling provides a critical window into post-injury tissue pathophysiology and therapeutic response (Leyh et al., 2021).

Photobiomodulation (PBM) therapy, which delivers low-level red to near-infrared light (600–1000 nm), has emerged as a promising non-invasive neuroprotective intervention for TBI (Hamblin, 2018; Gavish & Houreld, 2019). By stimulating mitochondrial cytochrome c oxidase, PBM boosts ATP production, mitigates oxidative stress, and promotes microglial polarization toward neurorepair phenotypes. However, evaluating PBM therapeutic efficacy across brain tissue slices requires objective, high-throughput, and spatially sensitive microglial quantification methods—a capability currently lacking in standard digital pathology protocols.

Recent baseline research at our institution by Presaizen (2026) established a 4-class soma-centric classification pipeline using YOLOv11 and DINOv2 on rat brain slices. While providing an initial proof-of-concept, Presaizen (2026) highlighted fundamental computational limitations: bounding-box object detectors ($64\times64$ crops) restrict analysis to the central soma, discarding up to 70% of distal process arborization. Consequently, bounding-box models suffer from biological misclassification between Resting and Resolution states and fail to detect shattered dystrophic microglia lacking a central soma anchor. Addressing these limitations through a topological, foundation-model-guided AI framework is essential for establishing unbiased biomarkers of neuroinflammation and optimizing PBM light therapy protocols.

---

## 2. Research Question and Hypothesis

### 2.1 Research Questions
* **RQ1**: Can foundation-model segmentation (e.g., Cellpose 3.0 / SAM-Microscopy) combined with Graph Neural Networks (GNNs) overcome soma-centric bounding box limitations to accurately capture fragmented microglial arborization?
* **RQ2**: Does incorporating spatial process topology resolve the persistent biological confusion between Resting and Resolution microglial states?
* **RQ3**: Can a continuous, multi-parametric activation index derived from graph topological representations provide superior sensitivity in quantifying Photobiomodulation (PBM) therapeutic response compared to standard discrete classification?

### 2.2 Research Hypothesis
It is hypothesized that transitioning from soma-centric bounding boxes to foundation-model-based polygonal segmentation (Cellpose 3.0 / SAM-Microscopy) and modeling the spatial neighborhood of fragmented distal processes using Graph Neural Networks (GNNs) will resolve the confusion between Resting and Resolution states by capturing the full cellular silhouette. Furthermore, by representing process fragments as nodes in a spatial proximity graph, the framework will significantly increase the recall of dystrophic/shattered cells lacking a unified soma anchor, yielding a sensitive, continuous activation index for PBM evaluation.

---

## 3. Research Objectives

* **Objective 1 (Dataset Re-annotation)**: Re-annotate the institutional benchmark dataset of 4,874 cells using fine polygonal masks to capture distal processes, beaded arborization, and shattered process fragments excluded by YOLO bounding boxes.
* **Objective 2 (Foundation Segmentation)**: Deploy and fine-tune foundation segmentation models (Cellpose 3.0 and SAM-Microscopy) to extract full microglial silhouettes without relying on a central soma anchor.
* **Objective 3 (Graph Topology Construction)**: Construct spatial proximity graphs connecting soma nodes and process fragment nodes, training a Graph Neural Network (GNN) to reconstruct shattered dystrophic cells into unified biological entities.
* **Objective 4 (Self-Supervised Feature Space)**: Implement a contrastive self-supervised representation learning space (DINOv2 / Masked Autoencoders) fine-tuned on segmented microglial masks to separate morphologically subtle activation states.
* **Objective 5 (Clinical & Experimental Validation)**: Formulate a continuous Multi-Parametric Activation Index (0–1 scale) and validate its sensitivity against experimental PBM-treated and LPS-challenged rodent TBI brain slices.

---

## 4. Literature Review

### 4.1 Microglia and Neuroinflammation
Microglia constitute 10%–15% of all glial cells in the CNS and serve as the frontline defense against traumatic, ischemic, and neurodegenerative insults. Under homeostatic conditions, surveilling microglia display an arborized morphology with delicate processes extending tens of micrometers from a small soma. Following TBI, inflammatory signaling cascades induce rapid structural metamorphosis: retracting processes, swelling somas, and transitioning into activated ameboid macrophages capable of phagocytosis.

### 4.2 Photobiomodulation (PBM) Therapy
Photobiomodulation (PBM) applies low-level light in the red to near-infrared spectrum (600–1000 nm) to modulate cellular function. In TBI models, PBM photon absorption by mitochondrial cytochrome c oxidase boosts ATP synthesis, attenuates reactive oxygen species (ROS), and shifts microglial polarization from pro-inflammatory (M1-like) to pro-resolving (M2-like) phenotypes. Quantifying these morphodynamic shifts across brain slices is essential for optimizing PBM therapeutic protocols.

### 4.3 Morphometric Analysis of Microglia
Traditional microglial morphometry relies on manual thresholding and skeletonization (e.g., ImageJ FracLac or HALO modules). Key quantitative metrics include fractal dimension ($D_f$), lacunarity, soma-to-cell area ratio, total process length, and number of branch endpoints. While informative, manual and parametric morphometry suffers from severe inter-rater variability, labor intensity, and failure in dense tissue slices with overlapping processes.

### 4.4 Automated Detection and Deep Learning
Recent computational advances have applied Convolutional Neural Networks (CNNs) and object detectors (YOLOv8, YOLOv11) to microglial quantification (Anwer et al., 2023; Morera et al., 2024; Hsu et al., 2025 - StainAI). While high-throughput bounding-box detectors excel at counting central somas, they cropped out distal process arborization ($64\times64$ crops), discarding up to 70% of the morphological information required to distinguish subtle functional states.

### 4.5 Foundation Models and Image Segmentation
Foundation models trained on millions of biological images—such as Cellpose 3.0 (Pachitariu & Stringer, 2024) and Segment Anything Model for Microscopy (SAM-Microscopy / SAM 2)—have revolutionized cellular segmentation. By predicting spatial gradient flows and vector fields, Cellpose segment zero-shot cell bodies and extended process branches without bounding-box constraints, providing the ideal input for topological analysis.

### 4.6 Graph Neural Networks (GNNs) in Cellular Topology
Graph Neural Networks (GNNs) model complex non-Euclidean spatial relationships. In neurobiology, representing segmented cellular somas and process fragments as nodes in a spatial proximity graph ($G = (V, E)$) enables Message Passing Neural Networks (MPNNs) or Graph Attention Networks (GATs) to learn topological connectivity. GNNs enable the reconstruction of "shattered" dystrophic microglia—a critical bottleneck in neurodegeneration research.

### 4.7 Literature Gap & Summary
Despite rapid progress in deep learning for digital pathology, existing microglial pipelines remain strictly soma-centric and bounding-box constrained. No current framework integrates foundation-model segmentation with spatial GNN topological modeling to resolve Resting vs. Resolution state confusion or reconstruct fragmented dystrophic cells. This project addresses this critical gap.

---

## 5. Preliminary Work and Study Rationale

This project builds directly upon the baseline M.Sc. thesis project completed at our institution by Tali Presaizen (Jan 2026), supervised by Dr. Sharon Yalov-Handzel and Dr. Lilach Gavish.

The baseline study established a soma-centric annotated dataset of 4,874 microglial cells extracted from phase-contrast and fluorescence microscopy images of PBM-treated rat brain slices. The dataset categorized cells into four discrete morphological states: Resting, Surveilling, Activated, and Resolution.

The baseline architecture utilized a YOLOv11 object detector paired with a DINOv2 Vision Transformer feature extractor for 4-class classification. While achieving respectable baseline metrics (mAP@0.5 ≈ 0.76, macro-F1 ≈ 0.69), the study revealed critical performance bottlenecks:

1. **Distal Information Loss**: Soma-centric $64\times64$ bounding boxes cropped out distal process branches, discarding 70% of morphological information.
2. **Resting vs. Resolution Ambiguity**: The confusion matrix revealed severe misclassification between Resting and Resolution states (F1 < 0.62), as both states share similar soma sizes but differ vastly in distal process topology.
3. **Failure on Dystrophic Microglia**: Dystrophic/shattered microglia in injured brain tissue lack a central soma anchor, causing YOLO to miss up to 45% of fragmented cellular entities.

These baseline findings provide the direct rationale for our proposed Topological AI Pipeline.

---

## 6. Proposed Methodology

### 6.1 Pipeline Architecture
The proposed framework consists of five integrated computational stages:
* **Stage 1 (Polygonal Re-annotation)**: Convert bounding boxes to pixel-wise polygonal masks capturing somas and distal process fragments.
* **Stage 2 (Foundation Segmentation)**: Fine-tune Cellpose 3.0 / SAM-Microscopy for whole-cell silhouette segmentation.
* **Stage 3 (GNN Topological Graph Construction)**: Construct spatial proximity k-NN graphs connecting somas and process fragment nodes.
* **Stage 4 (Self-Supervised Feature Learning)**: Train DINOv2 / Masked Autoencoders (MAE) on segmented masks for self-supervised feature extraction.
* **Stage 5 (Activation Index Computation)**: Aggregate graph embeddings into a continuous Multi-Parametric Activation Index (0–1 scale).

### 6.2 GNN Construction & Topological Feature Learning
Segmented cell somas ($V_{\text{soma}}$) and distal process fragments ($V_{\text{fragment}}$) are represented as graph nodes $V = V_{\text{soma}} \cup V_{\text{fragment}}$. Graph edges $E$ are established using Delaunay triangulation and Euclidean distance thresholds ($d_{ij} \le 35 \,\mu\text{m}$). Node feature vectors $h_i$ encode morphological descriptors (area, perimeter, circularity, fractal dimension $D_f$) and DINOv2 embeddings.

A Graph Attention Network (GATv2) with multi-head attention performs message passing to aggregate process neighborhood topology into node-level and graph-level representations $h_G = \text{Readout}(\{h_i\})$.

---

## 7. Work Plan and Project Stages

The proposed research will be executed across six structured project stages over a total estimated duration of **26 weeks (~6.5 months)**, adhering to the departmental proposal guidelines:

### Stage 1 – Dataset Re-annotation & Preprocessing
* **Task**: Re-annotate the institutional benchmark dataset of 4,874 microglial cells using fine polygonal masks in CVAT/Labelme, capturing somas, distal processes, and beaded fragments excluded by YOLO bounding boxes.
* **Goal**: Establish a high-quality, fragment-first annotated polygonal dataset for supervised foundation model training.
* **Estimated Duration**: 4 weeks

### Stage 2 – Foundation Model Fine-Tuning & Segmentation
* **Task**: Implement and fine-tune Cellpose 3.0 and SAM-Microscopy on the polygonal dataset. Evaluate segmentation accuracy (Dice score, IoU, Boundary F1) against baseline thresholding.
* **Goal**: Deploy a robust zero-shot foundation segmentation pipeline that extracts complete microglial silhouettes.
* **Estimated Duration**: 4 weeks

### Stage 3 – Graph Neural Network (GNN) Construction & Topological Modeling
* **Task**: Construct spatial k-NN and Delaunay proximity graphs connecting somas and process fragment nodes. Implement and train GATv2/MPNN GNN architectures for topological message passing.
* **Goal**: Reconstruct shattered dystrophic microglia into single biological entities and capture full arborization topology.
* **Estimated Duration**: 5 weeks

### Stage 4 – Self-Supervised Contrastive Feature Learning
* **Task**: Pre-train and fine-tune DINOv2 Vision Transformer and Masked Autoencoder (MAE) backbones on segmented microglial masks to build a self-supervised morphological feature space.
* **Goal**: Generate rich, low-dimensional morphological embeddings robust to staining and optical variations.
* **Estimated Duration**: 4 weeks

### Stage 5 – Multi-Parametric Activation Index & Model Evaluation
* **Task**: Develop a continuous Multi-Parametric Activation Index (0–1 scale) aggregating graph embeddings. Perform comprehensive ablation studies comparing against YOLOv11+DINOv2 baselines.
* **Goal**: Validate model performance across discrete classification (4-class F1) and continuous activation scoring.
* **Estimated Duration**: 5 weeks

### Stage 6 – Thesis Writing, Validation & Defense Preparation
* **Task**: Perform statistical sensitivity validation on PBM-treated TBI rat brain slices, write the final M.Sc. thesis document, prepare peer-reviewed publication manuscripts, and defend the thesis.
* **Goal**: Complete and submit the final M.Sc. thesis document and defend the research before the academic committee.
* **Estimated Duration**: 4 weeks

---

## 8. Evaluation Plan and Benchmarking

The framework will be evaluated across three complementary quantitative tiers:
1. **Segmentation Metrics**: Dice Coefficient, Intersection over Union (IoU), and Boundary-F1 score compared against manual polygonal ground truth.
2. **Morphological Classification**: Macro-F1 score, Per-class Precision/Recall, and Confusion Matrix analysis across Resting, Surveilling, Activated, and Resolution states (benchmarked against YOLOv11 baseline F1=0.69).
3. **Biological & Clinical Sensitivity**: Pearson correlation ($r$) and Spearman rank ($\rho$) between the computed Activation Index (0–1) and biological PBM light dosage ($\text{J/cm}^2$) across TBI tissue slices.

---

## 9. Expected Scientific and Technological Contribution

1. **Novel Methodological Paradigm**: First AI framework combining foundation-model segmentation with Graph Neural Networks for microglial morphometry.
2. **Dystrophic Microglia Reconstruction**: Solves the critical bottleneck of detecting shattered dystrophic microglia lacking a central soma anchor.
3. **Open-Source Benchmark Dataset**: Provides an open-source, fragment-first polygonal dataset of 4,874 microglial cells for the research community.
4. **Translational Neurobiology Impact**: Delivers a scalable, reproducible tool for quantifying Photobiomodulation (PBM) neuroprotective efficacy in TBI.

---

## 10. References

1. Anwer, D. M., Gubinelli, F., Kurt, Y. A., et al. (2023). A comparison of machine learning approaches for the quantification of microglial cells in the brain of mice, rats and non-human primates. *PLOS ONE*, 18(4), e0284480.
2. Dewan, M. C., Rattani, A., Gupta, S., et al. (2018). Estimating the global incidence of traumatic brain injury. *Journal of Neurosurgery*, 130(4), 1080-1097.
3. Gavish, L., & Houreld, N. N. (2019). Therapeutic Efficacy of Photobiomodulation (PBM) in Wound Healing and Neuroinflammation. *Photomedicine and Laser Surgery*, 37(3), 150-162.
4. Hamblin, M. R. (2018). Photobiomodulation for traumatic brain injury and neurodegenerative diseases. *Photonics & Lasers in Medicine*, 7(3), 231-244.
5. Hoge, C. W., McGurk, D., Thomas, J. L., et al. (2008). Mild traumatic brain injury in U.S. Soldiers returning from Iraq. *New England Journal of Medicine*, 358(5), 453-463.
6. Hsu, C.-H., Hsu, Y.-Y., Chang, B.-M., et al. (2025). StainAI: quantitative mapping of stained microglia and insights into brain-wide neuroinflammation and therapeutic effects in cardiac arrest. *Communications Biology*, 8, 7926.
7. Kim, J., Pavlidis, P., & Vogel Ciernia, A. (2024). Development of a High-Throughput Pipeline to Characterize Microglia Morphological States at a Single-Cell Resolution. *eNeuro*, 11(6), ENEURO.0010-24.2024.
8. Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 4015-4026.
9. Leyh, J., Schafer, M. K., et al. (2021). Microglial morphodynamics in traumatic brain injury and recovery. *Glia*, 69(8), 1950-1965.
10. Maas, A. I., Menon, D. K., Adelson, P. D., et al. (2017). Traumatic brain injury: integrated approaches to improve prevention, clinical care, and research. *The Lancet Neurology*, 16(12), 987-1048.
11. Morera, H., Dave, P., Kolinko, Y., et al. (2024). A novel deep learning-based method for automatic stereology of microglia cells from low magnification images. *Neurotoxicology and Teratology*, 102, 107336.
12. Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning Robust Visual Features Without Supervision. *arXiv preprint arXiv:2304.07193*.
13. Pachitariu, M., & Stringer, C. (2024). Cellpose 3.0: accurate segmentation of biological images using foundation models. *Nature Methods*, 21(4), 701-710.
14. Presaizen, T. (2026). *AI-Powered Microglial Classification for Activation Scoring*. Master's Thesis, School of Data Science: Intelligent Systems, Afeka Academic College of Engineering & Hebrew University of Jerusalem.
15. Salter, M. W., & Beggs, S. (2014). Sublime microglia: expanding roles for the guardians of the CNS. *Cell*, 158(1), 15-24.
16. Veličković, P., Cucurull, G., Casanova, A., et al. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.
17. Wolf, S. A., Boddeke, H. W., & Kettenmann, H. (2017). Microglia in Physiology and Pathology. *Physiological Reviews*, 97(4), 1339-1393.
18. Xiong, H., Zheng, S., Qi, X., Liu, J. (2025). μGlia-Flow, an automatic workflow for microglia segmentation and classification. *Journal of Neuroscience Methods*, 402, 110022.
19. Zähringer, A., Vinnakota, J. M., Wertheimer, T., et al. (2025). AIstain: Enhancing microglial phagocytosis analysis through deep learning. *Cell Reports Methods*, 5(11), 101207.
