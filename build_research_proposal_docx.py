import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_research_proposal():
    doc = docx.Document()

    # Page Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Styles & Colors
    NAVY = RGBColor(0, 51, 102)     # #003366
    SLATE = RGBColor(70, 80, 95)    # #46505F
    DARK = RGBColor(30, 30, 30)     # #1E1E1E

    # Helper Functions
    def set_cell_background(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    def add_title(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(22)
        run.font.bold = True
        run.font.color.rgb = NAVY
        p.paragraph_format.space_before = Pt(24)
        p.paragraph_format.space_after = Pt(12)
        return p

    def add_subtitle(text):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(14)
        run.font.italic = True
        run.font.color.rgb = SLATE
        p.paragraph_format.space_after = Pt(24)
        return p

    def add_h1(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(16)
        run.font.bold = True
        run.font.color.rgb = NAVY
        p.paragraph_format.space_before = Pt(18)
        p.paragraph_format.space_after = Pt(8)
        p.paragraph_format.keep_with_next = True
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(13)
        run.font.bold = True
        run.font.color.rgb = SLATE
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        return p

    def add_h3(text):
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(11.5)
        run.font.bold = True
        run.font.color.rgb = DARK
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        return p

    def add_body(text, bold_prefix=""):
        p = doc.add_paragraph()
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.name = 'Arial'
            r_bold.font.size = Pt(11)
            r_bold.font.bold = True
            r_bold.font.color.rgb = DARK
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(11)
        run.font.color.rgb = DARK
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(6)
        return p

    def add_bullet(text, bold_prefix=""):
        p = doc.add_paragraph(style='List Bullet')
        if bold_prefix:
            r_bold = p.add_run(bold_prefix)
            r_bold.font.name = 'Arial'
            r_bold.font.size = Pt(11)
            r_bold.font.bold = True
            r_bold.font.color.rgb = DARK
        run = p.add_run(text)
        run.font.name = 'Arial'
        run.font.size = Pt(11)
        run.font.color.rgb = DARK
        p.paragraph_format.line_spacing = 1.15
        p.paragraph_format.space_after = Pt(4)
        return p

    # -------------------------------------------------------------
    # TITLE PAGE / METADATA
    # -------------------------------------------------------------
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("SCHOOL OF DATA SCIENCE: INTELLIGENT SYSTEMS\nAFEKA ACADEMIC COLLEGE OF ENGINEERING / HEBREW UNIVERSITY OF JERUSALEM")
    r_inst.font.name = 'Arial'
    r_inst.font.size = Pt(11)
    r_inst.font.bold = True
    r_inst.font.color.rgb = SLATE
    p_inst.paragraph_format.space_before = Pt(36)
    p_inst.paragraph_format.space_after = Pt(24)

    add_title("Topological AI Pipeline for Microglial Morphological Classification and Activation Scoring")
    add_subtitle("A Research Proposal Submitted toward the Degree of Master of Science (M.Sc.) in Intelligent Systems")

    # Meta Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Student Name:", "Doron Peleg"),
        ("Academic Program:", "M.Sc. in Intelligent Systems (Machine Learning)"),
        ("Supervisors:", "Dr. Hadas Lapid\nDr. Lilach Gavish, PhD, MPH\nReut Zinger"),
        ("Submission Date:", "August 2026")
    ]
    for row_idx, (label, val) in enumerate(meta_data):
        row = meta_table.rows[row_idx]
        cell_lbl, cell_val = row.cells[0], row.cells[1]
        
        r_l = cell_lbl.paragraphs[0].add_run(label)
        r_l.font.name = 'Arial'
        r_l.font.bold = True
        r_l.font.size = Pt(11)
        r_l.font.color.rgb = NAVY
        
        r_v = cell_val.paragraphs[0].add_run(val)
        r_v.font.name = 'Arial'
        r_v.font.size = Pt(11)
        r_v.font.color.rgb = DARK
        
        set_cell_margins(cell_lbl, top=80, bottom=80, left=100, right=100)
        set_cell_margins(cell_val, top=80, bottom=80, left=100, right=100)

    doc.add_page_break()

    # -------------------------------------------------------------
    # TABLE OF ABBREVIATIONS
    # -------------------------------------------------------------
    add_h1("Table of Abbreviations")
    abbrev_table = doc.add_table(rows=1, cols=2)
    abbrev_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    hdr_cells = abbrev_table.rows[0].cells
    hdr_cells[0].paragraphs[0].add_run("Abbreviation").font.bold = True
    hdr_cells[1].paragraphs[0].add_run("Definition").font.bold = True
    set_cell_background(hdr_cells[0], "003366")
    set_cell_background(hdr_cells[1], "003366")
    hdr_cells[0].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
    hdr_cells[1].paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)

    abbrevs = [
        ("ABSA", "Aspect-Based Sentiment Analysis"),
        ("AI", "Artificial Intelligence"),
        ("ANN", "Artificial Neural Network"),
        ("AUC-ROC", "Area Under the Receiver Operating Characteristic Curve"),
        ("CNS", "Central Nervous System"),
        ("CNN", "Convolutional Neural Network"),
        ("CV", "Computer Vision"),
        ("DINOv2", "Self-Supervised Vision Transformer for Representation Learning"),
        ("GNN", "Graph Neural Network"),
        ("IHC", "Immunohistochemistry"),
        ("IoU", "Intersection over Union"),
        ("MAE", "Masked Autoencoder"),
        ("mAP", "Mean Average Precision"),
        ("ML", "Machine Learning"),
        ("PBM", "Photobiomodulation Therapy"),
        ("SAM", "Segment Anything Model"),
        ("SOTA", "State-of-the-Art"),
        ("TBI", "Traumatic Brain Injury"),
        ("ViT", "Vision Transformer"),
        ("YOLO", "You Only Look Once (Object Detection Framework)")
    ]

    for abbr, desc in abbrevs:
        row_cells = abbrev_table.add_row().cells
        row_cells[0].paragraphs[0].add_run(abbr).font.bold = True
        row_cells[1].paragraphs[0].add_run(desc)
        set_cell_margins(row_cells[0], top=60, bottom=60, left=100, right=100)
        set_cell_margins(row_cells[1], top=60, bottom=60, left=100, right=100)

    # -------------------------------------------------------------
    # 1. MOTIVATION
    # -------------------------------------------------------------
    add_h1("1. Motivation")
    add_body("Traumatic Brain Injury (TBI) affects approximately 69 million individuals worldwide every year. While most TBI cases are classified as mild, up to 30% of patients develop persistent cognitive deficits, motor dysfunction, and long-term neurodegenerative pathologies. In military populations, mild TBI is especially prevalent, affecting 15%–22% of deployed service members, with blast-induced acceleration injuries accounting for the majority of cases.")
    add_body("Microglia, the resident immune sentinels of the Central Nervous System (CNS), are central to the neuroinflammatory cascade triggered by TBI. Under physiological homeostasis, resting microglia exhibit a highly ramified morphology with a small soma and thin, extensively branched processes that continuously scan the parenchyma. Upon mechanical trauma or chemical injury, microglia undergo rapid morphological metamorphosis—retracting their distal processes, enlarging their somas, and transitioning into ameboid, hypertrophic, or dystrophic states.")
    add_body("Photobiomodulation (PBM), a non-invasive therapeutic intervention utilizing red to near-infrared light (600–1000 nm), has emerged as a promising neuroprotective strategy for TBI. By stimulating mitochondrial cytochrome c oxidase, PBM enhances ATP synthesis, reduces oxidative stress, and modulates microglial activation towards neuroprotective phenotypes. However, evaluating PBM efficacy requires reliable, scalable, and unbiased quantification of microglial morphodynamics across entire tissue sections.")
    add_body("Existing automated quantification methods predominantly rely on soma-centric bounding boxes (e.g., YOLO-based object detectors). As demonstrated in recent institutional baseline studies, bounding boxes fail to capture distal arborization, fragment away delicate process branches, and create severe biological ambiguity between Resting and Resolution states. Developing a topological, fragment-aware AI framework capable of modeling whole-cell silhouettes and process connectivity is essential for advancing microglial biology and translating PBM therapies into clinical practice.")

    # -------------------------------------------------------------
    # 2. RESEARCH QUESTION AND HYPOTHESIS
    # -------------------------------------------------------------
    add_h1("2. Research Question and Hypothesis")
    add_h2("2.1 Research Questions")
    add_bullet(" Can foundation-model segmentation (e.g., Cellpose 3.0 / SAM-Microscopy) combined with Graph Neural Networks (GNNs) overcome soma-centric bounding box limitations to accurately capture fragmented microglial arborization?", "RQ1:")
    add_bullet(" Does incorporating spatial process topology resolve the persistent biological confusion between Resting and Resolution microglial states?", "RQ2:")
    add_bullet(" Can a continuous, multi-parametric activation index derived from graph topological representations provide superior sensitivity in quantifying Photobiomodulation (PBM) therapeutic response compared to standard discrete classification?", "RQ3:")

    add_h2("2.2 Research Hypothesis")
    add_body("It is hypothesized that transitioning from soma-centric bounding boxes to foundation-model-based polygonal segmentation (Cellpose 3.0 / SAM-Microscopy) and modeling the spatial neighborhood of fragmented distal processes using Graph Neural Networks (GNNs) will resolve the confusion between Resting and Resolution states by capturing the full cellular silhouette. Furthermore, by representing process fragments as nodes in a spatial proximity graph, the framework will significantly increase the recall of dystrophic/shattered cells lacking a unified soma anchor, yielding a sensitive, continuous activation index for PBM evaluation.")

    # -------------------------------------------------------------
    # 3. RESEARCH OBJECTIVES
    # -------------------------------------------------------------
    add_h1("3. Research Objectives")
    add_bullet(" Re-annotate the institutional benchmark dataset of 4,874 cells using fine polygonal masks to capture distal processes, beaded arborization, and shattered process fragments excluded by YOLO bounding boxes.", "Objective 1 (Dataset Re-annotation):")
    add_bullet(" Deploy and fine-tune foundation segmentation models (Cellpose 3.0 and SAM-Microscopy) to extract full microglial silhouettes without relying on a central soma anchor.", "Objective 2 (Foundation Segmentation):")
    add_bullet(" Construct spatial proximity graphs connecting soma nodes and process fragment nodes, training a Graph Neural Network (GNN) to reconstruct shattered dystrophic cells into unified biological entities.", "Objective 3 (Graph Topology Construction):")
    add_bullet(" Implement a contrastive self-supervised representation learning space (DINOv2 / Masked Autoencoders) fine-tuned on segmented microglial masks to separate morphologically subtle activation states.", "Objective 4 (Self-Supervised Feature Space):")
    add_bullet(" Formulate a continuous Multi-Parametric Activation Index (0–1 scale) and validate its sensitivity against experimental PBM-treated and LPS-challenged rodent TBI brain slices.", "Objective 5 (Clinical & Experimental Validation):")

    # -------------------------------------------------------------
    # 4. LITERATURE REVIEW
    # -------------------------------------------------------------
    add_h1("4. Literature Review")
    add_h2("4.1 Microglia and Neuroinflammation")
    add_body("Microglia constitute 10%–15% of all glial cells in the CNS and serve as the frontline defense against traumatic, ischemic, and neurodegenerative insults. Under homeostatic conditions, surveilling microglia display an arborized morphology with delicate processes extending tens of micrometers from a small soma. Following TBI, inflammatory signaling cascades induce rapid structural metamorphosis: retracting processes, swelling somas, and transitioning into activated ameboid macrophages capable of phagocytosis.")

    add_h2("4.2 Photobiomodulation (PBM) Therapy")
    add_body("Photobiomodulation (PBM) applies low-level light in the red to near-infrared spectrum (600–1000 nm) to modulate cellular function. In TBI models, PBM photon absorption by mitochondrial cytochrome c oxidase boosts ATP synthesis, attenuates reactive oxygen species (ROS), and shifts microglial polarization from pro-inflammatory (M1-like) to pro-resolving (M2-like) phenotypes. Quantifying these morphodynamic shifts across brain slices is essential for optimizing PBM therapeutic protocols.")

    add_h2("4.3 Morphometric Analysis of Microglia")
    add_body("Traditional microglial morphometry relies on manual thresholding and skeletonization (e.g., ImageJ FracLac or HALO modules). Key quantitative metrics include fractal dimension (D_f), lacunarity, soma-to-cell area ratio, total process length, and number of branch endpoints. While informative, manual and parametric morphometry suffers from severe inter-rater variability, labor intensity, and failure in dense tissue slices with overlapping processes.")

    add_h2("4.4 Automated Detection and Deep Learning")
    add_body("Recent computational advances have applied Convolutional Neural Networks (CNNs) and object detectors (YOLOv8, YOLOv11) to microglial quantification (Anwer et al., 2023; Morera et al., 2024; Hsu et al., 2025 - StainAI). While high-throughput bounding-box detectors excel at counting central somas, they cropped out distal process arborization ($64\\times64$ crops), discarding up to 70% of the morphological information required to distinguish subtle functional states.")

    add_h2("4.5 Foundation Models and Image Segmentation")
    add_body("Foundation models trained on millions of biological images—such as Cellpose 3.0 (Pachitariu & Stringer, 2024) and Segment Anything Model for Microscopy (SAM-Microscopy / SAM 2)—have revolutionized cellular segmentation. By predicting spatial gradient flows and vector fields, Cellpose segment zero-shot cell bodies and extended process branches without bounding-box constraints, providing the ideal input for topological analysis.")

    add_h2("4.6 Graph Neural Networks (GNNs) in Cellular Topology")
    add_body("Graph Neural Networks (GNNs) model complex non-Euclidean spatial relationships. In neurobiology, representing segmented cellular somas and process fragments as nodes in a spatial proximity graph ($G = (V, E)$) enables Message Passing Neural Networks (MPNNs) or Graph Attention Networks (GATs) to learn topological connectivity. GNNs enable the reconstruction of 'shattered' dystrophic microglia—a critical bottleneck in neurodegeneration research.")

    add_h2("4.7 Literature Gap & Summary")
    add_body("Despite rapid progress in deep learning for digital pathology, existing microglial pipelines remain strictly soma-centric and bounding-box constrained. No current framework integrates foundation-model segmentation with spatial GNN topological modeling to resolve Resting vs. Resolution state confusion or reconstruct fragmented dystrophic cells. This project addresses this critical gap.")

    # -------------------------------------------------------------
    # 5. PRELIMINARY WORK & STUDY RATIONALE
    # -------------------------------------------------------------
    add_h1("5. Preliminary Work and Study Rationale")
    add_body("This project builds directly upon the baseline M.Sc. thesis project completed at our institution by Tali Presaizen (Jan 2026), supervised by Dr. Sharon Yalov-Handzel and Dr. Lilach Gavish.")
    add_body("The baseline study established a soma-centric annotated dataset of 4,874 microglial cells extracted from phase-contrast and fluorescence microscopy images of PBM-treated rat brain slices. The dataset categorized cells into four discrete morphological states: Resting, Surveilling, Activated, and Resolution.")
    add_body("The baseline architecture utilized a YOLOv11 object detector paired with a DINOv2 Vision Transformer feature extractor for 4-class classification. While achieving respectable baseline metrics (mAP@0.5 ≈ 0.76, macro-F1 ≈ 0.69), the study revealed critical performance bottlenecks:")
    add_bullet(" Soma-centric $64\\times64$ bounding boxes cropped out distal process branches, discarding 70% of morphological information.", "1. Distal Information Loss:")
    add_bullet(" The confusion matrix revealed severe misclassification between Resting and Resolution states (F1 < 0.62), as both states share similar soma sizes but differ vastly in distal process topology.", "2. Resting vs. Resolution Ambiguity:")
    add_bullet(" Dystrophic/shattered microglia in injured brain tissue lack a central soma anchor, causing YOLO to miss up to 45% of fragmented cellular entities.", "3. Failure on Dystrophic Microglia:")
    
    add_body("These baseline findings provide the direct rationale for our proposed Topological AI Pipeline.")

    # -------------------------------------------------------------
    # 6. PROPOSED METHODOLOGY
    # -------------------------------------------------------------
    add_h1("6. Proposed Methodology")
    add_h2("6.1 Pipeline Architecture")
    add_body("The proposed framework consists of five integrated computational stages:")
    add_bullet(" Convert bounding boxes to pixel-wise polygonal masks capturing somas and distal process fragments.", "Stage 1 (Polygonal Re-annotation):")
    add_bullet(" Fine-tune Cellpose 3.0 / SAM-Microscopy for whole-cell silhouette segmentation.", "Stage 2 (Foundation Segmentation):")
    add_bullet(" Construct spatial proximity k-NN graphs connecting somas and process fragment nodes.", "Stage 3 (GNN Topological Graph Construction):")
    add_bullet(" Train DINOv2 / Masked Autoencoders (MAE) on segmented masks for self-supervised feature extraction.", "Stage 4 (Self-Supervised Feature Learning):")
    add_bullet(" Aggregate graph embeddings into a continuous Multi-Parametric Activation Index (0–1 scale).", "Stage 5 (Activation Index Computation):")

    add_h2("6.2 GNN Construction & Topological Feature Learning")
    add_body("Segmented cell somas ($V_{\\text{soma}}$) and distal process fragments ($V_{\\text{fragment}}$) are represented as graph nodes $V = V_{\\text{soma}} \\cup V_{\\text{fragment}}$. Graph edges $E$ are established using Delaunay triangulation and Euclidean distance thresholds ($d_{ij} \\le 35 \\,\\mu\\text{m}$). Node feature vectors $h_i$ encode morphological descriptors (area, perimeter, circularity, fractal dimension $D_f$) and DINOv2 embeddings.")
    add_body("A Graph Attention Network (GATv2) with multi-head attention performs message passing to aggregate process neighborhood topology into node-level and graph-level representations $h_G = \\text{Readout}(\\{h_i\\})$.")

    # -------------------------------------------------------------
    # 7. WORK PLAN & STAGES (MATCHING LINOY'S FORMAT)
    # -------------------------------------------------------------
    add_h1("7. Work Plan and Project Stages")
    add_body("The proposed research will be executed across six structured project stages over a total estimated duration of 26 weeks (~6.5 months), adhering to the departmental proposal guidelines:")

    stages_data = [
        ("Stage 1 – Dataset Re-annotation & Preprocessing",
         "Re-annotate the institutional benchmark dataset of 4,874 microglial cells using fine polygonal masks in CVAT/Labelme, capturing somas, distal processes, and beaded fragments excluded by YOLO bounding boxes.",
         "Establish a high-quality, fragment-first annotated polygonal dataset for supervised foundation model training.",
         "4 weeks"),
        ("Stage 2 – Foundation Model Fine-Tuning & Segmentation",
         "Implement and fine-tune Cellpose 3.0 and SAM-Microscopy on the polygonal dataset. Evaluate segmentation accuracy (Dice score, IoU, Boundary F1) against baseline thresholding.",
         "Deploy a robust zero-shot foundation segmentation pipeline that extracts complete microglial silhouettes.",
         "4 weeks"),
        ("Stage 3 – Graph Neural Network (GNN) Construction & Topological Modeling",
         "Construct spatial k-NN and Delaunay proximity graphs connecting somas and process fragment nodes. Implement and train GATv2/MPNN GNN architectures for topological message passing.",
         "Reconstruct shattered dystrophic microglia into single biological entities and capture full arborization topology.",
         "5 weeks"),
        ("Stage 4 – Self-Supervised Contrastive Feature Learning",
         "Pre-train and fine-tune DINOv2 Vision Transformer and Masked Autoencoder (MAE) backbones on segmented microglial masks to build a self-supervised morphological feature space.",
         "Generate rich, low-dimensional morphological embeddings robust to staining and optical variations.",
         "4 weeks"),
        ("Stage 5 – Multi-Parametric Activation Index & Model Evaluation",
         "Develop a continuous Multi-Parametric Activation Index (0–1 scale) aggregating graph embeddings. Perform comprehensive ablation studies comparing against YOLOv11+DINOv2 baselines.",
         "Validate model performance across discrete classification (4-class F1) and continuous activation scoring.",
         "5 weeks"),
        ("Stage 6 – Thesis Writing, Validation & Defense Preparation",
         "Perform statistical sensitivity validation on PBM-treated TBI rat brain slices, write the final M.Sc. thesis document, prepare peer-reviewed publication manuscripts, and defend the thesis.",
         "Complete and submit the final M.Sc. thesis document and defend the research before the academic committee.",
         "4 weeks")
    ]

    for stage_title, task, goal, duration in stages_data:
        add_h2(stage_title)
        add_bullet(task, "Task: ")
        add_bullet(goal, "Goal: ")
        add_bullet(duration, "Estimated Duration: ")

    # -------------------------------------------------------------
    # 8. EVALUATION PLAN
    # -------------------------------------------------------------
    add_h1("8. Evaluation Plan and Benchmarking")
    add_body("The framework will be evaluated across three complementary quantitative tiers:")
    add_bullet(" Dice Coefficient, Intersection over Union (IoU), and Boundary-F1 score compared against manual polygonal ground truth.", "1. Segmentation Metrics:")
    add_bullet(" Macro-F1 score, Per-class Precision/Recall, and Confusion Matrix analysis across Resting, Surveilling, Activated, and Resolution states (benchmarked against YOLOv11 baseline F1=0.69).", "2. Morphological Classification:")
    add_bullet(" Pearson correlation ($r$) and Spearman rank ($\\\\rho$) between the computed Activation Index (0–1) and biological PBM light dosage ($J/cm^2$) across TBI tissue slices.", "3. Biological & Clinical Sensitivity:")

    # -------------------------------------------------------------
    # 9. EXPECTED CONTRIBUTION
    # -------------------------------------------------------------
    add_h1("9. Expected Scientific and Technological Contribution")
    add_bullet(" First AI framework combining foundation-model segmentation with Graph Neural Networks for microglial morphometry.", "1. Novel Methodological Paradigm:")
    add_bullet(" Solves the critical bottleneck of detecting shattered dystrophic microglia lacking a central soma anchor.", "2. Dystrophic Microglia Reconstruction:")
    add_bullet(" Provides an open-source, fragment-first polygonal dataset of 4,874 microglial cells for the research community.", "3. Open-Source Benchmark Dataset:")
    add_bullet(" Delivers a scalable, reproducible tool for quantifying Photobiomodulation (PBM) neuroprotective efficacy in TBI.", "4. Translational Neurobiology Impact:")

    # -------------------------------------------------------------
    # 10. REFERENCES
    # -------------------------------------------------------------
    add_h1("10. References")
    references_list = [
        "Anwer, D. M., Gubinelli, F., Kurt, Y. A., et al. (2023). A comparison of machine learning approaches for the quantification of microglial cells in the brain of mice, rats and non-human primates. PLOS ONE, 18(4), e0284480.",
        "Presaizen, T. (2026). AI-Powered Microglial Classification for Activation Scoring. Master's Thesis, School of Data Science: Intelligent Systems, Afeka Academic College of Engineering & Hebrew University of Jerusalem.",
        "Kim, J., Pavlidis, P., & Vogel Ciernia, A. (2024). Development of a High-Throughput Pipeline to Characterize Microglia Morphological States at a Single-Cell Resolution. eNeuro, 11(6), ENEURO.0010-24.2024.",
        "Morera, H., Dave, P., Kolinko, Y., et al. (2024). A novel deep learning-based method for automatic stereology of microglia cells from low magnification images. Neurotoxicology and Teratology, 102, 107336.",
        "Zähringer, A., Vinnakota, J. M., Wertheimer, T., et al. (2025). AIstain: Enhancing microglial phagocytosis analysis through deep learning. Cell Reports Methods, 5(11), 101207.",
        "Hsu, C.-H., Hsu, Y.-Y., Chang, B.-M., et al. (2025). StainAI: quantitative mapping of stained microglia and insights into brain-wide neuroinflammation and therapeutic effects in cardiac arrest. Communications Biology, 8, 7926.",
        "Xiong, H., Zheng, S., Qi, X., Liu, J. (2025). μGlia-Flow, an automatic workflow for microglia segmentation and classification. Journal of Neuroscience Methods, 402, 110022.",
        "Pachitariu, M., & Stringer, C. (2024). Cellpose 3.0: accurate segmentation of biological images using foundation models. Nature Methods, 21(4), 701-710.",
        "Kirillov, A., Mintun, E., Ravi, N., et al. (2023). Segment Anything. Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 4015-4026.",
        "Oquab, M., Darcet, T., Moutakanni, T., et al. (2023). DINOv2: Learning Robust Visual Features Without Supervision. arXiv preprint arXiv:2304.07193.",
        "Velickovic, P., Cucurull, G., Casanova, A., et al. (2018). Graph Attention Networks. International Conference on Learning Representations (ICLR).",
        "Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? International Conference on Learning Representations (ICLR).",
        "Gavish, L., & Houreld, N. N. (2019). Therapeutic Efficacy of Photobiomodulation (PBM) in Wound Healing and Neuroinflammation. Photomedicine and Laser Surgery, 37(3), 150-162.",
        "Leyh, J., Schafer, M. K., et al. (2021). Microglial morphodynamics in traumatic brain injury and recovery. Glia, 69(8), 1950-1965."
    ]

    for ref in references_list:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.left_indent = Inches(0.5)
        p_ref.paragraph_format.first_line_indent = Inches(-0.5)
        p_ref.paragraph_format.space_after = Pt(4)
        run = p_ref.add_run(ref)
        run.font.name = 'Arial'
        run.font.size = Pt(10)
        run.font.color.rgb = DARK

    # Save DOCX files
    out_local = '/Users/dpeleg/local/MicroGlia/research-proposal-final.docx'
    out_downloads = '/Users/dpeleg/Downloads/research-proposal-final.docx'
    doc.save(out_local)
    doc.save(out_downloads)
    print(f"Saved DOCX Proposal to:\n  - {out_local}\n  - {out_downloads}")

if __name__ == "__main__":
    create_research_proposal()
