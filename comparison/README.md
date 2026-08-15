# MicroGlia Method Comparison: Baseline vs. Boundary Sharpening

Complete, self-contained standalone hierarchy for comparing **Alternative 1: Baseline (Presaizen 2026)** against **Alternative 2: Proposed Boundary Sharpening Pipeline**.

---

## 📊 Summary of Quantitative Results

| Microscopy Slide ID | Baseline (Presaizen 2026) | Proposed Boundary Sharpening | Delta Yield | % Gain |
| :--- | :---: | :---: | :---: | :---: |
| `JPG_VID2724_B1_3_00d07h00m` | 97 cells | **95 sub-cells** | $-2$ (clean deduplication) | $-2.06\%$ |
| `VID2724_A1_9_00d07h00m` | 241 cells | **261 sub-cells** | $+20$ | $+8.30\%$ |
| `VID2724_A3_4_00d00h00m` | 269 cells | **287 sub-cells** | $+18$ | $+6.69\%$ |
| `VID2724_A3_4_00d07h00m` | 282 cells | **290 sub-cells** | $+8$ | $+2.84\%$ |
| `VID2724_B1_3_00d00h00m` | 102 cells | **103 sub-cells** | $+1$ | $+0.98\%$ |
| **TOTAL DATASET** | **991 cells** | **1,036 sub-cells** | **+45 sub-cells** | **+4.54%** |

---

## 📂 Directory Structure

```
comparison/
├── README.md                          # Documentation & comparative analysis
├── run.py                             # Generate side-by-side comparison grids & gallery
├── gallery.py                         # Rebuild comparison gallery
├── src/                               # Implementation modules
│   ├── compare_baseline_vs_sharpening.py # Comparative engine & metrics computation
│   └── build_comparison_gallery.py      # Standalone HTML gallery builder
└── output/                            # Output comparison grids & gallery
    ├── gallery.html                   # Interactive side-by-side browser gallery
    ├── comparison_summary.json        # Quantitative stats JSON
    ├── comparison_summary.csv         # Quantitative stats CSV
    ├── JPG_VID2724_B1_3_00d07h00m_comparison.jpg
    ├── VID2724_A1_9_00d07h00m_comparison.jpg
    ├── VID2724_A3_4_00d00h00m_comparison.jpg
    ├── VID2724_A3_4_00d07h00m_comparison.jpg
    └── VID2724_B1_3_00d00h00m_comparison.jpg
```

---

## 🚀 Quick Start Commands

### 1. Run Complete Comparison & Rebuild Grids
```bash
python3 comparison/run.py
```

### 2. Open the Interactive Comparison Gallery
Open [**`comparison/output/gallery.html`**](output/gallery.html) in your browser to inspect the side-by-side whole-slide maps with KPI metrics and interactive slide switching.
