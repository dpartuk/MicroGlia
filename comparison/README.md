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

### 3. Build the Master Gallery (one page, a tab for every gallery)
Build each method's gallery first. Galleries that do not exist yet are left out.
```bash
python3 comparison/master_gallery.py
```
Open [**`comparison/output/master_gallery.html`**](output/master_gallery.html). Click a tab, or press Alt + 1–9, to switch galleries.
Each tab keeps its slide and cell position when you switch away and back. Add `#baseline`, `#sharpened`, `#contour`, `#dpgnet` or `#explorer` to the URL to open a specific tab.

### 4. Build the Multi-Method Explorer (all four methods, cell by cell)
Run each method's `process.py` first. The explorer reads every available `<method>/output/<slide>/manifest.csv`.
```bash
python3 comparison/methods_gallery.py
```
Open [**`comparison/output/methods_gallery.html`**](output/methods_gallery.html) to:
* Overlay the cell outlines of any combination of methods on each slide.
* Select two methods (A vs B) and colour each cell of A as matched, split, merged, or only in A.
* Click a cell to see its crop and the matching crops from every other method, with IoU.
* Open the **Cell table** view: one row per ground-truth (Contour) cell, compared across all methods. Click a row to locate the cell on the slide.

The same command writes the table to [`comparison/output/cell_by_cell_comparison.csv`](output/cell_by_cell_comparison.csv).
Each method has four columns: `status` (`matched`, `split`, `merged`, `missed`, or `extra` for a cell that is not in the ground truth), `cell_ids`, `area`, and `iou`.

The explorer references the method output folders by relative path. Keep the repository layout intact when you open it.
