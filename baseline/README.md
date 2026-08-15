# Baseline Extraction Method (Presaizen 2026)

Complete, self-contained standalone hierarchy for **Alternative 1: Baseline YOLOv11 Soma Bounding-Box Extraction Pipeline**.

---

## 📂 Directory Structure

```
baseline/
├── README.md                      # Documentation & architecture overview
├── process.py                     # Batch extraction launcher script
├── debug.py                       # Per-cell debugger launcher script
├── gallery.py                     # Interactive HTML review gallery builder
├── src/                           # Core implementation modules
│   ├── extract_cells_baseline.py  # Core baseline extraction engine with per-cell debug trace
│   ├── process_baseline.py        # Batch processor across raw microscopy images
│   ├── build_gallery.py           # Multi-dataset offline HTML gallery generator
│   └── debug_cell.py              # Per-cell step-by-step CLI tool
└── output/                        # Extracted cells, manifests, debug traces, and gallery
    ├── gallery.html               # Master interactive review gallery with whole-slide correlation
    ├── JPG_VID2724_B1_3_00d07h00m/
    │   ├── manifest.csv           # BBox, dimensions, area, filenames, and flags
    │   ├── cell_001_extracted.jpg # RGB crop on white canvas
    │   ├── cell_001_marked.jpg    # Whole-slide location correlation image
    │   └── debug/
    ├── VID2724_A1_9_00d07h00m/
    ├── VID2724_A3_4_00d00h00m/
    ├── VID2724_A3_4_00d07h00m/
    └── VID2724_B1_3_00d00h00m/
```

---

## 🚀 Quick Start Commands

### 1. Run Full Baseline Batch Extraction
```bash
python3 baseline/process.py
```
*Discovers all microscopy images in `Data/raw-data/`, extracts all baseline cells (991 total), and saves manifests.*

### 2. Debug Any Specific Cell ID
```bash
# Debug cell #7 in default slide
python3 baseline/debug.py --cell 7

# Debug cell #14 on a specific slide
python3 baseline/debug.py --slide VID2724_A1_9_00d07h00m --cell 14

# List all available slides
python3 baseline/debug.py --list
```
*Outputs intermediate images tied directly to baseline code steps into `output/<slide>/debug/cell_<id>/`.*

### 3. Open the Interactive Web Gallery
Open [**`baseline/output/gallery.html`**](output/gallery.html) in your browser:
* **`1`**: Extracted Crop
* **`2`**: Debug Flowchart Trace (if generated)
* **`←` / `→`**: Step through baseline cells
* **`↑` / `↓`**: Switch between whole-slide datasets

### 4. Rebuild Gallery HTML
```bash
python3 baseline/gallery.py
```
