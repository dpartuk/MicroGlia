# Boundary Sharpening Method (Alternative 2)

Complete, self-contained standalone hierarchy for **Alternative 2: Multi-Scale Boundary Sharpening & Sub-Cell Extraction Pipeline**.

---

## 📂 Directory Structure

```
sharpened/
├── README.md                      # Documentation & architecture overview
├── process.py                     # Batch extraction launcher script
├── debug.py                       # Per-cell debugger launcher script
├── gallery.py                     # Interactive HTML gallery builder
├── src/                           # Core implementation modules
│   ├── boundary_sharpening_pipeline.py  # Core 8-step extraction engine with per-cell debug trace
│   ├── process_boundary_sharpening.py   # Batch processor across raw microscopy images
│   ├── whole_slide_boundary_sharpening.py # Whole-slide border sharpener
│   ├── build_gallery.py                 # Multi-dataset offline HTML gallery generator
│   └── debug_cell.py                    # Per-cell step-by-step CLI tool
└── output/                        # Extracted sub-cells, manifests, debug traces, and gallery
    ├── gallery.html               # Master interactive review gallery with whole-slide correlation
    ├── JPG_VID2724_B1_3_00d07h00m/
    │   ├── manifest.csv           # BBox, dimensions, area, filenames, and flags
    │   ├── subcell_001_original_extracted.jpg
    │   ├── subcell_001_sharpened_extracted.jpg
    │   ├── subcell_001_marked.jpg
    │   └── debug/
    │       └── cell_007/          # 16-step intermediate debug outputs & flowchart
    ├── VID2724_A1_9_00d07h00m/
    ├── VID2724_A3_4_00d00h00m/
    ├── VID2724_A3_4_00d07h00m/
    └── VID2724_B1_3_00d00h00m/
```

---

## 🚀 Quick Start Commands

### 1. Run Full Dataset Batch Extraction
```bash
python3 sharpened/process.py
```
*Discovers all microscopy images in `Data/raw-data/`, extracts all sub-cells (~0.4s per slide), and saves manifests.*

### 2. Debug Any Specific Cell ID
```bash
# Debug cell #7 in default slide
python3 sharpened/debug.py --cell 7

# Debug cell #14 on a specific slide
python3 sharpened/debug.py --slide VID2724_A1_9_00d07h00m --cell 14

# List all available slides
python3 sharpened/debug.py --list
```
*Outputs 16 intermediate images tied directly to code steps into `output/<slide>/debug/cell_<id>/`.*

### 3. Open the Interactive Web Gallery
Open [**`sharpened/output/gallery.html`**](output/gallery.html) in your browser:
* **`1`**: RGB Crop
* **`2`**: Sharpened Silhouette Mask
* **`3`**: Dual Side-by-Side View
* **`4`**: 16-Step Debug Flowchart Trace
* **`←` / `→`**: Step through sub-cells
* **`↑` / `↓`**: Switch between whole-slide datasets

### 4. Rebuild Gallery HTML
```bash
python3 sharpened/gallery.py
```
