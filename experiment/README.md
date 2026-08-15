# MicroGlia Experimental Method: Cellular Process & Bounding Box Refinement

Dedicated experimental sandbox branched from the frozen **Boundary Sharpening Method**.
Used for developing, testing, and validating multi-scale morphological gap-closing, proximity-based process attachment merging, and expanded bounding box padding.

---

## 📂 Directory Structure

```
experiment/
├── README.md                          # Documentation & experiment tracking
├── process.py                         # Full dataset batch extraction runner
├── debug.py                           # Per-cell step-by-step debugger CLI (with Step 0 initial BBox)
├── gallery.py                         # Rebuilds offline experimental gallery
│
├── src/                               # Python modules
│   ├── boundary_sharpening_pipeline.py # Experimental core pipeline & gap closing
│   ├── process_boundary_sharpening.py # Batch processing runner
│   ├── build_gallery.py                # Standalone HTML gallery builder
│   └── debug_cell.py                   # Per-cell step-by-step CLI tool
│
└── output/                            # Output extractions, debug traces, and gallery
    └── gallery.html                   # Interactive review gallery
```

---

## 🚀 Quick Start Commands

1. **Run Full Experimental Batch Extraction**:
   ```bash
   python3 experiment/process.py
   ```

2. **Debug a Specific Cell**:
   ```bash
   python3 experiment/debug.py --cell 13
   ```

3. **Open the Experimental Review Gallery**:
   * [**`experiment/output/gallery.html`**](output/gallery.html) *(Open directly in browser)*

4. **Rebuild Experimental Gallery HTML**:
   ```bash
   python3 experiment/gallery.py
   ```
