# MicroGlia Contour Method: Cellular Process & Bounding Box Refinement

Dedicated contour pipeline branched from the frozen **Boundary Sharpening Method**.
Used for developing, testing, and validating multi-scale morphological gap-closing, proximity-based process attachment merging, and expanded bounding box padding.

---

## 📂 Directory Structure

```
contour/
├── README.md                          # Documentation & pipeline tracking
├── process.py                         # Full dataset batch extraction runner
├── debug.py                           # Per-cell step-by-step debugger CLI (with Step 0 initial BBox)
├── gallery.py                         # Rebuilds offline contour gallery
│
├── src/                               # Python modules
│   ├── boundary_sharpening_pipeline.py # Contour core pipeline & gap closing
│   ├── process_boundary_sharpening.py # Batch processing runner
│   ├── build_gallery.py                # Standalone HTML gallery builder
│   └── debug_cell.py                   # Per-cell step-by-step CLI tool
│
└── output/                            # Output extractions, debug traces, and gallery
    └── gallery.html                   # Interactive review gallery
```

---

## 🚀 Quick Start Commands

1. **Run Full Contour Batch Extraction**:
   ```bash
   python3 contour/process.py
   ```

2. **Debug a Specific Cell**:
   ```bash
   python3 contour/debug.py --cell 13
   ```

3. **Open the Contour Review Gallery**:
   * [**`contour/output/gallery.html`**](output/gallery.html) *(Open directly in browser)*

4. **Rebuild Contour Gallery HTML**:
   ```bash
   python3 contour/gallery.py
   ```
