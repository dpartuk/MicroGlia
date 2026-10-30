# Mask Contour Method (Phase 2b)

Phase 2 experiment. It uses **only the MASK image** (the image with the cyan outlines):

| Role | File |
| :--- | :--- |
| MASK | `Data/pair-data/1/VID3553_MASK_B2_2_00d00h00m.tif` |

The RAW image (`VID3553_B2_2_00d00h00m.tif`) is not used.
Each cell is the area enclosed by a cyan outline, **without the outline itself**.
The crop is taken from the MASK image.

This method is similar to the Phase 1 `contour/` method: it crops the area inside a contour.
The difference from `pair_contour/` (frozen): there, the outline and the aura were kept and the pixels came from the RAW image.

The Phase 1 folders and `pair_contour/` are frozen. This experiment does not change them.
The gallery builder is a copy of `pair_contour/src/build_gallery.py`, which is a copy of `contour/src/build_gallery.py`.

---

## 📂 Directory Structure

```
mask_contour/
├── README.md
├── process.py                  # Extraction launcher (default: VID3553_MASK_B2_2)
├── gallery.py                  # Review gallery builder
├── compare_upscaled.py         # Cell-by-cell comparison of a base run and an upscaled run
├── src/
│   ├── extract_mask_cells.py   # Core pipeline
│   └── build_gallery.py        # Copy of the contour gallery, adapted
└── output/                     # Ignored by git
    ├── gallery.html                    # Same file as mask_contour_gallery.html
    ├── mask_contour_gallery.html
    ├── upscale_comparison.csv          # From compare_upscaled.py
    ├── VID3553_MASK_B2_2_00d00h00m_upscaled_x4/   # Same layout, upscaled run
    └── VID3553_MASK_B2_2_00d00h00m/
        ├── 00_original_reference.jpg      # MASK image
        ├── 01_cyan_core_lines.png         # Detected outline pixels
        ├── 02_cell_outline_overview.jpg   # Interior outlines (green), unclaimed outline pixels (red)
        ├── whole_slide_ground_truth_mask.png
        ├── whole_slide_label_map.png      # 16-bit, pixel value = cell_id
        ├── manifest.csv
        ├── summary.json
        ├── subcell_XXX_original_extracted.png  # Interior pixels, magenta outside
        ├── subcell_XXX_annotated.png           # Same box with the outline
        ├── subcell_XXX_mask.png                # Binary interior mask
        ├── subcell_XXX_marked.jpg              # Whole MASK image with the cell box
        └── debug/cell_XXX/                     # Step images, step grid, cell_debug_summary.json
```

---

## ⚙️ Pipeline

1. **Cyan line detection.** `cyanness = min(B, G) − R`. Grey pixels give 0.
   The solid line ("core") has cyanness ≥ 100. Every pixel with cyanness > 0 is outline ("halo").
2. **Enclosed regions.** Connected components of the non-core pixels, with 4-connectivity.
   The largest component is the background. Each other region of 60–30,000 px is a cell.
3. **Background pocket rejection.** A region with interior std < 6 is background enclosed between outlines.
4. **Interior mask.** The enclosed region minus every cyan-tinted pixel.
   In the MASK image every non-grey pixel is cyan-tinted, so no outline pixel stays in the cell.
5. **Cut from the MASK image.** The crop box has 18 px padding (same as `contour/`).
   Pixels outside the mask are magenta (255, 0, 255). The MASK image holds no magenta.
   Crops are PNG, so the cut pixels are exact MASK image values.

---

## 📊 Results (VID3553_MASK_B2_2)

| Metric | Value |
| :--- | :---: |
| Cells extracted | 384 |
| Background pockets rejected | 0 |
| Outline pixels next to a cell | 99.84% |
| Open outline fragments (≥ 10 px) | 3 |
| Outline pixels removed from the regions | 20,995 |
| Cyan-tinted pixels left in the cuts | 0 |
| Mean cell area (interior only) | 575.1 px² |
| Cut pixels that differ from the MASK image | 0 |
| Non-magenta pixels outside the masks | 0 |

For comparison, `pair_contour/` on the same image gives 384 cells with a mean area of 1,007.1 px² (outline + aura).

---

## 🔍 AI-Upscaled Image (×4)

Input: `Data/pair-data/upscaled/VID3553_MASK_B2_2_00d00h00m.tif` (5632 × 4160, an AI-upscaled copy of the MASK image).

**What the upscaler changed**
* The image is no longer grey: 91% of pixels have some colour. Background cyanness reaches about 20.
* The cyan line is 6–8 px wide, with an orange-red colour fringe beside it.
* Some line segments turned white, red or faint, so 15 outlines no longer close.
* The cell interiors are smoothed and redrawn. Fine texture is lost or invented.

**Settings for the upscaled image** (the defaults keep the original behaviour):

| Option | Value | Reason |
| :--- | :---: | :--- |
| `--scale` | 4 | Area limits ×16, padding ×4 |
| `--halo-level` | 30 | With the default (1), colour noise removes large parts of the cells |
| `--fringe-chroma` / `--fringe-radius` | 30 / 8 | Removes the colour fringe next to the line |
| `--core-level` | 60 | Keeps faint line segments |
| `--close-gaps` | 5 | Bridges the gaps in the line. Without it, 15 cells leak into the background |

**Results** (areas in original-image pixels, from `compare_upscaled.py`):

| Metric | Original | Upscaled ×4 |
| :--- | :---: | :---: |
| Cells | 384 | 385 |
| Matched one-to-one | — | 382 |
| Missed (original cells 131, 271) | — | 2 |
| Extra (slivers between two lines) | — | 3 |
| Split / merged | — | 0 / 0 |
| Mean IoU of matched cells | — | 0.801 |
| Median area ratio upscaled / original | — | 0.876 |
| Mean cell area | 575.1 px² | 511.0 px² |

**Conclusion.** The upscaled image gives the same cells but no better data:
* The cells are about 12% smaller, because the wider line and fringe take more of the cell edge.
* The interior pixels are generated by the upscaler, not measured.
* The upscaled run needs five extra settings, and it still misses 2 cells and adds 3 false ones.

---

### Known limits

* The MASK image is contrast-enhanced, so the cut pixels are not the original intensities.
* The white aura outside the outline is not part of the cut.
* Visible cells without a cyan outline are not cut.

---

## 🚀 Quick Start Commands

```bash
# Extract with debug traces for every cell (about 5 s)
.venv/bin/python mask_contour/process.py

# Debug traces for some cells only, or none
.venv/bin/python mask_contour/process.py --debug-cell 7 14
.venv/bin/python mask_contour/process.py --no-debug

# Another MASK image
.venv/bin/python mask_contour/process.py --mask Data/pair-data/1/VID3553_MASK_B2_1_00d00h00m.tif

# AI-upscaled (x4) MASK image, then compare it with the original run
.venv/bin/python mask_contour/process.py --mask Data/pair-data/upscaled/VID3553_MASK_B2_2_00d00h00m.tif \
    --name VID3553_MASK_B2_2_00d00h00m_upscaled_x4 --scale 4 --core-level 60 --close-gaps 5 \
    --halo-level 30 --fringe-chroma 30 --fringe-radius 8
.venv/bin/python mask_contour/compare_upscaled.py

# Rebuild the gallery
.venv/bin/python mask_contour/gallery.py
```

Open [`mask_contour/output/gallery.html`](output/gallery.html) in a browser.
It has the same layout as the Contour gallery:
* **`1`**: Interior cut (no outline)
* **`2`**: Same box with the outline
* **`3`**: Dual view
* **`D`**: Step-by-step debug trace
* **`←` / `→`**: Step through cells
* **`↑` / `↓`**: Switch between images
