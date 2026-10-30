# Pair Contour Method (Phase 2)

Phase 2 experiment. It uses one image pair from `Data/pair-data/1/`:

| Role | File | Content |
| :--- | :--- | :--- |
| RAW | `VID3553_B2_2_00d00h00m.tif` | The real pixel data |
| MASK | `VID3553_MASK_B2_2_00d00h00m.tif` | The same field of view, contrast-enhanced, with cyan cell outlines |

The cyan outlines in the MASK image define the cells.
The pixels are cut from the RAW image.
The cyan line is drawn on the cell edge, so the RAW cut also recovers the edge pixels that the line hides.

The Phase 1 folders (`baseline/`, `sharpened/`, `contour/`, `dpgnet/`, `comparison/`) are frozen. This experiment does not change them.
The gallery builder is a copy of `contour/src/build_gallery.py`, adapted to the pair data.

---

## 📂 Directory Structure

```
pair_contour/
├── README.md
├── process.py                  # Extraction launcher (default: pair B2_2)
├── gallery.py                  # Review gallery builder
├── src/
│   ├── extract_pair_cells.py   # Core pipeline
│   └── build_gallery.py        # Copy of contour/src/build_gallery.py, adapted
└── output/                     # Ignored by git
    ├── gallery.html                    # Same file as pair_contour_gallery.html
    ├── pair_contour_gallery.html
    └── VID3553_B2_2_00d00h00m/
        ├── 00_original_reference.jpg      # RAW image
        ├── 00_annotated_reference.jpg     # MASK image
        ├── 01_cyan_core_lines.png         # Detected outline pixels
        ├── 02_cell_outline_overview.jpg   # RAW + cell outlines (green), unclaimed outline pixels (red)
        ├── 03_taken_vs_ignored.png        # Cut cells filled green, outlined cells not cut filled red
        ├── whole_slide_ground_truth_mask.png
        ├── whole_slide_label_map.png      # 16-bit, pixel value = cell_id
        ├── manifest.csv
        ├── summary.json
        ├── subcell_XXX_original_extracted.png  # RAW pixels inside the cell, magenta outside
        ├── subcell_XXX_annotated.png           # Same box from the MASK image
        ├── subcell_XXX_mask.png                # Binary cell mask (outline + aura)
        ├── subcell_XXX_marked.jpg              # Whole RAW image with the cell box
        └── debug/cell_XXX/                     # Step images, step grid, cell_debug_summary.json
```

---

## ⚙️ Pipeline

1. **Cyan line detection.** `cyanness = min(B, G) − R`. Grey pixels give 0.
   The solid line ("core") has cyanness ≥ 100. The anti-aliased edge ("halo") has cyanness > 0.
2. **Enclosed regions.** Connected components of the non-core pixels, with 4-connectivity.
   The line is 1 px wide and steps diagonally, so 8-connectivity leaks through every outline.
   The largest component is the background. Each other region of 60–30,000 px is a cell.
1b. **Gap bridging.** Skeleton endpoints of the core line that lie within 8 px of each other are joined
   with a 1-px line (`--max-gap`, 0 = off). Unpaired endpoints are short spurs and stay as they are.
   B2_1 and B2_2 have no real gaps (7 and 12 endpoints, all spurs), so nothing is bridged there.
   Test: a 5 px gap cut into the outline of cell 39 loses the cell without bridging (383 cells)
   and recovers it with bridging (384 cells, IoU 0.998 with the original cell).
3. **Background pocket rejection.** A region with RAW texture std < 6 is background enclosed between outlines.
4. **Cell mask.** The region plus the outline pixels within 3 px of it.
   Two touching cells share one line, so both cells get the shared line pixels.
> **Since 2026-10-30 the cut stops at the contour:** the mask is the area inside the cyan contour plus the
> contour line itself (steps 1–4a). Steps 4b (aura growth) and 4c (contact completion) cropped too much
> and are **off by default**. They stay available as options (`--aura-free-px 2 --aura-px 10`,
> `--contact-fit-residual 1.5`).

4b. **Aura growth (off by default).** The phase-contrast white aura often lies outside the cyan line.
   The mask grows 2 px in any direction, then up to 10 px more through RAW pixels that are
   brighter than the background (5×5 Gaussian-smoothed RAW ≥ background median + 8).
   It grows only into the cell's own zone: pixels closer to this cell than to any other cell.
   This stops the aura from running along a neighbour's bright rim.
   A 3×3 opening then removes spokes and speckles thinner than 3 px.
4c. **Contact completion (off by default).** Where two cells touch, the hand-drawn outline follows the contact line and
   cuts into the cell (example: cells 2 and 7), and aura growth stops at the neighbour.
   For round cells (solidity ≥ 0.9, at least 35% of the edge free), a shape is fitted to the free edge:
   an ellipse when the free edge goes around ≥ 60% of the cell, else a circle (an ellipse on a short arc
   stretches over the neighbours). If the fit error is ≤ 1.5 px and the shape is ≤ 1.5× the mask, the shape
   pixels **inside neighbouring cells**, within 8 px of the contact edge, are added; holes are filled.
   Background is never added here, and irregular cells are left as they are.
   The added pixels also belong to the neighbour: touching cells overlap in this zone.
   These pixels are inferred from the shape, not seen. `--contact-min-free 0.5` is stricter
   (fixes cell 2 but not cell 7; 110 cells instead of 142). `--contact-fit-residual 0` turns the step off.
4d. **Border exclusion.** A cell whose final mask reaches the image edge is not cut: part of it lies
   outside the image. The project goal is fewer mistakes, not more cells. With the default contour-only
   mask, 39 cells are excluded (50 when aura growth is on, because some auras reach the edge). Excluded cells still block their
   neighbours in steps 4b–4c. Cell numbers are kept, so the numbering has gaps (for example, cell 1 is excluded).
   The excluded cells are drawn in grey in `02_cell_outline_overview.jpg` and listed in `summary.json`.
5. **Cut from RAW.** The crop box has 18 px padding (same as `contour/`).
   Pixels outside the mask are **magenta (255, 0, 255)**. The RAW images are pure grey (B = G = R),
   so magenta never occurs in cell data, and white cell auras stay visible.
   Crops are PNG, so the cut pixels are exact RAW values.
6. **QA: signal left outside.** Local background = Gaussian-weighted (σ 20 px) mean of background pixels.
   Noise σ = MAD of RAW − local background. A pixel is *signal* if |RAW − local background| > 3σ.
   * `outside_signal_pct`: share of signal pixels in a 4 px ring outside the final mask (other cells excluded).
     High = the cut missed part of the cell or its aura. Flag (`qa_flag`) above 25%.
   * `edge_background_pct`: share of the mask's outer edge that is plain background (|dev| ≤ 1σ). High = over-growth.
   * `crowded`: neighbours cover more than half of the ring. The score is kept but not flagged or averaged.
   * Per image: uncovered signal blobs ≥ 150 px (orange boxes in `02_cell_outline_overview.jpg`)
     are candidate cells without an outline. Smaller blobs are counted as debris.
   * Debug step 6 (`09_step6_qa_signal_outside.png`): green = mask edge, blue = ring, red = signal left in the ring.

---

## 📊 Results (pair B2_2, default: contour-only mask)

| Metric | Value |
| :--- | :---: |
| Cells extracted | 345 |
| Cells excluded (cut by the image edge) | 39 |
| Background pockets rejected | 0 |
| Outline pixels claimed by a cell | 99.87% |
| Outline endpoints / gaps bridged | 7 / 0 |
| Mean cell area (inside + contour line) | 823.7 px² |
| Mask pixels beyond the contour | 0 |
| Cut pixels that differ from RAW | 0 |
| Noise σ (local background) | 1.42 grey levels |
| Signal left outside, median | 34.3% of ring pixels |
| Uncovered signal blobs ≥ 150 px (candidate cells without an outline) | 32 (+ 109 small debris) |

**QA note.** With a contour-only mask, the "signal left outside" check mostly measures the white aura,
which is now left out by design. Its flag (> 25%, 240 cells) no longer marks mistakes.

Earlier settings on the same image (for reference):

| Setting | Cells | Mean area | Signal left outside (median) |
| :--- | :---: | :---: | :---: |
| Contour + aura | 384 (no border exclusion) | 1,007.1 px² | 12.9% |
| Contour + aura + contact completion, border exclusion | 334 | 1,050.6 px² | 11.2% |
| **Contour only, border exclusion (default)** | **345** | **823.7 px²** | 34.3% |

### Known limits

* Some visible cells in the MASK image have no cyan outline. The pipeline cuts only outlined cells.
* Aura growth uses one global brightness threshold. Faint parts of an aura (below background + 8) stay outside the cut.
  The QA confirms this: a median 12.9% of the ring just outside each mask still holds cell signal.
* Two touching cells split the background between them at the midline, so a shared bright wall is split too.

---

## 🚀 Quick Start Commands

```bash
# Extract pair B2_2 with debug traces for every cell (about 5 s)
.venv/bin/python pair_contour/process.py

# Stricter contact completion, or none
.venv/bin/python pair_contour/process.py --contact-min-free 0.5
.venv/bin/python pair_contour/process.py --contact-fit-residual 0

# Tune the aura growth
.venv/bin/python pair_contour/process.py --aura-free-px 2 --aura-px 10 --aura-delta 8

# Debug traces for some cells only, or none
.venv/bin/python pair_contour/process.py --debug-cell 7 14
.venv/bin/python pair_contour/process.py --no-debug

# Another pair (the MASK path is derived: VID3553_B2_1 -> VID3553_MASK_B2_1)
.venv/bin/python pair_contour/process.py --raw Data/pair-data/1/VID3553_B2_1_00d00h00m.tif

# Rebuild the gallery
.venv/bin/python pair_contour/gallery.py
```

Open [`pair_contour/output/gallery.html`](output/gallery.html) in a browser.
It has the same layout as the Contour gallery:
* **`1`**: RAW cut
* **`2`**: MASK image crop (with the cyan outline)
* **`3`**: Dual view
* **Overview first:** the gallery opens on the taken-vs-ignored map (green = cut, red = outlined but ignored,
  with the reason counts). Press **`O`**, **`Esc`** or *Review cells* to close it; **`O`** or *Overview* reopens it.
* **`D`**: Step-by-step debug trace
* **`−` / `+`**: Crop zoom (1×–8×, default 3×; a crop is never stretched beyond this)
* **`P`**: Hard pixels instead of smooth rendering
* **`←` / `→`**: Step through cells
* **`↑` / `↓`**: Switch between pairs
