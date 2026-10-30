# Enhance Experiment (Phase 2c)

Goal: improve the quality of the RAW image **without changing its features**.
Input pair: `Data/pair-data/1/VID3553_B2_2_00d00h00m.tif` (RAW) and `VID3553_MASK_B2_2_00d00h00m.tif`.
Only the RAW pixels are enhanced. The MASK outlines give the cell masks for validation only.

All earlier experiments are frozen. This experiment does not change them.
It needs `scikit-image` and `bm3d` (installed in `.venv` with `uv pip install --python .venv/bin/python scikit-image bm3d`).

---

## 🔬 Diagnosis of the RAW image

| Check | Result | Meaning |
| :--- | :--- | :--- |
| Format | 8-bit grey, LZW (lossless) | No compression damage |
| JPEG blocking (8-px edge ratio) | 1.00 | None |
| Noise σ (wavelet MAD) | 1.14 grey levels | Very low |
| Flat-background std | 2.04 grey levels | Very low |
| Grey levels used | 253 (most data in 65–182) | Adequate |
| Spectrum, low band → Nyquist band | falls about 15× | **The image is soft** |

The image is limited by **softness** (optics and resolution: a cell is about 25 px wide), not by noise.
Processing can remove the small amount of noise. It cannot add real detail.

---

## ⚙️ Candidates (`src/enhance_methods.py`)

| Name | Method |
| :--- | :--- |
| `bm3d` | BM3D denoising, σ estimated from the image |
| `nlm` | OpenCV non-local means, h = σ |
| `bm3d_rl` | BM3D + mild Richardson–Lucy deconvolution (Gaussian PSF σ 0.7, 2 iterations). Negative control. |
| `bm3d_x2` | BM3D + Lanczos ×2 interpolation (more pixels, no new information) |

All candidates run in float32. Each one is saved as an 8-bit PNG for viewing and as a 16-bit TIFF (value × 256), which keeps the sub-grey-level precision of the result.

---

## ✅ Validation (`src/validate.py`)

Every check runs on the original pixel grid. The ×2 candidate is reduced back with Lanczos first.
Lanczos up then Lanczos down returns the original within 0.09 grey levels. Area averaging gives 0.92, so it is not used.

| Check | Limit | Why |
| :--- | :---: | :--- |
| `residual_edge_corr`: correlation of the removed signal with the cell edges | ≤ 0.10 | Removed signal must be noise, not structure |
| `cell_mean_abs_shift`: change of per-cell mean intensity | ≤ 0.50 grey levels | Intensity features stay the same |
| `cell_texture_ratio`: per-cell texture std, after / before | 0.95–1.05 | Texture features stay the same |
| `edge_strength_ratio`: edge strength on the outline band | 0.95–1.05 | Cell boundaries stay the same |
| `aura_contrast_ratio`: white-aura contrast vs. background | ≥ 0.95 | The aura stays visible |

---

## 📊 Results (VID3553_B2_2, 384 cells)

| Candidate | Result | Bg std | ResEdge | Texture | Edges | Aura | SSIM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| original | — | 2.04 | — | 1.000 | 1.000 | 1.000 | 1.0000 |
| `bm3d` | **PASS** | 1.78 | −0.54 | 0.999 | 0.998 | 0.967 | 0.9907 |
| `nlm` | PASS | 1.82 | −0.38 | 1.000 | 0.999 | 0.962 | 0.9901 |
| `bm3d_rl` | FAIL | 2.00 | 0.64 | 1.008 | 1.064 | 0.825 | 0.9845 |
| `bm3d_x2` | PASS | 1.77 | −0.37 | 0.995 | 0.995 | 0.973 | 0.9906 |

**Findings**
* **BM3D is the best denoiser here.** It lowers the background noise by 13% and keeps every cell feature within 0.2%. Its residual is even grain in the background; the cells are untouched.
* **The gain is small,** because the image has little noise to remove. BM3D removes only 0.77 grey levels (std).
* **Sharpening invents structure.** Every Richardson–Lucy setting tested (PSF σ 0.5–1.0, 2–10 iterations) failed: the residual follows the cell edges (0.6–0.7), texture or edges change by up to 50%, and the aura loses 9–34%.
* **×2 interpolation is safe but adds no information.** Use it only when a later step needs more pixels (for example sub-pixel outlines).
* Side effect of BM3D: tiny debris dots lose a little contrast (faint rings in its residual).

**Recommendation:** use `bm3d_16bit.tif` as the enhanced RAW image. For real detail, the data must improve at acquisition: a higher bit-depth export or a higher magnification.

---

## 📂 Directory Structure

```
enhance/
├── README.md
├── process.py                  # Runs candidates, validation and the report
├── src/
│   ├── enhance_methods.py      # Noise estimate and the 4 candidates
│   ├── validate.py             # Cell masks from the MASK outlines, metrics, pass limits
│   └── build_report.py         # Offline report.html
└── output/                     # Ignored by git
    └── VID3553_B2_2_00d00h00m/
        ├── report.html                     # Swipe viewer + metrics table
        ├── original.png
        ├── <candidate>.png                 # 8-bit view
        ├── <candidate>_16bit.tif           # value × 256, keeps the float precision
        ├── <candidate>_residual.png        # 128 + 20 × (original − enhanced)
        ├── metrics.json
        └── metrics.csv
```

---

## 🚀 Quick Start

```bash
.venv/bin/python enhance/process.py                 # about 10 s
open enhance/output/VID3553_B2_2_00d00h00m/report.html
```

In the report, drag on the image to move the divider (original on the left).
Keys: `1` enhanced, `2` residual, `↑` / `↓` candidate, `+` / `−` zoom, `P` pixel view.
