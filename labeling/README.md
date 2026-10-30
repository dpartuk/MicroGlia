# Expert Labeling (Phase 2)

Experts label the cells that `pair_contour/` cuts. The page runs in a browser.
It works in two modes:

| Mode | Where | Where labels go |
| :--- | :--- | :--- |
| Shared store | Private claude.ai page, experts invited as Editors | One shared database. Each rater sees only their own labels. |
| Browser only | `labeling/web/index.html` opened from a folder or zip | This browser. The rater sends the JSON export. |

---

## 📂 Directory Structure

```
labeling/
├── README.md
├── export_web.py            # pair_contour output -> web bundle + reference masks
├── merge_labels.py          # JSON exports -> labels.csv + Cohen's kappa
├── src/labeling_page.html   # The labeling page (source)
├── web/                     # Ignored by git: index.html, slides.js, slides/<slide>.js (about 1.3 MB per slide)
└── reference/<slide>/       # Ignored by git, never published: cells_index.csv, masks.npz
```

---

## ⚙️ Design

* **Label set.** 1 Resting, 2 Surveilling, 3 Activated, 4 Resolution, 5 Dystrophic (Presaizen 2026),
  X = not a cell or bad cut, U = unsure. Confidence (Q / W / E = low / medium / high) sits above the classes and
  starts at High for each cell. An optional free note (N).
* **Start over.** A top-bar button deletes all labels of this rater (shared store and browser), after an in-page confirmation.
* **Layout.** On a desktop window (≥ 901 × 560 px) the page fits the window with no scroll. Checked at 1920×1080, 1400×900 and 1280×720.
* **Cell key.** Labels are keyed by `slide@cx_cy` (mask centroid), not by cell number.
  `merge_labels.py` matches each key to the current pipeline output by mask IoU (default ≥ 0.5).
  A label below the threshold stays unmatched and is reported.
* **Raters.** One rater labels all cells (decision 2026-10-30). The code also supports more raters:
  `merge_labels.py` then reports Cohen's kappa per rater pair.
* **Order.** The rater works slide by slide. Inside a slide, the order is random, seeded by rater id + slide name.
  Adding slides later does not change the order of earlier slides.
* **Second look.** 10% of each slide's cells come back a second time, at least 30 items later (`--repeat-fraction`).
  The page does not show the first label. `merge_labels.py` reports intra-rater kappa from these.
* **Images.** Each slide file holds the full RAW image once (lossless PNG) plus, per cell, the mask bits and
  the outline. The browser draws the RAW cut (neutral tint outside the mask, padding 18 px as in pair_contour),
  the RAW surroundings with the outline (key C toggles it), and the position on the slide.
  A headless-Chrome check confirmed that the decoded pixels inside every mask equal RAW (913 cells, 0 mismatches).
* **Shared store layout.** `labels/<rater id>` holds the rater code. `labels/<rater id>/slides/<slide>` holds
  `cells` and `repeats` maps. Only the owner reads all raters. Each rater reads and writes only their own subtree.
  The page also keeps a copy in the browser.

### Checks done (2026-10-30)

* Headless Chrome: keys, auto-advance, notes, reload persistence, export, light and dark theme, 400 px width.
* Published page: the access rules hide one rater's documents from Editors; the owner reads all.
* Merge with two synthetic raters: 760 labels, all matched at IoU 1.0.
* Merge after a pipeline change (aura on, 334 cells): median IoU 0.80. The 11 cells that the change moves
  to the image border are reported as unmatched.

---

## 🚀 Quick Start Commands

```bash
# Build the bundle for all slides in pair_contour/output (default), or for named slide folders
.venv/bin/python labeling/export_web.py
.venv/bin/python labeling/export_web.py pair_contour/output/VID3553_B2_1_00d00h00m pair_contour/output/VID3553_B2_2_00d00h00m

# Merge rater exports, or slide documents downloaded from the shared store
.venv/bin/python labeling/merge_labels.py exports/labels_*.json
.venv/bin/python labeling/merge_labels.py "dump/labels/*/slides/*.json"
```

Published page: https://claude.ai/artifact/7vzmvdo8PKKW37XxJYymWT (private; shared with the rater as Editor).
Rater instructions: https://claude.ai/artifact/AYecc2epoNu7SLvDSksEd3
To add slides: run pair_contour on them, run `export_web.py`, then republish `labeling/web/index.html`
with the new `slides/<slide>.js` files. More than about 40 new slides need two publishes (64 MB per publish).
