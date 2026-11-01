# Expert Labeling (Phase 2)

Experts label the cells that `pair_probe/` cuts (`bundle.PIPE_METHOD`; `pair_contour/` until 2026-11-01). The page runs in a browser.
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
├── bundle.py                # Shared helpers: mask/slide hashes, lock file, mask comparison
├── export_web.py            # pipeline output -> web bundle; --mark-published locks published slides; --keep-labels
├── merge_labels.py          # JSON exports or store documents -> labels.csv + Cohen's kappa
├── check_stable.py          # Reruns the pipeline with the locked settings, compares with the published masks
├── published_slides.json    # The lock (in git): slide hash, cells, settings, commit per published slide
├── src/labeling_page.html   # The labeling page (source)
└── web/                     # Ignored by git: index.html, slides.js, slides/<slide>.js (about 1.3 MB per slide)
```

---

## ⚙️ Design

* **Label set.** 1 Resting, 2 Surveilling, 3 Activated, 4 Resolution, 5 Dystrophic (Presaizen 2026),
  X = not a cell or bad cut, U = unsure. Confidence (Q / W / E = low / medium / high) sits above the classes and
  starts at High for each cell. An optional free note (N).
* **Start over.** A top-bar button deletes all labels of this rater (shared store and browser), after an in-page confirmation.
* **Layout.** On a desktop window (≥ 901 × 560 px) the page fits the window with no scroll. Checked at 1920×1080, 1400×900 and 1280×720.
* **Cell key.** Labels are keyed by `slide@cx_cy` (mask centroid), not by cell number.
  A labeled cell that gets a new cut keeps its old key (`--keep-labels`, below).
  Each label also stores `mh`, the hash of the exact mask the rater saw.
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
  `cells` and `repeats` maps and the `slide_hash`. Only the owner reads all raters. Each rater reads and writes only their own subtree.
  The page also keeps a copy in the browser.

## 🔒 Cell Stability

Labels must always point to the cells the rater saw. Four rules protect this:

| Rule | Where | Effect |
| :--- | :--- | :--- |
| A. Locked slides | `export_web.py` + `published_slides.json` | A published slide is never rewritten with other masks. The script prints identical / changed / removed / new and keeps the file. `--force` overrides. |
| B. Page = reference | `merge_labels.py` | Masks come from the published slide files, checked against the lock. Each label row gets a status: `matched`, `no_current_cell`, `unknown_key`, `mask_hash_mismatch` or `no_reference`. |
| C. Recorded source | slide files, labels | Each slide file holds its hash, the pipeline settings and the commit. Each label holds its mask hash. |
| D. Stability check | `check_stable.py` | Reruns the pipeline with the locked settings in a temporary folder (or `--current`) and compares. Exit code 1 = a difference. |

| E. Kept labels | `export_web.py --force --keep-labels <store files>` | A replaced slide keeps the rater's labels. Each labeled cell is matched to one new cell (IoU ≥ 0.5, one to one). The new cell keeps the old key and stores the old mask hash as `prev_mh`. `merge_labels.py` reports these labels as `matched_previous_cut`. |

**Publish steps:** `check_stable.py` → `export_web.py` → publish `index.html` + `slides.js` + `slides/*.js` →
`export_web.py --mark-published` → commit `published_slides.json`.

**New machine:** read the published slide files with the Artifact tool (`action: read`, `paths: ["slides/<slide>.js", …]`)
into `labeling/web/slides/`. The lock in git confirms them: `merge_labels.py` and `check_stable.py` refuse a file whose hash differs.

### Change to pair_probe cuts (2026-11-01)

The user decided to keep the rater's labels when the cuts change: the rater is an expert, and the labels
stay valid for a slightly different cut of the same cell. B2_1 had 36 labels (no second looks); B2_2 had none.
* All 36 labeled cells matched one pair_probe cell each (IoU 0.70–0.98, same cell numbers) and keep their key.
* B2_1: 568 → 564 cells; B2_2: 345 → 333 cells (the halo band puts some cells on the image edge).
* `merge_labels.py`: 36 `matched_previous_cut`. `check_stable.py --current`: 564 / 333 identical.
* Headless Chrome with the 36 labels in the browser store: B2_1 shows 36 / 620 done.
* The cell order of each slide changes (a new cell set). "Next unlabeled" finds the open cells.

### All 64 slides (2026-11-01)

* All pairs of `Data/pair-data/1`–`4` are on the page: 64 slides, 27,849 cells (30,636 items with second looks).
* The slide list is sorted by data folder, then by slide number (`B2_<n>`).
* Two cells with the same centroid (a drawn outline inside another, `VID3553_B2_4_00d04h00m` cells 322 and 324):
  the later cell gets the key suffix `_2`.
* The bundle is 89 MB, so it went up in two publishes (one publish takes at most 64 MB). The first publish added
  slide files only; the second added the rest and `slides.js`, so the page never listed a slide without its file.
* The label document of the largest slide (1,271 cells + 127 second looks) is about 193 KiB of the 256 KiB limit.

### Checks done (2026-10-30)

* Headless Chrome: keys, auto-advance, notes, reload persistence, export, light and dark theme, 400 px width.
* Published page: the access rules hide one rater's documents from Editors; the owner reads all.
* Merge with two synthetic raters: 760 labels, all matched at IoU 1.0.
* Merge after a pipeline change (aura on, 334 cells): median IoU 0.80. The 11 cells that the change moves
  to the image border are reported as unmatched.
* Stability (Version 5): the files read back from the live page are byte-identical to `labeling/web/slides/`.
  A fresh rerun gives 913 / 913 identical masks. With aura on, `export_web.py` kept the published B2_2 file and
  reported 334 changed + 11 removed; `check_stable.py --current` exited 1; the merge gave 334 `matched` +
  11 `no_current_cell`. A tampered mask hash, an unknown key and a slide file with a wrong hash were all reported.

---

## 🚀 Quick Start Commands

```bash
# Build the bundle for all slides in pair_contour/output (default), or for named slide folders
.venv/bin/python labeling/export_web.py
.venv/bin/python labeling/export_web.py pair_contour/output/VID3553_B2_1_00d00h00m pair_contour/output/VID3553_B2_2_00d00h00m

# Check that the pipeline still gives the published cells (before any republish)
.venv/bin/python labeling/check_stable.py

# After a publish: lock the published slides, then commit labeling/published_slides.json
.venv/bin/python labeling/export_web.py --mark-published

# Merge rater exports, or slide documents downloaded from the shared store
.venv/bin/python labeling/merge_labels.py exports/labels_*.json
.venv/bin/python labeling/merge_labels.py "dump/labels/*/slides/*.json"
```

Published page: https://claude.ai/artifact/7vzmvdo8PKKW37XxJYymWT (private; shared with the rater as Editor).
Rater instructions: https://claude.ai/artifact/AYecc2epoNu7SLvDSksEd3
To add slides: run pair_contour on them, then follow the publish steps above. Locked slides stay as they are. More than about 40 new slides need two publishes (64 MB per publish).
