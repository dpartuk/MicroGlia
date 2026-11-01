# Pair Probe Method (Phase 2)

Phase 2 experiment. It starts from `pair_contour/` (frozen) and fixes **problem 1**:
parts of the cell halo (aura) or body lie just outside the hand-drawn contour, so the cut leaves them out.
The expert rater reported this on 2026-11-01 (B2_1 cells 147, 201, 549, 261, 432, 224).

**Problem 2** (thin processes that leave the cell body and are cut by the contour) needs a different rule.
It is not handled yet.

| Role | File | Content |
| :--- | :--- | :--- |
| RAW | `VID3553_B2_2_00d00h00m.tif` | The real pixel data |
| MASK | `VID3553_MASK_B2_2_00d00h00m.tif` | The same field of view, contrast-enhanced, with cyan cell outlines |

The contour marks **where** the cell is. The RAW image decides where it **ends**.

Differences from `pair_contour/`:
* Aura growth and contact completion (both off by default in `pair_contour/`) are removed.
* Step 4b is the **halo band**, step 4b2 the **slow entry** into neighbours (below).
* With `--halo-px 0 --halo-leak-px 0 --entry-px 0` the output is identical to `pair_contour/` (label maps checked pixel by pixel on B2_1).
* Cell numbers are the same as in `pair_contour/`, so a cell ID means the same cell in both galleries.

---

## 📂 Directory Structure

```
pair_probe/
├── README.md
├── process.py                  # Extraction launcher (default: pair B2_2)
├── gallery.py                  # Review gallery builder
├── src/
│   ├── extract_pair_cells.py   # Copy of pair_contour/src/extract_pair_cells.py with the halo band
│   └── build_gallery.py        # Copy of pair_contour/src/build_gallery.py
└── output/                     # Ignored by git; same files as pair_contour/output/
```

---

## ⚙️ Step 4b: Halo Band

Steps 1–4a (cyan line detection, gap bridging, enclosed regions, pocket rejection, contour mask) are the same as in
`pair_contour/`. Then:

1. **Signal map.** A pixel is signal if the smoothed RAW (Gaussian, σ = 1 px) differs from the local background
   by more than 3 noise σ, brighter (halo) or darker (cell body). The noise σ comes from the **unsmoothed** RAW
   (MAD of the background, about 1.41 grey levels), so the threshold is about 4.2 grey levels.
   The MAD of the smoothed image is much smaller (0.29) and gave a threshold below one grey level
   (9.5% false alarms on background).
2. **Growth (`--halo-px 6`).** The contour mask of each cell is its seed. All cells grow together, 1 px per step
   (8-connected), through signal pixels only, for up to 6 steps.
   A pixel goes to the cell that reaches it first. A pixel that two cells reach in the same step is
   **contested** and goes to neither. Growth never enters another cell's contour mask.
   Why 6 px: on B2_1 the signal share outside the contour falls from 66% (1 px) to its plateau at 6 px
   (B2_2: 7–8 px).
3. **Leak rule (`--halo-leak-px 4`).** The growth continues 4 more steps, only as a test. A real halo ends inside
   the band. Where this cell's signal goes on past 6 px, the band touches something else (a neighbour with no
   contour, debris, a process). Band pixels within 4 px of that signal are removed.
4. **Neck rule (`--halo-open-px 2`).** An opening (disk, radius 2 px) of contour mask + band cuts thin necks.
   Band parts that no longer touch the contour mask are removed (attached debris).
5. **Gap fill (`--halo-close-px 2`).** A closing (disk, radius 2 px) fills faint gaps of the band. It adds only pixels
   within 6 px of the contour, outside the leak zone, not in another cell and not contested.
6. **Area rule (`--halo-max-added 1.0`).** A cell whose band adds more than 100% of its contour mask area keeps
   its contour-only mask (`halo_status = reverted_area`). On B2_1 and B2_2 no cell fails this rule.

Manifest columns: `halo_px` (pixels added), `halo_filled_px` (of these, added by the gap fill),
`halo_status` (`grown`, `no_signal`, `reverted_area`).

### Step 4b2: Slow Entry into Neighbours (`--entry-px 6`)

The drawn boundary between two touching cells can lie inside one of them. Then part of its halo (and of its edge)
is inside the neighbour's contour (example: B2_1 cell 549, rater #21). The halo band never enters another contour,
so that side of the halo stays out.

After the halo band, a second growth enters other cells' contour masks, the pixels their halo growth took first
(a shared aura: B2_1 cell 261, rater #26), and contested pixels, 1 px per step, up to
6 steps. It starts from this cell's own bright pixels and takes a pixel only if:
1. it is **bright** (halo): smoothed RAW above the local background by more than 3 σ. The neighbour's dark body stops it;
2. it is within 6 px of this cell's contour;
3. it is closer to this cell's contour than to the centre of the neighbour it lies in;
4. it is **on this cell's halo ring**: its distance from the cell centre is inside the 10th–90th percentile range of
   the halo pixels the band found on the free side.

A cell gets no entry if it has fewer than 20 free-side halo pixels (`--entry-min-ring-px`).
If that ring is wider than 5 px (`--entry-max-ring-width`), the cell has no ring (elongated or irregular cell,
example: B2_1 cell 261, rater #26). Rule 4 is then a **band parallel to the contour**: the pixel must be within
the halo width found on the free side (90th percentile of the free-side halo pixels' distance to the contour,
at least `--entry-band-min-px` 2, at most 6 px). Set `entry_band_fallback=False` (function argument) to turn the band off.
The entered pixels stay in the neighbour too: touching cells overlap there. The label map gives an overlapping
pixel to one cell only; the per-cell mask files hold the full masks.

Then holes that the cell and its entered ring enclose, within 6 px of the contour, are filled. This takes the
cell's own dark edge inside the neighbour's contour (between the body and the ring; it is not bright).
Manifest column: `entry_px` (ring + filled gap).

Checked versions (B2_1):

| Version | Cells with entry | Pixels | Result |
| :--- | :---: | :---: | :--- |
| Rules 1–3 only | 518 | 62,613 | Far too much: deep into neighbour bodies (224, 147, 64, 65) |
| + ring (rule 4) | 386 | 21,782 | 549 correct; blobs on non-round cells (317, 75, 236) |
| + ring width ≤ 5 px | 191 | 6,628 | 549 full ring but a gap at its own dark edge; 317, 75, 236 no entry; 170 doubtful (+53 px) |
| + fill enclosed gap | 191 | 7,344 | 549 complete, no gap; 261 (elongated) gets no entry |
| + parallel band for cells with no ring | 397 | 19,179 | 261 gets part of its lower aura; 317, 75, 236 get thin strips, no blobs; 170 still doubtful (+58 px) |
| **+ enter the neighbours' halo pixels too (default)** | **399** | **22,927** | 261: the lower cell's halo band no longer keeps the shared aura alone |
| Same, band minimum 4 px | 400 | 34,433 | 261 gets the full lower aura (812 px²); 317 takes larger pieces of both neighbours, 75 and 86 take strips |
| Same, band minimum 6 px | 400 | 43,877 | More of the same; 455 takes a blob |
| Default + strongly bright pixels (> 10 σ) past the band, up to 6 px (`--entry-strong-k 10`) | 399 | 34,340 | 261 gets most of its bottom aura (801 px²); 317, 75, 86 take the neighbours' halos |
| Same, > 6 σ (`--entry-strong-k 6`) | 400 | 38,742 | 261: 807 px²; more of the neighbours' halos |

Brightness cannot tell whose aura a pixel is: the neighbours' halos are as bright as the cell's own aura.
The strong-bright test is off by default (`--entry-strong-k 0`).

Debug step `06_step4b_halo_band.png` (on RAW): white = contour mask edge, yellow = signal pixels added,
green = gaps filled, blue = signal pixels removed by the leak or neck rule, red = found but not used
(area rule), magenta = contested, orange = slow entry into a neighbour.

---

## 📊 Results (default settings: halo band + slow entry)

| Metric | B2_1 | B2_2 |
| :--- | :---: | :---: |
| Cells extracted (`pair_contour` → `pair_probe`) | 568 → 564 | 345 → 333 |
| Cells excluded at the image edge | 30 → 34 | 39 → 51 |
| Mean cell area (px²) | 752.0 → 846.6 | 823.7 → 903.5 |
| Area added per cell, median / 90th percentile | 11.9% / 28.8% | 9.0% / 26.1% |
| Cells with slow entry / pixels | 399 / 22,927 | 205 / 9,778 |
| Signal left outside, median (QA) | 39.0% → 23.3% | 34.3% → 21.6% |
| Contested pixels (given to no cell) | 9,175 | 4,436 |

More cells reach the image edge because the band grows the mask; those cells are excluded (border rule).

### Halo band variants checked on B2_1 (2026-11-01, before slow entry)

Each rule was added one at a time and checked visually on the six rater cells and on cells 21, 53, 107, 142, 170,
245, 454 and 460.

| Variant | Cells | Mean area | Pixels added | What it does |
| :--- | :---: | :---: | :---: | :--- |
| Contour only | 568 | 752 | 0 | `pair_contour/` |
| Growth only | 564 | 826 | 41,856 | Takes the halo, but also debris (432) and neighbour parts (454, 142, 460) |
| + leak rule | 564 | 818 | 37,546 | Small change alone (part of 460) |
| + neck rule | 566 | 800 | 26,757 | Removes the debris and the neighbour parts; also some real halo |
| **+ gap fill (default)** | **564** | **806** | **30,030** | Puts back a smooth halo edge (549, 53, 245) |
| Leak zone 2 px | 564 | 809 | 31,527 | A little more halo (21), more of the neighbour at 460 |
| Area rule 25% | 564 | 803 | 28,371 | 14 cells keep their contour mask |

Decision 2026-11-01: keep the default. Cells enclosed by neighbours are **not** filtered
(a "3+ blocked directions" rule removed 9–24% of the cells; limiting it to round cells was rejected too).

### Known limits

* Halo between touching cells: taken by slow entry only for cells with a measurable thin ring; otherwise it stays out.
* The cell's own body inside a neighbour's contour is taken only where it is enclosed by the entered ring.
  Example: the right side of cell 147 (rater #10) lies 3–4 px past its contour; that gap is contested, and
  the cell has 19 free-side halo pixels (minimum 20), so it gets no entry.
* Cell 460 (B2_1) keeps a few pixels of a neighbour with no contour.
* Cell 170 (B2_1) gets a pale area below the cell (+24%); it is not clear if it belongs to the cell.
* Faint parts of the halo below the threshold stay out.
* The leak rule also stops the band where a process leaves the cell (problem 2).

---

## ❌ Tried and Dropped: Thin Processes (Problem 2)

On 2026-11-01 a separate experiment (`pair_process/`, now deleted) added free processes: thin signal
lines outside all drawn outlines that leave a cut and end in a free tip. `pair_probe` stays the current method.

Candidates: connected signal components (> 3 σ) outside all cuts and outlines that touch a cut, not a
closed shape, width ≤ 9 px (90th percentile along the skeleton), skeleton ≥ 15 px, length ≥ 2 × width,
and a free tip ≥ 8 px from the cut.

| Owner rule tried | Result (B2_1) |
| :--- | :--- |
| Touches one cut only | 6 cells (24, 36, 104, 325, 328, 441). The process at 24 belongs to the long cell 19 |
| Root clearance 6 px from other cells | No change |
| Narrowing: cell covers ≤ 25% of an 8 px disk at the root | Too strict: kept only 36 and 441 |
| Line continuation ≥ 7 px into the cell | Not stable: kept 24 (wrong), rejected 325 (wrong) |
| No rule: user review of every proposal (one per possible cell, near limit 12 px) | 84 proposals. The user decided 29: 4 accepted (19, 36, 66, 104), 25 rejected |

Conclusion: most candidates are not processes of the proposed cell. The step was dropped.
Earlier rejected ideas for problem 1: the climb rule ("grow until it gets darker") and the line-based
entry. Both gave worse cuts.

---

## 🚀 Quick Start Commands

```bash
# Extract pair B2_2 with debug traces for every cell
.venv/bin/python pair_probe/process.py

# Another pair (the MASK path is derived: VID3553_B2_1 -> VID3553_MASK_B2_1)
.venv/bin/python pair_probe/process.py --raw Data/pair-data/1/VID3553_B2_1_00d00h00m.tif

# Halo band and slow entry off: same output as pair_contour/
.venv/bin/python pair_probe/process.py --halo-px 0 --halo-leak-px 0 --entry-px 0

# Halo band only, no slow entry
.venv/bin/python pair_probe/process.py --entry-px 0

# Stricter area rule
.venv/bin/python pair_probe/process.py --halo-max-added 0.25

# Rebuild the gallery
.venv/bin/python pair_probe/gallery.py
```

Open [`pair_probe/output/gallery.html`](output/gallery.html) in a browser. It has the same keys as the
`pair_contour/` gallery (`1`/`2`/`3` views, `D` debug trace, `O` overview, `←`/`→` cells, `↑`/`↓` pairs).
