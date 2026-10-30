# Handoff: MicroGlia Phase 2 — continue from the previous machine

You are continuing an M.Sc. research project (Doron Peleg, Afeka / HUJI; supervisor Dr. Hadas Lapid,
advisor Dr. Lilach Gavish). Goal of the current phase: cut single microglia cells out of
phase-contrast microscopy images to build a dataset that experts will label. The repo is
github.com/dpartuk/MicroGlia (branch main, last commit b839070). Read README.md, then
pair_contour/README.md (the main method), before you change anything.

## Working rules (from the user)
- Write replies and docs in ASD-STE100 Simplified Technical English (see AGENTS.md): sentences
  under 20 words, active voice, one instruction per sentence, same term for the same thing.
- Never `git commit` or `git push` without explicit approval for that specific action.
- **Optimize for fewer mistakes, not more cells.** Prefer strict filters. Report how many cells a
  rule removes. Offer looser options instead of making them the default.
- Earlier experiments are frozen. Make a new folder for a new experiment instead of changing an old one.
- Check results visually (render crops/debug panels and look at them) before you report success.

## Setup on this machine
```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt   # .venv has no pip; always use uv
```
Raw data must be at `Data/pair-data/1..4/` (git-ignored). Each folder has 16 pairs:
`VID3553_B2_N_00d00h00m.tif` (RAW) + `VID3553_MASK_B2_N_00d00h00m.tif` (same image, contrast-stretched
~2x around grey 128, with hand-drawn cyan cell outlines). Folders 1–4 = two videos (VID3553, VID3552)
at time points 00d00h, 00d04h, 01d00h. Read TIFFs with cv2.imread (tifffile needs imagecodecs for LZW).
`dpgnet/output/checkpoints/best_model.pth` is not in git (only needed to re-run DPGNet).

## Experiments (Phase 2)
1. **pair_contour/ — the chosen method.** Cells come from the MASK cyan outlines; pixels are cut from
   RAW. Default pipeline (pair B2_2):
   - cyan line: `min(B,G) − R`; core ≥ 100; enclosed regions = 4-connected components of non-core pixels
     (8-connectivity leaks through the 1-px diagonal lines); largest = background; 60–30,000 px kept.
   - gap bridging: skeleton endpoints ≤ 8 px apart are joined (B2_1/B2_2 have no real gaps; tested on a
     synthetic gap).
   - **mask = inside of the contour + the contour line itself, nothing beyond** (user decision).
     Aura growth (`--aura-free-px 2 --aura-px 10`) and contact completion (`--contact-fit-residual 1.5`)
     exist but are OFF by default because they cropped too much.
   - border exclusion: any cell whose mask reaches the image edge is not cut (39 on B2_2).
     Cell numbers are fixed after aura step sorting, so numbering has gaps on purpose.
   - cut onto a magenta (255,0,255) canvas as PNG (RAW is pure grey, so magenta never occurs).
   - QA "signal left outside" (|RAW − local background| > 3σ in a 4 px ring). With the contour-only
     mask it mainly measures the excluded aura (median 34%, 240 flagged), so its flag no longer marks
     mistakes. **Open decision:** drop the flag, or redefine it to detect real leaks only.
   - outputs: manifest.csv, summary.json, debug traces per cell, 02_cell_outline_overview.jpg,
     03_taken_vs_ignored.png (green = cut, red = outlined but ignored).
   - Result B2_2: 345 cells cut, 39 excluded at the edge, mean area 824 px², 0 pixels beyond the contour,
     cut pixels identical to RAW.
   - gallery (`pair_contour/gallery.py` → output/gallery.html): copy of the frozen contour gallery;
     opens on the taken/ignored overview (O / Esc); 1 RAW cut, 2 MASK crop, 3 dual, D debug,
     −/+ crop zoom (default 3×, smooth), P hard pixels. JS was syntax-checked but **never tested in a
     real browser** — check it.
   Run: `.venv/bin/python pair_contour/process.py && .venv/bin/python pair_contour/gallery.py` (~5 s).
2. **mask_contour/** — interior only, without the line, pixels from the MASK image (clipped, 2× noise).
   Also has an AI-upscaled ×4 comparison: upscaling repaints texture and adds nothing; rejected.
3. **enhance/** — RAW is soft, not noisy (σ ≈ 1.1). BM3D passes all feature checks but gains little;
   deconvolution invents structure and fails. Recommendation: no enhancement step for now.

Key finding: the RAW image has the better quality; MASK is RAW contrast-stretched with lines burned in.
Use MASK only for the outlines.

## Next work (agreed direction, not started)
1. **Manual labeling of the cut cells by an expert.** Proposed design: a labeling mode inside the
   pair_contour gallery — keys 1–5 for Resting / Surveilling / Activated / Resolution / Dystrophic
   (Presaizen 2026 Table 1 rubric visible), X = not a cell / bad crop, U = unsure, optional confidence
   and note; autosave in the browser; export JSON/CSV; a merge script into labels.csv.
   **Labels must be keyed by slide name + outline position (centroid/box), not by cell number**, and
   matched by overlap, because cell numbers change when the pipeline changes. Add random order and
   ~10% double labeling for inter-rater agreement (Cohen's κ).
   Still to ask the user: final label set; how many experts and where they work.
2. **Expose it externally.** Options discussed: shared drive zip with export/merge; a private
   claude.ai page with a shared label database (this machine uses a PERSONAL claude.ai account, which is
   why we moved); a university server with Label Studio/CVAT later. Before publishing anything, confirm
   the lab allows uploading the cell images to claude.ai. A web export should drop the debug traces and
   per-cell full-slide images (~240 MB → ~10–20 MB) and draw boxes client-side.
3. Later: run pair_contour on all 64 pairs with a per-image summary (user deferred this until the method
   is final).

Start by: setting up the venv, running pair_contour on pair B2_2, opening the gallery in a browser to
verify it works, then asking the user the open labeling questions above.
