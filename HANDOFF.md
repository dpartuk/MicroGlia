# Handoff: MicroGlia Phase 2 — expert labeling is live

You are continuing an M.Sc. research project (Doron Peleg, Afeka / HUJI; supervisor Dr. Hadas Lapid,
advisor Dr. Lilach Gavish). Phase 2 cuts single microglia cells out of phase-contrast images and has an
expert label them. The repo is github.com/dpartuk/MicroGlia (branch main). Read README.md,
pair_contour/README.md (the cutting method) and labeling/README.md (the labeling tool) before you change anything.

## Working rules (from the user)
- Write replies and docs in ASD-STE100 Simplified Technical English (see AGENTS.md): sentences
  under 20 words, active voice, one instruction per sentence, same term for the same thing.
- Never `git commit` or `git push` without explicit approval for that specific action.
- **Optimize for fewer mistakes, not more cells.** Prefer strict filters. Report how many cells a
  rule removes. Offer looser options instead of making them the default.
- Earlier experiments are frozen. Make a new folder for a new experiment instead of changing an old one.
- Check results visually (render crops/debug panels and look at them) before you report success.
  For web pages: test in headless Chrome (puppeteer-core + /Applications/Google Chrome.app) and look at screenshots.

## Setup on a new machine
```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt   # .venv has no pip; always use uv
```
Raw data must be at `Data/pair-data/1..4/` (git-ignored): 64 RAW + MASK pairs, 16 per folder.
`pair_contour/output/`, `labeling/web/` and `labeling/reference/` are git-ignored. Rebuild them:
```bash
.venv/bin/python pair_contour/process.py                                                    # B2_2: 345 cells
.venv/bin/python pair_contour/process.py --raw Data/pair-data/1/VID3553_B2_1_00d00h00m.tif  # B2_1: 568 cells
.venv/bin/python pair_contour/gallery.py
.venv/bin/python labeling/export_web.py      # all slides in pair_contour/output -> labeling/web + reference
```
The cell keys depend only on the masks, so a rebuild with the same settings gives the same keys.

## State on 2026-10-30

### pair_contour (cutting method) — unchanged defaults
Contour-only mask, border exclusion, magenta canvas. Run on **2 of 64 pairs only**: B2_2 (345 cells, 39 at the edge)
and B2_1 (568 cells, 30 at the edge, 1 rejected for size). The user deferred the run on all 64 pairs.
Decision 2026-10-30: the gallery shows "signal left outside" as "info only" and no longer shows FLAGGED;
`qa_flag` stays in manifest.csv. The gallery JS was tested in headless Chrome: all keys work, no errors.

### labeling/ (new) — live with one rater
- **Published page:** https://claude.ai/artifact/7vzmvdo8PKKW37XxJYymWT (Version 4). The user shared it with the rater.
- **Rater instructions (Claude Doc):** https://claude.ai/artifact/AYecc2epoNu7SLvDSksEd3 — sections: open the page,
  confidence + class, export, start over, stop and continue. An open comment asks the user: email or a shared
  Drive folder for the export file. The doc says email for now.
- **Decisions:** one rater labels all cells (no inter-rater kappa). The rater is a domain expert: no class definitions
  in the instructions. Labels: 1 Resting, 2 Surveilling, 3 Activated, 4 Resolution, 5 Dystrophic, X not a cell / bad cut,
  U unsure. Confidence Low / Medium / High, default High, picked before the class key. 10% of each slide's cells
  come back as a "second look" (intra-rater kappa). The data upload to claude.ai is approved by the user.
- **Bundle:** per slide one `slides/<slide>.js` (full RAW as lossless PNG + per-cell mask bits + outline), ~1.3 MB.
  The browser draws the cut and surroundings. Checked: decoded pixels in every mask equal RAW (913 cells, 0 mismatches).
  64 slides ≈ 85 MB: fits one page version (256 MB, 511 files) but needs 2 publishes (64 MB per publish).
- **Shared store layout:** `labels/<rater id>` = {rater, updated}; `labels/<rater id>/slides/<slide>` = {slide, cells, repeats}.
  Rules: `labels` read/write owner; `labels/{self}` read/write interact. Verified: Editors cannot read another
  rater's documents; the owner reads all. On a personal claude.ai account the rater must be invited as **Editor**.
- **Not yet verified:** a real save and a Start over from the live page (only possible inside claude.ai).
  First task next time: read the store and confirm the rater's labels arrive:
  `ArtifactData list collection "labels"` with the page URL, then `labels/<id>/slides`.
- **Merge:** `labeling/merge_labels.py` takes page exports or store documents saved by ArtifactData `out_dir`
  (`<dir>/labels/<id>/slides/<slide>.json`). It matches labels to the current pipeline output by mask IoU ≥ 0.5 and
  writes labeling/labels.csv + kappa. Tested: same output IoU 1.0; with aura on, the 11 cells that move to the edge
  are reported unmatched.

## Next work
1. Confirm the rater's first labels arrive in the store (see above). Then confirm Start over deletes from the store.
2. Answer the open doc comment (email vs Drive) and update section 3 of the instructions if needed.
3. When the user approves: run pair_contour on the other 62 pairs, check each overview visually, export, and
   republish in two batches. New slides do not change the order of slides already labeled.
4. When labels come in: dump the store with ArtifactData (`out_dir`) and run merge_labels.py.
