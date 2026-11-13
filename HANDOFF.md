# Handoff: MicroGlia Phase 2 — all slides on the labeling page, extra Resolution page live

You are continuing an M.Sc. research project (Doron Peleg, Afeka / HUJI; supervisor Dr. Hadas Lapid,
advisor Dr. Lilach Gavish). Phase 2 cuts single microglia cells out of phase-contrast images and has an
expert label them. The repo is github.com/dpartuk/MicroGlia (branch main). Read README.md,
pair_probe/README.md (the current cutting method) and labeling/README.md (the labeling tool) before you change anything.

## Working rules (from the user)
- Write replies and docs in ASD-STE100 Simplified Technical English (see AGENTS.md): sentences
  under 20 words, active voice, one instruction per sentence, same term for the same thing.
- Never `git commit` or `git push` without explicit approval for that specific action.
- Leave `README_FOR_PROFESSOR.md` (deleted on disk) and the untracked `README_Setup.md` out of commits.
- Prefer fewer mistakes over more cells. Judge added area strictly. A little overlap between touching cells is OK.
- Earlier experiments are frozen. Make a new folder for a new experiment instead of changing an old one.
- Check results visually (render crops and look at them) before you report success.
  For web pages: test in headless Chrome (puppeteer-core + /Applications/Google Chrome.app) and look at screenshots.
- Rater-facing text must not explain the morphology classes. One expert rater ("R.N").

## Setup on a new machine
```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt   # .venv has no pip; always use uv
uv pip install --python .venv/bin/python scikit-learn           # for labeling/select_resolution.py
```
Raw data must be at `Data/pair-data/1..4/` (git-ignored): 64 RAW + MASK pairs, 16 per folder.
**Folders 2 and 4:** the MASK file names say `00d00h00m`, but the RAW images are at 4 h / 1 day. The masks are drawn
on the RAW of their own folder (checked: grey correlation 0.99). Pass `--mask` explicitly for these folders:
```bash
.venv/bin/python pair_probe/process.py --no-debug --raw Data/pair-data/2/VID3553_B2_1_00d04h00m.tif \
    --mask Data/pair-data/2/VID3553_MASK_B2_1_00d00h00m.tif
```
`pair_probe/output/`, `labeling/web/` and `labeling/web_extra/` are git-ignored. Get the published slide files with the
Artifact tool (`action: read`, `paths: ["slides/<slide>.js", ...]`) into `labeling/web/slides/` (and `labeling/web_extra/slides/`).
`labeling/check_stable.py` must print "all published cells are unchanged".

## State on 2026-11-13

### Cutting method: pair_probe (current)
- pair_contour + halo band + slow entry into touching neighbours (problem 1, cells cut off at the outline).
- Problem 2 (thin processes) was tried in pair_process/ and dropped; the user rejected 25 of 29 reviewed proposals.
  pair_process/ was deleted. Record: "Tried and Dropped" in pair_probe/README.md.
- Run on all 64 pairs (debug traces only for B2_1 and B2_2 of folder 1): 27,849 cells
  (folder 1: 5,914; 2: 6,587; 3: 7,249; 4: 8,099). Gallery: pair_probe/output/gallery.html.

### Labeling page (original) — all 64 slides
- https://claude.ai/artifact/7vzmvdo8PKKW37XxJYymWT. Lock: labeling/published_slides.json (64 slides, commit 8320ca2).
- B2_1 and B2_2 were republished with pair_probe cuts on 2026-11-01. The rater's 36 labels were kept with
  `export_web.py --force --keep-labels` (old cell keys kept, old mask hash in `prev_mh`, merge status `matched_previous_cut`).
  The rater has since relabeled those cells.
- Duplicate centroid (VID3553_B2_4_00d04h00m cells 322/324, nested outlines): the later cell gets key suffix `_2`.
- Progress on 2026-11-13: 2,195 first looks on folder 1 B2_1–B2_6 (all cells of those slides) + 219 second looks.
  Resting 916, Surveilling 635, Activated 418, Resolution 114 (5.2%), Dystrophic 31 (1.4%), not a cell 77, unsure 4.
  Intra-rater kappa 0.75. All labels have High confidence and no notes.

### Extra page: Resolution set 1 — live, not yet used
- https://claude.ai/artifact/WrMMHRUhyZ8Hf5UWHdvXC4. Separate artifact and store; the original page is not affected.
- `labeling/select_resolution.py`: logistic model on shape/texture measures (leave-one-slide-out AUC 0.98,
  top 5% precision 71%). Selection `labeling/extra/resolution_set1.csv`: 1,500 unlabeled cells, at most 50% per folder,
  58 slides, no decoys, 10% second looks. Lock: labeling/published_extra.json.
- A cell may be labeled on both pages (user decision: a consistency check). Class shares come only from the original page.
- On 2026-11-13 its store was empty. **The user must share the page with the rater** (same access as the original page).
- Merge: `merge_labels.py <files> --slides-dir labeling/web_extra/slides --lock labeling/published_extra.json --page extra`.

### Other artifacts
- Rater instructions (Claude Doc): https://claude.ai/artifact/AYecc2epoNu7SLvDSksEd3
- Process review page (obsolete, process step dropped): https://claude.ai/artifact/TQrBmJjUGFXKYSmFaxeNUe

## Read labels
```
ArtifactData list collection "labels" (page URL) -> rater id
ArtifactData list collection "labels/<rater id>/slides" with out_dir -> json files
.venv/bin/python labeling/merge_labels.py "<out_dir>/labels/*/slides/*.json" --out <csv>
```

## Next work
1. Confirm the rater uses the extra page (store not empty, status "Connected to shared store").
2. When extra labels arrive: merge with `--page extra`, report Resolution precision of the selection per folder.
3. Dystrophic is still rare (1.4%) and simple measures do not separate it. Possible next step: image-embedding
   similarity search from the Dystrophic examples, once more examples exist.
4. Possible reminder to the rater: use the confidence keys (Q / W / E); all labels so far are High.
