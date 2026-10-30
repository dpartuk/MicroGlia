#!/usr/bin/env python3
"""
build_methods_gallery.py

Generates an offline, interactive multi-method gallery at
comparison/output/methods_gallery.html

For every slide, the gallery overlays the cell outlines of all available
methods (baseline, sharpened, contour, dpgnet) on the original image.
Two methods can be selected as "A" and "B"; each cell is then classified
by its pixel overlap with the other method:

  matched   - exactly one partner cell, and that partner has only this cell
  split     - the cell overlaps 2+ cells of the other method
  merged    - the cell shares its single partner with other cells
  only      - the cell has no partner in the other method

Cell outlines are reconstructed from each method's white-background crop
(manifest.csv + extracted crop image), so the gallery reads only the
existing pipeline outputs and needs no change to the pipelines.
"""

import csv
import glob
import json
import os

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(COMPARISON_ROOT, ".."))
OUTPUT_DIR = os.path.join(COMPARISON_ROOT, "output")
GALLERY_PATH = os.path.join(OUTPUT_DIR, "methods_gallery.html")

# (key, display name, colour, crop column in manifest.csv)
METHODS = [
    ("baseline", "Baseline", "#f59e0b", "filename"),
    ("sharpened", "Sharpened", "#a78bfa", "original_filename"),
    ("contour", "Contour (GT)", "#4ade80", "original_filename"),
    ("dpgnet", "DPGNet", "#f472b6", "original_filename"),
]

# Pixels at or above this value in all channels are the white crop canvas.
WHITE_LEVEL = 235
# Minimum overlap, as a fraction of the smaller cell, to count two cells as partners.
MIN_OVERLAP_FRACTION = 0.10


def recover_cell_mask(crop_bgr):
    """Rebuild the filled cell mask from a crop whose background is white."""
    non_white = (crop_bgr.min(axis=2) < WHITE_LEVEL).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(non_white, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(closed)
    cv2.drawContours(mask, contours, -1, 255, cv2.FILLED)
    return mask, contours


def load_method_slide(method_key, crop_col, slide_dir, img_shape):
    """Load one method's cells for one slide: label image + per-cell records."""
    h, w = img_shape[:2]
    labels = np.zeros((h, w), dtype=np.int32)
    cells = []
    manifest_path = os.path.join(slide_dir, "manifest.csv")
    rel_dir = os.path.relpath(slide_dir, OUTPUT_DIR)

    with open(manifest_path, newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        cell_id = int(row["cell_id"])
        x1, y1 = int(row["x1"]), int(row["y1"])
        crop = cv2.imread(os.path.join(slide_dir, row[crop_col]))
        if crop is None:
            continue
        mask, contours = recover_cell_mask(crop)
        ch, cw = mask.shape
        region = labels[y1:y1 + ch, x1:x1 + cw]
        # Later cells do not overwrite earlier ones; overlaps inside one method are rare.
        region[(mask > 0) & (region == 0)] = cell_id

        polygons = []
        for c in contours:
            if cv2.contourArea(c) < 4:
                continue
            approx = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
            polygons.append([[int(px + x1), int(py + y1)] for px, py in approx])

        record = {
            "id": cell_id,
            "box": [x1, y1, int(row["x2"]), int(row["y2"])],
            "area": int(row["area"]),
            "maskArea": int((mask > 0).sum()),
            "polys": polygons,
            "crop": f"{rel_dir}/{row[crop_col]}",
        }
        if row.get("flagged_possible_merge") == "True":
            record["flagged"] = True
        if "mean_uncertainty" in row:
            record["uncertainty"] = float(row["mean_uncertainty"])
        cells.append(record)

    return labels, cells


def compute_overlaps(labels_a, cells_a, labels_b, cells_b):
    """For each cell of A, list the cells of B it overlaps: [[b_id, shared_px], ...]."""
    area_a = {c["id"]: c["maskArea"] for c in cells_a}
    area_b = {c["id"]: c["maskArea"] for c in cells_b}
    both = (labels_a > 0) & (labels_b > 0)
    pairs = np.stack([labels_a[both], labels_b[both]], axis=1)
    result = {c["id"]: [] for c in cells_a}
    if len(pairs) == 0:
        return result
    uniq, counts = np.unique(pairs, axis=0, return_counts=True)
    for (a_id, b_id), shared in zip(uniq, counts):
        a_id, b_id, shared = int(a_id), int(b_id), int(shared)
        smaller = max(1, min(area_a.get(a_id, 1), area_b.get(b_id, 1)))
        if shared / smaller >= MIN_OVERLAP_FRACTION:
            result[a_id].append([b_id, shared])
    return result


def discover_slides():
    """Return slide names that have output for at least one method."""
    slides = set()
    for key, _, _, _ in METHODS:
        for manifest in glob.glob(os.path.join(PROJECT_ROOT, key, "output", "*", "manifest.csv")):
            slides.add(os.path.basename(os.path.dirname(manifest)))
    return sorted(slides)


def build_slide(slide):
    """Collect all methods' cells and pairwise overlaps for one slide."""
    reference = None
    present = []
    for key, _, _, crop_col in METHODS:
        slide_dir = os.path.join(PROJECT_ROOT, key, "output", slide)
        if os.path.isfile(os.path.join(slide_dir, "manifest.csv")):
            present.append((key, crop_col, slide_dir))
            ref_path = os.path.join(slide_dir, "00_original_reference.jpg")
            if reference is None and os.path.isfile(ref_path):
                reference = ref_path

    if reference is None:
        return None
    img = cv2.imread(reference)
    h, w = img.shape[:2]

    labels = {}
    methods = {}
    for key, crop_col, slide_dir in present:
        lab, cells = load_method_slide(key, crop_col, slide_dir, img.shape)
        labels[key] = lab
        methods[key] = {"cells": cells}

    for key_a in methods:
        overlaps = {}
        for key_b in methods:
            if key_a == key_b:
                continue
            overlaps[key_b] = compute_overlaps(
                labels[key_a], methods[key_a]["cells"], labels[key_b], methods[key_b]["cells"]
            )
        for cell in methods[key_a]["cells"]:
            cell["ov"] = {key_b: overlaps[key_b][cell["id"]] for key_b in overlaps}

    return {
        "name": slide,
        "width": w,
        "height": h,
        "image": os.path.relpath(reference, OUTPUT_DIR),
        "methods": methods,
    }


def build_gallery():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    slides = []
    for slide in discover_slides():
        data = build_slide(slide)
        if data is None:
            continue
        counts = ", ".join(f"{k}={len(v['cells'])}" for k, v in data["methods"].items())
        print(f"  - {slide}: {counts}")
        slides.append(data)

    # Imported here because build_cell_comparison imports this module.
    from build_cell_comparison import build_report
    rows_by_slide = build_report(slides)
    for data in slides:
        data["rows"] = rows_by_slide.get(data["name"], [])

    method_meta = [{"key": k, "name": n, "color": c} for k, n, c, _ in METHODS]
    payload = json.dumps({"methods": method_meta, "slides": slides}, separators=(",", ":"))
    html = HTML_TEMPLATE.replace("__GALLERY_DATA__", payload)

    with open(GALLERY_PATH, "w") as f:
        f.write(html)
    print(f"\n[SUCCESS] Multi-method gallery written to: {GALLERY_PATH}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Method Explorer | MicroGlia</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1117;
    --panel-bg: #181b22;
    --panel-sub: #212631;
    --border: #2e3545;
    --text: #f0f3f8;
    --muted: #8b949e;
    --matched: #10b981;
    --split: #f59e0b;
    --merged: #c084fc;
    --only: #ef4444;
  }
  * { box-sizing: border-box; }
  [hidden] { display: none !important; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
    height: 100vh;
    overflow: hidden;
    font-size: 13px;
  }
  header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    flex-wrap: wrap;
    gap: 10px 18px;
    align-items: center;
  }
  header h1 { font-size: 16px; margin: 0 8px 0 0; }
  .group { display: flex; gap: 6px; align-items: center; }
  .group > span.lbl { color: var(--muted); }
  select, button {
    background: var(--panel-sub);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 5px 9px;
    font: inherit;
    cursor: pointer;
  }
  select:hover, button:hover { border-color: #4b5568; }
  button.on { outline: 2px solid currentColor; outline-offset: -2px; }
  .seg { display: inline-flex; }
  .seg button { border-radius: 0; }
  .seg button:first-child { border-radius: 6px 0 0 6px; }
  .seg button:last-child { border-radius: 0 6px 6px 0; }
  .seg button.active { background: #334155; }
  main { flex: 1; display: flex; min-height: 0; }
  #stage {
    flex: 1;
    overflow: auto;
    position: relative;
    background: #000;
  }
  #canvas { position: relative; transform-origin: 0 0; }
  #canvas img { display: block; user-select: none; }
  #canvas svg { position: absolute; left: 0; top: 0; }
  #canvas polygon { fill: transparent; stroke-width: 1.5; cursor: pointer; vector-effect: non-scaling-stroke; }
  #canvas polygon:hover { fill: rgba(255,255,255,0.18); }
  #canvas polygon.sel { stroke-width: 3; fill: rgba(255,255,255,0.25); }
  #canvas polygon.dim { opacity: 0.15; }
  aside {
    width: 380px;
    flex: 0 0 380px;
    border-left: 1px solid var(--border);
    background: var(--panel-bg);
    overflow-y: auto;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  aside h2 { font-size: 13px; margin: 0 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 4px 6px; text-align: right; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; vertical-align: -1px; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 14px; color: var(--muted); }
  .cellcard { background: var(--panel-sub); border: 1px solid var(--border); border-radius: 8px; padding: 8px; }
  .cellcard .hd { display: flex; justify-content: space-between; margin-bottom: 6px; }
  .crops { display: flex; flex-wrap: wrap; gap: 6px; }
  .crops figure { margin: 0; background: #fff; border-radius: 4px; padding: 2px; cursor: pointer; }
  .crops img { display: block; max-width: 160px; max-height: 160px; }
  .crops figcaption { color: #333; font-size: 11px; text-align: center; }
  .status { font-weight: 600; }
  .muted { color: var(--muted); }
  .hint { color: var(--muted); line-height: 1.5; }
  #tableView { flex: 1; overflow: auto; padding: 0 12px 12px; }
  #cellTable { width: auto; min-width: 100%; }
  #cellTable th { position: sticky; top: 0; background: var(--bg); padding-top: 10px; }
  #cellTable th.mhead { text-align: center; border-bottom: 2px solid currentColor; }
  #cellTable tbody tr { cursor: pointer; }
  #cellTable tbody tr:hover { background: var(--panel-sub); }
  #cellTable td.st { text-align: left; font-weight: 600; }
  .st-matched { color: var(--matched); }
  .st-split { color: var(--split); }
  .st-merged { color: var(--merged); }
  .st-missed, .st-extra { color: var(--only); }
</style>
</head>
<body>
<header>
  <h1>Method Explorer</h1>
  <div class="group">
    <span class="lbl">Slide</span>
    <select id="slideSelect"></select>
    <button id="prevBtn" title="Previous slide">&larr;</button>
    <button id="nextBtn" title="Next slide">&rarr;</button>
  </div>
  <div class="group">
    <span class="lbl">View</span>
    <div class="seg" id="modeSeg">
      <button data-mode="overlay" class="active" title="Show outlines of the selected methods in their own colours">Overlay</button>
      <button data-mode="diff" title="Colour the cells of method A by how they match method B">A vs B diff</button>
      <button data-mode="table" title="One row per ground-truth cell, compared across all methods">Cell table</button>
    </div>
  </div>
  <div class="group" id="tableGroup" hidden>
    <span class="lbl">Show</span>
    <select id="rowFilter">
      <option value="all">All rows</option>
      <option value="diff">Rows with any difference</option>
      <option value="split">Split</option>
      <option value="merged">Merged</option>
      <option value="missed">Missed</option>
      <option value="extra">Extra (not in GT)</option>
    </select>
    <span class="muted" id="rowCount"></span>
  </div>
  <div class="group" id="overlayGroup">
    <span class="lbl">Methods</span>
    <span id="methodToggles"></span>
  </div>
  <div class="group" id="diffGroup" hidden>
    <span class="lbl">A</span><select id="aSelect"></select>
    <button id="swapBtn" title="Swap A and B">&#8646;</button>
    <span class="lbl">B</span><select id="bSelect"></select>
    <label class="muted"><input type="checkbox" id="showB" checked> show B outlines</label>
  </div>
  <div class="group">
    <span class="lbl">Zoom</span>
    <select id="zoomSelect">
      <option value="fit">Fit</option>
      <option value="1">100%</option>
      <option value="1.5">150%</option>
      <option value="2">200%</option>
      <option value="3">300%</option>
    </select>
  </div>
</header>
<main>
  <div id="stage"><div id="canvas"><img id="slideImg" alt=""><svg id="overlay"></svg></div></div>
  <div id="tableView" hidden><table id="cellTable"></table></div>
  <aside>
    <section>
      <h2>Cell counts</h2>
      <table id="countTable"></table>
    </section>
    <section id="diffSummary" hidden>
      <h2>A vs B match summary</h2>
      <div class="legend" id="diffLegend"></div>
      <table id="diffTable" style="margin-top:6px"></table>
    </section>
    <section>
      <h2>Selected cell</h2>
      <div id="cellDetail" class="hint">Click a cell outline to compare it across all methods.</div>
    </section>
    <section class="hint">
      <h2>How to read</h2>
      <b>Overlay</b> draws each method's cells in its own colour.<br>
      <b>A vs B diff</b> colours the cells of A by their partner cells in B:<br>
      <span style="color:var(--matched)">matched</span> &ndash; one-to-one;
      <span style="color:var(--split)">split</span> &ndash; B splits it into 2+ cells;
      <span style="color:var(--merged)">merged</span> &ndash; B joins it with other A cells;
      <span style="color:var(--only)">only in A</span> &ndash; B has no cell there.<br>
      <b>Cell table</b> has one row per ground-truth (Contour) cell. Each method shows its partner
      cell IDs, area, IoU and status (matched, split, merged, missed). Extra rows list cells
      that a method found where the ground truth has no cell. Click a row to see the cell on the slide.<br>
      Two cells are partners when they share at least 10% of the smaller cell's pixels.
      Outlines are rebuilt from each method's crop images.
    </section>
  </aside>
</main>
<script>
const DATA = __GALLERY_DATA__;
const STATUS_COLORS = { matched: 'var(--matched)', split: 'var(--split)', merged: 'var(--merged)', only: 'var(--only)' };
const STATUS_LABELS = { matched: 'matched', split: 'split in B', merged: 'merged in B', only: 'only in A' };
const methodMeta = Object.fromEntries(DATA.methods.map(m => [m.key, m]));

const state = {
  slide: 0,
  mode: 'overlay',
  visible: new Set(DATA.methods.map(m => m.key)),
  a: 'baseline',
  b: 'dpgnet',
  showB: true,
  zoom: 'fit',
  selected: null,  // {method, id}
};

const $ = id => document.getElementById(id);
const SVG_NS = 'http://www.w3.org/2000/svg';

function slide() { return DATA.slides[state.slide]; }
function available() { return DATA.methods.filter(m => slide().methods[m.key]); }
function cellsOf(key) { return (slide().methods[key] || { cells: [] }).cells; }
function cellById(key, id) { return cellsOf(key).find(c => c.id === id); }

function classify(cell, a, b) {
  const partners = (cell.ov && cell.ov[b]) || [];
  if (partners.length === 0) return 'only';
  if (partners.length >= 2) return 'split';
  const partner = cellById(b, partners[0][0]);
  const back = (partner && partner.ov && partner.ov[a]) || [];
  return back.length >= 2 ? 'merged' : 'matched';
}

function iou(cellA, cellB, shared) {
  return shared / (cellA.maskArea + cellB.maskArea - shared);
}

function initControls() {
  DATA.slides.forEach((s, i) => {
    const opt = document.createElement('option');
    opt.value = i; opt.textContent = s.name;
    $('slideSelect').appendChild(opt);
  });
  $('slideSelect').addEventListener('change', e => goTo(parseInt(e.target.value)));
  $('prevBtn').addEventListener('click', () => goTo((state.slide - 1 + DATA.slides.length) % DATA.slides.length));
  $('nextBtn').addEventListener('click', () => goTo((state.slide + 1) % DATA.slides.length));

  document.querySelectorAll('#modeSeg button').forEach(btn => btn.addEventListener('click', () => {
    state.mode = btn.dataset.mode;
    document.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('active', b === btn));
    render();
  }));
  $('swapBtn').addEventListener('click', () => { [state.a, state.b] = [state.b, state.a]; render(); });
  $('aSelect').addEventListener('change', e => { state.a = e.target.value; render(); });
  $('bSelect').addEventListener('change', e => { state.b = e.target.value; render(); });
  $('showB').addEventListener('change', e => { state.showB = e.target.checked; render(); });
  $('zoomSelect').addEventListener('change', e => { state.zoom = e.target.value; applyZoom(); });
  $('rowFilter').addEventListener('change', renderTable);
  window.addEventListener('resize', applyZoom);
  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'SELECT') return;
    if (e.key === 'ArrowRight') $('nextBtn').click();
    if (e.key === 'ArrowLeft') $('prevBtn').click();
  });
}

function goTo(i) {
  state.slide = i;
  state.selected = null;
  render();
}

function fillMethodSelect(sel, value) {
  sel.innerHTML = '';
  available().forEach(m => {
    const opt = document.createElement('option');
    opt.value = m.key; opt.textContent = m.name;
    sel.appendChild(opt);
  });
  sel.value = value;
}

function normaliseSelection() {
  const keys = available().map(m => m.key);
  if (!keys.includes(state.a)) state.a = keys[0];
  if (!keys.includes(state.b) || state.b === state.a) state.b = keys.find(k => k !== state.a) || state.a;
}

function renderToggles() {
  const box = $('methodToggles');
  box.innerHTML = '';
  available().forEach(m => {
    const btn = document.createElement('button');
    btn.textContent = m.name;
    btn.style.color = m.color;
    btn.classList.toggle('on', state.visible.has(m.key));
    btn.title = 'Show or hide this method';
    btn.addEventListener('click', () => {
      state.visible.has(m.key) ? state.visible.delete(m.key) : state.visible.add(m.key);
      render();
    });
    box.appendChild(btn);
  });
}

function addPolys(svg, cell, key, stroke, extraClass) {
  cell.polys.forEach(poly => {
    const el = document.createElementNS(SVG_NS, 'polygon');
    el.setAttribute('points', poly.map(p => p.join(',')).join(' '));
    el.style.stroke = stroke;
    if (extraClass) el.classList.add(extraClass);
    if (state.selected && state.selected.method === key && state.selected.id === cell.id) el.classList.add('sel');
    el.addEventListener('click', ev => { ev.stopPropagation(); select(key, cell.id); });
    const title = document.createElementNS(SVG_NS, 'title');
    title.textContent = `${methodMeta[key].name} #${cell.id} – area ${cell.area} px²`;
    el.appendChild(title);
    svg.appendChild(el);
  });
}

function renderOverlay() {
  const s = slide();
  const svg = $('overlay');
  svg.innerHTML = '';
  svg.setAttribute('width', s.width);
  svg.setAttribute('height', s.height);
  svg.setAttribute('viewBox', `0 0 ${s.width} ${s.height}`);

  if (state.mode === 'overlay') {
    available().forEach(m => {
      if (!state.visible.has(m.key)) return;
      cellsOf(m.key).forEach(c => addPolys(svg, c, m.key, m.color));
    });
  } else {
    if (state.showB) cellsOf(state.b).forEach(c => addPolys(svg, c, state.b, '#9ca3af', 'dim'));
    cellsOf(state.a).forEach(c => addPolys(svg, c, state.a, STATUS_COLORS[classify(c, state.a, state.b)]));
  }
}

function renderCounts() {
  const rows = available().map(m => {
    const cells = cellsOf(m.key);
    const total = cells.reduce((t, c) => t + c.area, 0);
    const mean = cells.length ? Math.round(total / cells.length) : 0;
    return `<tr><td><span class="swatch" style="background:${m.color}"></span>${m.name}</td>` +
      `<td>${cells.length}</td><td>${mean.toLocaleString()}</td><td>${total.toLocaleString()}</td></tr>`;
  });
  $('countTable').innerHTML = '<tr><th>Method</th><th>Cells</th><th>Mean px²</th><th>Total px²</th></tr>' + rows.join('');
}

function renderDiffSummary() {
  const show = state.mode === 'diff';
  $('diffSummary').hidden = !show;
  if (!show) return;
  const tally = (a, b) => {
    const t = { matched: 0, split: 0, merged: 0, only: 0 };
    cellsOf(a).forEach(c => t[classify(c, a, b)]++);
    return t;
  };
  const ta = tally(state.a, state.b);
  const tb = tally(state.b, state.a);
  const na = methodMeta[state.a].name, nb = methodMeta[state.b].name;
  $('diffLegend').innerHTML = Object.keys(STATUS_LABELS)
    .map(k => `<span><span class="swatch" style="background:${STATUS_COLORS[k]}"></span>${STATUS_LABELS[k]}</span>`).join('');
  const labelsFor = (x, y) => ({ matched: 'matched', split: `split by ${y}`, merged: `merged by ${y}`, only: `only in ${x}` });
  const la = labelsFor(na, nb), lb = labelsFor(nb, na);
  $('diffTable').innerHTML =
    `<tr><th>${na} cells</th><th>#</th><th>${nb} cells</th><th>#</th></tr>` +
    Object.keys(STATUS_LABELS).map(k =>
      `<tr><td style="color:${STATUS_COLORS[k]}">${la[k]}</td><td>${ta[k]}</td>` +
      `<td style="color:${STATUS_COLORS[k]}">${lb[k]}</td><td>${tb[k]}</td></tr>`).join('');
}

function select(key, id) {
  state.selected = { method: key, id };
  renderOverlay();
  renderDetail();
}

function cropFigure(key, cell, extra) {
  const m = methodMeta[key];
  return `<figure onclick="select('${key}', ${cell.id})" title="Select this cell">` +
    `<img src="${cell.crop}" alt="${m.name} cell ${cell.id}">` +
    `<figcaption><span class="swatch" style="background:${m.color}"></span>${m.name} #${cell.id}${extra || ''}</figcaption></figure>`;
}

function renderDetail() {
  const box = $('cellDetail');
  if (!state.selected) { box.className = 'hint'; box.innerHTML = 'Click a cell outline to compare it across all methods.'; return; }
  const { method, id } = state.selected;
  const cell = cellById(method, id);
  if (!cell) { state.selected = null; renderDetail(); return; }
  box.className = '';
  const m = methodMeta[method];
  const facts = [
    `area <b>${cell.area}</b> px²`,
    `box ${cell.box.join(', ')}`,
  ];
  if (cell.uncertainty !== undefined) facts.push(`mean uncertainty ${cell.uncertainty.toFixed(3)}`);
  if (cell.flagged) facts.push('<span style="color:var(--split)">flagged: possible merge</span>');

  let html = `<div class="cellcard"><div class="hd"><b style="color:${m.color}">${m.name} #${cell.id}</b></div>` +
    `<div class="muted" style="margin-bottom:6px">${facts.join(' &middot; ')}</div>` +
    `<div class="crops">${cropFigure(method, cell)}</div></div>`;

  available().forEach(o => {
    if (o.key === method) return;
    const partners = (cell.ov && cell.ov[o.key]) || [];
    const status = classify(cell, method, o.key);
    const label = { matched: 'matched', split: `split into ${partners.length} cells`, merged: 'merged with other cells', only: 'no matching cell' }[status];
    const figs = partners.map(([pid, shared]) => {
      const p = cellById(o.key, pid);
      return p ? cropFigure(o.key, p, ` &middot; IoU ${iou(cell, p, shared).toFixed(2)}`) : '';
    }).join('');
    html += `<div class="cellcard"><div class="hd"><b style="color:${o.color}">${o.name}</b>` +
      `<span class="status" style="color:${STATUS_COLORS[status]}">${label}</span></div>` +
      (figs ? `<div class="crops">${figs}</div>` : '<div class="muted">This method found no cell at this location.</div>') + '</div>';
  });
  box.innerHTML = html;
}

function renderTable() {
  const s = slide();
  const others = available().filter(m => m.key !== 'contour');
  const filter = $('rowFilter').value;
  const statuses = r => others.map(m => r[`${m.key}_status`]).filter(Boolean);
  const rows = (s.rows || []).filter(r => {
    const st = statuses(r);
    if (filter === 'all') return true;
    if (filter === 'diff') return st.some(x => x !== 'matched');
    return st.includes(filter);
  });
  $('rowCount').textContent = `${rows.length} of ${(s.rows || []).length} rows`;

  const gt = methodMeta['contour'];
  let head = `<thead><tr><th colspan="2" class="mhead" style="color:${gt.color}">${gt.name}</th>` +
    others.map(m => `<th colspan="4" class="mhead" style="color:${m.color}">${m.name}</th>`).join('') + '</tr>' +
    '<tr><th>Cell</th><th>Area</th>' + others.map(() => '<th>Status</th><th>Cells</th><th>Area</th><th>IoU</th>').join('') + '</tr></thead>';
  const body = rows.map(r => {
    const target = r.gt_cell_id !== ''
      ? `contour:${r.gt_cell_id}`
      : others.filter(m => r[`${m.key}_status`] === 'extra').map(m => `${m.key}:${r[`${m.key}_cell_ids`]}`)[0];
    const cols = others.map(m => {
      const st = r[`${m.key}_status`] || '';
      return `<td class="st st-${st}">${st}</td><td>${r[`${m.key}_cell_ids`] || ''}</td>` +
        `<td>${r[`${m.key}_area`] === '' || r[`${m.key}_area`] === undefined ? '' : r[`${m.key}_area`]}</td>` +
        `<td>${r[`${m.key}_iou`] === '' || r[`${m.key}_iou`] === undefined ? '' : Number(r[`${m.key}_iou`]).toFixed(2)}</td>`;
    }).join('');
    return `<tr data-target="${target}"><td>${r.gt_cell_id === '' ? '&ndash;' : '#' + r.gt_cell_id}</td><td>${r.gt_area}</td>${cols}</tr>`;
  }).join('');
  $('cellTable').innerHTML = head + `<tbody>${body}</tbody>`;
  $('cellTable').querySelectorAll('tbody tr').forEach(tr => tr.addEventListener('click', () => {
    const [key, id] = tr.dataset.target.split(':');
    locateCell(key, parseInt(id));
  }));
}

function locateCell(key, id) {
  state.mode = 'overlay';
  document.querySelectorAll('#modeSeg button').forEach(b => b.classList.toggle('active', b.dataset.mode === 'overlay'));
  state.visible = new Set(available().map(m => m.key));
  state.selected = { method: key, id };
  if (state.zoom === 'fit') { state.zoom = '2'; $('zoomSelect').value = '2'; }
  render();
  const cell = cellById(key, id);
  if (!cell) return;
  const z = parseFloat(state.zoom);
  const stage = $('stage');
  stage.scrollLeft = ((cell.box[0] + cell.box[2]) / 2) * z - stage.clientWidth / 2;
  stage.scrollTop = ((cell.box[1] + cell.box[3]) / 2) * z - stage.clientHeight / 2;
}

function applyZoom() {
  const s = slide();
  const stage = $('stage');
  let z = state.zoom === 'fit'
    ? Math.min((stage.clientWidth - 4) / s.width, (stage.clientHeight - 4) / s.height)
    : parseFloat(state.zoom);
  const img = $('slideImg');
  img.style.width = (s.width * z) + 'px';
  img.style.height = (s.height * z) + 'px';
  const svg = $('overlay');
  svg.style.width = img.style.width;
  svg.style.height = img.style.height;
}

function render() {
  const s = slide();
  normaliseSelection();
  $('slideSelect').value = state.slide;
  $('overlayGroup').hidden = state.mode !== 'overlay';
  $('diffGroup').hidden = state.mode !== 'diff';
  $('tableGroup').hidden = state.mode !== 'table';
  $('stage').hidden = state.mode === 'table';
  $('tableView').hidden = state.mode !== 'table';
  if (state.mode === 'table') renderTable();
  fillMethodSelect($('aSelect'), state.a);
  fillMethodSelect($('bSelect'), state.b);
  $('showB').checked = state.showB;
  const img = $('slideImg');
  if (img.getAttribute('src') !== s.image) img.setAttribute('src', s.image);
  renderToggles();
  renderOverlay();
  renderCounts();
  renderDiffSummary();
  renderDetail();
  applyZoom();
}

initControls();
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build_gallery()
