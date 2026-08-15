#!/usr/bin/env python3
"""
build_gallery.py

Scans baseline/output/ for every image dataset produced by extract_cells_baseline.py
and generates a standalone, offline HTML gallery with a full-screen Modal Debug Overlay at:
  - baseline/output/gallery.html
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_ROOTS = [
    os.path.join(BASELINE_ROOT, "output")
]

STEP_DESCRIPTIONS = {
    "01_step1_raw_rgb_roi.jpg": ("Step 1: Raw RGB ROI", "Original raw microscopy tissue crop around the cell body.", "I_RGB(x, y)"),
    "02_step1_contrast_map.jpg": ("Step 1: Contrast CLAHE Map", "Multi-tile local adaptive CLAHE contrast mapping over cyan/grayscale signal.", "CLAHE(I, clipLimit=3.5, tileGrid=(8,8))"),
    "03_step2_contrast_sharpened.jpg": ("Step 2: Contrast Sharpened", "Unsharp masking contrast boosting.", "1.8 * CLAHE - 0.8 * GaussBlur(5x5, 1.0)"),
    "04_step2_binary_mask.jpg": ("Step 2: Otsu Binary Mask", "Adaptive Otsu thresholded binary cellular boundaries.", "Threshold(Contrast_Sharp, 0.65 * Otsu)"),
    "05_step2_closed_mask.jpg": ("Step 2: Morphology Closed Mask", "3x3 morphological closing.", "MorphologyClose(BinMask, Ellipse(3x3))"),
    "06_step3_cell_contour_mask.jpg": ("Step 3: Cell Contour & Hull Mask", "Filtered non-composite cell contour polygon mask.", "Contour(Area >= 80, Deduplication)"),
    "07_step4_final_extracted.jpg": ("Step 4: Final Extracted Cell", "Extracted cell composited onto white canvas.", "Mask * I_RGB + (1 - Mask) * 255"),
    "08_step4_slide_location_marked.jpg": ("Step 4: Whole Slide Marked", "Whole-slide reference map with marked bounding box.", "WholeSlide + BBox"),
    "08b_step4_zoomed_location_context.jpg": ("Step 4: Zoomed Slide Context", "500x500 zoomed context in tissue neighborhood.", "ROI(x1-250:x2+250, y1-250:y2+250)"),
    "09_debug_flowchart_panel.jpg": ("Step 9: Full 3x3 Collage Matrix", "3x3 diagnostic summary matrix compiling all stages.", "Collage(9 stages, 3x3 Grid)")
}


def discover_datasets(output_root):
    datasets = []
    if not os.path.isdir(output_root):
        return datasets
    for entry in sorted(os.listdir(output_root)):
        full_path = os.path.join(output_root, entry)
        if not os.path.isdir(full_path):
            continue
        manifest_path = os.path.join(full_path, "manifest.csv")
        if os.path.isfile(manifest_path):
            datasets.append((entry, manifest_path))
    return datasets


def load_manifest_rows(manifest_path):
    rows = []
    dataset_dir = os.path.dirname(manifest_path)
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cell_id"])
            debug_dir = os.path.join(dataset_dir, "debug", f"cell_{cid:03d}")
            has_debug = os.path.isdir(debug_dir)

            meta_file = os.path.join(debug_dir, "cell_debug_summary.json")
            debug_summary_text = ""
            if os.path.exists(meta_file):
                try:
                    with open(meta_file) as mf:
                        meta_obj = json.load(mf)
                        debug_summary_text = json.dumps(meta_obj, indent=2)
                except Exception:
                    pass

            step_files = []
            if has_debug:
                step_files = sorted([
                    f for f in os.listdir(debug_dir)
                    if f.lower().endswith((".jpg", ".png", ".jpeg"))
                ])

            rows.append({
                "cell_id": cid,
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "width": int(row.get("width", int(row["x2"]) - int(row["x1"]))),
                "height": int(row.get("height", int(row["y2"]) - int(row["y1"]))),
                "area": int(row["area"]),
                "filename": row["filename"],
                "location_filename": row["location_filename"],
                "flagged": row.get("flagged_possible_merge", "false").strip().lower() == "true",
                "has_debug": has_debug,
                "debug_folder": f"debug/cell_{cid:03d}",
                "debug_steps": step_files,
                "debug_summary": debug_summary_text
            })
    rows.sort(key=lambda r: r["cell_id"])
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Baseline (Presaizen 2026) | MicroGlia</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #0b0d13;
    --panel-bg: #151821;
    --panel-sub: #1e222d;
    --border: #2b3140;
    --text: #f0f3f8;
    --muted: #8b949e;
    --accent: #f59e0b;
    --accent-bg: rgba(245, 158, 11, 0.12);
    --debug-accent: #fbbf24;
    --flag-bg: #422006;
    --flag-border: #f59e0b;
    --flag-text: #fde68a;
    --thumb-selected: #f59e0b;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    height: 100%;
    margin: 0;
  }}
  body {{
    padding: 8px 12px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }}

  .top-band {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow: hidden;
  }}
  .titlebar {{
    display: flex;
    align-items: baseline;
    gap: 12px;
    flex-wrap: wrap;
    justify-content: space-between;
  }}
  .title-left {{
    display: flex;
    align-items: baseline;
    gap: 10px;
  }}
  h1 {{
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
    color: var(--accent);
    letter-spacing: -0.02em;
  }}
  .subtitle {{
    color: var(--muted);
    font-size: 0.78rem;
    margin: 0;
  }}
  .badge-method {{
    background: #451a03;
    color: #fde68a;
    border: 1px solid #b45309;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 600;
  }}

  .toolbar {{
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    padding: 5px 10px;
    border-radius: 8px;
  }}
  button, select {{
    font-size: 0.80rem;
    padding: 4px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel-sub);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  button:hover, select:hover {{
    background: var(--accent-bg);
    border-color: var(--accent);
    color: var(--accent);
  }}
  button:active {{ transform: translateY(1px); }}
  button.active-mode {{
    background: var(--accent);
    color: #000;
    font-weight: 600;
    border-color: var(--accent);
  }}
  button#debugBtn {{
    border-color: var(--debug-accent);
    color: var(--debug-accent);
    font-weight: 600;
  }}
  button#debugBtn:hover {{
    background: var(--debug-accent);
    color: #000;
  }}

  select#datasetSelect {{
    font-weight: 600;
    max-width: 280px;
  }}
  .dataset-picker {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .dataset-picker label {{
    font-size: 0.76rem;
    color: var(--muted);
  }}
  .nav-group, .view-group {{
    display: flex;
    gap: 4px;
  }}
  #position {{
    font-weight: 700;
    font-size: 0.82rem;
    min-width: 100px;
    text-align: center;
    color: var(--text);
  }}
  .flag-badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    background: var(--flag-bg);
    border: 1px solid var(--flag-border);
    color: var(--flag-text);
    font-weight: 600;
    font-size: 0.72rem;
  }}
  .flag-badge.hidden {{ display: none; }}

  .meta {{
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    padding: 4px 10px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.75rem;
    align-items: center;
  }}
  .meta div span.label {{
    color: var(--muted);
    margin-right: 4px;
  }}
  .meta div span.val {{
    font-weight: 600;
    color: var(--text);
  }}

  .viewer {{
    display: flex;
    gap: 10px;
    flex-wrap: nowrap;
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }}
  .panel {{
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 10px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }}
  .panel-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }}
  .panel-title {{
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .panel-file {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.68rem;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .panel.cell-panel {{ flex: 0 0 340px; max-width: 380px; }}
  .panel.location-panel {{ flex: 2 1 500px; }}

  .img-box {{
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    position: relative;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: #000;
  }}
  .img-box img {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}

  .cell-panel .img-box.checker-bg {{
    background:
      linear-gradient(45deg, #1f232b 25%, transparent 25%),
      linear-gradient(-45deg, #1f232b 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #1f232b 75%),
      linear-gradient(-45deg, transparent 75%, #1f232b 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
    background-color: #ffffff;
  }}
  .cell-panel .img-box img {{ image-rendering: pixelated; }}

  .thumbs-wrap {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 2px;
    overflow: hidden;
  }}
  .thumbs-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    flex-wrap: wrap;
  }}
  .thumbs-wrap h2 {{
    font-size: 0.76rem;
    margin: 0;
    color: var(--muted);
    font-weight: 600;
  }}
  .thumbs-hint {{ color: var(--muted); font-size: 0.70rem; }}
  .thumbs {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 4px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    height: 70px;
  }}
  .thumb {{
    flex: 0 0 auto;
    width: 56px;
    height: 56px;
    border: 2px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    padding: 0;
    background: #111;
    position: relative;
    transition: transform 0.1s ease, border-color 0.1s ease;
  }}
  .thumb:hover {{ transform: scale(1.05); }}
  .thumb img {{
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    border-radius: 4px;
    background: #fff;
  }}
  .thumb.selected {{
    border-color: var(--thumb-selected);
    box-shadow: 0 0 8px rgba(245, 158, 11, 0.4);
  }}
  .thumb.flagged::after {{
    content: "";
    position: absolute;
    top: 2px;
    right: 2px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--flag-border);
    box-shadow: 0 0 0 1px #000;
  }}

  /* Full Screen Debug Modal Overlay */
  .debug-overlay {{
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.82);
    backdrop-filter: blur(4px);
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 16px 24px;
    z-index: 100;
    transition: opacity 0.15s ease;
  }}
  .debug-overlay.hidden {{ display: none; }}
  .debug-modal {{
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    width: 100%;
    height: 100%;
    max-width: 1400px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    overflow: hidden;
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6);
  }}
  .debug-head {{
    flex: none;
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    background: var(--panel-sub);
  }}
  .debug-title {{
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--debug-accent);
  }}
  .debug-step-name {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.82rem;
    color: var(--text);
    overflow-wrap: anywhere;
  }}
  .debug-step-desc {{
    font-size: 0.74rem;
    color: var(--muted);
  }}
  .debug-spacer {{ flex: 1 1 auto; }}
  #debugSummaryBtn {{
    font-size: 0.80rem;
    padding: 4px 12px;
    font-weight: 600;
    border-color: var(--accent);
    color: var(--accent);
  }}
  #debugCloseBtn {{
    font-size: 1.3rem;
    line-height: 1;
    padding: 2px 10px;
    border-radius: 6px;
    color: var(--muted);
  }}
  #debugCloseBtn:hover {{
    color: #fff;
    background: rgba(239, 68, 68, 0.2);
    border-color: #ef4444;
  }}
  .debug-body {{
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    gap: 14px;
    padding: 14px;
    overflow: hidden;
    position: relative;
  }}
  .debug-stage {{
    flex: 1 1 auto;
    min-width: 0;
    min-height: 0;
    position: relative;
    background:
      linear-gradient(45deg, #181b22 25%, transparent 25%),
      linear-gradient(-45deg, #181b22 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #181b22 75%),
      linear-gradient(-45deg, transparent 75%, #181b22 75%);
    background-size: 20px 20px;
    background-position: 0 0, 0 10px, 10px -10px, -10px 0px;
    background-color: #0f1117;
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
  }}
  .debug-stage img {{
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
  }}
  .debug-summary {{
    flex: 0 0 360px;
    margin: 0;
    padding: 12px;
    overflow: auto;
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.75rem;
    line-height: 1.45;
    background: #08090d;
    border: 1px solid var(--border);
    border-radius: 8px;
    color: #fbbf24;
    white-space: pre-wrap;
  }}
  .debug-summary.hidden {{ display: none; }}
  .debug-steps {{
    flex: none;
    display: flex;
    gap: 8px;
    padding: 10px 16px 14px 16px;
    overflow-x: auto;
    overflow-y: hidden;
    border-top: 1px solid var(--border);
    background: var(--panel-sub);
    scrollbar-width: thin;
  }}
  .debug-step {{
    flex: none;
    width: 96px;
    padding: 4px;
    border: 2px solid transparent;
    border-radius: 6px;
    background: var(--bg);
    cursor: pointer;
    text-align: center;
    transition: all 0.12s ease;
  }}
  .debug-step:hover {{
    border-color: var(--debug-accent);
    transform: translateY(-2px);
  }}
  .debug-step.selected {{
    border-color: var(--debug-accent);
    background: rgba(245, 158, 11, 0.15);
    box-shadow: 0 0 10px rgba(245, 158, 11, 0.4);
  }}
  .debug-step img {{
    display: block;
    width: 100%;
    height: 60px;
    object-fit: contain;
    image-rendering: pixelated;
    background: #111;
    border-radius: 4px;
  }}
  .debug-step span {{
    display: block;
    margin-top: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    color: var(--text);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  @media (max-width: 850px) {{
    .viewer {{ flex-direction: column; }}
    .panel.cell-panel {{ flex: 1 1 0; max-width: 100%; }}
    .panel.location-panel {{ flex: 2 1 0; }}
  }}
</style>
</head>
<body>

<header class="top-band">
  <div class="titlebar">
    <div class="title-left">
      <h1>Baseline (Presaizen 2026)</h1>
      <div class="subtitle" id="subtitle">&mdash; extraction gallery with whole-slide correlation</div>
    </div>
    <span class="badge-method">Baseline Method (Presaizen 2026)</span>
  </div>

  <div class="toolbar">
    <div class="dataset-picker">
      <label for="datasetSelect">Slide:</label>
      <select id="datasetSelect"></select>
    </div>
    <div class="nav-group">
      <button id="prevBtn" title="Previous cell (Left Arrow)">&larr; Prev</button>
      <button id="nextBtn" title="Next cell (Right Arrow)">Next &rarr;</button>
    </div>
    <span id="position">Cell - / -</span>

    <button id="debugBtn" title="Show the per-step pipeline debug trace for this cell (Key D)">Debug</button>
    <span id="flagBadge" class="flag-badge hidden">FLAGGED: possible merged cluster</span>
  </div>

  <div class="meta">
    <div><span class="label">Cell ID:</span><span class="val" id="metaId">-</span></div>
    <div><span class="label">BBox (x1,y1)-(x2,y2):</span><span class="val" id="metaBbox">-</span></div>
    <div><span class="label">Dimensions:</span><span class="val" id="metaDim">-</span></div>
    <div><span class="label">Area:</span><span class="val" id="metaArea">-</span> px&sup2;</div>
    <div><span class="label">Flagged:</span><span class="val" id="metaFlag">-</span></div>
  </div>
</header>

<main class="viewer">
  <div class="panel cell-panel" id="cellPanel">
    <div class="panel-header">
      <span class="panel-title">Extracted Cell Crop</span>
      <span class="panel-file" id="cellFile">-</span>
    </div>
    <div class="img-box checker-bg" id="cropBox">
      <img id="cellImg" src="" alt="cell crop">
    </div>
  </div>

  <div class="panel location-panel">
    <div class="panel-header">
      <span class="panel-title">Source Slide Location Correlation</span>
      <span class="panel-file" id="locationFile">-</span>
    </div>
    <div class="img-box">
      <img id="locationImg" src="" alt="location in whole slide">
    </div>
  </div>
</main>

<footer class="thumbs-wrap">
  <div class="thumbs-header">
    <h2>Baseline Cell Index Strip (click to jump)</h2>
    <span class="thumbs-hint">Keys: &larr;/&rarr; step cell | D=Debug Modal | &uarr;/&darr; switch slide</span>
  </div>
  <div class="thumbs" id="thumbs"></div>
</footer>

<!-- Full-Screen Debug Modal Overlay -->
<div class="debug-overlay hidden" id="debugOverlay">
  <div class="debug-modal">
    <div class="debug-head">
      <strong class="debug-title" id="debugModalTitle">Debug trace - cell</strong>
      <span class="debug-step-name" id="debugStepName">-</span>
      <span class="debug-step-desc" id="debugStepDesc">-</span>
      <span class="debug-spacer"></span>
      <button id="debugSummaryBtn" title="Show computed values and metadata for this cell">Values</button>
      <button id="debugCloseBtn" title="Close modal (Esc)">&times;</button>
    </div>
    <div class="debug-body">
      <div class="debug-stage" id="debugStage">
        <img id="debugImg" src="" alt="debug stage">
      </div>
      <pre class="debug-summary hidden" id="debugSummary"></pre>
    </div>
    <div class="debug-steps" id="debugSteps"></div>
  </div>
</div>

<script>
const DATASETS = {datasets_json};
const STEP_INFO = {step_info_json};
const DATASET_NAMES = Object.keys(DATASETS);

let currentDataset = DATASET_NAMES[0] || "";
let currentIndex = 0;
let debugStepIndex = 0;

const cellImg = document.getElementById('cellImg');
const locationImg = document.getElementById('locationImg');
const cellFile = document.getElementById('cellFile');
const locationFile = document.getElementById('locationFile');
const positionEl = document.getElementById('position');
const flagBadge = document.getElementById('flagBadge');

const metaId = document.getElementById('metaId');
const metaBbox = document.getElementById('metaBbox');
const metaDim = document.getElementById('metaDim');
const metaArea = document.getElementById('metaArea');
const metaFlag = document.getElementById('metaFlag');

const thumbsEl = document.getElementById('thumbs');
const subtitleEl = document.getElementById('subtitle');
const datasetSelect = document.getElementById('datasetSelect');
const debugBtn = document.getElementById('debugBtn');

// Debug Modal Elements
const debugOverlay = document.getElementById('debugOverlay');
const debugModalTitle = document.getElementById('debugModalTitle');
const debugStepName = document.getElementById('debugStepName');
const debugStepDesc = document.getElementById('debugStepDesc');
const debugImg = document.getElementById('debugImg');
const debugStage = document.getElementById('debugStage');
const debugSteps = document.getElementById('debugSteps');
const debugSummary = document.getElementById('debugSummary');
const debugSummaryBtn = document.getElementById('debugSummaryBtn');
const debugCloseBtn = document.getElementById('debugCloseBtn');

function currentCells() {{
  return DATASETS[currentDataset] || [];
}}

function render() {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  const prefix = currentDataset + '/';

  cellImg.src = prefix + c.filename;
  cellFile.textContent = c.filename;

  locationImg.src = prefix + c.location_filename;
  locationFile.textContent = c.location_filename;

  positionEl.textContent = 'Cell ' + (currentIndex + 1) + ' / ' + cells.length;

  metaId.textContent = 'cell_' + String(c.cell_id).padStart(3, '0');
  metaBbox.textContent = '(' + c.x1 + ', ' + c.y1 + ') - (' + c.x2 + ', ' + c.y2 + ')';
  metaDim.textContent = c.width + ' x ' + c.height + ' px';
  metaArea.textContent = c.area;
  metaFlag.textContent = c.flagged ? 'True (Merge Cluster)' : 'False';

  if (c.flagged) {{
    flagBadge.classList.remove('hidden');
  }} else {{
    flagBadge.classList.add('hidden');
  }}

  debugBtn.disabled = !c.has_debug;
  debugBtn.title = c.has_debug ? 'Show step-by-step debug trace for this cell (D)' : 'No debug trace available';

  const thumbs = thumbsEl.querySelectorAll('.thumb');
  thumbs.forEach((t, i) => {{
    if (i === currentIndex) {{
      t.classList.add('selected');
      t.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
    }} else {{
      t.classList.remove('selected');
    }}
  }});
}}

/* ---------- DEBUG MODAL FUNCTIONS ---------- */
function isDebugOpen() {{
  return !debugOverlay.classList.contains('hidden');
}}

function openDebug() {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  if (!c.has_debug || !c.debug_steps.length) return;

  debugModalTitle.textContent = 'Debug Trace \\u2014 Cell #' + c.cell_id + ' (cell_' + String(c.cell_id).padStart(3, '0') + ')';
  debugSummary.textContent = c.debug_summary || '(No debug summary values found)';
  debugSummary.classList.add('hidden');
  debugSummaryBtn.textContent = 'Values';

  debugSteps.innerHTML = '';
  const frag = document.createDocumentFragment();
  const folderPrefix = currentDataset + '/' + c.debug_folder + '/';

  c.debug_steps.forEach((stepFile, idx) => {{
    const btn = document.createElement('button');
    btn.className = 'debug-step' + (idx === debugStepIndex ? ' selected' : '');
    btn.title = stepFile;
    btn.addEventListener('click', () => showDebugStep(idx));

    const img = document.createElement('img');
    img.src = folderPrefix + stepFile;
    img.alt = stepFile;
    img.loading = 'lazy';

    const label = document.createElement('span');
    const info = STEP_INFO[stepFile];
    label.textContent = info ? info[0] : stepFile.split('_')[0];

    btn.appendChild(img);
    btn.appendChild(label);
    frag.appendChild(btn);
  }});

  debugSteps.appendChild(frag);
  debugOverlay.classList.remove('hidden');
  showDebugStep(0);
}}

function closeDebug() {{
  debugOverlay.classList.add('hidden');
}}

function showDebugStep(idx) {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  if (!c.has_debug || !c.debug_steps.length) return;

  debugStepIndex = Math.max(0, Math.min(idx, c.debug_steps.length - 1));
  const stepFile = c.debug_steps[debugStepIndex];
  const folderPrefix = currentDataset + '/' + c.debug_folder + '/';

  debugImg.src = folderPrefix + stepFile;
  debugImg.alt = stepFile;

  const info = STEP_INFO[stepFile];
  if (info) {{
    debugStepName.textContent = info[0] + '  (' + stepFile + ')';
    debugStepDesc.textContent = '\\u2014 ' + info[1];
  }} else {{
    debugStepName.textContent = stepFile;
    debugStepDesc.textContent = '';
  }}

  const stepBtns = debugSteps.querySelectorAll('.debug-step');
  stepBtns.forEach((btn, j) => {{
    if (j === debugStepIndex) {{
      btn.classList.add('selected');
      btn.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
    }} else {{
      btn.classList.remove('selected');
    }}
  }});
}}

function toggleValues() {{
  const isHidden = debugSummary.classList.toggle('hidden');
  debugSummaryBtn.textContent = isHidden ? 'Values' : 'Hide Values';
}}

debugBtn.addEventListener('click', openDebug);
debugCloseBtn.addEventListener('click', closeDebug);
debugSummaryBtn.addEventListener('click', toggleValues);

debugOverlay.addEventListener('click', (e) => {{
  if (e.target === debugOverlay) closeDebug();
}});

let lastDebugWheel = 0;
debugStage.addEventListener('wheel', (e) => {{
  if (!isDebugOpen()) return;
  e.preventDefault();
  const now = Date.now();
  if (now - lastDebugWheel < 100) return;
  lastDebugWheel = now;
  if (e.deltaY > 0 || e.deltaX > 0) {{
    showDebugStep(debugStepIndex + 1);
  }} else if (e.deltaY < 0 || e.deltaX < 0) {{
    showDebugStep(debugStepIndex - 1);
  }}
}}, {{ passive: false }});

function goTo(index) {{
  const cells = currentCells();
  if (!cells.length) return;
  if (index < 0) index = 0;
  if (index > cells.length - 1) index = cells.length - 1;
  currentIndex = index;
  render();
  if (isDebugOpen()) openDebug();
}}

function buildThumbs() {{
  thumbsEl.innerHTML = '';
  const prefix = currentDataset + '/';
  const frag = document.createDocumentFragment();
  currentCells().forEach((c, i) => {{
    const btn = document.createElement('button');
    btn.className = 'thumb' + (c.flagged ? ' flagged' : '');
    btn.title = 'Cell ' + c.cell_id + (c.flagged ? ' (flagged)' : '');
    btn.addEventListener('click', () => goTo(i));

    const img = document.createElement('img');
    img.src = prefix + c.filename;
    img.alt = 'cell ' + c.cell_id + ' thumbnail';
    img.loading = 'lazy';

    btn.appendChild(img);
    frag.appendChild(btn);
  }});
  thumbsEl.appendChild(frag);
}}

function selectDataset(name) {{
  if (!DATASETS.hasOwnProperty(name)) return;
  currentDataset = name;
  currentIndex = 0;
  datasetSelect.value = name;
  subtitleEl.textContent = currentDataset + ' \\u2014 ' + DATASETS[name].length + ' baseline cells with whole-slide correlation';
  buildThumbs();
  render();
  if (isDebugOpen()) openDebug();
}}

function buildDatasetSelect() {{
  const frag = document.createDocumentFragment();
  DATASET_NAMES.forEach((name) => {{
    const opt = document.createElement('option');
    opt.value = name;
    opt.textContent = name + ' (' + DATASETS[name].length + ' cells)';
    frag.appendChild(opt);
  }});
  datasetSelect.appendChild(frag);
  datasetSelect.value = currentDataset;
}}

datasetSelect.addEventListener('change', (e) => selectDataset(e.target.value));

document.getElementById('prevBtn').addEventListener('click', () => goTo(currentIndex - 1));
document.getElementById('nextBtn').addEventListener('click', () => goTo(currentIndex + 1));

document.addEventListener('keydown', (e) => {{
  if (e.key === 'Escape') {{
    if (isDebugOpen()) closeDebug();
  }} else if (e.key.toLowerCase() === 'd') {{
    if (isDebugOpen()) closeDebug();
    else openDebug();
  }} else if (isDebugOpen()) {{
    if (e.key === 'ArrowLeft' || e.key === '[') showDebugStep(debugStepIndex - 1);
    else if (e.key === 'ArrowRight' || e.key === ']') showDebugStep(debugStepIndex + 1);
  }} else {{
    if (e.key === 'ArrowLeft') {{ goTo(currentIndex - 1); }}
    else if (e.key === 'ArrowRight') {{ goTo(currentIndex + 1); }}
    else if (e.key === 'ArrowUp') {{
      const curIdx = DATASET_NAMES.indexOf(currentDataset);
      if (curIdx > 0) selectDataset(DATASET_NAMES[curIdx - 1]);
    }} else if (e.key === 'ArrowDown') {{
      const curIdx = DATASET_NAMES.indexOf(currentDataset);
      if (curIdx < DATASET_NAMES.length - 1) selectDataset(DATASET_NAMES[curIdx + 1]);
    }}
  }}
}});

buildDatasetSelect();
if (currentDataset) {{
  subtitleEl.textContent = currentDataset + ' \\u2014 ' + DATASETS[currentDataset].length + ' baseline cells with whole-slide correlation';
  buildThumbs();
  render();
}}
</script>

</body>
</html>
"""

def build_gallery():
    for out_root in OUTPUT_ROOTS:
        if not os.path.exists(out_root):
            continue
        datasets = discover_datasets(out_root)
        if not datasets:
            continue

        all_data = {}
        for name, manifest_path in datasets:
            all_data[name] = load_manifest_rows(manifest_path)

        datasets_json = json.dumps(all_data, separators=(",", ":"))
        step_info_json = json.dumps(STEP_DESCRIPTIONS, separators=(",", ":"))
        html = HTML_TEMPLATE.format(datasets_json=datasets_json, step_info_json=step_info_json)

        gpath = os.path.join(out_root, "gallery.html")
        with open(gpath, "w") as f:
            f.write(html)
        print(f"Wrote {gpath} with {len(all_data)} dataset(s):")
        for name in all_data:
            print(f"  - {name}: {len(all_data[name])} cells")

if __name__ == "__main__":
    build_gallery()
