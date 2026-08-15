#!/usr/bin/env python3
"""
build_gallery.py

Scans baseline/output/ for every image dataset produced by extract_cells_baseline.py
and generates a standalone, offline HTML gallery with interactive Step-by-Step Debug Inspector
at baseline/output/gallery.html
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_ROOTS = [
    os.path.join(BASELINE_ROOT, "output")
]

DEBUG_STEPS_INFO = [
    {
        "id": 1,
        "key": "01_step1_raw_rgb_roi.jpg",
        "name": "Step 1: Raw RGB ROI",
        "desc": "Original raw microscopy tissue crop around the cell body.",
        "math": "I_RGB(x, y)"
    },
    {
        "id": 2,
        "key": "02_step1_contrast_map.jpg",
        "name": "Step 1: Contrast CLAHE Map",
        "desc": "Multi-tile local adaptive CLAHE contrast mapping over cyan/grayscale signal.",
        "math": "CLAHE(I, clipLimit=3.5, tileGrid=(8,8))"
    },
    {
        "id": 3,
        "key": "03_step2_contrast_sharpened.jpg",
        "name": "Step 2: Contrast Sharpened",
        "desc": "Unsharp masking contrast boosting.",
        "math": "1.8 * CLAHE - 0.8 * GaussBlur(5x5, 1.0)"
    },
    {
        "id": 4,
        "key": "04_step2_binary_mask.jpg",
        "name": "Step 2: Otsu Binary Mask",
        "desc": "Adaptive Otsu thresholded binary cellular boundaries.",
        "math": "Threshold(Contrast_Sharp, 0.65 * Otsu)"
    },
    {
        "id": 5,
        "key": "05_step2_closed_mask.jpg",
        "name": "Step 2: Morphology Closed Mask",
        "desc": "3x3 morphological closing.",
        "math": "MorphologyClose(BinMask, Ellipse(3x3))"
    },
    {
        "id": 6,
        "key": "06_step3_cell_contour_mask.jpg",
        "name": "Step 3: Cell Contour & Hull Mask",
        "desc": "Filtered non-composite cell contour polygon mask.",
        "math": "Contour(Area >= 80, Deduplication)"
    },
    {
        "id": 7,
        "key": "07_step4_final_extracted.jpg",
        "name": "Step 4: Final Extracted Cell",
        "desc": "Extracted cell composited onto white canvas.",
        "math": "Mask * I_RGB + (1 - Mask) * 255"
    },
    {
        "id": 8,
        "key": "08b_step4_zoomed_location_context.jpg",
        "name": "Step 4: Zoomed Slide Context",
        "desc": "500x500 zoomed context in tissue neighborhood.",
        "math": "ROI(x1-250:x2+250, y1-250:y2+250)"
    },
    {
        "id": 9,
        "key": "09_debug_flowchart_panel.jpg",
        "name": "Step 9: Full 3x3 Collage Matrix",
        "desc": "3x3 diagnostic summary matrix compiling all stages.",
        "math": "Collage(9 stages, 3x3 Grid)"
    }
]


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
            has_debug = os.path.exists(debug_dir)

            meta_file = os.path.join(debug_dir, "cell_debug_summary.json")
            debug_meta = {}
            if os.path.exists(meta_file):
                try:
                    with open(meta_file) as mf:
                        debug_meta = json.load(mf)
                except Exception:
                    pass

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
                "debug_prefix": f"debug/cell_{cid:03d}/" if has_debug else None,
                "debug_meta": debug_meta
            })
    rows.sort(key=lambda r: r["cell_id"])
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MicroGlia Baseline Gallery - Presaizen (2026)</title>
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
    padding: 10px 14px;
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
  button#modeDebug.active-mode {{
    background: var(--debug-accent);
    color: #000;
    font-weight: 700;
    border-color: var(--debug-accent);
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
  .panel.cell-panel.wide-active {{ flex: 1 1 540px; max-width: 650px; }}
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

  /* Debug Inspector */
  .debug-container {{
    display: flex;
    flex-direction: column;
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
    gap: 6px;
  }}
  .debug-stepper {{
    display: flex;
    gap: 4px;
    overflow-x: auto;
    padding: 3px 0;
    flex: 0 0 auto;
    border-bottom: 1px solid var(--border);
  }}
  .step-pill {{
    padding: 3px 8px;
    font-size: 0.68rem;
    font-weight: 600;
    background: var(--panel-sub);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--muted);
    cursor: pointer;
    white-space: nowrap;
    transition: all 0.12s ease;
  }}
  .step-pill:hover {{
    color: var(--debug-accent);
    border-color: var(--debug-accent);
  }}
  .step-pill.active {{
    background: var(--debug-accent);
    color: #000;
    font-weight: 700;
    border-color: var(--debug-accent);
  }}
  .debug-split {{
    display: flex;
    gap: 8px;
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
  }}
  .debug-pane {{
    flex: 1 1 50%;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
  }}
  .debug-pane .pane-label {{
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--debug-accent);
    margin-bottom: 2px;
  }}
  .debug-info-card {{
    background: var(--panel-sub);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    flex: 0 0 auto;
    font-size: 0.72rem;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .debug-info-card .info-title {{
    font-weight: 700;
    color: var(--debug-accent);
  }}
  .debug-info-card .info-desc {{ color: var(--text); }}
  .debug-info-card .info-math {{
    font-family: ui-monospace, monospace;
    color: var(--muted);
    font-size: 0.68rem;
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
      <h1>MicroGlia Baseline Gallery & Debug Inspector</h1>
      <div class="subtitle" id="subtitle">&mdash; Presaizen (2026) extraction with step-by-step deep dive</div>
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

    <div class="view-group">
      <button id="modeCrop" class="active-mode" title="View Extracted Cell Crop (Key 1)">Extracted Crop</button>
      <button id="modeDebug" style="border-color: var(--debug-accent); color: var(--debug-accent);" title="Dive Into Baseline Algorithmic Steps (Key 2 or D)">🔬 Step-by-Step Deep Dive</button>
    </div>

    <span id="flagBadge" class="flag-badge hidden">FLAGGED: possible merged cluster</span>
  </div>

  <div class="meta">
    <div><span class="label">Cell ID:</span><span class="val" id="metaId">-</span></div>
    <div><span class="label">BBox (x1,y1)-(x2,y2):</span><span class="val" id="metaBbox">-</span></div>
    <div><span class="label">Dimensions:</span><span class="val" id="metaDim">-</span></div>
    <div><span class="label">Area:</span><span class="val" id="metaArea">-</span> px&sup2;</div>
    <div><span class="label">Otsu Thresh:</span><span class="val" id="metaOtsu">-</span></div>
    <div><span class="label">Flagged:</span><span class="val" id="metaFlag">-</span></div>
  </div>
</header>

<main class="viewer">
  <div class="panel cell-panel" id="cellPanel">
    <div class="panel-header">
      <span class="panel-title" id="cropPanelTitle">Extracted Cell Crop</span>
      <span class="panel-file" id="cellFile">-</span>
    </div>

    <!-- Normal Crop Box -->
    <div class="img-box checker-bg" id="cropBox">
      <img id="cellImg" src="" alt="cell crop">
    </div>

    <!-- Debug Inspector -->
    <div class="debug-container" id="debugContainer" style="display: none;">
      <div class="debug-stepper" id="debugStepper"></div>

      <div class="debug-split">
        <div class="debug-pane">
          <span class="pane-label" id="debugStageLabel">Active Algorithmic Stage</span>
          <div class="img-box" id="debugStageBox">
            <img id="debugStageImg" src="" alt="debug stage output">
          </div>
        </div>
        <div class="debug-pane">
          <span class="pane-label" style="color: var(--muted);">Original Raw Reference</span>
          <div class="img-box checker-bg">
            <img id="debugRefImg" src="" alt="original reference crop">
          </div>
        </div>
      </div>

      <div class="debug-info-card">
        <div class="info-title" id="debugInfoTitle">Step Title</div>
        <div class="info-desc" id="debugInfoDesc">Description of transformation</div>
        <div class="info-math" id="debugInfoMath">Formula</div>
      </div>
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
    <span class="thumbs-hint">Keys: &larr;/&rarr; step cell | 1=Crop, 2=Debug | [/] step debug stages | &uarr;/&darr; switch slide</span>
  </div>
  <div class="thumbs" id="thumbs"></div>
</footer>

<script>
const DATASETS = {datasets_json};
const DEBUG_STEPS = {debug_steps_json};
const DATASET_NAMES = Object.keys(DATASETS);

let currentDataset = DATASET_NAMES[0] || "";
let currentIndex = 0;
let isDebugMode = false;
let currentDebugStepIdx = 0;

const cellImg = document.getElementById('cellImg');
const cropBox = document.getElementById('cropBox');
const debugContainer = document.getElementById('debugContainer');
const cellPanel = document.getElementById('cellPanel');
const cropPanelTitle = document.getElementById('cropPanelTitle');

const debugStageImg = document.getElementById('debugStageImg');
const debugRefImg = document.getElementById('debugRefImg');
const debugStageLabel = document.getElementById('debugStageLabel');
const debugInfoTitle = document.getElementById('debugInfoTitle');
const debugInfoDesc = document.getElementById('debugInfoDesc');
const debugInfoMath = document.getElementById('debugInfoMath');
const debugStepper = document.getElementById('debugStepper');

const locationImg = document.getElementById('locationImg');
const cellFile = document.getElementById('cellFile');
const locationFile = document.getElementById('locationFile');
const positionEl = document.getElementById('position');
const flagBadge = document.getElementById('flagBadge');

const metaId = document.getElementById('metaId');
const metaBbox = document.getElementById('metaBbox');
const metaDim = document.getElementById('metaDim');
const metaArea = document.getElementById('metaArea');
const metaOtsu = document.getElementById('metaOtsu');
const metaFlag = document.getElementById('metaFlag');

const thumbsEl = document.getElementById('thumbs');
const subtitleEl = document.getElementById('subtitle');
const datasetSelect = document.getElementById('datasetSelect');

const modeCropBtn = document.getElementById('modeCrop');
const modeDebugBtn = document.getElementById('modeDebug');

function currentCells() {{
  return DATASETS[currentDataset] || [];
}}

function buildDebugStepper() {{
  debugStepper.innerHTML = '';
  const frag = document.createDocumentFragment();
  DEBUG_STEPS.forEach((step, idx) => {{
    const pill = document.createElement('button');
    pill.className = 'step-pill' + (idx === currentDebugStepIdx ? ' active' : '');
    pill.textContent = step.id + '. ' + step.name.split(':')[1].trim();
    pill.title = step.name + ' - ' + step.desc;
    pill.addEventListener('click', () => setDebugStep(idx));
    frag.appendChild(pill);
  }});
  debugStepper.appendChild(frag);
}}

function setDebugStep(idx) {{
  if (idx < 0) idx = 0;
  if (idx > DEBUG_STEPS.length - 1) idx = DEBUG_STEPS.length - 1;
  currentDebugStepIdx = idx;

  const pills = debugStepper.querySelectorAll('.step-pill');
  pills.forEach((p, i) => {{
    p.classList.toggle('active', i === currentDebugStepIdx);
    if (i === currentDebugStepIdx) {{
      p.scrollIntoView({{ behavior: 'smooth', inline: 'center', block: 'nearest' }});
    }}
  }});

  renderDebugStage();
}}

function setViewMode(debug) {{
  isDebugMode = debug;
  modeCropBtn.classList.toggle('active-mode', !debug);
  modeDebugBtn.classList.toggle('active-mode', debug);

  if (debug) {{
    cropBox.style.display = 'none';
    debugContainer.style.display = 'flex';
    cellPanel.className = 'panel cell-panel wide-active';
    cropPanelTitle.textContent = '🔬 Baseline Step-by-Step Deep Dive';
    renderDebugStage();
  }} else {{
    cropBox.style.display = 'block';
    debugContainer.style.display = 'none';
    cellPanel.className = 'panel cell-panel';
    cropPanelTitle.textContent = 'Extracted Cell Crop';
    render();
  }}
}}

function renderDebugStage() {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  const step = DEBUG_STEPS[currentDebugStepIdx];
  const prefix = currentDataset + '/' + (c.debug_prefix || '');

  debugStageImg.src = prefix + step.key;
  debugRefImg.src = prefix + '01_step1_raw_rgb_roi.jpg';

  debugStageLabel.textContent = step.name;
  debugInfoTitle.textContent = step.name;
  debugInfoDesc.textContent = step.desc;
  debugInfoMath.textContent = 'Formula / Config: ' + step.math;

  cellFile.textContent = (c.debug_prefix || '') + step.key;
}}

function render() {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  const prefix = currentDataset + '/';

  if (!isDebugMode) {{
    cellImg.src = prefix + c.filename;
    cellFile.textContent = c.filename;
  }} else {{
    renderDebugStage();
  }}

  locationImg.src = prefix + c.location_filename;
  locationFile.textContent = c.location_filename;

  positionEl.textContent = 'Cell ' + (currentIndex + 1) + ' / ' + cells.length;

  metaId.textContent = 'cell_' + String(c.cell_id).padStart(3, '0');
  metaBbox.textContent = '(' + c.x1 + ', ' + c.y1 + ') - (' + c.x2 + ', ' + c.y2 + ')';
  metaDim.textContent = c.width + ' x ' + c.height + ' px';
  metaArea.textContent = c.area;

  if (c.debug_meta && c.debug_meta.otsu_threshold) {{
    metaOtsu.textContent = c.debug_meta.otsu_threshold + ' (Calib: ' + c.debug_meta.calibrated_threshold + ')';
  }} else {{
    metaOtsu.textContent = 'Calibrated';
  }}

  metaFlag.textContent = c.flagged ? 'True (Merge Cluster)' : 'False';

  if (c.flagged) {{
    flagBadge.classList.remove('hidden');
  }} else {{
    flagBadge.classList.add('hidden');
  }}

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

function goTo(index) {{
  const cells = currentCells();
  if (!cells.length) return;
  if (index < 0) index = 0;
  if (index > cells.length - 1) index = cells.length - 1;
  currentIndex = index;
  render();
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

modeCropBtn.addEventListener('click', () => setViewMode(false));
modeDebugBtn.addEventListener('click', () => setViewMode(true));

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ goTo(currentIndex - 1); }}
  else if (e.key === 'ArrowRight') {{ goTo(currentIndex + 1); }}
  else if (e.key === '1') {{ setViewMode(false); }}
  else if (e.key === '2' || e.key.toLowerCase() === 'd') {{ setViewMode(true); }}
  else if (e.key === '[' || e.key === '-') {{ if (isDebugMode) setDebugStep(currentDebugStepIdx - 1); }}
  else if (e.key === ']' || e.key === '=' || e.key === '+') {{ if (isDebugMode) setDebugStep(currentDebugStepIdx + 1); }}
  else if (e.key === 'ArrowUp') {{
    const curIdx = DATASET_NAMES.indexOf(currentDataset);
    if (curIdx > 0) selectDataset(DATASET_NAMES[curIdx - 1]);
  }} else if (e.key === 'ArrowDown') {{
    const curIdx = DATASET_NAMES.indexOf(currentDataset);
    if (curIdx < DATASET_NAMES.length - 1) selectDataset(DATASET_NAMES[curIdx + 1]);
  }}
}});

buildDatasetSelect();
buildDebugStepper();
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
        debug_steps_json = json.dumps(DEBUG_STEPS_INFO, separators=(",", ":"))
        html = HTML_TEMPLATE.format(datasets_json=datasets_json, debug_steps_json=debug_steps_json)

        gpath = os.path.join(out_root, "gallery.html")
        with open(gpath, "w") as f:
            f.write(html)
        print(f"Wrote {gpath} with {len(all_data)} dataset(s):")
        for name in all_data:
            print(f"  - {name}: {len(all_data[name])} cells")

if __name__ == "__main__":
    build_gallery()
