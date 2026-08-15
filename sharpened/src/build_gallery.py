#!/usr/bin/env python3
"""
build_gallery.py

Scans sharpened/output/ (and Data/boundary-sharpening-output/) for every image dataset
and generates a standalone, offline HTML gallery at:
  - sharpened/output/gallery.html
  - Data/boundary-sharpening-output/gallery.html
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
SHARPENED_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

OUTPUT_ROOTS = [
    os.path.join(SHARPENED_ROOT, "output")
]

def discover_datasets(output_root):
    """Return sorted list of (dataset_name, manifest_path) for every immediate
    subdirectory of output_root that contains a manifest.csv."""
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
    """Read manifest.csv into row dictionary shape for gallery JS."""
    rows = []
    dataset_dir = os.path.dirname(manifest_path)
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = int(row["cell_id"])
            debug_path = os.path.join(dataset_dir, "debug", f"cell_{cid:03d}", "16_debug_step_by_step_panel.jpg")
            has_debug = os.path.exists(debug_path)
            rows.append({
                "cell_id": cid,
                "x1": int(row["x1"]),
                "y1": int(row["y1"]),
                "x2": int(row["x2"]),
                "y2": int(row["y2"]),
                "width": int(row.get("width", int(row["x2"]) - int(row["x1"]))),
                "height": int(row.get("height", int(row["y2"]) - int(row["y1"]))),
                "area": int(row["area"]),
                "original_filename": row["original_filename"],
                "sharpened_filename": row["sharpened_filename"],
                "location_filename": row["location_filename"],
                "flagged": row.get("flagged_possible_merge", "false").strip().lower() == "true",
                "has_debug": has_debug,
                "debug_panel": f"debug/cell_{cid:03d}/16_debug_step_by_step_panel.jpg" if has_debug else None
            })
    rows.sort(key=lambda r: r["cell_id"])
    return rows


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MicroGlia Cell Gallery & Slide Correlation</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #0f1117;
    --panel-bg: #181b22;
    --panel-sub: #212631;
    --border: #2e3545;
    --text: #f0f3f8;
    --muted: #8b949e;
    --accent: #38bdf8;
    --accent-bg: rgba(56, 189, 248, 0.12);
    --flag-bg: #422006;
    --flag-border: #f59e0b;
    --flag-text: #fde68a;
    --thumb-selected: #38bdf8;
    --badge-green: #10b981;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{
    height: 100%;
    margin: 0;
  }}
  body {{
    padding: 12px 16px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}

  .top-band {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
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
    font-size: 0.8rem;
    margin: 0;
  }}
  .badge-method {{
    background: #064e3b;
    color: #a7f3d0;
    border: 1px solid #059669;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
  }}

  .toolbar {{
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 8px;
  }}
  button, select {{
    font-size: 0.82rem;
    padding: 5px 12px;
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

  select#datasetSelect {{
    font-weight: 600;
    max-width: 320px;
  }}
  .dataset-picker {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}
  .dataset-picker label {{
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .nav-group, .view-group {{
    display: flex;
    gap: 4px;
  }}
  #position {{
    font-weight: 700;
    font-size: 0.85rem;
    min-width: 110px;
    text-align: center;
    color: var(--text);
  }}
  .flag-badge {{
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    background: var(--flag-bg);
    border: 1px solid var(--flag-border);
    color: var(--flag-text);
    font-weight: 600;
    font-size: 0.75rem;
  }}
  .flag-badge.hidden {{ display: none; }}

  .meta {{
    display: flex;
    gap: 16px;
    flex-wrap: wrap;
    padding: 5px 12px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    font-size: 0.78rem;
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
    gap: 12px;
    flex-wrap: nowrap;
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
  }}
  .panel {{
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
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
    margin-bottom: 6px;
  }}
  .panel-title {{
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .panel-file {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    font-size: 0.70rem;
    color: var(--muted);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }}

  .panel.cell-panel {{ flex: 0 0 340px; max-width: 380px; }}
  .panel.cell-panel.dual-active {{ flex: 0 0 540px; max-width: 580px; }}
  .panel.location-panel {{ flex: 2 1 600px; }}

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

  .dual-crop-box {{
    display: flex;
    gap: 8px;
    flex: 1 1 auto;
    min-height: 0;
    min-width: 0;
  }}
  .dual-sub-box {{
    flex: 1 1 50%;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    position: relative;
  }}
  .dual-sub-box .sub-label {{
    font-size: 0.68rem;
    color: var(--muted);
    text-align: center;
    margin-bottom: 3px;
  }}
  .dual-sub-box .img-box {{
    background: #ffffff;
  }}
  .dual-sub-box .img-box img {{
    image-rendering: pixelated;
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
  .cell-panel .img-box img {{
    image-rendering: pixelated;
  }}

  .thumbs-wrap {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 4px;
    overflow: hidden;
  }}
  .thumbs-header {{
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .thumbs-wrap h2 {{
    font-size: 0.80rem;
    margin: 0;
    color: var(--muted);
    font-weight: 600;
  }}
  .thumbs-hint {{
    color: var(--muted);
    font-size: 0.72rem;
  }}
  .thumbs {{
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 6px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    height: 76px;
  }}
  .thumb {{
    flex: 0 0 auto;
    width: 60px;
    height: 60px;
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
    box-shadow: 0 0 8px rgba(56, 189, 248, 0.4);
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
      <h1>MicroGlia Cell Gallery</h1>
      <div class="subtitle" id="subtitle">&mdash; extracted cells with whole-slide correlation</div>
    </div>
    <span class="badge-method">Boundary Sharpening Pipeline (Method 2)</span>
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
      <button id="modeOrig" class="active-mode" title="View Original RGB Crop (Key 1)">RGB Crop</button>
      <button id="modeSharp" title="View Sharpened Silhouette Mask (Key 2)">Sharpened Mask</button>
      <button id="modeDual" title="View Both Side-by-Side (Key 3)">Dual View</button>
      <button id="modeDebug" style="display:none; border-color:#00e5ff; color:#00e5ff;" title="View 16-Step Debug Flowchart (Key 4)">🔍 Debug Flowchart</button>
    </div>

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
      <span class="panel-title" id="cropPanelTitle">Cell Crop (RGB)</span>
      <span class="panel-file" id="cellFile">-</span>
    </div>

    <div class="img-box checker-bg" id="singleCropBox">
      <img id="cellImg" src="" alt="cell crop">
    </div>

    <div class="dual-crop-box" id="dualCropBox" style="display: none;">
      <div class="dual-sub-box">
        <span class="sub-label">Original RGB Crop</span>
        <div class="img-box checker-bg">
          <img id="dualOrigImg" src="" alt="original crop">
        </div>
      </div>
      <div class="dual-sub-box">
        <span class="sub-label">Sharpened Silhouette</span>
        <div class="img-box">
          <img id="dualSharpImg" src="" alt="sharpened crop">
        </div>
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
    <h2>Sub-Cell Index Strip (click to jump)</h2>
    <span class="thumbs-hint">Keys: &larr;/&rarr; step cell | 1=RGB, 2=Sharpened, 3=Dual, 4=Debug | &uarr;/&darr; switch slide</span>
  </div>
  <div class="thumbs" id="thumbs"></div>
</footer>

<script>
const DATASETS = {datasets_json};
const DATASET_NAMES = Object.keys(DATASETS);

let currentDataset = DATASET_NAMES[0] || "";
let currentIndex = 0;
let cropViewMode = "orig";

const cellImg = document.getElementById('cellImg');
const dualOrigImg = document.getElementById('dualOrigImg');
const dualSharpImg = document.getElementById('dualSharpImg');
const singleCropBox = document.getElementById('singleCropBox');
const dualCropBox = document.getElementById('dualCropBox');
const cellPanel = document.getElementById('cellPanel');
const cropPanelTitle = document.getElementById('cropPanelTitle');

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

const modeOrigBtn = document.getElementById('modeOrig');
const modeSharpBtn = document.getElementById('modeSharp');
const modeDualBtn = document.getElementById('modeDual');
const modeDebugBtn = document.getElementById('modeDebug');

function currentCells() {{
  return DATASETS[currentDataset] || [];
}}

function setViewMode(mode) {{
  cropViewMode = mode;
  modeOrigBtn.classList.toggle('active-mode', mode === 'orig');
  modeSharpBtn.classList.toggle('active-mode', mode === 'sharp');
  modeDualBtn.classList.toggle('active-mode', mode === 'dual');
  modeDebugBtn.classList.toggle('active-mode', mode === 'debug');

  if (mode === 'dual') {{
    singleCropBox.style.display = 'none';
    dualCropBox.style.display = 'flex';
    cellPanel.classList.add('dual-active');
    cropPanelTitle.textContent = 'Dual Cell Crops (RGB + Sharpened)';
  }} else if (mode === 'debug') {{
    singleCropBox.style.display = 'block';
    dualCropBox.style.display = 'none';
    cellPanel.classList.add('dual-active');
    cropPanelTitle.textContent = '16-Step Debug Flowchart Trace';
  }} else {{
    singleCropBox.style.display = 'block';
    dualCropBox.style.display = 'none';
    cellPanel.classList.remove('dual-active');
    cropPanelTitle.textContent = mode === 'orig' ? 'Cell Crop (Original RGB)' : 'Cell Crop (Sharpened Silhouette)';
  }}
  render();
}}

function render() {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  const prefix = currentDataset + '/';

  const origPath = prefix + c.original_filename;
  const sharpPath = prefix + c.sharpened_filename;
  const locPath = prefix + c.location_filename;
  const debugPath = c.debug_panel ? (prefix + c.debug_panel) : null;

  if (c.has_debug) {{
    modeDebugBtn.style.display = 'inline-block';
  }} else {{
    modeDebugBtn.style.display = 'none';
    if (cropViewMode === 'debug') setViewMode('orig');
  }}

  if (cropViewMode === 'orig') {{
    cellImg.src = origPath;
    cellFile.textContent = c.original_filename;
  }} else if (cropViewMode === 'sharp') {{
    cellImg.src = sharpPath;
    cellFile.textContent = c.sharpened_filename;
  }} else if (cropViewMode === 'debug' && debugPath) {{
    cellImg.src = debugPath;
    cellFile.textContent = c.debug_panel;
  }} else {{
    dualOrigImg.src = origPath;
    dualSharpImg.src = sharpPath;
    cellFile.textContent = c.original_filename + ' | ' + c.sharpened_filename;
  }}

  locationImg.src = locPath;
  locationFile.textContent = c.location_filename;

  positionEl.textContent = 'Cell ' + (currentIndex + 1) + ' / ' + cells.length;

  metaId.textContent = 'subcell_' + String(c.cell_id).padStart(3, '0');
  metaBbox.textContent = '(' + c.x1 + ', ' + c.y1 + ') - (' + c.x2 + ', ' + c.y2 + ')';
  metaDim.textContent = c.width + ' x ' + c.height + ' px';
  metaArea.textContent = c.area;
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
    img.src = prefix + c.original_filename;
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
  subtitleEl.textContent = currentDataset + ' \\u2014 ' + DATASETS[name].length + ' sub-cells with whole-slide correlation';
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

modeOrigBtn.addEventListener('click', () => setViewMode('orig'));
modeSharpBtn.addEventListener('click', () => setViewMode('sharp'));
modeDualBtn.addEventListener('click', () => setViewMode('dual'));
modeDebugBtn.addEventListener('click', () => setViewMode('debug'));

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft') {{ goTo(currentIndex - 1); }}
  else if (e.key === 'ArrowRight') {{ goTo(currentIndex + 1); }}
  else if (e.key === '1') {{ setViewMode('orig'); }}
  else if (e.key === '2') {{ setViewMode('sharp'); }}
  else if (e.key === '3') {{ setViewMode('dual'); }}
  else if (e.key === '4') {{ setViewMode('debug'); }}
  else if (e.key === 'ArrowUp') {{
    const curIdx = DATASET_NAMES.indexOf(currentDataset);
    if (curIdx > 0) selectDataset(DATASET_NAMES[curIdx - 1]);
  }} else if (e.key === 'ArrowDown') {{
    const curIdx = DATASET_NAMES.indexOf(currentDataset);
    if (curIdx < DATASET_NAMES.length - 1) selectDataset(DATASET_NAMES[curIdx + 1]);
  }}
}});

buildDatasetSelect();
if (currentDataset) {{
  subtitleEl.textContent = currentDataset + ' \\u2014 ' + DATASETS[currentDataset].length + ' sub-cells with whole-slide correlation';
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
        html = HTML_TEMPLATE.format(datasets_json=datasets_json)

        gpath = os.path.join(out_root, "gallery.html")
        with open(gpath, "w") as f:
            f.write(html)
        print(f"Wrote {gpath} with {len(all_data)} dataset(s):")
        for name in all_data:
            print(f"  - {name}: {len(all_data[name])} cells")

if __name__ == "__main__":
    build_gallery()
