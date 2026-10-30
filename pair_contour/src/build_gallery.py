#!/usr/bin/env python3
"""
build_gallery.py

Scans pair_contour/output/ for every image pair produced by
extract_pair_cells.py and generates a standalone, offline HTML gallery
with a full-screen Modal Debug Overlay at:
  - pair_contour/output/gallery.html

Copied from contour/src/build_gallery.py (frozen) and adapted to the pair data:
the second crop is the MASK image crop instead of the sharpened silhouette,
and the badge marks cells that touch the image border.
"""

import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIR_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_ROOTS = [
    os.path.join(PAIR_ROOT, "output")
]

STEP_DESCRIPTIONS = {
    "00_step0_bounding_box_location.jpg": ("Step 0: Cell BBox Context", "500x500 RAW neighborhood with the bounding box of the cell being extracted.", "Target ROI: (x1, y1) - (x2, y2)"),
    "01_step1_mask_image_roi.png": ("Step 1: MASK Image ROI", "Crop of the MASK image showing the hand-drawn cyan outline of the cell.", "I_MASK(x, y)"),
    "02_step1_cyanness_map.png": ("Step 1: Cyanness Map", "Cyan signal of the MASK image. Grey pixels give exactly 0.", "clip(min(B, G) - R, 0, 255)"),
    "03_step1_cyan_core_lines.png": ("Step 1: Cyan Core Lines", "Solid outline pixels used to separate the cells.", "Cyanness >= 100"),
    "04_step2_enclosed_region.png": ("Step 2: Enclosed Region", "Region enclosed by the outline (4-connected component of non-line pixels).", "ConnectedComponents(~Core, conn=4)"),
    "05_step4a_region_plus_outline.png": ("Step 4a: Region + Outline", "Enclosed region plus the outline pixels around it, which hide the cell edge in the MASK image.", "Region | (Dilate(Region, r=3) & Cyanness > 0)"),
    "06_step4b_aura_growth.png": ("Step 4b: Aura Growth", "White = outline mask. Yellow = added aura: 2 px free margin, then up to 10 px through bright RAW pixels, only in pixels closer to this cell than to any other cell.", "Open(Grow(Mask, RAW_smooth >= bg_median + 8) within own zone)"),
    "06c_step4c_contact_completion.png": ("Step 4c: Contact Completion", "White = mask after aura growth. Red = pixels added where the cell touches a neighbour: an ellipse (or a circle, when the free edge is a short arc) fitted to the free edge, kept only inside neighbouring cells, up to 8 px from the contact. Round cells only.", "Shape(free edge) & neighbours & depth <= 8, if fit error <= 1.5 px"),
    "07_step5a_raw_roi.png": ("Step 5a: RAW ROI", "Same box cut from the RAW image, with no outline drawn on it.", "I_RAW(x, y)"),
    "08_step5b_cut_from_raw.png": ("Step 5b: Cut from RAW", "RAW pixels inside the cell mask (with aura) on a magenta canvas.", "Mask * I_RAW + (1 - Mask) * Magenta"),
    "09_step6_qa_signal_outside.png": ("Step 6: QA Signal Left Outside", "Green = mask edge. Blue = 4 px ring outside the mask (neighbour cells excluded). Red = ring pixels that differ from the local background by more than 3 noise sigma: cell signal the cut missed.", "|RAW - local background| > 3 sigma, in Dilate(Mask, 4) - all masks"),
    "99_debug_step_by_step_panel.jpg": ("Complete Step Grid", "Collage of every intermediate step for this cell.", "Collage(steps, 4 per row)")
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
                "original_filename": row["original_filename"],
                "annotated_filename": row["annotated_filename"],
                "location_filename": row["location_filename"],
                "flagged": row.get("touches_border", "false").strip().lower() == "true",
                "outside_signal_pct": row.get("outside_signal_pct", ""),
                "crowded": row.get("crowded", "false").strip().lower() == "true",
                "qa_flag": row.get("qa_flag", "false").strip().lower() == "true",
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
<title>Method: Pair Contour (Phase 2) | MicroGlia</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #0b0d13;
    --panel-bg: #151821;
    --panel-sub: #1e222d;
    --border: #2b3140;
    --text: #f0f3f8;
    --muted: #8b949e;
    --accent: #38bdf8;
    --accent-bg: rgba(56, 189, 248, 0.12);
    --debug-accent: #00e5ff;
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

  /* ---------- Top band: title, toolbar, metadata ---------- */
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
    background: #064e3b;
    color: #a7f3d0;
    border: 1px solid #059669;
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

  /* ---------- Middle band: main viewer ---------- */
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

  /* Dual Crop Container */
  .dual-crop-box {{
    display: flex;
    gap: 6px;
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
    font-size: 0.65rem;
    color: var(--muted);
    text-align: center;
    margin-bottom: 2px;
  }}
  .dual-sub-box .img-box {{
    background: #ffffff;
  }}
  .dual-sub-box .img-box img {{
    image-rendering: auto;
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
  /* Crops are shown at a capped zoom (set in JS), centred, smooth unless pixel view is on */
  .cell-panel .img-box img {{
    top: 50%;
    left: 50%;
    width: auto;
    height: auto;
    transform: translate(-50%, -50%);
    image-rendering: auto;
  }}
  .cell-panel.pixel-view .img-box img {{
    image-rendering: pixelated;
  }}
  .crop-zoom {{ display: inline-flex; align-items: center; gap: 4px; }}
  .crop-zoom span {{ min-width: 2.6em; text-align: center; font-variant-numeric: tabular-nums; }}

  /* ---------- Bottom band: thumbnail strip ---------- */
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
  .thumbs-hint {{
    color: var(--muted);
    font-size: 0.70rem;
  }}
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

  /* ---------- FULL SCREEN MODAL DEBUG OVERLAY ---------- */
  /* ---------- Overview overlay: taken vs. ignored map, shown first ---------- */
  .overview-overlay {{
    position: fixed;
    inset: 0;
    z-index: 60;
    background: rgba(0, 0, 0, 0.9);
    display: flex;
    flex-direction: column;
    gap: 10px;
    padding: 14px 18px;
  }}
  .overview-overlay.hidden {{ display: none; }}
  .overview-head {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  .overview-head h2 {{ margin: 0; font-size: 1rem; }}
  .legend-chip {{ display: inline-flex; align-items: center; gap: 6px; }}
  .legend-chip i {{ display: inline-block; width: 14px; height: 14px; border-radius: 3px; }}
  .overview-note {{ color: var(--muted); font-size: 0.8rem; }}
  #overviewStart {{ margin-left: auto; }}
  .overview-body {{ flex: 1 1 auto; min-height: 0; position: relative; }}
  .overview-body img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }}

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
    color: #38bdf8;
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
    background: rgba(0, 229, 255, 0.12);
    box-shadow: 0 0 10px rgba(0, 229, 255, 0.4);
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
      <h1>Method: Pair Contour (Phase 2)</h1>
      <div class="subtitle" id="subtitle">&mdash; cells outlined in the MASK image, cut from the RAW image</div>
    </div>
    <span class="badge-method">Pair Contour</span>
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
      <button id="modeOrig" class="active-mode" title="View the cell cut from the RAW image (Key 1)">RAW Cut</button>
      <button id="modeSharp" title="View the same box in the MASK image (Key 2)">MASK Image</button>
      <button id="modeDual" title="View Both Side-by-Side (Key 3)">Dual View</button>
      <button id="debugBtn" title="Show the per-step pipeline debug trace for this cell (Key D)">Debug</button>
      <button id="overviewBtn" title="Show which cells are taken and which are ignored (Key O)">Overview</button>
    </div>
    <div class="view-group crop-zoom" title="Crop zoom: keys - and +">
      <button id="cropZoomOut">&minus;</button>
      <span id="cropZoomLbl">3&times;</span>
      <button id="cropZoomIn">+</button>
      <button id="pixelBtn" title="Show hard pixels instead of smooth rendering (Key P)">Pixels</button>
    </div>

    <span id="flagBadge" class="flag-badge hidden">Touches image border: partial cell</span>
  </div>

  <div class="meta">
    <div><span class="label">Cell ID:</span><span class="val" id="metaId">-</span></div>
    <div><span class="label">BBox (x1,y1)-(x2,y2):</span><span class="val" id="metaBbox">-</span></div>
    <div><span class="label">Dimensions:</span><span class="val" id="metaDim">-</span></div>
    <div><span class="label">Area:</span><span class="val" id="metaArea">-</span> px&sup2;</div>
    <div><span class="label">Touches border:</span><span class="val" id="metaFlag">-</span></div>
    <div><span class="label">Signal left outside:</span><span class="val" id="metaQa">-</span></div>
  </div>
</header>

<main class="viewer">
  <!-- Left Panel: Cell Crop -->
  <div class="panel cell-panel" id="cellPanel">
    <div class="panel-header">
      <span class="panel-title" id="cropPanelTitle">Cell Cut (RAW Image)</span>
      <span class="panel-file" id="cellFile">-</span>
    </div>

    <!-- Mode 1 & 2: Single Crop Box -->
    <div class="img-box checker-bg" id="singleCropBox">
      <img id="cellImg" src="" alt="cell crop">
    </div>

    <!-- Mode 3: Dual Crop Mode -->
    <div class="dual-crop-box" id="dualCropBox" style="display: none;">
      <div class="dual-sub-box">
        <span class="sub-label">Cut from RAW</span>
        <div class="img-box checker-bg">
          <img id="dualOrigImg" src="" alt="original crop">
        </div>
      </div>
      <div class="dual-sub-box">
        <span class="sub-label">MASK Image (cyan outline)</span>
        <div class="img-box">
          <img id="dualSharpImg" src="" alt="mask image crop">
        </div>
      </div>
    </div>
  </div>

  <!-- Right Panel: Whole Slide Location Correlation -->
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
    <span class="thumbs-hint">Keys: &larr;/&rarr; step cell | 1=RAW Cut, 2=MASK Image, 3=Dual, D=Debug Modal, O=Overview | &minus;/+ crop zoom, P=pixels | &uarr;/&darr; switch slide</span>
  </div>
  <div class="thumbs" id="thumbs"></div>
</footer>

<!-- Full-Screen Debug Modal Overlay -->
<!-- Overview Overlay: taken vs. ignored cells (shown first) -->
<div class="overview-overlay" id="overviewOverlay">
  <div class="overview-head">
    <h2 id="overviewTitle">Overview</h2>
    <span class="legend-chip"><i style="background:#3cc83c"></i><span id="ovTaken">Taken</span></span>
    <span class="legend-chip"><i style="background:#e62828"></i><span id="ovIgnored">Ignored</span></span>
    <span class="overview-note">Cells with no colour have no cyan outline, so the method does not see them.</span>
    <button id="overviewStart" title="Close the overview (O or Esc)">Review cells &rarr;</button>
  </div>
  <div class="overview-body"><img id="overviewImg" src="" alt="taken vs ignored cells"></div>
</div>

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
const OVERVIEWS = {overviews_json};

let currentDataset = DATASET_NAMES[0] || "";
let currentIndex = 0;
let cropViewMode = "orig"; // "orig" | "sharp" | "dual"
let debugStepIndex = 0;

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
const metaQa = document.getElementById('metaQa');

const thumbsEl = document.getElementById('thumbs');
const subtitleEl = document.getElementById('subtitle');
const datasetSelect = document.getElementById('datasetSelect');

const modeOrigBtn = document.getElementById('modeOrig');
const modeSharpBtn = document.getElementById('modeSharp');
const modeDualBtn = document.getElementById('modeDual');
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

function setViewMode(mode) {{
  cropViewMode = mode;
  modeOrigBtn.classList.toggle('active-mode', mode === 'orig');
  modeSharpBtn.classList.toggle('active-mode', mode === 'sharp');
  modeDualBtn.classList.toggle('active-mode', mode === 'dual');

  if (mode === 'dual') {{
    singleCropBox.style.display = 'none';
    dualCropBox.style.display = 'flex';
    cellPanel.className = 'panel cell-panel wide-active' + (pixelView ? ' pixel-view' : '');
    cropPanelTitle.textContent = 'Dual View (RAW Cut + MASK Image)';
  }} else {{
    singleCropBox.style.display = 'block';
    dualCropBox.style.display = 'none';
    cellPanel.className = 'panel cell-panel' + (pixelView ? ' pixel-view' : '');
    cropPanelTitle.textContent = mode === 'orig' ? 'Cell Cut (RAW Image)' : 'Cell Crop (MASK Image)';
  }}
  render();
}}

/* ---------- CROP ZOOM: fit the crop in its box, but never above cropZoom x natural size ---------- */
const CROP_ZOOMS = [1, 1.5, 2, 3, 4, 6, 8];
let cropZoom = 3;
let pixelView = false;
const cropZoomLbl = document.getElementById('cropZoomLbl');
const pixelBtn = document.getElementById('pixelBtn');

function fitCrop(img) {{
  const box = img.parentElement;
  if (!img.naturalWidth || !box.clientWidth) return;
  const scale = Math.min(box.clientWidth / img.naturalWidth, box.clientHeight / img.naturalHeight, cropZoom);
  img.style.width = Math.round(img.naturalWidth * scale) + 'px';
  img.style.height = Math.round(img.naturalHeight * scale) + 'px';
}}
function fitAllCrops() {{ [cellImg, dualOrigImg, dualSharpImg].forEach(fitCrop); }}
[cellImg, dualOrigImg, dualSharpImg].forEach(img => img.addEventListener('load', () => fitCrop(img)));
window.addEventListener('resize', fitAllCrops);

function setCropZoom(step) {{
  const i = Math.max(0, Math.min(CROP_ZOOMS.length - 1, CROP_ZOOMS.indexOf(cropZoom) + step));
  cropZoom = CROP_ZOOMS[i];
  cropZoomLbl.textContent = cropZoom + '\u00d7';
  fitAllCrops();
}}
function togglePixelView() {{
  pixelView = !pixelView;
  cellPanel.classList.toggle('pixel-view', pixelView);
  pixelBtn.classList.toggle('active-mode', pixelView);
}}
document.getElementById('cropZoomOut').addEventListener('click', () => setCropZoom(-1));
document.getElementById('cropZoomIn').addEventListener('click', () => setCropZoom(1));
pixelBtn.addEventListener('click', togglePixelView);

function render() {{
  const cells = currentCells();
  if (!cells.length) return;
  const c = cells[currentIndex];
  const prefix = currentDataset + '/';

  const origPath = prefix + c.original_filename;
  const sharpPath = prefix + c.annotated_filename;
  const locPath = prefix + c.location_filename;

  if (cropViewMode === 'orig') {{
    cellImg.src = origPath;
    cellFile.textContent = c.original_filename;
  }} else if (cropViewMode === 'sharp') {{
    cellImg.src = sharpPath;
    cellFile.textContent = c.annotated_filename;
  }} else if (cropViewMode === 'dual') {{
    dualOrigImg.src = origPath;
    dualSharpImg.src = sharpPath;
    cellFile.textContent = c.original_filename + ' | ' + c.annotated_filename;
  }}

  // Box sizes can change with the view mode; cached images fire no load event
  requestAnimationFrame(fitAllCrops);

  locationImg.src = locPath;
  locationFile.textContent = c.location_filename;

  positionEl.textContent = 'Cell ' + (currentIndex + 1) + ' / ' + cells.length;

  metaId.textContent = 'subcell_' + String(c.cell_id).padStart(3, '0');
  metaBbox.textContent = '(' + c.x1 + ', ' + c.y1 + ') - (' + c.x2 + ', ' + c.y2 + ')';
  metaDim.textContent = c.width + ' x ' + c.height + ' px';
  metaArea.textContent = c.area;
  metaFlag.textContent = c.flagged ? 'True (partial cell)' : 'False';
  // The qa_flag is not shown: with the contour-only mask it measures the excluded aura, not mistakes.
  metaQa.textContent = c.outside_signal_pct === '' ? '-' :
    c.outside_signal_pct + '% of ring' + (c.crowded ? ' (crowded: not scored)' : ' (info only)');

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

  debugModalTitle.textContent = 'Debug Trace \\u2014 Cell #' + c.cell_id + ' (subcell_' + String(c.cell_id).padStart(3, '0') + ')';
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

// Scroll wheel stepping inside debug modal
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
    btn.title = 'Cell ' + c.cell_id + (c.flagged ? ' (touches border)' : '');
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

/* ---------- OVERVIEW OVERLAY ---------- */
const overviewOverlay = document.getElementById('overviewOverlay');
function isOverviewOpen() {{ return !overviewOverlay.classList.contains('hidden'); }}
function updateOverview() {{
  const ov = OVERVIEWS[currentDataset] || {{}};
  document.getElementById('overviewTitle').textContent = 'Overview \u2014 ' + currentDataset;
  document.getElementById('ovTaken').textContent = 'Taken: ' + (ov.taken ?? '-') + ' cells';
  document.getElementById('ovIgnored').textContent = 'Ignored: ' + (ov.ignored_border ?? 0) + ' cut by the image edge, ' +
    (ov.ignored_pockets ?? 0) + ' empty pockets, ' + (ov.ignored_size ?? 0) + ' wrong size';
  document.getElementById('overviewImg').src = ov.map ? currentDataset + '/' + ov.map : '';
}}
function openOverview() {{ updateOverview(); overviewOverlay.classList.remove('hidden'); }}
function closeOverview() {{ overviewOverlay.classList.add('hidden'); }}
document.getElementById('overviewStart').addEventListener('click', closeOverview);
document.getElementById('overviewBtn').addEventListener('click', openOverview);

function selectDataset(name) {{
  if (!DATASETS.hasOwnProperty(name)) return;
  currentDataset = name;
  currentIndex = 0;
  datasetSelect.value = name;
  subtitleEl.textContent = currentDataset + ' \\u2014 ' + DATASETS[name].length + ' sub-cells with whole-slide correlation';
  buildThumbs();
  render();
  updateOverview();
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

modeOrigBtn.addEventListener('click', () => setViewMode('orig'));
modeSharpBtn.addEventListener('click', () => setViewMode('sharp'));
modeDualBtn.addEventListener('click', () => setViewMode('dual'));

document.addEventListener('keydown', (e) => {{
  if (isOverviewOpen()) {{
    if (e.key === 'Escape' || e.key.toLowerCase() === 'o') closeOverview();
    else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {{
      const curIdx = DATASET_NAMES.indexOf(currentDataset) + (e.key === 'ArrowUp' ? -1 : 1);
      if (curIdx >= 0 && curIdx < DATASET_NAMES.length) selectDataset(DATASET_NAMES[curIdx]);
    }}
    return;
  }}
  if (e.key.toLowerCase() === 'o' && !isDebugOpen()) {{ openOverview(); return; }}
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
    else if (e.key === '1') {{ setViewMode('orig'); }}
    else if (e.key === '2') {{ setViewMode('sharp'); }}
    else if (e.key === '3') {{ setViewMode('dual'); }}
    else if (e.key === '-') {{ setCropZoom(-1); }}
    else if (e.key === '+' || e.key === '=') {{ setCropZoom(1); }}
    else if (e.key.toLowerCase() === 'p') {{ togglePixelView(); }}
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
  subtitleEl.textContent = currentDataset + ' \\u2014 ' + DATASETS[currentDataset].length + ' sub-cells with whole-slide correlation';
  buildThumbs();
  render();
  updateOverview();
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

        overviews = {}
        for name, manifest_path in datasets:
            summary_path = os.path.join(os.path.dirname(manifest_path), "summary.json")
            summary = {}
            if os.path.isfile(summary_path):
                with open(summary_path) as f:
                    summary = json.load(f)
            has_map = os.path.isfile(os.path.join(os.path.dirname(manifest_path), "03_taken_vs_ignored.png"))
            overviews[name] = {
                "map": "03_taken_vs_ignored.png" if has_map else None,
                "taken": len(all_data[name]),
                "ignored_border": summary.get("cells_excluded_border", 0),
                "ignored_pockets": summary.get("rejected_background_pockets", 0),
                "ignored_size": summary.get("regions_ignored_size", 0),
            }

        datasets_json = json.dumps(all_data, separators=(",", ":"))
        step_info_json = json.dumps(STEP_DESCRIPTIONS, separators=(",", ":"))
        html = HTML_TEMPLATE.format(datasets_json=datasets_json, step_info_json=step_info_json,
                                    overviews_json=json.dumps(overviews, separators=(",", ":")))

        named_path = os.path.join(out_root, "pair_contour_gallery.html")
        with open(named_path, "w") as f:
            f.write(html)
        gpath = os.path.join(out_root, "gallery.html")
        with open(gpath, "w") as f:
            f.write(html)
        print(f"Wrote {named_path} and {gpath} with {len(all_data)} dataset(s):")
        for name in all_data:
            print(f"  - {name}: {len(all_data[name])} cells")

if __name__ == "__main__":
    build_gallery()
