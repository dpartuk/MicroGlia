#!/usr/bin/env python3
"""
build_comparison_gallery.py

Generates an offline, interactive side-by-side Method Comparison Gallery
at comparison/output/gallery.html
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(COMPARISON_ROOT, "output")
SUMMARY_JSON_PATH = os.path.join(OUTPUT_DIR, "comparison_summary.json")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Method Comparison: Baseline (Presaizen 2026) vs. Boundary Sharpening</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #0f1117;
    --panel-bg: #181b22;
    --panel-sub: #212631;
    --border: #2e3545;
    --text: #f0f3f8;
    --muted: #8b949e;
    --accent-base: #f59e0b;
    --accent-sharp: #38bdf8;
    --green: #10b981;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ height: 100%; margin: 0; }}
  body {{
    padding: 14px 20px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
    gap: 12px;
    height: 100vh;
    overflow: hidden;
  }}

  /* Top Header */
  .top-band {{
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }}
  .titlebar {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 12px;
  }}
  h1 {{
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    color: #fff;
  }}
  .titlebar .badges {{
    display: flex;
    gap: 8px;
  }}
  .badge {{
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-base {{ background: rgba(245, 158, 11, 0.15); border: 1px solid var(--accent-base); color: var(--accent-base); }}
  .badge-sharp {{ background: rgba(56, 189, 248, 0.15); border: 1px solid var(--accent-sharp); color: var(--accent-sharp); }}

  /* KPI Ribbon */
  .kpi-ribbon {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
  }}
  .kpi-card {{
    background: var(--panel-bg);
    border: 1px solid var(--border);
    padding: 8px 16px;
    border-radius: 8px;
    flex: 1 1 160px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .kpi-card .kpi-label {{
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }}
  .kpi-card .kpi-val {{
    font-size: 1.15rem;
    font-weight: 700;
  }}
  .kpi-card.base .kpi-val {{ color: var(--accent-base); }}
  .kpi-card.sharp .kpi-val {{ color: var(--accent-sharp); }}
  .kpi-card.gain .kpi-val {{ color: var(--green); }}

  /* Controls */
  .toolbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    padding: 6px 14px;
    border-radius: 8px;
  }}
  .picker {{
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  select, button {{
    font-size: 0.85rem;
    padding: 5px 12px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel-sub);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s ease;
  }}
  select:hover, button:hover {{
    border-color: var(--accent-sharp);
  }}
  select#slideSelect {{
    font-weight: 600;
    min-width: 280px;
  }}

  /* Main Comparison Viewer */
  .viewer {{
    flex: 1 1 auto;
    min-height: 0;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    overflow: hidden;
  }}
  .viewer-header {{
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--muted);
  }}
  .img-box {{
    flex: 1 1 auto;
    min-height: 0;
    position: relative;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: #000;
    overflow: hidden;
  }}
  .img-box img {{
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    object-fit: contain;
  }}

  /* Bottom Table */
  .summary-table-wrap {{
    flex: 0 0 auto;
    max-height: 130px;
    overflow-y: auto;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 6px 12px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.78rem;
  }}
  th, td {{
    padding: 4px 10px;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 0.70rem; }}
  tr.active-row {{ background: rgba(56, 189, 248, 0.1); font-weight: 600; }}
  .gain-text {{ color: var(--green); font-weight: 600; }}
</style>
</head>
<body>

<header class="top-band">
  <div class="titlebar">
    <h1>MicroGlia Method Comparison</h1>
    <div class="badges">
      <span class="badge badge-base">Baseline: Presaizen (2026)</span>
      <span class="badge badge-sharp">Proposed: Boundary Sharpening</span>
    </div>
  </div>

  <div class="kpi-ribbon">
    <div class="kpi-card base">
      <span class="kpi-label">Baseline Cell Yield</span>
      <span class="kpi-val" id="kpiBase">-</span>
    </div>
    <div class="kpi-card sharp">
      <span class="kpi-label">Sharpening Cell Yield</span>
      <span class="kpi-val" id="kpiSharp">-</span>
    </div>
    <div class="kpi-card gain">
      <span class="kpi-label">Net Sub-Cell Gain</span>
      <span class="kpi-val" id="kpiGain">-</span>
    </div>
    <div class="kpi-card gain">
      <span class="kpi-label">Percentage Gain</span>
      <span class="kpi-val" id="kpiPct">-</span>
    </div>
  </div>

  <div class="toolbar">
    <div class="picker">
      <label for="slideSelect">Select Slide:</label>
      <select id="slideSelect"></select>
    </div>
    <div class="nav">
      <button id="prevBtn">&larr; Previous Slide</button>
      <button id="nextBtn">Next Slide &rarr;</button>
    </div>
  </div>
</header>

<main class="viewer">
  <div class="viewer-header">
    <span id="viewTitle">Side-by-Side Whole-Slide Overview Map</span>
    <span id="viewFile">-</span>
  </div>
  <div class="img-box">
    <img id="compImg" src="" alt="side-by-side comparison map">
  </div>
</main>

<div class="summary-table-wrap">
  <table>
    <thead>
      <tr>
        <th>Slide Name</th>
        <th>Baseline Cells</th>
        <th>Boundary Sharpening</th>
        <th>Delta</th>
        <th>% Gain</th>
      </tr>
    </thead>
    <tbody id="tableBody"></tbody>
  </table>
</div>

<script>
const SUMMARY = {summary_json};
const SLIDES = SUMMARY.slides || [];
let currentIndex = 0;

const slideSelect = document.getElementById('slideSelect');
const compImg = document.getElementById('compImg');
const viewTitle = document.getElementById('viewTitle');
const viewFile = document.getElementById('viewFile');
const tableBody = document.getElementById('tableBody');

// Update KPIs
document.getElementById('kpiBase').textContent = SUMMARY.total_baseline_cells + ' cells';
document.getElementById('kpiSharp').textContent = SUMMARY.total_sharpened_cells + ' sub-cells';
document.getElementById('kpiGain').textContent = '+' + SUMMARY.net_gain + ' sub-cells';
document.getElementById('kpiPct').textContent = '+' + SUMMARY.percentage_gain + '%';

function render() {{
  if (!SLIDES.length) return;
  const s = SLIDES[currentIndex];
  compImg.src = s.comparison_image;
  viewFile.textContent = s.comparison_image;
  viewTitle.textContent = s.slide_name + ' | Baseline: ' + s.baseline_count + ' vs Sharpened: ' + s.sharpened_count + ' (' + s.delta_str + ')';
  slideSelect.value = currentIndex;

  // Highlight active row in table
  const rows = tableBody.querySelectorAll('tr');
  rows.forEach((r, idx) => {{
    r.classList.toggle('active-row', idx === currentIndex);
  }});
}}

function goTo(index) {{
  if (index < 0) index = 0;
  if (index > SLIDES.length - 1) index = SLIDES.length - 1;
  currentIndex = index;
  render();
}}

// Populate dropdown & table
SLIDES.forEach((s, idx) => {{
  const opt = document.createElement('option');
  opt.value = idx;
  opt.textContent = s.slide_name + ' (Baseline: ' + s.baseline_count + ' vs Sharpened: ' + s.sharpened_count + ')';
  slideSelect.appendChild(opt);

  const tr = document.createElement('tr');
  tr.style.cursor = 'pointer';
  tr.innerHTML = '<td>' + s.slide_name + '</td><td>' + s.baseline_count + '</td><td>' + s.sharpened_count + '</td><td class="gain-text">' + s.delta_str + '</td><td class="gain-text">+' + s.pct_change + '%</td>';
  tr.addEventListener('click', () => goTo(idx));
  tableBody.appendChild(tr);
}});

slideSelect.addEventListener('change', (e) => goTo(parseInt(e.target.value)));
document.getElementById('prevBtn').addEventListener('click', () => goTo(currentIndex - 1));
document.getElementById('nextBtn').addEventListener('click', () => goTo(currentIndex + 1));

document.addEventListener('keydown', (e) => {{
  if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') goTo(currentIndex - 1);
  else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') goTo(currentIndex + 1);
}});

render();
</script>

</body>
</html>
"""

def build_gallery():
    if not os.path.exists(SUMMARY_JSON_PATH):
        raise FileNotFoundError(f"Missing summary JSON: {SUMMARY_JSON_PATH}. Run compare_baseline_vs_sharpening.py first.")

    with open(SUMMARY_JSON_PATH) as f:
        summary_data = json.load(f)

    html = HTML_TEMPLATE.format(summary_json=json.dumps(summary_data, separators=(",", ":")))
    gallery_path = os.path.join(OUTPUT_DIR, "gallery.html")
    with open(gallery_path, "w") as f:
        f.write(html)
    print(f"Wrote comparison gallery -> {gallery_path}")

if __name__ == "__main__":
    build_gallery()
