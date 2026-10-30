#!/usr/bin/env python3
"""
build_report.py

Writes an offline comparison report (report.html) next to the enhance outputs:
  - metrics table with the pass limits, one row per candidate
  - swipe viewer: original on the left, the selected candidate (or its residual) on the right
"""

import json
import os


def build_report(out_dir, summary):
    path = os.path.join(out_dir, "report.html")
    payload = json.dumps(summary, separators=(",", ":"))
    with open(path, "w") as f:
        f.write(HTML_TEMPLATE.replace("__REPORT_DATA__", payload))
    return path


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Enhance Report | MicroGlia</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1117; --panel-bg: #181b22; --panel-sub: #212631; --border: #2e3545;
    --text: #f0f3f8; --muted: #8b949e; --pass: #10b981; --fail: #ef4444; --accent: #22d3ee;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); font-size: 13px;
    display: flex; flex-direction: column; height: 100vh; overflow: hidden;
  }
  header { padding: 10px 16px; border-bottom: 1px solid var(--border); display: flex; flex-wrap: wrap; gap: 10px 18px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 8px 0 0; }
  .group { display: flex; gap: 6px; align-items: center; }
  .muted { color: var(--muted); }
  select, button {
    background: var(--panel-sub); color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 5px 9px; font: inherit; cursor: pointer;
  }
  .seg { display: inline-flex; }
  .seg button { border-radius: 0; }
  .seg button:first-child { border-radius: 6px 0 0 6px; }
  .seg button:last-child { border-radius: 0 6px 6px 0; }
  .seg button.active { background: #334155; }
  main { flex: 1; display: flex; min-height: 0; }
  #stage { flex: 1; overflow: auto; background: #000; }
  #viewer { position: relative; display: inline-block; cursor: col-resize; }
  #viewer img { display: block; user-select: none; -webkit-user-drag: none; }
  #viewer.pixel img { image-rendering: pixelated; }
  #rightLayer { position: absolute; left: 0; top: 0; }
  #divider { position: absolute; top: 0; bottom: 0; width: 2px; background: var(--accent); pointer-events: none; }
  .tag { position: absolute; top: 6px; padding: 2px 6px; border-radius: 4px; background: rgba(0,0,0,.6); font-size: 11px; pointer-events: none; }
  #tagL { left: 6px; } #tagR { right: 6px; }
  aside { width: 470px; flex: 0 0 470px; border-left: 1px solid var(--border); background: var(--panel-bg); overflow-y: auto; padding: 12px 14px; display: flex; flex-direction: column; gap: 14px; }
  aside h2 { font-size: 13px; margin: 0 0 6px; color: var(--muted); text-transform: uppercase; letter-spacing: .05em; }
  table { border-collapse: collapse; width: 100%; }
  th, td { padding: 4px 5px; text-align: right; border-bottom: 1px solid var(--border); font-variant-numeric: tabular-nums; }
  th:first-child, td:first-child { text-align: left; }
  th { color: var(--muted); font-weight: 500; }
  tr.sel td { background: var(--panel-sub); }
  tbody tr { cursor: pointer; }
  .bad { color: var(--fail); font-weight: 600; }
  .pass { color: var(--pass); font-weight: 600; }
  .fail { color: var(--fail); font-weight: 600; }
  .hint { color: var(--muted); line-height: 1.5; }
  dl { display: grid; grid-template-columns: auto 1fr; gap: 3px 10px; margin: 0; }
  dt { color: var(--muted); }
  dd { margin: 0; }
</style>
</head>
<body>
<header>
  <h1>Enhance Report</h1>
  <span class="muted" id="subtitle"></span>
  <div class="group"><span class="muted">Candidate</span><select id="candSelect"></select></div>
  <div class="group">
    <span class="muted">Right side</span>
    <div class="seg" id="viewSeg">
      <button data-view="image">1 Enhanced</button>
      <button data-view="residual">2 Residual</button>
    </div>
  </div>
  <div class="group"><span class="muted">Zoom</span>
    <button id="zoomOut">&minus;</button><span id="zoomLbl">100%</span><button id="zoomIn">+</button>
    <button id="pixelBtn" title="Show pixels without smoothing (P)">Pixels</button>
  </div>
</header>
<main>
  <div id="stage">
    <div id="viewer">
      <img id="leftImg" alt="original">
      <div id="rightLayer"><img id="rightImg" alt="candidate"></div>
      <div id="divider"></div>
      <span class="tag" id="tagL">Original</span>
      <span class="tag" id="tagR"></span>
    </div>
  </div>
  <aside>
    <section>
      <h2>Candidates (click a row)</h2>
      <table id="metricTable"></table>
    </section>
    <section>
      <h2>Selected candidate</h2>
      <dl id="candInfo"></dl>
    </section>
    <section class="hint">
      <h2>How to read this</h2>
      <b>BgStd</b>: grey-level std of the flat background (noise left).<br>
      <b>ResEdge</b>: correlation of the removed signal with the cell edges. Must be &le; 0.10; a higher value means the method changed structure, not only noise.<br>
      <b>Texture / Edges</b>: per-cell texture std and outline edge strength, enhanced / original. Must stay within 0.95&ndash;1.05.<br>
      <b>Aura</b>: white-aura contrast, enhanced / original. Must be &ge; 0.95.<br>
      <b>Shift</b>: mean change of the per-cell mean intensity (grey levels). Must be &le; 0.50.<br>
      Residual view: 128 + gain &times; (original &minus; enhanced). Pure noise looks like even grain; visible cell shapes mean features were changed.<br>
      Drag on the image to move the divider. Keys: 1/2 view, &uarr;/&darr; candidate, +/&minus; zoom, P pixels.
    </section>
  </aside>
</main>
<script>
const R = __REPORT_DATA__;
const state = { cand: 0, view: "image", zoom: 1, split: 0.5, pixel: true };
const $ = id => document.getElementById(id);
const COLS = [
  ["BgStd", "flat_background_std"], ["ResEdge", "residual_edge_corr"], ["Shift", "cell_mean_abs_shift"],
  ["Texture", "cell_texture_ratio"], ["Edges", "edge_strength_ratio"], ["Aura", "aura_contrast_ratio"], ["SSIM", "ssim"],
];

function render() {
  const c = R.candidates[state.cand];
  const [w, h] = R.image_size;
  const W = w * state.zoom, H = h * state.zoom;
  $("leftImg").src = "original.png";
  $("rightImg").src = state.view === "image" ? `${c.name}.png` : `${c.name}_residual.png`;
  for (const img of [$("leftImg"), $("rightImg")]) { img.style.width = W + "px"; img.style.height = H + "px"; }
  $("viewer").style.width = W + "px"; $("viewer").style.height = H + "px";
  $("viewer").classList.toggle("pixel", state.pixel);
  $("rightLayer").style.clipPath = `inset(0 0 0 ${state.split * 100}%)`;
  $("divider").style.left = `calc(${state.split * 100}% - 1px)`;
  $("tagR").textContent = (state.view === "image" ? c.name : `${c.name} residual ×${R.residual_gain}`) + (c.scale > 1 ? ` (×${c.scale} shown at original size)` : "");
  $("zoomLbl").textContent = Math.round(state.zoom * 100) + "%";
  document.querySelectorAll("#viewSeg button").forEach(b => b.classList.toggle("active", b.dataset.view === state.view));
  $("pixelBtn").classList.toggle("active", state.pixel);
  $("candSelect").value = state.cand;
  document.querySelectorAll("#metricTable tbody tr").forEach((tr, i) => tr.classList.toggle("sel", +tr.dataset.idx === state.cand));
  $("candInfo").innerHTML = [
    ["Method", c.description],
    ["Result", c.passed ? '<span class="pass">PASS</span>' : `<span class="fail">FAIL</span> (${c.failed.join(", ")})`],
    ["PSNR vs original", c.metrics.psnr_db + " dB"],
    ["Residual std", c.metrics.residual_std + " grey levels"],
    ["Residual inside cells / background", c.metrics.residual_cell_ratio],
    ["Files", `${c.name}.png, ${c.name}_16bit.tif, ${c.name}_residual.png`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}

function failsLimit(key, value) {
  const lim = R.pass_limits[key];
  if (!lim || value === null) return false;
  const [kind, l] = lim;
  return kind === "max" ? value > l : kind === "min" ? value < l : (value < l[0] || value > l[1]);
}

function buildTable() {
  const head = "<thead><tr><th>Candidate</th><th>Result</th>" + COLS.map(([n]) => `<th>${n}</th>`).join("") + "</tr></thead>";
  const o = R.original;
  let body = `<tr><td>original</td><td class="muted">ref</td>` + COLS.map(([, k]) => `<td class="muted">${k === "residual_edge_corr" || k === "cell_mean_abs_shift" ? "–" : o[k]}</td>`).join("") + "</tr>";
  R.candidates.forEach((c, i) => {
    body += `<tr data-idx="${i}"><td>${c.name}</td><td class="${c.passed ? "pass" : "fail"}">${c.passed ? "PASS" : "FAIL"}</td>` +
      COLS.map(([, k]) => `<td class="${failsLimit(k, c.metrics[k]) ? "bad" : ""}">${c.metrics[k]}</td>`).join("") + "</tr>";
  });
  $("metricTable").innerHTML = head + "<tbody>" + body + "</tbody>";
  $("metricTable").querySelectorAll("tbody tr[data-idx]").forEach(tr => tr.addEventListener("click", () => { state.cand = +tr.dataset.idx; render(); }));
}

function setSplit(e) {
  const r = $("viewer").getBoundingClientRect();
  state.split = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
  render();
}

$("subtitle").textContent = `${R.raw_image} · noise σ ${R.noise_sigma} grey levels · ${R.original.cells_measured} cells`;
R.candidates.forEach((c, i) => $("candSelect").add(new Option(`${c.name} — ${c.passed ? "PASS" : "FAIL"}`, i)));
$("candSelect").addEventListener("change", e => { state.cand = +e.target.value; render(); });
document.querySelectorAll("#viewSeg button").forEach(b => b.addEventListener("click", () => { state.view = b.dataset.view; render(); }));
$("zoomIn").addEventListener("click", () => { state.zoom = Math.min(8, state.zoom * 1.5); render(); });
$("zoomOut").addEventListener("click", () => { state.zoom = Math.max(0.5, state.zoom / 1.5); render(); });
$("pixelBtn").addEventListener("click", () => { state.pixel = !state.pixel; render(); });
let dragging = false;
$("viewer").addEventListener("mousedown", e => { dragging = true; setSplit(e); });
window.addEventListener("mousemove", e => { if (dragging) setSplit(e); });
window.addEventListener("mouseup", () => { dragging = false; });
document.addEventListener("keydown", e => {
  if (e.target.tagName === "SELECT") return;
  if (e.key === "1") state.view = "image";
  else if (e.key === "2") state.view = "residual";
  else if (e.key === "ArrowDown") state.cand = Math.min(R.candidates.length - 1, state.cand + 1);
  else if (e.key === "ArrowUp") state.cand = Math.max(0, state.cand - 1);
  else if (e.key === "+" || e.key === "=") state.zoom = Math.min(8, state.zoom * 1.5);
  else if (e.key === "-") state.zoom = Math.max(0.5, state.zoom / 1.5);
  else if (e.key === "p" || e.key === "P") state.pixel = !state.pixel;
  else return;
  e.preventDefault(); render();
});
buildTable();
render();
</script>
</body>
</html>
"""
