#!/usr/bin/env python3
"""
build_master_gallery.py

Generates comparison/output/master_gallery.html: one page with a tab for
every gallery that exists (method explorer, each method's cell gallery,
and the side-by-side comparison). Each tab shows its gallery in a frame.
Galleries that are not generated yet are left out, so re-run this script
after a new method produces its gallery.
"""

import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(COMPARISON_ROOT, ".."))
OUTPUT_DIR = os.path.join(COMPARISON_ROOT, "output")
MASTER_PATH = os.path.join(OUTPUT_DIR, "master_gallery.html")

# (tab key, tab label, gallery path relative to the project root, colour)
PAGES = [
    ("explorer", "Method Explorer", "comparison/output/methods_gallery.html", "#f0f3f8"),
    ("baseline", "Baseline", "baseline/output/gallery.html", "#f59e0b"),
    ("sharpened", "Sharpened", "sharpened/output/gallery.html", "#a78bfa"),
    ("contour", "Contour (GT)", "contour/output/gallery.html", "#4ade80"),
    ("dpgnet", "DPGNet", "dpgnet/output/gallery.html", "#f472b6"),
    ("side-by-side", "Baseline vs Sharpened", "comparison/output/gallery.html", "#8b949e"),
]


def build_master_gallery():
    pages = []
    for key, label, rel_path, color in PAGES:
        if os.path.isfile(os.path.join(PROJECT_ROOT, rel_path)):
            src = os.path.relpath(os.path.join(PROJECT_ROOT, rel_path), OUTPUT_DIR)
            pages.append({"key": key, "label": label, "src": src, "color": color})
            print(f"  - {label}: {rel_path}")
        else:
            print(f"  - {label}: not generated yet, skipped")

    html = HTML_TEMPLATE.replace("__PAGES__", json.dumps(pages))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(MASTER_PATH, "w") as f:
        f.write(html)
    print(f"\n[SUCCESS] Master gallery written to: {MASTER_PATH}")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Master Gallery | MicroGlia</title>
<style>
  :root {
    color-scheme: dark;
    --bg: #0f1117;
    --panel-sub: #212631;
    --border: #2e3545;
    --text: #f0f3f8;
    --muted: #8b949e;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    display: flex;
    flex-direction: column;
    font-size: 13px;
  }
  nav {
    flex: 0 0 auto;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border-bottom: 1px solid var(--border);
  }
  nav .brand { font-weight: 700; font-size: 14px; margin-right: 12px; }
  nav button {
    background: transparent;
    color: var(--muted);
    border: 1px solid transparent;
    border-radius: 6px;
    padding: 5px 12px;
    font: inherit;
    cursor: pointer;
  }
  nav button:hover { color: var(--text); background: var(--panel-sub); }
  nav button.active { color: var(--text); background: var(--panel-sub); border-color: var(--border); }
  nav button .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
  nav .spacer { flex: 1; }
  nav a { color: var(--muted); text-decoration: none; }
  nav a:hover { color: var(--text); }
  nav .keys { color: var(--muted); margin-left: 12px; }
  main { flex: 1; position: relative; min-height: 0; }
  iframe { position: absolute; inset: 0; width: 100%; height: 100%; border: 0; }
</style>
</head>
<body>
<nav id="tabs">
  <span class="brand">MicroGlia</span>
  <span class="spacer"></span>
  <a id="openTab" target="_blank" title="Open the current gallery on its own">Open in new tab &#8599;</a>
  <span class="keys">Alt + 1&ndash;9 switches tabs</span>
</nav>
<main id="frames"></main>
<script>
const PAGES = __PAGES__;
const tabs = document.getElementById('tabs');
const frames = document.getElementById('frames');
const spacer = tabs.querySelector('.spacer');
const buttons = {};
const loaded = {};

PAGES.forEach((p, i) => {
  const btn = document.createElement('button');
  btn.innerHTML = `<span class="dot" style="background:${p.color}"></span>${p.label}`;
  btn.title = `${p.src} (Alt+${i + 1})`;
  btn.addEventListener('click', () => show(p.key));
  tabs.insertBefore(btn, spacer);
  buttons[p.key] = btn;
});

// Each gallery keeps its own frame, so its slide and cell selection stay
// in place when you switch away and back.
function show(key) {
  const page = PAGES.find(p => p.key === key) || PAGES[0];
  if (!loaded[page.key]) {
    const frame = document.createElement('iframe');
    frame.src = page.src;
    frame.title = page.label;
    frames.appendChild(frame);
    loaded[page.key] = frame;
  }
  Object.entries(loaded).forEach(([k, f]) => { f.hidden = k !== page.key; });
  Object.entries(buttons).forEach(([k, b]) => b.classList.toggle('active', k === page.key));
  document.getElementById('openTab').href = page.src;
  document.title = `${page.label} | MicroGlia`;
  if (location.hash !== '#' + page.key) history.replaceState(null, '', '#' + page.key);
}

document.addEventListener('keydown', e => {
  const n = parseInt(e.key);
  if (e.altKey && n >= 1 && n <= PAGES.length) show(PAGES[n - 1].key);
});
window.addEventListener('hashchange', () => show(location.hash.slice(1)));
show(location.hash.slice(1));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    build_master_gallery()
