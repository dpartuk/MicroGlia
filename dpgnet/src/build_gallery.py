#!/usr/bin/env python3
"""
build_gallery.py

Generates an offline, interactive review gallery for DPGNet microglia segmentation at:
  - dpgnet/output/gallery.html

Features:
  - Multi-dataset selector across all processed whole slides
  - Cell Inspection Mode with 4-way side-by-side view:
      1. Original RGB Extracted Crop
      2. DPGNet Predicted Binary Mask
      3. Boundary Uncertainty Heatmap (EDA Attention)
      4. Whole-Slide Location Context (Marked BBox)
  - Whole-Slide Overview Mode (Raw Slide, DPGNet Mask, Uncertainty Heatmap, Contour Overlay, Ground Truth)
  - Split 2x2 comparison grid for whole-slide inspection
  - Live sorting (ID, Area, Uncertainty) and filtering (Uncertainty levels, Area size)
  - Bottom thumbnail strip with auto-scrolling
  - Modal high-resolution zoom
  - Full keyboard shortcuts navigation
"""

import csv
import json
import os
import shutil
from typing import Dict, List, Any
import cv2


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DPGNET_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(DPGNET_ROOT, "output")
CONTOUR_OUTPUT_DIR = os.path.abspath(os.path.join(DPGNET_ROOT, "..", "contour", "output"))


def discover_datasets(output_root: str) -> List[Dict[str, Any]]:
    datasets = []
    if not os.path.isdir(output_root):
        return datasets

    for entry in sorted(os.listdir(output_root)):
        full_path = os.path.join(output_root, entry)
        if not os.path.isdir(full_path):
            continue

        manifest_path = os.path.join(full_path, "manifest.csv")
        if not os.path.isfile(manifest_path):
            continue

        # Check and copy contour ground truth mask if available
        gt_source = os.path.join(CONTOUR_OUTPUT_DIR, entry, "whole_slide_ground_truth_mask.png")
        gt_dest = os.path.join(full_path, "ground_truth_mask.png")
        has_gt = False
        if os.path.isfile(gt_source):
            if not os.path.isfile(gt_dest):
                shutil.copy2(gt_source, gt_dest)
            has_gt = True
        elif os.path.isfile(gt_dest):
            has_gt = True

        # Read slide image dimensions
        ref_img_path = os.path.join(full_path, "00_original_reference.jpg")
        img_w, img_h = 0, 0
        if os.path.isfile(ref_img_path):
            ref_img = cv2.imread(ref_img_path)
            if ref_img is not None:
                img_h, img_w = ref_img.shape[:2]

        cells = []
        with open(manifest_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cid = int(row["cell_id"])
                area = int(row["area"])
                unc = float(row.get("mean_uncertainty", 0.0))
                cells.append({
                    "cell_id": cid,
                    "x1": int(row["x1"]),
                    "y1": int(row["y1"]),
                    "x2": int(row["x2"]),
                    "y2": int(row["y2"]),
                    "width": int(row.get("width", int(row["x2"]) - int(row["x1"]))),
                    "height": int(row.get("height", int(row["y2"]) - int(row["y1"]))),
                    "area": area,
                    "mean_uncertainty": unc,
                    "original_filename": row["original_filename"],
                    "dpgnet_mask_filename": row["dpgnet_mask_filename"],
                    "uncertainty_filename": row["uncertainty_filename"],
                    "location_filename": row["location_filename"],
                })

        cells.sort(key=lambda c: c["cell_id"])

        # Compute summary stats
        total_cells = len(cells)
        mean_area = round(sum(c["area"] for c in cells) / max(total_cells, 1), 1)
        mean_unc = round(sum(c["mean_uncertainty"] for c in cells) / max(total_cells, 1), 4)

        datasets.append({
            "name": entry,
            "dir": entry,
            "img_width": img_w,
            "img_height": img_h,
            "has_gt": has_gt,
            "total_cells": total_cells,
            "mean_area": mean_area,
            "mean_uncertainty": mean_unc,
            "cells": cells,
        })

    return datasets


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DPGNet Review Gallery | Microglia Boundary-Aware Segmentation</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #0a0d14;
    --panel-bg: #131722;
    --panel-sub: #1b202e;
    --border: #262c3d;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --accent: #38bdf8;
    --accent-glow: rgba(56, 189, 248, 0.25);
    --accent-bg: rgba(56, 189, 248, 0.12);
    --badge-green: #10b981;
    --badge-purple: #a855f7;
    --badge-orange: #f59e0b;
    --badge-red: #ef4444;
    --unc-high: #f43f5e;
    --unc-med: #f59e0b;
    --unc-low: #10b981;
  }
  * { box-sizing: border-box; }
  html, body {
    height: 100%;
    margin: 0;
  }
  body {
    padding: 8px 14px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    height: 100dvh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  /* ---------- Top Band ---------- */
  .top-band {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .titlebar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 10px;
  }
  .title-left {
    display: flex;
    align-items: baseline;
    gap: 10px;
  }
  h1 {
    font-size: 1.15rem;
    font-weight: 700;
    margin: 0;
    color: var(--accent);
    letter-spacing: -0.01em;
  }
  .subtitle {
    font-size: 0.78rem;
    color: var(--text-muted);
  }
  .badges-group {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .badge {
    font-size: 0.70rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--panel-sub);
    color: var(--text-muted);
  }
  .badge-pvt {
    background: rgba(168, 85, 247, 0.15);
    border-color: var(--badge-purple);
    color: #d8b4fe;
  }
  .badge-eda {
    background: rgba(56, 189, 248, 0.15);
    border-color: var(--accent);
    color: #7dd3fc;
  }
  .badge-stat {
    background: rgba(16, 185, 129, 0.15);
    border-color: var(--badge-green);
    color: #6ee7b7;
  }

  /* Toolbar */
  .toolbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    padding: 6px 12px;
    border-radius: 8px;
    gap: 10px;
    flex-wrap: wrap;
  }
  .tool-group {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
  }
  button, select, input {
    font-size: 0.80rem;
    padding: 5px 10px;
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--panel-sub);
    color: var(--text);
    cursor: pointer;
    transition: all 0.15s ease;
  }
  button:hover, select:hover {
    background: var(--accent-bg);
    border-color: var(--accent);
    color: var(--accent);
  }
  button:active { transform: translateY(1px); }
  button.active-btn {
    background: var(--accent);
    color: #0b0d14;
    font-weight: 700;
    border-color: var(--accent);
  }
  select#datasetSelect {
    font-weight: 600;
    max-width: 280px;
  }
  .pos-indicator {
    font-weight: 700;
    font-size: 0.82rem;
    min-width: 110px;
    text-align: center;
    color: var(--text);
  }

  /* KPI / Metadata Strip */
  .kpi-strip {
    display: flex;
    gap: 8px;
    align-items: center;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    padding: 5px 12px;
    border-radius: 8px;
    font-size: 0.76rem;
    flex-wrap: wrap;
  }
  .kpi-item {
    display: flex;
    gap: 4px;
    align-items: center;
  }
  .kpi-label {
    color: var(--text-muted);
  }
  .kpi-val {
    font-weight: 600;
    color: var(--text);
  }
  .unc-badge {
    padding: 1px 7px;
    border-radius: 999px;
    font-size: 0.70rem;
    font-weight: 700;
  }
  .unc-badge.high { background: rgba(244, 63, 94, 0.2); color: #fb7185; border: 1px solid #f43f5e; }
  .unc-badge.med { background: rgba(245, 158, 11, 0.2); color: #fcd34d; border: 1px solid #f59e0b; }
  .unc-badge.low { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid #10b981; }

  /* ---------- Middle Band: Main Viewer ---------- */
  .main-content {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* View Mode 1: Cell Inspection 4-Panel Grid */
  .cell-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    height: 100%;
    min-height: 0;
  }
  @media (max-width: 1100px) {
    .cell-grid {
      grid-template-columns: repeat(2, 1fr);
      grid-template-rows: repeat(2, 1fr);
    }
  }
  .cell-panel {
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }
  .panel-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
    flex: 0 0 auto;
  }
  .panel-title {
    font-size: 0.76rem;
    font-weight: 700;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  .panel-subtitle {
    font-size: 0.68rem;
    color: var(--text-muted);
  }
  .img-container {
    flex: 1 1 auto;
    min-height: 0;
    position: relative;
    background: #000;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    cursor: zoom-in;
    display: flex;
    align-items: center;
    justify-content: center;
  }
  .img-container.checker-bg {
    background:
      linear-gradient(45deg, #1b202e 25%, transparent 25%),
      linear-gradient(-45deg, #1b202e 25%, transparent 25%),
      linear-gradient(45deg, transparent 75%, #1b202e 75%),
      linear-gradient(-45deg, transparent 75%, #1b202e 75%);
    background-size: 16px 16px;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px;
    background-color: #ffffff;
  }
  .img-container img {
    max-width: 100%;
    max-height: 100%;
    width: auto;
    height: auto;
    object-fit: contain;
    image-rendering: pixelated;
  }

  /* View Mode 2: Whole Slide Overview */
  .slide-view {
    display: none;
    flex-direction: column;
    gap: 8px;
    height: 100%;
    min-height: 0;
  }
  .slide-view.active {
    display: flex;
  }
  .slide-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 0 0 auto;
  }
  .slide-container {
    flex: 1 1 auto;
    min-height: 0;
    display: flex;
    gap: 10px;
    overflow: hidden;
  }
  .slide-single-box {
    flex: 1 1 auto;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px;
    display: flex;
    flex-direction: column;
    min-height: 0;
  }
  .slide-grid-box {
    display: none;
    grid-template-columns: repeat(2, 1fr);
    grid-template-rows: repeat(2, 1fr);
    gap: 8px;
    flex: 1 1 auto;
    min-height: 0;
  }
  .slide-grid-box.active {
    display: grid;
  }

  /* ---------- Bottom Band: Filmstrip Thumbnails ---------- */
  .bottom-strip {
    flex: 0 0 auto;
    display: flex;
    flex-direction: column;
    gap: 3px;
  }
  .strip-header {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    font-size: 0.72rem;
    color: var(--text-muted);
  }
  .thumbs-container {
    display: flex;
    gap: 6px;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 4px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    height: 72px;
  }
  .thumb-card {
    flex: 0 0 auto;
    width: 60px;
    height: 60px;
    background: #0f131a;
    border: 2px solid transparent;
    border-radius: 6px;
    cursor: pointer;
    position: relative;
    overflow: hidden;
    transition: transform 0.12s ease, border-color 0.12s ease;
  }
  .thumb-card:hover {
    transform: scale(1.06);
    border-color: var(--accent);
  }
  .thumb-card.selected {
    border-color: var(--accent);
    box-shadow: 0 0 10px var(--accent-glow);
  }
  .thumb-card img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    image-rendering: pixelated;
  }
  .thumb-badge {
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    background: rgba(0,0,0,0.75);
    color: #fff;
    font-size: 0.58rem;
    text-align: center;
    padding: 1px 0;
    font-weight: 700;
  }

  /* ---------- Fullscreen Zoom Modal ---------- */
  .modal-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.88);
    backdrop-filter: blur(4px);
    display: none;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  .modal-overlay.open { display: flex; }
  .modal-content {
    max-width: 94vw;
    max-height: 94vh;
    display: flex;
    flex-direction: column;
    gap: 8px;
    background: var(--panel-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
  }
  .modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .modal-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--accent);
  }
  .modal-close {
    background: transparent;
    border: none;
    color: var(--text-muted);
    font-size: 1.2rem;
    cursor: pointer;
  }
  .modal-img-box {
    flex: 1 1 auto;
    overflow: auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    border-radius: 6px;
    max-height: 85vh;
  }
  .modal-img-box img {
    max-width: 90vw;
    max-height: 82vh;
    object-fit: contain;
    image-rendering: pixelated;
  }
</style>
</head>
<body>

  <!-- Top Band -->
  <div class="top-band">
    <div class="titlebar">
      <div class="title-left">
        <h1>DPGNet Review Gallery</h1>
        <span class="subtitle">Boundary-Aware Segmentation & Boundary Uncertainty Perception</span>
      </div>
      <div class="badges-group">
        <span class="badge badge-pvt">Backbone: PVTv2-B2 (25.51M)</span>
        <span class="badge badge-eda">EDA Edge Difference Attention</span>
        <span class="badge badge-stat" id="totalSlidesBadge">5 Slides</span>
        <span class="badge badge-stat" id="totalCellsBadge">670 Cells</span>
      </div>
    </div>

    <!-- Main Toolbar -->
    <div class="toolbar">
      <div class="tool-group">
        <label for="datasetSelect" style="font-size:0.76rem; color:var(--text-muted);">Slide Dataset:</label>
        <select id="datasetSelect"></select>

        <button id="viewCellModeBtn" class="active-btn" onclick="setViewMode('cell')">🔬 Cell Inspection</button>
        <button id="viewSlideModeBtn" onclick="setViewMode('slide')">🗺️ Whole Slide</button>
      </div>

      <!-- Cell navigation controls -->
      <div class="tool-group" id="cellNavGroup">
        <button onclick="navFirst()" title="First cell (Home)">⏮</button>
        <button onclick="navPrev()" title="Previous cell (Left Arrow)">◀</button>
        <span class="pos-indicator" id="posIndicator">Cell 1 / 1</span>
        <button onclick="navNext()" title="Next cell (Right Arrow)">▶</button>
        <button onclick="navLast()" title="Last cell (End)">⏭</button>

        <span style="color:var(--border);">|</span>

        <label for="sortSelect" style="font-size:0.75rem; color:var(--text-muted);">Sort:</label>
        <select id="sortSelect" onchange="onSortChange()">
          <option value="id_asc">Cell ID (Asc)</option>
          <option value="id_desc">Cell ID (Desc)</option>
          <option value="area_desc">Area (Largest)</option>
          <option value="area_asc">Area (Smallest)</option>
          <option value="unc_desc">Uncertainty (Highest)</option>
          <option value="unc_asc">Uncertainty (Lowest)</option>
        </select>

        <label for="filterSelect" style="font-size:0.75rem; color:var(--text-muted);">Filter:</label>
        <select id="filterSelect" onchange="onFilterChange()">
          <option value="all">All Cells</option>
          <option value="high_unc">High Uncertainty (≥ 0.58)</option>
          <option value="med_unc">Medium Uncertainty (0.50 - 0.58)</option>
          <option value="low_unc">Low Uncertainty (< 0.50)</option>
          <option value="large">Large Cells (> 2000 px²)</option>
          <option value="small">Small Cells (< 500 px²)</option>
        </select>
      </div>

      <!-- Whole slide controls -->
      <div class="tool-group" id="slideNavGroup" style="display:none;">
        <button id="slideTabOrig" class="active-btn" onclick="setSlideLayer('orig')">1. Raw Slide</button>
        <button id="slideTabMask" onclick="setSlideLayer('mask')">2. DPGNet Mask</button>
        <button id="slideTabUnc" onclick="setSlideLayer('unc')">3. Uncertainty</button>
        <button id="slideTabOver" onclick="setSlideLayer('over')">4. Overlay</button>
        <button id="slideTabGt" onclick="setSlideLayer('gt')">5. Ground Truth</button>
        <button id="slideTabGrid" onclick="setSlideLayer('grid')">🔲 2x2 Grid</button>
      </div>
    </div>

    <!-- KPI Strip -->
    <div class="kpi-strip" id="kpiStrip">
      <div class="kpi-item"><span class="kpi-label">Cell ID:</span> <span class="kpi-val" id="kpiCellId">-</span></div>
      <div class="kpi-item"><span class="kpi-label">ROI:</span> <span class="kpi-val" id="kpiRoi">-</span></div>
      <div class="kpi-item"><span class="kpi-label">Dimensions:</span> <span class="kpi-val" id="kpiDims">-</span></div>
      <div class="kpi-item"><span class="kpi-label">Area:</span> <span class="kpi-val" id="kpiArea">-</span></div>
      <div class="kpi-item">
        <span class="kpi-label">Boundary Uncertainty:</span>
        <span class="kpi-val" id="kpiUncVal">-</span>
        <span class="unc-badge low" id="kpiUncBadge">LOW</span>
      </div>
      <span style="color:var(--border);">|</span>
      <div class="kpi-item"><span class="kpi-label">Slide Cells:</span> <span class="kpi-val" id="kpiSlideCells">-</span></div>
      <div class="kpi-item"><span class="kpi-label">Slide Mean Area:</span> <span class="kpi-val" id="kpiSlideMeanArea">-</span></div>
      <div class="kpi-item"><span class="kpi-label">Slide Mean Uncertainty:</span> <span class="kpi-val" id="kpiSlideMeanUnc">-</span></div>
    </div>
  </div>

  <!-- Middle Band: Main Viewer -->
  <div class="main-content">

    <!-- Mode 1: Cell Inspection (4 Panels) -->
    <div class="cell-grid" id="cellViewGrid">
      <!-- Panel 1: Original RGB -->
      <div class="cell-panel">
        <div class="panel-top">
          <span class="panel-title">Original RGB</span>
          <span class="panel-subtitle">Extracted Cell Crop</span>
        </div>
        <div class="img-container checker-bg" onclick="openZoom(currentCellOriginalUrl(), 'Original RGB Extracted Cell')">
          <img id="imgCellOriginal" src="" alt="Original RGB Cell" />
        </div>
      </div>

      <!-- Panel 2: DPGNet Binary Mask -->
      <div class="cell-panel">
        <div class="panel-top">
          <span class="panel-title">DPGNet Binary Mask</span>
          <span class="panel-subtitle">Predicted Segmentation</span>
        </div>
        <div class="img-container" onclick="openZoom(currentCellMaskUrl(), 'DPGNet Predicted Binary Mask')">
          <img id="imgCellMask" src="" alt="DPGNet Binary Mask" />
        </div>
      </div>

      <!-- Panel 3: Boundary Uncertainty Heatmap -->
      <div class="cell-panel">
        <div class="panel-top">
          <span class="panel-title">Boundary Uncertainty</span>
          <span class="panel-subtitle">Edge Difference Attention (EDA)</span>
        </div>
        <div class="img-container" onclick="openZoom(currentCellUncertaintyUrl(), 'EDA Boundary Uncertainty Heatmap')">
          <img id="imgCellUncertainty" src="" alt="Boundary Uncertainty Heatmap" />
        </div>
      </div>

      <!-- Panel 4: Whole-Slide Location Context -->
      <div class="cell-panel">
        <div class="panel-top">
          <span class="panel-title">Slide Location Context</span>
          <span class="panel-subtitle">Bounding Box Context</span>
        </div>
        <div class="img-container" onclick="openZoom(currentCellLocationUrl(), 'Whole Slide Location Context')">
          <img id="imgCellLocation" src="" alt="Slide Location Context" />
        </div>
      </div>
    </div>

    <!-- Mode 2: Whole Slide Overview -->
    <div class="slide-view" id="slideViewContainer">
      <!-- Single Slide View -->
      <div class="slide-single-box" id="slideSingleBox">
        <div class="panel-top">
          <span class="panel-title" id="slideSingleTitle">Raw Microscopy Slide</span>
          <span class="panel-subtitle" id="slideSingleSubtitle">Whole slide field-of-view</span>
        </div>
        <div class="img-container" onclick="openZoom(document.getElementById('slideSingleImg').src, document.getElementById('slideSingleTitle').innerText)">
          <img id="slideSingleImg" src="" alt="Whole Slide View" />
        </div>
      </div>

      <!-- 2x2 Grid View -->
      <div class="slide-grid-box" id="slideGridBox">
        <div class="cell-panel">
          <div class="panel-top"><span class="panel-title">1. Raw Slide</span></div>
          <div class="img-container" onclick="openZoom(getSlideUrl('orig'), 'Raw Slide')">
            <img id="slideGridOrig" src="" alt="Raw Slide" />
          </div>
        </div>
        <div class="cell-panel">
          <div class="panel-top"><span class="panel-title">2. DPGNet Mask</span></div>
          <div class="img-container" onclick="openZoom(getSlideUrl('mask'), 'DPGNet Mask')">
            <img id="slideGridMask" src="" alt="DPGNet Mask" />
          </div>
        </div>
        <div class="cell-panel">
          <div class="panel-top"><span class="panel-title">3. Uncertainty Heatmap</span></div>
          <div class="img-container" onclick="openZoom(getSlideUrl('unc'), 'Uncertainty Heatmap')">
            <img id="slideGridUnc" src="" alt="Uncertainty Heatmap" />
          </div>
        </div>
        <div class="cell-panel">
          <div class="panel-top"><span class="panel-title">4. Prediction Overlay</span></div>
          <div class="img-container" onclick="openZoom(getSlideUrl('over'), 'Prediction Overlay')">
            <img id="slideGridOver" src="" alt="Prediction Overlay" />
          </div>
        </div>
      </div>
    </div>

  </div>

  <!-- Bottom Band: Filmstrip Strip -->
  <div class="bottom-strip" id="bottomFilmstrip">
    <div class="strip-header">
      <span>CELL PREVIEWS FILMSTRIP (Click thumbnail to view; use ← / → arrow keys to navigate)</span>
      <span id="filmstripCounter">Showing 0 cells</span>
    </div>
    <div class="thumbs-container" id="thumbsContainer"></div>
  </div>

  <!-- Fullscreen Zoom Modal -->
  <div class="modal-overlay" id="zoomModal" onclick="closeZoom(event)">
    <div class="modal-content" onclick="event.stopPropagation()">
      <div class="modal-header">
        <span class="modal-title" id="modalTitle">Image Zoom</span>
        <button class="modal-close" onclick="closeZoom()">&times;</button>
      </div>
      <div class="modal-img-box">
        <img id="modalImg" src="" alt="Zoomed View" />
      </div>
    </div>
  </div>

<script>
  // Injected dataset models
  const DATASETS = __DATASETS_JSON__;

  let currentDatasetIndex = 0;
  let activeCells = [];
  let currentCellIndex = 0;
  let currentViewMode = "cell"; // "cell" or "slide"
  let currentSlideLayer = "orig"; // "orig", "mask", "unc", "over", "gt", "grid"

  // Initialize
  function init() {
    populateDatasetSelect();
    loadDataset(0);
    setupKeyboardListeners();
  }

  function populateDatasetSelect() {
    const sel = document.getElementById("datasetSelect");
    sel.innerHTML = "";
    DATASETS.forEach((ds, idx) => {
      const opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = `${ds.name} (${ds.total_cells} cells)`;
      sel.appendChild(opt);
    });
    sel.addEventListener("change", (e) => {
      loadDataset(parseInt(e.target.value, 10));
    });
  }

  function loadDataset(index) {
    currentDatasetIndex = index;
    const ds = DATASETS[currentDatasetIndex];

    // Update dataset dropdown
    document.getElementById("datasetSelect").value = index;

    // Apply sorting & filtering
    applyFilterAndSort();

    // Update whole slide images
    updateSlideViews();
  }

  function applyFilterAndSort() {
    const ds = DATASETS[currentDatasetIndex];
    const filter = document.getElementById("filterSelect").value;
    const sort = document.getElementById("sortSelect").value;

    let filtered = [...ds.cells];

    // Filter
    if (filter === "high_unc") {
      filtered = filtered.filter(c => c.mean_uncertainty >= 0.58);
    } else if (filter === "med_unc") {
      filtered = filtered.filter(c => c.mean_uncertainty >= 0.50 && c.mean_uncertainty < 0.58);
    } else if (filter === "low_unc") {
      filtered = filtered.filter(c => c.mean_uncertainty < 0.50);
    } else if (filter === "large") {
      filtered = filtered.filter(c => c.area >= 2000);
    } else if (filter === "small") {
      filtered = filtered.filter(c => c.area < 500);
    }

    // Sort
    if (sort === "id_asc") {
      filtered.sort((a, b) => a.cell_id - b.cell_id);
    } else if (sort === "id_desc") {
      filtered.sort((a, b) => b.cell_id - a.cell_id);
    } else if (sort === "area_desc") {
      filtered.sort((a, b) => b.area - a.area);
    } else if (sort === "area_asc") {
      filtered.sort((a, b) => a.area - b.area);
    } else if (sort === "unc_desc") {
      filtered.sort((a, b) => b.mean_uncertainty - a.mean_uncertainty);
    } else if (sort === "unc_asc") {
      filtered.sort((a, b) => a.mean_uncertainty - b.mean_uncertainty);
    }

    activeCells = filtered;
    currentCellIndex = 0;
    renderThumbnails();
    displayCurrentCell();
  }

  function onFilterChange() {
    applyFilterAndSort();
  }

  function onSortChange() {
    applyFilterAndSort();
  }

  function renderThumbnails() {
    const container = document.getElementById("thumbsContainer");
    container.innerHTML = "";
    const ds = DATASETS[currentDatasetIndex];

    document.getElementById("filmstripCounter").textContent =
      `Showing ${activeCells.length} of ${ds.total_cells} cells`;

    activeCells.forEach((c, idx) => {
      const card = document.createElement("div");
      card.className = "thumb-card" + (idx === currentCellIndex ? " selected" : "");
      card.id = `thumb-${idx}`;
      card.onclick = () => selectCellIndex(idx);

      const img = document.createElement("img");
      img.src = `${ds.dir}/${c.original_filename}`;
      img.loading = "lazy";
      img.alt = `Cell #${c.cell_id}`;

      const badge = document.createElement("div");
      badge.className = "thumb-badge";
      badge.textContent = `#${c.cell_id}`;

      card.appendChild(img);
      card.appendChild(badge);
      container.appendChild(card);
    });
  }

  function selectCellIndex(idx) {
    if (idx < 0 || idx >= activeCells.length) return;
    currentCellIndex = idx;
    displayCurrentCell();
    scrollThumbIntoView();
  }

  function scrollThumbIntoView() {
    // Remove old selected class
    document.querySelectorAll(".thumb-card.selected").forEach(el => el.classList.remove("selected"));
    const activeThumb = document.getElementById(`thumb-${currentCellIndex}`);
    if (activeThumb) {
      activeThumb.classList.add("selected");
      activeThumb.scrollIntoView({ behavior: "smooth", block: "nearest", inline: "center" });
    }
  }

  function displayCurrentCell() {
    const ds = DATASETS[currentDatasetIndex];
    if (activeCells.length === 0) {
      document.getElementById("posIndicator").textContent = "0 / 0";
      document.getElementById("imgCellOriginal").src = "";
      document.getElementById("imgCellMask").src = "";
      document.getElementById("imgCellUncertainty").src = "";
      document.getElementById("imgCellLocation").src = "";
      document.getElementById("kpiCellId").textContent = "None";
      return;
    }

    const c = activeCells[currentCellIndex];
    document.getElementById("posIndicator").textContent = `Cell ${currentCellIndex + 1} / ${activeCells.length}`;

    // Images
    document.getElementById("imgCellOriginal").src = `${ds.dir}/${c.original_filename}`;
    document.getElementById("imgCellMask").src = `${ds.dir}/${c.dpgnet_mask_filename}`;
    document.getElementById("imgCellUncertainty").src = `${ds.dir}/${c.uncertainty_filename}`;
    document.getElementById("imgCellLocation").src = `${ds.dir}/${c.location_filename}`;

    // KPI Strip
    document.getElementById("kpiCellId").textContent = `#${c.cell_id}`;
    document.getElementById("kpiRoi").textContent = `(${c.x1}, ${c.y1}) - (${c.x2}, ${c.y2})`;
    document.getElementById("kpiDims").textContent = `${c.width} × ${c.height} px`;
    document.getElementById("kpiArea").textContent = `${c.area.toLocaleString()} px²`;
    document.getElementById("kpiUncVal").textContent = c.mean_uncertainty.toFixed(4);

    const badge = document.getElementById("kpiUncBadge");
    if (c.mean_uncertainty >= 0.58) {
      badge.textContent = "HIGH";
      badge.className = "unc-badge high";
    } else if (c.mean_uncertainty >= 0.50) {
      badge.textContent = "MED";
      badge.className = "unc-badge med";
    } else {
      badge.textContent = "LOW";
      badge.className = "unc-badge low";
    }

    document.getElementById("kpiSlideCells").textContent = ds.total_cells;
    document.getElementById("kpiSlideMeanArea").textContent = `${ds.mean_area.toLocaleString()} px²`;
    document.getElementById("kpiSlideMeanUnc").textContent = ds.mean_uncertainty.toFixed(4);
  }

  // Navigation handlers
  function navPrev() {
    if (activeCells.length === 0) return;
    const nextIdx = (currentCellIndex - 1 + activeCells.length) % activeCells.length;
    selectCellIndex(nextIdx);
  }
  function navNext() {
    if (activeCells.length === 0) return;
    const nextIdx = (currentCellIndex + 1) % activeCells.length;
    selectCellIndex(nextIdx);
  }
  function navFirst() {
    if (activeCells.length === 0) return;
    selectCellIndex(0);
  }
  function navLast() {
    if (activeCells.length === 0) return;
    selectCellIndex(activeCells.length - 1);
  }

  // Image URL getters
  function currentCellOriginalUrl() {
    const ds = DATASETS[currentDatasetIndex];
    return activeCells.length > 0 ? `${ds.dir}/${activeCells[currentCellIndex].original_filename}` : "";
  }
  function currentCellMaskUrl() {
    const ds = DATASETS[currentDatasetIndex];
    return activeCells.length > 0 ? `${ds.dir}/${activeCells[currentCellIndex].dpgnet_mask_filename}` : "";
  }
  function currentCellUncertaintyUrl() {
    const ds = DATASETS[currentDatasetIndex];
    return activeCells.length > 0 ? `${ds.dir}/${activeCells[currentCellIndex].uncertainty_filename}` : "";
  }
  function currentCellLocationUrl() {
    const ds = DATASETS[currentDatasetIndex];
    return activeCells.length > 0 ? `${ds.dir}/${activeCells[currentCellIndex].location_filename}` : "";
  }

  // View Mode Switcher
  function setViewMode(mode) {
    currentViewMode = mode;
    const cellBtn = document.getElementById("viewCellModeBtn");
    const slideBtn = document.getElementById("viewSlideModeBtn");
    const cellGrid = document.getElementById("cellViewGrid");
    const slideView = document.getElementById("slideViewContainer");
    const cellNav = document.getElementById("cellNavGroup");
    const slideNav = document.getElementById("slideNavGroup");
    const filmstrip = document.getElementById("bottomFilmstrip");

    if (mode === "cell") {
      cellBtn.classList.add("active-btn");
      slideBtn.classList.remove("active-btn");
      cellGrid.style.display = "grid";
      slideView.classList.remove("active");
      cellNav.style.display = "flex";
      slideNav.style.display = "none";
      filmstrip.style.display = "flex";
    } else {
      slideBtn.classList.add("active-btn");
      cellBtn.classList.remove("active-btn");
      cellGrid.style.display = "none";
      slideView.classList.add("active");
      cellNav.style.display = "none";
      slideNav.style.display = "flex";
      filmstrip.style.display = "none";
      updateSlideViews();
    }
  }

  // Whole Slide Views
  function getSlideUrl(layer) {
    const ds = DATASETS[currentDatasetIndex];
    if (layer === "orig") return `${ds.dir}/00_original_reference.jpg`;
    if (layer === "mask") return `${ds.dir}/dpgnet_binary_segmentation.png`;
    if (layer === "unc") return `${ds.dir}/dpgnet_uncertainty_heatmap.jpg`;
    if (layer === "over") return `${ds.dir}/dpgnet_prediction_overlay.jpg`;
    if (layer === "gt") return ds.has_gt ? `${ds.dir}/ground_truth_mask.png` : `${ds.dir}/00_original_reference.jpg`;
    return "";
  }

  function setSlideLayer(layer) {
    currentSlideLayer = layer;
    const buttons = {
      "orig": "slideTabOrig",
      "mask": "slideTabMask",
      "unc": "slideTabUnc",
      "over": "slideTabOver",
      "gt": "slideTabGt",
      "grid": "slideTabGrid"
    };

    Object.keys(buttons).forEach(k => {
      const btn = document.getElementById(buttons[k]);
      if (btn) {
        if (k === layer) btn.classList.add("active-btn");
        else btn.classList.remove("active-btn");
      }
    });

    const singleBox = document.getElementById("slideSingleBox");
    const gridBox = document.getElementById("slideGridBox");

    if (layer === "grid") {
      singleBox.style.display = "none";
      gridBox.classList.add("active");
      updateSlideViews();
    } else {
      gridBox.classList.remove("active");
      singleBox.style.display = "flex";

      const titleMap = {
        "orig": "Raw Microscopy Slide",
        "mask": "DPGNet Predicted Binary Segmentation Mask",
        "unc": "EDA Boundary Uncertainty Heatmap",
        "over": "Prediction Overlay (Green Contours on Tissue)",
        "gt": "Validated Ground Truth Mask"
      };
      const subMap = {
        "orig": "Full field-of-view raw tissue reference",
        "mask": "Ensembled 5-stage DPGNet binary segmentation output",
        "unc": "Edge Difference Attention map highlighting boundary ambiguity",
        "over": "Slide composited with detected cellular contours",
        "gt": "Reference ground-truth binary mask"
      };

      document.getElementById("slideSingleTitle").textContent = titleMap[layer] || "Whole Slide";
      document.getElementById("slideSingleSubtitle").textContent = subMap[layer] || "";
      document.getElementById("slideSingleImg").src = getSlideUrl(layer);
    }
  }

  function updateSlideViews() {
    if (currentSlideLayer === "grid") {
      document.getElementById("slideGridOrig").src = getSlideUrl("orig");
      document.getElementById("slideGridMask").src = getSlideUrl("mask");
      document.getElementById("slideGridUnc").src = getSlideUrl("unc");
      document.getElementById("slideGridOver").src = getSlideUrl("over");
    } else {
      document.getElementById("slideSingleImg").src = getSlideUrl(currentSlideLayer);
    }
  }

  // Zoom Modal
  function openZoom(src, title) {
    if (!src) return;
    document.getElementById("modalImg").src = src;
    document.getElementById("modalTitle").textContent = title || "Image Inspection";
    document.getElementById("zoomModal").classList.add("open");
  }

  function closeZoom(e) {
    document.getElementById("zoomModal").classList.remove("open");
  }

  // Keyboard Navigation
  function setupKeyboardListeners() {
    window.addEventListener("keydown", (e) => {
      // If modal open, Escape closes it
      if (document.getElementById("zoomModal").classList.contains("open")) {
        if (e.key === "Escape") {
          closeZoom();
        }
        return;
      }

      if (e.key === "ArrowLeft" || e.key === "a" || e.key === "[") {
        navPrev();
      } else if (e.key === "ArrowRight" || e.key === "d" || e.key === "]") {
        navNext();
      } else if (e.key === "Home") {
        navFirst();
      } else if (e.key === "End") {
        navLast();
      } else if (e.key === "m" || e.key === "M") {
        setViewMode(currentViewMode === "cell" ? "slide" : "cell");
      }
    });
  }

  window.onload = init;
</script>
</body>
</html>
"""


def build_gallery():
    """
    Scan dpgnet/output/ and generate standalone gallery.html
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    datasets = discover_datasets(OUTPUT_DIR)

    if not datasets:
        print(f"Warning: No valid datasets with manifest.csv found in {OUTPUT_DIR}")
        return

    total_cells = sum(d["total_cells"] for d in datasets)
    print(f"Discovered {len(datasets)} slide datasets with {total_cells} total cells.")

    datasets_json = json.dumps(datasets, indent=2)
    html_content = HTML_TEMPLATE.replace("__DATASETS_JSON__", datasets_json)

    out_file = os.path.join(OUTPUT_DIR, "gallery.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Successfully generated DPGNet review gallery: {out_file}")


if __name__ == "__main__":
    build_gallery()
