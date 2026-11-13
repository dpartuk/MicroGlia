"""
Select unlabeled cells that are probably Resolution, for the extra labeling page.

Phase 2 experiment (decision 2026-11-13). The rater's labels are unbalanced: about 5% Resolution.
A logistic model (Resolution vs all other classes) is trained on the rater's first-look labels with
shape and texture measures of each pair_probe cell. It scores every cell of the published slides that
has no label yet, and the top cells go to labeling/extra/<name>.csv. export_extra.py builds the page.

Measures per cell: log area, perimeter, solidity, roundness, aspect, extent, body thickness, number of
parts; RAW mean, std, 10th / 90th percentile, mean gradient, Laplacian std inside the mask; and the
pipeline columns interior_std, ring_px, halo_px, entry_px, outside_signal_pct, edge_background_pct.

Test: each labeled slide is scored by a model trained on the other labeled slides.
Selection: the top --count unlabeled cells by score, at most --max-folder-share from one data folder.

  python3 labeling/select_resolution.py "<store dir>/labels/*/slides/*.json" --name resolution_set1
"""

import argparse
import collections
import csv
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import bundle

EXTRA_DIR = os.path.join(bundle.ROOT, "labeling", "extra")
TARGET = "resolution"
MAN_COLS = ["interior_std", "ring_px", "halo_px", "entry_px", "outside_signal_pct", "edge_background_pct"]


def slide_features(slide):
    """{cell_id: {measure: value}} and the data folder of one pipeline slide."""
    sdir = os.path.join(bundle.PIPE_DIR, slide)
    summary = json.load(open(os.path.join(sdir, "summary.json")))
    raw_path = glob.glob(os.path.join(bundle.ROOT, "Data", "pair-data", "*", summary["raw_image"]))[0]
    folder = os.path.basename(os.path.dirname(raw_path))
    g = cv2.imread(raw_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    grad = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    lap = cv2.Laplacian(g, cv2.CV_32F)
    out = {}
    for r in csv.DictReader(open(os.path.join(sdir, "manifest.csv"))):
        x1, y1 = int(r["x1"]), int(r["y1"])
        m = cv2.imread(os.path.join(sdir, r["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 0
        h, w = m.shape
        win = (slice(y1, y1 + h), slice(x1, x1 + w))
        gi, gr, la = g[win][m], grad[win][m], lap[win][m]
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(contours, key=cv2.contourArea)
        area, per = float(m.sum()), cv2.arcLength(c, True)
        hull = cv2.contourArea(cv2.convexHull(c))
        (_, _), (rw, rh), _ = cv2.minAreaRect(c)
        dist = cv2.distanceTransform(np.pad(m, 1).astype(np.uint8), cv2.DIST_L2, 5)
        f = {"log_area": np.log(area), "perimeter": per, "solidity": area / max(hull, 1.0),
             "roundness": 4 * np.pi * area / max(per, 1.0) ** 2, "aspect": max(rw, rh) / max(min(rw, rh), 1.0),
             "extent": area / max(rw * rh, 1.0), "max_thickness": float(dist.max()), "n_parts": len(contours),
             "mean": float(gi.mean()), "std": float(gi.std()), "p10": float(np.percentile(gi, 10)),
             "p90": float(np.percentile(gi, 90)), "grad_mean": float(gr.mean()), "lap_std": float(la.std())}
        for k in MAN_COLS:
            f[k] = float(r[k])
        out[int(r["cell_id"])] = f
    return slide, folder, out


def read_labels(files):
    """{cell_key: label} of first looks, from store slide documents or page exports."""
    labels = {}
    for path in files:
        doc = json.load(open(path))
        doc = doc.get("data", doc)
        items = doc["cells"].items() if "cells" in doc else ((r["cell_key"], r) for r in doc["labels"] if not r.get("repeat"))
        for key, rec in items:
            if rec.get("label"):
                labels[key] = rec["label"]
    return labels


def main():
    parser = argparse.ArgumentParser(description="Select probable Resolution cells for the extra labeling page")
    parser.add_argument("labels", nargs="+", help="Store slide documents or page exports (globs allowed)")
    parser.add_argument("--name", default="resolution_set1", help="Output: labeling/extra/<name>.csv")
    parser.add_argument("--count", type=int, default=1500, help="Number of cells to select")
    parser.add_argument("--max-folder-share", type=float, default=0.5, help="Largest share of the selection from one data folder")
    args = parser.parse_args()

    labels = read_labels(sorted({p for pat in args.labels for p in glob.glob(pat)}))
    # Published cells: key, cell id and slide come from the published slide files (the cells the rater sees)
    lock = bundle.load_lock()
    cells = []
    for slide in sorted(lock):
        data = bundle.read_slide_js(os.path.join(bundle.SLIDES_DIR, slide + ".js"))
        if data["hash"] != lock[slide]["hash"]:
            raise RuntimeError(f"{slide}: slide file differs from the lock")
        cells += [(slide, c["cell_id"], c["key"]) for c in data["cells"]]
    with ProcessPoolExecutor() as ex:
        feats = {slide: (folder, f) for slide, folder, f in ex.map(slide_features, sorted(lock))}
    names = list(next(iter(next(iter(feats.values()))[1].values())))
    X = np.array([[feats[s][1][cid][k] for k in names] for s, cid, _ in cells])
    folders = np.array([feats[s][0] for s, _, _ in cells])
    slides = np.array([s for s, _, _ in cells])
    labeled = np.array([key in labels for _, _, key in cells])
    y = np.array([labels.get(key) == TARGET for _, _, key in cells])
    print(f"Published cells {len(cells)}, labeled {labeled.sum()}, {TARGET} {y[labeled].sum()} ({100 * y[labeled].mean():.1f}%)")

    def model():
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000))

    held = np.zeros(len(cells))
    for s in np.unique(slides[labeled]):
        train, test = labeled & (slides != s), labeled & (slides == s)
        held[test] = model().fit(X[train], y[train]).decision_function(X[test])
    order = np.argsort(-held[labeled])
    yl = y[labeled]
    print(f"Leave-one-slide-out: AUC {roc_auc_score(yl, held[labeled]):.3f}")
    for frac in (0.05, 0.10, 0.20):
        top = order[:int(frac * len(order))]
        print(f"  top {int(frac * 100)}%: precision {100 * yl[top].mean():.0f}%, recall {100 * yl[top].sum() / yl.sum():.0f}%")

    score = model().fit(X[labeled], y[labeled]).decision_function(X)
    cap = int(args.max_folder_share * args.count)
    taken, per_folder = [], collections.Counter()
    for i in np.argsort(-score):
        if labeled[i] or per_folder[folders[i]] >= cap:
            continue
        taken.append(i)
        per_folder[folders[i]] += 1
        if len(taken) == args.count:
            break
    print(f"Selected {len(taken)} cells, by folder {dict(sorted(per_folder.items()))}, lowest score {score[taken[-1]]:.2f}")

    os.makedirs(EXTRA_DIR, exist_ok=True)
    path = os.path.join(EXTRA_DIR, args.name + ".csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["slide", "folder", "cell_id", "cell_key", "score"])
        for i in sorted(taken, key=lambda i: (slides[i], cells[i][1])):
            w.writerow([slides[i], folders[i], cells[i][1], cells[i][2], round(float(score[i]), 3)])
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
