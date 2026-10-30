"""
Shared helpers for the labeling bundle: encoding, hashes, the lock file and mask comparison.

Identity of a cell:
    cell_key   = <slide>@<cx>_<cy>          (mask centroid, what labels are stored under)
    mask hash  = sha1(box + mask bits)[:12] (the exact pixels the rater saw, stored with each label)
    slide hash = sha1(sorted key:mask hash)[:12]

labeling/published_slides.json (in git) is the lock: one entry per published slide with its slide hash,
cell count, pipeline settings and pipeline commit. A locked slide is never re-exported with other masks
unless export_web.py gets --force.
"""

import base64
import csv
import hashlib
import json
import os
import subprocess

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPE_DIR = os.path.join(ROOT, "pair_contour", "output")
WEB_DIR = os.path.join(ROOT, "labeling", "web")
SLIDES_DIR = os.path.join(WEB_DIR, "slides")
LOCK_PATH = os.path.join(ROOT, "labeling", "published_slides.json")
MIN_IOU = 0.5


def mask_hash(box, bits_b64):
    return hashlib.sha1((",".join(map(str, box)) + ":" + bits_b64).encode("ascii")).hexdigest()[:12]


def slide_hash(cells):
    lines = sorted(c["key"] + ":" + c["mh"] for c in cells)
    return hashlib.sha1("\n".join(lines).encode("ascii")).hexdigest()[:12]


def pipeline_commit():
    """Short commit of the repo plus '-dirty' when pair_contour/ has uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "pair_contour/process.py", "pair_contour/src"],
                               cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_lock():
    if not os.path.exists(LOCK_PATH):
        return {}
    return json.load(open(LOCK_PATH))


def save_lock(lock):
    with open(LOCK_PATH, "w") as f:
        json.dump(dict(sorted(lock.items())), f, indent=1)
        f.write("\n")


def read_slide_js(path):
    """Parse a slides/<slide>.js file written by export_web.py."""
    text = open(path).read()
    start = text.index("] = ") + 4
    return json.loads(text[start:text.rstrip().rindex(";")])


def decode_mask(cell, width, height):
    x1, y1, x2, y2 = cell["box"]
    w, h = x2 - x1, y2 - y1
    bits = np.unpackbits(np.frombuffer(base64.b64decode(cell["bits"]), dtype=np.uint8))[: w * h].astype(bool)
    full = np.zeros((height, width), dtype=bool)
    full[y1:y2, x1:x2] = bits.reshape(h, w)
    return full


def slide_file_masks(slide_data):
    """{cell_key: full-slide mask} from a parsed slide file."""
    return {c["key"]: decode_mask(c, slide_data["width"], slide_data["height"]) for c in slide_data["cells"]}


def pipeline_masks(slide_dir):
    """[(cell_id, full-slide mask)] from a pair_contour slide output folder."""
    summary = json.load(open(os.path.join(slide_dir, "summary.json")))
    img_w, img_h = summary["image_size"]
    cells = []
    with open(os.path.join(slide_dir, "manifest.csv")) as f:
        for row in csv.DictReader(f):
            x1, y1, x2, y2 = (int(row[k]) for k in ("x1", "y1", "x2", "y2"))
            full = np.zeros((img_h, img_w), dtype=bool)
            full[y1:y2, x1:x2] = cv2.imread(os.path.join(slide_dir, row["mask_filename"]), cv2.IMREAD_GRAYSCALE) > 0
            cells.append((int(row["cell_id"]), full))
    return cells


def best_match(mask, candidates):
    """Best IoU of `mask` against [(id, mask)]: (id, iou), or (None, 0.0)."""
    best_id, best_iou = None, 0.0
    ys, xs = np.nonzero(mask)
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    for cid, cm in candidates:
        inter = np.count_nonzero(mask[y1:y2, x1:x2] & cm[y1:y2, x1:x2])
        if inter == 0:
            continue
        iou = inter / np.count_nonzero(mask | cm)
        if iou > best_iou:
            best_id, best_iou = cid, iou
    return best_id, best_iou


def compare(frozen, current):
    """
    Compare frozen masks {key: mask} with current masks [(id, mask)].
    Returns {"identical": [...], "changed": [(key, id, iou)], "removed": [key], "new": [id]}.
    identical = a current mask equal pixel by pixel; changed = best IoU >= MIN_IOU but not equal.
    """
    out = {"identical": [], "changed": [], "removed": [], "new": []}
    by_id, used = dict(current), set()
    for key, fm in frozen.items():
        cid, iou = best_match(fm, current)
        if cid is None or iou < MIN_IOU:
            out["removed"].append(key)
            continue
        used.add(cid)
        if np.array_equal(fm, by_id[cid]):
            out["identical"].append(key)
        else:
            out["changed"].append((key, cid, round(iou, 3)))
    out["new"] = [cid for cid, _ in current if cid not in used]
    return out


def report(name, diff):
    n = {k: len(v) for k, v in diff.items()}
    ok = n["changed"] == n["removed"] == n["new"] == 0
    print(f"  {name}: {n['identical']} identical, {n['changed']} changed, {n['removed']} removed, {n['new']} new"
          + ("" if ok else "  <-- DIFFERS"))
    for key, cid, iou in diff["changed"][:5]:
        print(f"      changed {key} -> cell {cid} (IoU {iou})")
    for key in diff["removed"][:5]:
        print(f"      removed {key}")
    return ok
