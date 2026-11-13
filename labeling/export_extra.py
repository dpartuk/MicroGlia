"""
Build the web bundle of an extra labeling page from a cell selection (labeling/extra/<name>.csv).

Phase 2 experiment (decision 2026-11-13). The extra page is a separate artifact with its own label
store; the original page and its labels do not change. Its slide files hold only the selected cells,
copied from the published slide files of the original page: same RAW image, same masks, same cell keys
and mask hashes. So labels of both pages merge by cell key, and a cell labeled on both pages is found.

    labeling/web_extra/
        index.html          # labeling/src/labeling_page.html with the extra-set title
        slides.js           # the slides that have selected cells, in the order of the original page
        slides/<slide>.js   # the selected cells of one slide

The extra page has its own lock: labeling/published_extra.json (--mark-published after a publish).

  python3 labeling/export_extra.py --name resolution_set1
  python3 labeling/export_extra.py --name resolution_set1 --mark-published
"""

import argparse
import csv
import datetime
import glob
import json
import os
import shutil

import bundle
import export_web

WEB_DIR = os.path.join(bundle.ROOT, "labeling", "web_extra")
SLIDES_DIR = os.path.join(WEB_DIR, "slides")
LOCK_PATH = os.path.join(bundle.ROOT, "labeling", "published_extra.json")
TITLE = "Microglia Cell Labeling"
EXTRA_TITLE = "Microglia Cell Labeling · Extra Set"


def build(name, repeat_fraction):
    selected = {}
    for r in csv.DictReader(open(os.path.join(bundle.ROOT, "labeling", "extra", name + ".csv"))):
        selected.setdefault(r["slide"], set()).add(r["cell_key"])
    lock = bundle.load_lock()
    if os.path.isdir(SLIDES_DIR):
        shutil.rmtree(SLIDES_DIR)
    os.makedirs(SLIDES_DIR)
    index = []
    for path in glob.glob(os.path.join(bundle.SLIDES_DIR, "*.js")):
        data = bundle.read_slide_js(path)
        if data["name"] not in selected:
            continue
        if data["hash"] != lock.get(data["name"], {}).get("hash"):
            raise RuntimeError(f"{data['name']}: published slide file differs from the lock")
        cells = [c for c in data["cells"] if c["key"] in selected[data["name"]]]
        missing = selected[data["name"]] - {c["key"] for c in cells}
        if missing:
            raise RuntimeError(f"{data['name']}: selected keys not in the slide file: {sorted(missing)[:3]}")
        data = dict(data, cells=cells, hash=bundle.slide_hash(cells), extra_set=name, source_hash=data["hash"],
                    exported=datetime.date.today().isoformat())
        with open(os.path.join(SLIDES_DIR, data["name"] + ".js"), "w") as f:
            f.write("(window.SLIDE_DATA = window.SLIDE_DATA || {})[" + json.dumps(data["name"]) + "] = ")
            json.dump(data, f, separators=(",", ":"))
            f.write(";\n")
        index.append((export_web.slide_order(data), {"name": data["name"], "hash": data["hash"], "width": data["width"],
                                                     "height": data["height"], "cells": len(cells)}))
    slides = [s for _, s in sorted(index, key=lambda x: x[0])]
    with open(os.path.join(WEB_DIR, "slides.js"), "w") as f:
        f.write("window.LABELING = ")
        json.dump({"version": 3, "repeat_fraction": repeat_fraction, "extra_set": name, "slides": slides}, f, indent=1)
        f.write(";\n")
    page = open(export_web.PAGE_SRC).read()
    if page.count(TITLE) != 2:
        raise RuntimeError("The page source title changed; update TITLE in export_extra.py")
    with open(os.path.join(WEB_DIR, "index.html"), "w") as f:
        f.write(page.replace(TITLE, EXTRA_TITLE))
    print(f"Wrote {WEB_DIR}: {len(slides)} slides, {sum(s['cells'] for s in slides)} cells")


def mark_published(name):
    lock = json.load(open(LOCK_PATH)) if os.path.exists(LOCK_PATH) else {}
    today = datetime.date.today().isoformat()
    for path in sorted(glob.glob(os.path.join(SLIDES_DIR, "*.js"))):
        d = bundle.read_slide_js(path)
        lock[d["name"]] = {"hash": d["hash"], "cells": len(d["cells"]), "extra_set": name,
                           "source_hash": d["source_hash"], "published": today}
    with open(LOCK_PATH, "w") as f:
        json.dump(dict(sorted(lock.items())), f, indent=1)
        f.write("\n")
    print(f"Wrote {LOCK_PATH}: {len(lock)} slides. Commit it to git.")


def main():
    parser = argparse.ArgumentParser(description="Build the extra labeling page bundle from a cell selection")
    parser.add_argument("--name", default="resolution_set1", help="Selection: labeling/extra/<name>.csv")
    parser.add_argument("--repeat-fraction", type=float, default=0.10, help="Share of each slide's cells shown a second time")
    parser.add_argument("--mark-published", action="store_true", help="Only lock the slide files in labeling/web_extra/slides")
    args = parser.parse_args()
    if args.mark_published:
        mark_published(args.name)
    else:
        build(args.name, args.repeat_fraction)


if __name__ == "__main__":
    main()
