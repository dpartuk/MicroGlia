"""
Check that the pipeline still gives the published cells.

For each slide in labeling/published_slides.json:
  1. The slide file in labeling/web/slides must have the locked slide hash.
  2. pair_contour runs again with the LOCKED settings into a temporary folder
     (pair_contour/output is not touched). With --current, the existing pair_contour/output is used instead.
  3. The new masks are compared with the published masks: identical, changed (IoU >= 0.5, not equal),
     removed (no match) and new (no published mask).

Exit code 0 = every slide identical. Exit code 1 = a difference: do not republish those slides,
or republish them only with export_web.py --force after a decision.
"""

import argparse
import glob
import inspect
import os
import sys
import tempfile

import bundle

sys.path.insert(0, os.path.join(bundle.ROOT, "pair_contour", "src"))
from extract_pair_cells import extract_pair_cells, mask_path_for  # noqa: E402


def rerun(entry, out_dir):
    hits = glob.glob(os.path.join(bundle.ROOT, "Data", "pair-data", "*", entry["raw_image"]))
    if not hits:
        raise FileNotFoundError("RAW image not found: " + entry["raw_image"])
    accepted = inspect.signature(extract_pair_cells).parameters
    settings = {k: v for k, v in entry["settings"].items() if k in accepted}
    ignored = sorted(set(entry["settings"]) - set(settings))
    if ignored:
        print(f"      settings not accepted by extract_pair_cells, ignored: {ignored}")
    extract_pair_cells(raw_path=hits[0], mask_path=mask_path_for(hits[0]), output_dir=out_dir, debug_cell_ids=None, **settings)


def main():
    parser = argparse.ArgumentParser(description="Compare the pipeline output with the published labeling slides")
    parser.add_argument("slides", nargs="*", help="Slide names (default: every locked slide)")
    parser.add_argument("--current", action="store_true", help="Use pair_contour/output instead of a fresh run")
    args = parser.parse_args()

    lock = bundle.load_lock()
    names = args.slides or sorted(lock)
    if not names:
        print("No locked slides. Run export_web.py --mark-published after a publish.")
        return 1
    print(f"Pipeline commit now: {bundle.pipeline_commit()}")
    all_ok = True
    for name in names:
        entry = lock.get(name)
        path = os.path.join(bundle.SLIDES_DIR, name + ".js")
        if entry is None or not os.path.exists(path):
            print(f"  {name}: {'not locked' if entry is None else 'no slide file in ' + bundle.SLIDES_DIR}")
            all_ok = False
            continue
        data = bundle.read_slide_js(path)
        if data.get("hash") != entry["hash"]:
            print(f"  {name}: slide file hash {data.get('hash')} differs from the lock {entry['hash']}")
            all_ok = False
            continue
        frozen = bundle.slide_file_masks(data)
        if args.current:
            current = bundle.pipeline_masks(os.path.join(bundle.PIPE_DIR, name))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                out_dir = os.path.join(tmp, name)
                devnull = open(os.devnull, "w")
                stdout, sys.stdout = sys.stdout, devnull
                try:
                    rerun(entry, out_dir)
                finally:
                    sys.stdout = stdout
                    devnull.close()
                current = bundle.pipeline_masks(out_dir)
        all_ok &= bundle.report(f"{name} (locked commit {entry['commit']})", bundle.compare(frozen, current))
    print("RESULT: " + ("all published cells are unchanged" if all_ok else "DIFFERENCES FOUND - see above"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
