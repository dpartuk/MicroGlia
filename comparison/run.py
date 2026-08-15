#!/usr/bin/env python3
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from compare_baseline_vs_sharpening import compare_baseline_vs_sharpening
from build_comparison_gallery import build_gallery

if __name__ == "__main__":
    compare_baseline_vs_sharpening()
    build_gallery()
