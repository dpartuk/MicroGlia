#!/usr/bin/env python3
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(SCRIPT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from process_boundary_sharpening import process_boundary_sharpening_dataset

if __name__ == "__main__":
    process_boundary_sharpening_dataset()
