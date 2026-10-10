#!/usr/bin/env python3
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dpgnet.src.infer import run_batch_inference

if __name__ == "__main__":
    run_batch_inference()
