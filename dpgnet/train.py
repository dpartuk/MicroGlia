#!/usr/bin/env python3
import os
import sys

# Ensure repository root is in Python module search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dpgnet.src.train import main

if __name__ == "__main__":
    main()
