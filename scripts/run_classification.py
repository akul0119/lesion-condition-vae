#!/usr/bin/env python3
"""Standalone script to run TBI vs PTE classification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.classification import main

if __name__ == "__main__":
    main()
