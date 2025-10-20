#!/usr/bin/env python3
"""Standalone script to run tract geometry analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.geometry.comprehensive_tract_geometry_analysis import main

if __name__ == "__main__":
    main()
