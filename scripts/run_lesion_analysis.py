#!/usr/bin/env python3
"""Standalone script to run lesion SH analysis."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lesion.lesion_sh_heme_comprehensive import main

if __name__ == "__main__":
    main()
