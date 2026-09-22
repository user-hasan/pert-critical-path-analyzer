"""
Allow launching with: python -m pert_analyzer.gui
"""

from __future__ import annotations

import sys

from pert_analyzer.gui.app import main

if __name__ == "__main__":
    sys.exit(main())
