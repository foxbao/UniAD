#!/usr/bin/env python
"""Compatibility wrapper for the KL forecasting annotation utility."""

from pathlib import Path
import sys

TOOLS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

from data_converter.add_forecasting import main  # noqa: E402


if __name__ == '__main__':
    main()
