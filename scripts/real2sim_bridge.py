#!/usr/bin/env python3
"""Compatibility entry point for the local real2sim bridge CLI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from botyard_real2sim.cli.bridge import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
