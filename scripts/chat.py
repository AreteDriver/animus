#!/usr/bin/env python3
"""Animus chat agent — thin shim over packages/core/animus/__main__.py.

This script sets up the Python path so Animus Core is importable from the
monorepo checkout, then delegates to the unified CLI entry point.

Usage:
    python scripts/chat.py
    # or from the core venv:
    packages/core/.venv/bin/python scripts/chat.py
"""

from __future__ import annotations

import os
import sys

# Ensure animus package is importable from monorepo
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "packages", "core"))

from animus.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
