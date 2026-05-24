#!/usr/bin/env python3
"""
Football Analysis System — entry point.

Run from the project root:
    python run.py

Or as a module:
    python -m app.main
"""
import sys
from pathlib import Path

# Ensure the project root is on sys.path so `app` and `pipeline` are importable
sys.path.insert(0, str(Path(__file__).parent))

from app.main import main

if __name__ == "__main__":
    main()
