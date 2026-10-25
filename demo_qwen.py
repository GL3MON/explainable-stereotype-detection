#!/usr/bin/env python3
"""Run the three-task Qwen3.5 terminal demo from a source checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stereodetect.inference.qwen_demo import main

if __name__ == "__main__":
    raise SystemExit(main())
