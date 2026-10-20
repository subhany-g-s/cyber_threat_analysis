"""
scripts/run_pipeline.py -- thin entry point; all logic lives in src.service / src.cli.
Usage:  python scripts/run_pipeline.py --help     (or the root-level run_pipeline.py)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
