"""Root-level launcher: python run_pipeline.py --input data\\sample_2000_events.json"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
