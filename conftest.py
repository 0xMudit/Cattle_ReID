# Ensures the repo root is on sys.path so tests can import training.*
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))