"""
DEPRECATED — superseded by test_model_v3.py (letterbox preprocessing,
class count inferred from the checkpoint).

This shim keeps old commands working by running the v3 evaluation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.test_model_v3 import main  # noqa: E402

if __name__ == "__main__":
    print("[warn] test_model_v2.py is deprecated — use test_model_v3.py")
    main()