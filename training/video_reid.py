"""
DEPRECATED — superseded by video_reid_v5.py (DeepSORT + multi-embedding
gallery, fixed frame timing, dated outputs).

This shim keeps old commands working by running the v5 pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.video_reid_v5 import main  # noqa: E402

if __name__ == "__main__":
    print("[warn] video_reid.py is deprecated — use video_reid_v5.py "
          "(DeepSORT + multi-embedding gallery)")
    main()