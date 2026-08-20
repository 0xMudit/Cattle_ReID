"""
DEPRECATED — superseded by train_v3.py (adds focal loss, hard-negative mining,
mixup, synthetic augmentation, torch.compile, and correct batch sizing).

This shim keeps old commands working by running the v3 pipeline.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_v3 import train, parse_args  # noqa: E402

if __name__ == "__main__":
    print("[warn] train_hanwoo.py is deprecated — use train_v3.py "
          "(identical behavior, more features)")
    train(parse_args())