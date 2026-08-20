#!/usr/bin/env python3
"""Download model weights required by the cattle Re-ID pipeline.

Downloads four files into cattle_osnet/models/ and cattle_osnet/:
  1. osnet_x1_0_imagenet.pth  — OSNet Re-ID backbone (KaiyangZhou/deep-person-reid)
  2. cow_pose.pt              — YOLOv8m-pose fine-tuned for cow keypoints
  3. yolov8m.pt               — YOLOv8m detection (cow class 21, COCO)
  4. yolov8n.pt               — YOLOv8n detection (cow class 21, COCO, lightweight)

Usage:
    python scripts/download_weights.py          # download all
    python scripts/download_weights.py --osnet  # OSNet only
    python scripts/download_weights.py --pose   # cow pose only
    python scripts/download_weights.py --yolo   # YOLOv8m detection
    python scripts/download_weights.py --yolo-nano  # YOLOv8n detection
"""

import argparse
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(REPO, "cattle_osnet", "models")

WEIGHTS = {
    "osnet": {
        "repo": "0xmudit/cattle-reid-weights",
        "filename": "osnet_x1_0_imagenet.pth",
        "dest": os.path.join(MODELS_DIR, "osnet_x1_0_imagenet.pth"),
        "desc": "OSNet x1.0 ImageNet-pretrained (Re-ID backbone)",
    },
    "pose": {
        "repo": "0xmudit/cattle-reid-weights",
        "filename": "cow_pose.pt",
        "dest": os.path.join(MODELS_DIR, "cow_pose.pt"),
        "desc": "YOLOv8m-pose fine-tuned for 12 cow keypoints",
    },
    "yolo": {
        "repo": "0xmudit/cattle-reid-weights",
        "filename": "yolov8n.pt",
        "dest": os.path.join(REPO, "cattle_osnet", "yolov8n.pt"),
        "desc": "YOLOv8n nano (COCO detection, cow class 19)",
    },
}


def download_weight(key, force=False):
    """Download a single weight file via huggingface_hub."""
    from huggingface_hub import hf_hub_download

    w = WEIGHTS[key]
    if os.path.exists(w["dest"]) and not force:
        print(f"  [skip] {key}: {w['dest']} already exists")
        return True

    print(f"  Downloading {w['desc']}...")
    try:
        path = hf_hub_download(repo_id=w["repo"], filename=w["filename"])
    except Exception as e:
        print(f"  [FAIL] {key}: {e}", file=sys.stderr)
        return False

    # hf_hub_download caches it; we need a copy in the right place
    import shutil
    os.makedirs(os.path.dirname(w["dest"]), exist_ok=True)
    shutil.copy2(path, w["dest"])
    size_mb = os.path.getsize(w["dest"]) / (1024 * 1024)
    print(f"  [OK]   {key}: {w['dest']} ({size_mb:.1f} MB)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Download model weights for cattle Re-ID")
    ap.add_argument("--osnet", action="store_true", help="download OSNet weights only")
    ap.add_argument("--pose", action="store_true", help="download cow pose weights only")
    ap.add_argument("--yolo", action="store_true", help="download YOLOv8n weights only")
    ap.add_argument("--force", action="store_true", help="re-download even if file exists")
    args = ap.parse_args()

    # If no specific flag, download all
    download_all = not (args.osnet or args.pose or args.yolo)
    keys = ["osnet", "pose", "yolo"] if download_all else (
        (["osnet"] if args.osnet else []) +
        (["pose"] if args.pose else []) +
        (["yolo"] if args.yolo else [])
    )

    print(f"Downloading {len(keys)} weight file(s) to {MODELS_DIR}...\n")
    ok = 0
    for k in keys:
        if download_weight(k, force=args.force):
            ok += 1

    print(f"\nDone: {ok}/{len(keys)} downloaded successfully.")
    if ok < len(keys):
        sys.exit(1)


if __name__ == "__main__":
    main()
