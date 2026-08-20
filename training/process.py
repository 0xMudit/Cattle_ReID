#!/usr/bin/env python3
"""YOLO detection, crop extraction, augmentation, and train/gallery/query split.

Detection runs on GPU when available (RTX 5080) — pass --device cpu to force CPU.
"""
import os
import random
import argparse
import shutil
import cv2
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO
from glob import glob
from pathlib import Path
from tqdm import tqdm

from .config import CFG


def split_paths(paths, seed=42):
    """Seeded random 60/20/20 split of one identity's image paths.

    Returns (train, gallery, query) lists. Handles small identities:
      n < 3 -> all to train, no eval images.
    Every image lands in exactly one split; no image-level leakage.
    """
    paths = sorted(paths)
    rng = random.Random(seed)
    rng.shuffle(paths)
    n = len(paths)
    if n < 3:
        return paths, [], []
    n_train = max(1, int(n * 0.6))
    n_gal = max(1, int(n * 0.2))
    n_query = n - n_train - n_gal
    if n_query < 1:
        n_query = 1
        n_gal = max(1, n - n_train - n_query)
    return paths[:n_train], paths[n_train:n_train + n_gal], paths[n_train + n_gal:]


class Prep:
    def __init__(self, yolo_model, h=None, w=None, letterbox=True, conf=0.25):
        self.m = yolo_model
        self.h = h or CFG["h"]
        self.w = w or CFG["w"]
        self.letterbox = letterbox
        self.conf = conf
        self.aug = A.Compose([
            A.GaussNoise(var_limit=(10, 80), p=0.33),
            A.Blur(blur_limit=3, p=0.33),
            A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.2),
                                       contrast_limit=(-0.3, 0.2), p=0.3),
            A.CLAHE(clip_limit=4, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, sat=0.3, p=0.33),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.33),
            A.HorizontalFlip(p=0.5),
        ])

    def detect(self, img):
        r = self.m(img, verbose=False, conf=self.conf)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=int)
        keep = [
            bb for bb, ci in zip(boxes.xyxy.cpu().numpy(),
                                 boxes.cls.cpu().numpy().astype(int))
            if ci == CFG["cow_cls"] or r.names[ci] == "cow"
        ]
        if not keep:
            return np.empty((0, 4), dtype=int)
        return np.asarray(keep, dtype=int)

    def crop(self, img, bb):
        x1, y1, x2, y2 = bb
        crop = img[y1:y2, x1:x2]
        if not self.letterbox:
            return cv2.resize(crop, (self.w, self.h))
        h0, w0 = crop.shape[:2]
        if h0 == 0 or w0 == 0:
            return np.zeros((self.h, self.w, 3), dtype=np.uint8)
        scale = min(self.h / h0, self.w / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        pad_w = (self.w - new_w) // 2
        pad_h = (self.h - new_h) // 2
        out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return out

    def aug_img(self, img, n=3):
        imgs = [img]
        for _ in range(n):
            imgs.append(self.aug(image=img)["image"])
        return imgs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="YOLO crop extraction + split")
    p.add_argument("--yolo", type=str,
                   default=str(Path(CFG["proj"]) / "cattle_osnet" / "yolov8n.pt"),
                   help="YOLO weights (GPU inference by default)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"],
                   help="Detection device (auto = GPU when available)")
    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence")
    p.add_argument("--aug-n", type=int, default=CFG["aug_n"],
                   help="Augmented copies per train image")
    p.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True,
                   help="Wipe output splits before processing (prevents leakage)")
    p.add_argument("--seed", type=int, default=42, help="Split shuffle seed")
    return p.parse_args(argv)


def process_images(clean=True, yolo_path=None, device="auto", conf=0.25,
                   aug_n=None, seed=42):
    """Process raw images into train/gallery/query splits.

    Args:
        clean: If True, wipe output directories before processing.
               This prevents identity leakage from accumulated files
               across multiple runs.
        yolo_path: Path to YOLO weights (defaults to cattle_osnet/yolov8n.pt).
        device: "auto" (GPU when available), "cuda" or "cpu".
        conf: YOLO detection confidence threshold.
        aug_n: Augmented copies per train image.
        seed: Random seed for the per-identity split shuffle.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo = YOLO(yolo_path or os.path.join(CFG["proj"], "cattle_osnet", "yolov8n.pt"))
    yolo.to(device)
    print(f"YOLO loaded on {device}")

    prep = Prep(yolo, conf=conf)
    aug_n = aug_n if aug_n is not None else CFG["aug_n"]

    raw = CFG["data_raw"]
    proc = CFG["data_proc"]

    if clean:
        for d in ["train", "gallery", "query"]:
            p = os.path.join(proc, d)
            if os.path.isdir(p):
                shutil.rmtree(p)
                print(f"  [clean] removed {p}")
            os.makedirs(p, exist_ok=True)
        print("  Output directories cleaned")
    else:
        for d in ["train", "gallery", "query"]:
            os.makedirs(os.path.join(proc, d), exist_ok=True)

    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(glob(os.path.join(raw, "**", ext), recursive=True))

    cows = {}
    for p in imgs:
        parent = Path(p).parent.name
        if parent.isdigit():
            cows.setdefault(int(parent), []).append(p)
        else:
            cows.setdefault(parent, []).append(p)

    print(f"Found {len(cows)} cows with {sum(len(v) for v in cows.values())} images total")

    if len(cows) == 0:
        print("ERROR: No cows found. Extract dataset first: python -m training.download")
        return

    cow_ids = sorted(cows.keys())
    label_map = {cid: i for i, cid in enumerate(cow_ids)}

    print(f"Splitting {len(cow_ids)} identities: 60% train / 20% gallery / 20% query "
          f"per identity (seed={seed})")

    counts = {"train": 0, "gallery": 0, "query": 0}
    id_splits = {"train": set(), "gallery": set(), "query": set()}

    for cid, paths in tqdm(cows.items(), desc="Processing cows"):
        train_imgs, gal_imgs, query_imgs = split_paths(paths, seed)

        pid = label_map[cid]
        for img_list, split_name, augment in [
            (train_imgs, "train", True),
            (gal_imgs, "gallery", False),
            (query_imgs, "query", False),
        ]:
            if img_list:
                id_splits[split_name].add(pid)
            dest = os.path.join(proc, split_name)
            for p in img_list:
                im = cv2.imread(p)
                if im is None:
                    continue
                bbs = prep.detect(im)
                if len(bbs) == 0:
                    continue
                for bb in bbs:
                    cropped = prep.crop(im, bb)
                    images = prep.aug_img(cropped, aug_n) if augment else [cropped]
                    for img_aug in images:
                        counts[split_name] += 1
                        name = f"c0_p{pid}_{counts[split_name]}.jpg"
                        cv2.imwrite(os.path.join(dest, name), img_aug)

    train_ids = id_splits["train"]
    gallery_ids = id_splits["gallery"]
    query_ids = id_splits["query"]

    print(f"\n{'='*60}")
    print(f"  IDENTITY SPLIT VERIFICATION")
    print(f"{'='*60}")
    print(f"  Train identities:    {len(train_ids)}")
    print(f"  Gallery identities:  {len(gallery_ids)}")
    print(f"  Query identities:    {len(query_ids)}")
    print(f"  Note: same identities in all splits is normal for closed-set ReID.")
    print(f"  What matters: no individual IMAGE appears in multiple splits.")

    print(f"\n  Created: {counts}")
    for split in ["train", "gallery", "query"]:
        d = os.path.join(proc, split)
        n_files = len([f for f in os.listdir(d) if f.endswith('.jpg')])
        print(f"  {split}: {n_files} files, {len(id_splits[split])} identities")


if __name__ == "__main__":
    args = parse_args()
    process_images(clean=args.clean, yolo_path=args.yolo, device=args.device,
                   conf=args.conf, aug_n=args.aug_n, seed=args.seed)