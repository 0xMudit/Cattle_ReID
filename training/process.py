#!/usr/bin/env python3
"""YOLO detection, crop extraction, augmentation, and train/gallery/query split."""
import os
import cv2
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO
from glob import glob
from pathlib import Path
from tqdm import tqdm

from .config import CFG


class Prep:
    def __init__(self, yolo_model, h=None, w=None, letterbox=True):
        self.m = yolo_model
        self.h = h or CFG["h"]
        self.w = w or CFG["w"]
        self.letterbox = letterbox
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
        r = self.m(img, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=int)
        mask = boxes.cls.cpu().numpy() == CFG["cow_cls"]
        if not mask.any():
            return np.empty((0, 4), dtype=int)
        return boxes.xyxy.cpu().numpy()[mask].astype(int)

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


def process_images():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    yolo = YOLO(os.path.join(CFG["proj"], "cattle_osnet", "yolov8n.pt")).to(device)
    print("YOLOv8 loaded")

    prep = Prep(yolo)

    raw = CFG["data_raw"]
    proc = CFG["data_proc"]

    imgs = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        imgs.extend(glob(os.path.join(raw, "**", ext), recursive=True))

    cows = {}
    for p in imgs:
        for part in Path(p).parts:
            if part.isdigit():
                cows.setdefault(int(part), []).append(p)
                break

    print(f"Found {len(cows)} cows with {sum(len(v) for v in cows.values())} images total")

    if len(cows) == 0:
        print("ERROR: No cows found. Extract dataset first: python -m training.download")
        return

    cow_ids = sorted(cows.keys())
    n = len(cow_ids)
    n_train = int(n * 0.7)
    n_gal = int(n * 0.15)

    train_ids = set(cow_ids[:n_train])
    gallery_ids = set(cow_ids[n_train:n_train + n_gal])
    query_ids = set(cow_ids[n_train + n_gal:])

    print(f"Split: {len(train_ids)} train, {len(gallery_ids)} gallery, {len(query_ids)} query identities")

    for d in ["train", "gallery", "query"]:
        os.makedirs(os.path.join(proc, d), exist_ok=True)

    counts = {"train": 0, "gallery": 0, "query": 0}

    for cid, paths in tqdm(cows.items(), desc="Processing cows"):
        if cid in train_ids:
            subset = paths[:CFG["max_train_per_cow"]]
            dest = os.path.join(proc, "train")
            augment = True
        elif cid in gallery_ids:
            subset = paths[:CFG["max_gallery_per_cow"]]
            dest = os.path.join(proc, "gallery")
            augment = False
        elif cid in query_ids:
            subset = paths[:CFG["max_query_per_cow"]]
            dest = os.path.join(proc, "query")
            augment = False
        else:
            continue

        for p in subset:
            im = cv2.imread(p)
            if im is None:
                continue
            bbs = prep.detect(im)
            if len(bbs) == 0:
                continue
            cropped = prep.crop(im, bbs[0])
            images = prep.aug_img(cropped, CFG["aug_n"]) if augment else [cropped]
            for img_aug in images:
                split = os.path.basename(dest)
                counts[split] += 1
                name = f"c0_p{cid}_{counts[split]}.jpg"
                cv2.imwrite(os.path.join(dest, name), img_aug)

    print(f"Created: {counts}")
    for split in ["train", "gallery", "query"]:
        d = os.path.join(proc, split)
        print(f"  {split}: {len(os.listdir(d))} files")


if __name__ == "__main__":
    process_images()
