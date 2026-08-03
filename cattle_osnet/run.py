#!/usr/bin/env python3
"""
Cattle Re-ID — zero-shot OSNet feature extraction and matching.

Pipeline:
  1. Build gallery: gallery/<cow_id>/*.jpg  ->  mean embedding per cow
  2. Embed query images: queries/*.jpg       ->  512-dim vectors
  3. Match: nearest gallery mean by cosine similarity + L2 distance

No training needed — uses OSNet x1.0 pretrained on ImageNet.
Drop your own cow photos into gallery/<cow_id>/ and queries/.
"""

import os
import sys
import glob
import pickle
import argparse
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
from osnet import osnet_x1_0  # noqa: E402

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
WEIGHTS = os.path.join(MODEL_DIR, 'osnet_x1_0_imagenet.pth')
GALLERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gallery')
QUERY_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queries')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
GALLERY_PKL = os.path.join(OUTPUT_DIR, 'gallery.pkl')

IMG_H, IMG_W = 256, 128
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
COS_THRESHOLD = 0.6   # below this, a query is "Unknown"


class Extractor:
    """OSNet x1.0 feature extractor (512-dim, ImageNet-pretrained)."""

    def __init__(self, weights=WEIGHTS, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = osnet_x1_0(num_classes=1000, pretrained=False)
        state = torch.load(weights, map_location=self.device, weights_only=True)
        if 'state_dict' in state:
            state = state['state_dict']
        state = {k[7:] if k.startswith('module.') else k: v for k, v in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()

    def embed(self, img_bgr):
        """BGR numpy image -> 512-dim normalized embedding."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMG_W, IMG_H))
        x = rgb.astype(np.float32) / 255.0
        for c in range(3):
            x[..., c] = (x[..., c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            v = self.model(x)
        v = v.cpu().numpy().flatten().astype(np.float64)
        return v / (np.linalg.norm(v) + 1e-12)


def build_gallery(ext, gallery_dir):
    """Each subfolder of gallery_dir is one cow identity."""
    gal = {}
    for cow_id in sorted(os.listdir(gallery_dir)):
        d = os.path.join(gallery_dir, cow_id)
        if not os.path.isdir(d):
            continue
        paths = []
        for e in ('*.jpg', '*.jpeg', '*.png'):
            paths += glob.glob(os.path.join(d, e))
        if not paths:
            continue
        embs = np.stack([ext.embed(cv2.imread(p)) for p in paths if cv2.imread(p) is not None])
        gal[cow_id] = {
            'mean': embs.mean(axis=0),
            'embs': embs,
            'n': len(embs),
            'paths': paths,
        }
        print(f'  gallery: {cow_id:20s} {len(embs)} images')
    return gal


def match(query_emb, gal):
    """Return list of (cow_id, cosine, l2) sorted best first."""
    scores = []
    for cow_id, g in gal.items():
        cos = float(np.dot(query_emb, g['mean']))
        l2 = float(np.sqrt(np.mean((query_emb - g['mean']) ** 2)))
        scores.append((cow_id, cos, l2))
    scores.sort(key=lambda t: -t[1])
    return scores


def main():
    ap = argparse.ArgumentParser(description='Zero-shot cattle Re-ID with OSNet')
    ap.add_argument('--image', '-i', default=None,
                    help='single query image; otherwise scans queries/')
    ap.add_argument('--gallery', default=GALLERY_DIR, help='gallery folder')
    ap.add_argument('--threshold', type=float, default=COS_THRESHOLD,
                    help='cosine threshold for Known/Unknown')
    ap.add_argument('--rebuild', action='store_true',
                    help='force gallery rebuild even if gallery.pkl exists')
    ap.add_argument('--topk', type=int, default=3)
    args = ap.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ext = Extractor()

    gal = None
    if os.path.exists(GALLERY_PKL) and not args.rebuild:
        with open(GALLERY_PKL, 'rb') as f:
            gal = pickle.load(f)
        print(f'Loaded gallery from {GALLERY_PKL} ({len(gal)} cows)')
    if gal is None:
        print('Building gallery...')
        gal = build_gallery(ext, args.gallery)
        with open(GALLERY_PKL, 'wb') as f:
            pickle.dump(gal, f)
        print(f'Saved gallery -> {GALLERY_PKL}')

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f'Cannot read {args.image}')
            sys.exit(1)
        q = [args.image]
    else:
        q = []
        for e in ('*.jpg', '*.jpeg', '*.png'):
            q += glob.glob(os.path.join(QUERY_DIR, e))
        q = sorted(q)

    if not q:
        print('No query images found. Put images in queries/ or pass --image.')
        return
    if not gal:
        print('Gallery empty. Put photos in gallery/<cow_id>/ first.')
        return

    print(f'Processing {len(q)} query image(s)...\n')
    for path in q:
        emb = ext.embed(cv2.imread(path))
        scores = match(emb, gal)
        top = scores[0]
        label = top[0] if top[1] >= args.threshold else 'Unknown'
        print(f'  {os.path.basename(path):35s} -> {label:15s} '
              f'(cos={top[1]:.3f} l2={top[2]:.3f})')
        for cow_id, cos, l2 in scores[:args.topk]:
            print(f'      #{cow_id:15s} cos={cos:.3f} l2={l2:.3f}')
        print()


if __name__ == '__main__':
    main()
