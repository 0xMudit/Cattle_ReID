#!/usr/bin/env python3
"""
Extract cow crops from videos: sample frames, YOLO-detect cows, track each
cow across sampled frames, and save crops per cow identity.

Usage:
    python prep_videos.py --videos "C:/.../Dataset" --out output/vidcrops
"""

import os
import sys
import json
import glob
import argparse
import cv2
import numpy as np

COW_CLS = 19
DET_CONF = 0.25


def expand_box(x1, y1, x2, y2, margin=0.05):
    w, h = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - margin * w))
    y1 = max(0, int(y1 - margin * h))
    x2 = int(x2 + margin * w)
    y2 = int(y2 + margin * h)
    return x1, y1, x2, y2


class Tracker:
    """Assigns a stable id to each cow across sampled frames (centroid matching)."""

    def __init__(self, match_scale=0.5):
        self.match_scale = match_scale
        self.tracks = {}   # id -> dict(centroid, size, crops=[frame_idx])
        self.next_id = 0

    def step(self, boxes, frame_idx):
        """boxes: list of (x1,y1,x2,y2). Returns list of (track_id, crop_box)."""
        out = []
        for (x1, y1, x2, y2) in boxes:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            w, h = x2 - x1, y2 - y1
            best_id, best_d = None, None
            for tid, t in self.tracks.items():
                d = np.hypot(t['cx'] - cx, t['cy'] - cy)
                thr = self.match_scale * max(t['w'], t['h'])
                if d < thr and (best_d is None or d < best_d):
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'crops': []}
            t = self.tracks[best_id]
            t['cx'], t['cy'] = 0.7 * t['cx'] + 0.3 * cx, 0.7 * t['cy'] + 0.3 * cy
            t['w'], t['h'] = w, h
            t['crops'].append(frame_idx)
            out.append((best_id, (x1, y1, x2, y2)))
        return out


def detect_cows(yolo, img_bgr, conf=0.25):
    r = yolo(img_bgr, verbose=False)[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        return []
    mask = boxes.cls.cpu().numpy() == COW_CLS
    confs = boxes.conf.cpu().numpy()
    xyxy = boxes.xyxy.cpu().numpy()
    return [tuple(int(v) for v in b) for b, c, m in zip(xyxy, confs, mask) if m and c >= conf]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--videos', required=True, help='folder with .mp4 files')
    ap.add_argument('--out', default='output/vidcrops', help='output folder')
    ap.add_argument('--step', type=int, default=50, help='sample every N frames (2s @25fps)')
    ap.add_argument('--conf', type=float, default=DET_CONF)
    ap.add_argument('--max-frames', type=int, default=0, help='limit frames per video (0=all)')
    args = ap.parse_args()

    from ultralytics import YOLO
    yolo = YOLO('yolov8n.pt')

    os.makedirs(args.out, exist_ok=True)
    meta = {}

    for vp in sorted(glob.glob(os.path.join(args.videos, '*.mp4'))):
        vname = os.path.splitext(os.path.basename(vp))[0]
        cap = cv2.VideoCapture(vp)
        tracker = Tracker()
        frame_idx = 0
        crop_count = 0
        sample_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % args.step != 0:
                frame_idx += 1
                continue
            boxes = detect_cows(yolo, frame, conf=args.conf)
            if boxes:
                hits = tracker.step(boxes, frame_idx)
                for tid, box in hits:
                    x1, y1, x2, y2 = expand_box(*box)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue
                    d = os.path.join(args.out, f'{vname}_cow_{tid}')
                    os.makedirs(d, exist_ok=True)
                    cv2.imwrite(os.path.join(d, f'f{frame_idx:06d}.jpg'), crop)
                    crop_count += 1
            sample_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            frame_idx += 1
        cap.release()

        keep = {tid: {'frames': t['crops']} for tid, t in tracker.tracks.items() if len(t['crops']) >= 2}
        meta[vname] = {'total_samples': sample_idx, 'tracks': keep}
        print(f'{vname}: {sample_idx} sampled frames, {crop_count} crops, '
              f'{len(tracker.tracks)} raw tracks -> {len(keep)} tracks with >=2 crops')

    with open(os.path.join(args.out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('Saved meta ->', os.path.join(args.out, 'meta.json'))


if __name__ == '__main__':
    main()
