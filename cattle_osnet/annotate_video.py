#!/usr/bin/env python3
"""
Annotate a source video with cow boxes, stable track tags, and best-effort
skeletons, then render an H.264 mp4 — the paper's Fig.5/Fig.12 style on video.

Designed to run on a weak CPU (6 GB RAM): it processes at a tunable resolution,
inference size, and frame-sample rate. For full 2.7K output use Kaggle.

Usage:
    python annotate_video.py --src ../Dataset/ch07m_...mp4 --out annotated_ch07.mp4
    python annotate_video.py --src <vid> --start 60 --dur 30 --fps 5 --scale 1280
    python annotate_video.py --src <vid> --out out.mp4 --max-cows 12 --no-pose

Options:
    --src PATH       input video (default first mp4 in ../Dataset)
    --out PATH       output mp4 (default <src>_annotated.mp4)
    --start SEC      start offset (default 0)
    --dur SEC        seconds to process (default: whole video)
    --fps N          process every Nth frame (output fps = src_fps/N) (default 1)
    --scale N        resize long edge to N px before detection (default 1600)
    --imgsz N        YOLO inference size (default 1280)
    --conf F         detection confidence (default 0.15)
    --max-cows N     keep top N detections per frame (default 30)
    --track / --no-track        enable/disable stable track IDs (default on)
    --pose  / --no-pose         enable/disable skeleton overlay (default on)
    --outfps N       override output fps (default src_fps / fps)
"""

import os
import sys
import time
import glob
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
DET_MODEL = os.path.join(BASE, 'yolov8n.pt')
POSE_MODEL = os.path.join(BASE, 'models', 'cow_pose.pt')
DEFAULT_SRC = glob.glob(os.path.join(BASE, '..', 'Dataset', '*.mp4'))
DEFAULT_SRC = DEFAULT_SRC[0] if DEFAULT_SRC else None

CONF = 0.15
KP_NAMES = ["Nose", "R_Eye", "L_Eye", "Neck", "LF_Hoof", "RF_Hoof",
            "LB_Hoof", "RB_Hoof", "Backbone", "TailRoot", "BackPose", "Stomach"]
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (3, 8), (8, 9),
    (3, 4), (3, 5),
    (9, 6), (9, 7),
    (8, 10), (10, 11),
]
SEGOE_UI = r"C:\Windows\Fonts\segoeui.ttf"
SEGOE_UI_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"


def _font(size, bold=True):
    path = SEGOE_UI_BOLD if bold else SEGOE_UI
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


class Tracker:
    """Greedy IoU tracker giving stable cow IDs across sampled frames."""

    def __init__(self, iou_thr=0.25, expire_frames=180):
        self.iou_thr = iou_thr
        self.expire_frames = expire_frames
        self.tracks = {}
        self.next_id = 1
        self.fidx = 0

    @staticmethod
    def iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, boxes):
        self.fidx += 1
        assign = []
        used = set()
        # greedy: best-scoring track first
        for tid, rec in sorted(self.tracks.items(), key=lambda kv: -kv[1][1]):
            if tid in used:
                continue
            box = rec[0]
            best_i, best_iou = None, 0.0
            for i, b in enumerate(boxes):
                if i in used:
                    continue
                v = self.iou(box, b)
                if v > best_iou:
                    best_iou, best_i = v, i
            if best_i is not None and best_iou >= self.iou_thr:
                used.add(best_i)
                self.tracks[tid] = (boxes[best_i], 0.0, self.fidx)
                assign.append((tid, best_i))
        for i, b in enumerate(boxes):
            if i not in used:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = (b, 0.0, self.fidx)
                assign.append((tid, i))
        for tid in list(self.tracks):
            if self.fidx - self.tracks[tid][2] > self.expire_frames:
                del self.tracks[tid]
        return assign


def run_pose(pose_model, img, box, margin=0.3):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    mw, mh = margin * (x2 - x1), margin * (y2 - y1)
    cx1, cy1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
    cx2, cy2 = min(w, int(x2 + mw)), min(h, int(y2 + mh))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    r = pose_model(crop, verbose=False, conf=0.25, imgsz=640)[0]
    if r.keypoints is None or len(r.keypoints) == 0:
        return None
    kp = r.keypoints.data.cpu().numpy()
    kp = kp[np.argmax((kp[:, :, 2] > 0.25).sum(axis=1))]  # best instance
    vis = (kp[:, 2] > 0.25)
    if vis.sum() < 4:
        return None
    kp_img = kp.copy()
    kp_img[:, 0] += cx1
    kp_img[:, 1] += cy1
    return kp_img, vis


def annotate_frame(frame, det_model, pose_model, tracker, args, frame_idx):
    fh, fw = frame.shape[:2]
    scale = max(1, int(fw / args.scale) if fw > args.scale else 1)
    if scale > 1:
        small = cv2.resize(frame, (fw // scale, fh // scale), interpolation=cv2.INTER_AREA)
    else:
        small = frame
    r = det_model(small, verbose=False, conf=args.conf, imgsz=args.imgsz)[0]
    cows = []
    if r.boxes is not None and len(r.boxes):
        b = r.boxes.xyxy.cpu().numpy()
        c = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        names = r.names
        for bb, cc, ci in zip(b, c, cls):
            if names[ci] == 'cow':
                bb = bb / scale  # back to original coords
                cows.append(bb)
        if len(cows) > args.max_cows:
            cows = sorted(cows, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]))[-args.max_cows:]
    boxes = [tuple(int(v) for v in bb) for bb in cows]
    assign = tracker.update(boxes) if args.track else [(i, i) for i in range(len(boxes))]

    # pose only on the largest cows (small/far cows rarely give usable skeletons)
    pose_boxes = []
    if args.pose and boxes:
        area = sorted(boxes, key=lambda bb: (bb[2] - bb[0]) * (bb[3] - bb[1]), reverse=True)
        pose_boxes = set(area[:min(args.pose_max, len(area))])

    pal = [(255, 90, 90), (72, 205, 184), (255, 201, 60), (120, 220, 120),
           (150, 110, 230), (255, 150, 100), (80, 170, 255), (230, 110, 200)]
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil, 'RGBA')
    fs = max(16, fw // 90)
    f_bold = _font(fs, bold=True)

    for tid, bi in assign:
        box = boxes[bi]
        color = pal[tid % len(pal)]
        draw.rectangle(box, outline=color, width=max(2, fw // 800))
        label = f"Cow_{tid:02d}"
        tw, th = draw.textbbox((0, 0), label, font=f_bold)[2:]
        draw.rectangle([box[0], box[1] - th - 6, box[0] + tw + 6, box[1]], fill=color)
        draw.text((box[0] + 3, box[1] - th - 3), label, fill=(20, 20, 20), font=f_bold)
        if args.pose and box in pose_boxes:
            pk = run_pose(pose_model, frame, box)
            if pk is not None:
                kp, vis = pk
                for (a, b) in SKELETON:
                    if vis[a] and vis[b]:
                        draw.line((kp[a][0], kp[a][1], kp[b][0], kp[b][1]),
                                  fill=(255, 255, 255, 255), width=max(2, fs // 8))
                for j in range(len(kp)):
                    if vis[j]:
                        r0 = max(3, fs // 6)
                        draw.ellipse([kp[j][0] - r0, kp[j][1] - r0,
                                      kp[j][0] + r0, kp[j][1] + r0],
                                     fill=(255, 255, 255, 255), outline=(0, 0, 0, 255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC)
    ap.add_argument('--out')
    ap.add_argument('--start', type=float, default=0.0)
    ap.add_argument('--dur', type=float, default=0.0)
    ap.add_argument('--fps', type=int, default=1)
    ap.add_argument('--scale', type=int, default=1600)
    ap.add_argument('--imgsz', type=int, default=1280)
    ap.add_argument('--conf', type=float, default=CONF)
    ap.add_argument('--max-cows', type=int, default=30)
    ap.add_argument('--pose-max', type=int, default=30,
                    help='run skeleton pose on the N largest cows only (default 30)')
    ap.add_argument('--track', dest='track', action='store_true', default=True)
    ap.add_argument('--no-track', dest='track', action='store_false')
    ap.add_argument('--pose', dest='pose', action='store_true', default=True)
    ap.add_argument('--no-pose', dest='pose', action='store_false')
    ap.add_argument('--outfps', type=float, default=0.0)
    args = ap.parse_args()

    if not args.src:
        sys.exit("no --src and no mp4 in ../Dataset")
    out = args.out or os.path.splitext(os.path.basename(args.src))[0] + '_annotated.mp4'

    det_model = YOLO(DET_MODEL)
    pose_model = YOLO(POSE_MODEL) if args.pose else None
    tracker = Tracker()

    cap = cv2.VideoCapture(args.src)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.src}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000)

    dur_frames = int(args.dur * src_fps) if args.dur > 0 else total_frames
    outfps = args.outfps if args.outfps > 0 else src_fps / args.fps

    writer = None
    fh = fw = None
    t0 = time.time()
    fidx = 0
    processed = 0
    while fidx < dur_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % args.fps == 0:
            if writer is None:
                fh, fw = frame.shape[:2]
                scale = max(1, int(fw / args.scale) if fw > args.scale else 1)
                wh = (fw // scale, fh // scale)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out, fourcc, float(outfps), wh)
                if not writer.isOpened():
                    sys.exit(f"cannot write {out}")
            ann = annotate_frame(frame, det_model, pose_model, tracker, args, fidx)
            if ann.shape[1] != wh[0]:
                ann = cv2.resize(ann, wh, interpolation=cv2.INTER_AREA)
            writer.write(ann)
            processed += 1
            if processed % 10 == 0:
                el = time.time() - t0
                print(f"  {processed} frames, {el:.0f}s elapsed, "
                      f"{processed / max(el, 0.01):.2f} frames/s", flush=True)
        fidx += 1
    cap.release()
    if writer:
        writer.release()
        print(f"done: {out} ({processed} frames @ {outfps:.1f} fps, "
              f"{processed / max(time.time() - t0, 0.01):.2f} frames/s)")


if __name__ == '__main__':
    main()
