#!/usr/bin/env python3
"""
Annotate frames with cow skeletons + head tags (no bounding boxes).

Improvements over v1:
  - Enlarged adaptive crops (25% margin, imgsz 640/1280 for far cows)
  - Multi-scale pose ensemble: per-crop pass + one full-frame pass merged
  - Lower keypoint threshold (0.10)
  - Post-processing: spine/body interpolation + hoof snapping, hollow = estimated
  - --smooth: temporal EMA per cow (consistent tags + stable skeletons over a
    frame sequence)

Usage:
    python annotate.py [--src DIR] [--out DIR] [--conf 0.15] [--smooth]
"""

import os
import sys
import glob
import argparse
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE, 'models')
DET_MODEL = os.path.join(BASE, 'yolov8n.pt')
POSE_MODEL = os.path.join(MODEL_DIR, 'cow_pose.pt')
DEFAULT_SRC = os.path.join(BASE, 'output', 'frames')
DEFAULT_OUT = os.path.join(BASE, 'output', 'annotated')

CONF = 0.15
KPT_CONF = 0.10
COW_CLS = 19

KP_NAMES = ["Nose", "R_Eye", "L_Eye", "Neck", "LF_Hoof", "RF_Hoof",
            "LB_Hoof", "RB_Hoof", "Backbone", "TailRoot", "BackPose", "Stomach"]
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (3, 8), (8, 9),
    (3, 4), (3, 5),
    (9, 6), (9, 7),
    (8, 10), (10, 11),
]
SPINE_CHAINS = [[3, 8, 9], [8, 10, 11], [3, 8, 10]]
HOOVES = {4: 0.25, 5: 0.75, 6: 0.25, 7: 0.75}

SEGOE_UI = r"C:\Windows\Fonts\segoeui.ttf"
SEGOE_UI_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"

PALETTE = [
    (255, 90, 90), (72, 205, 184), (255, 201, 60), (120, 220, 120),
    (150, 110, 230), (255, 150, 100), (80, 170, 255), (230, 110, 200),
]


def font(size, bold=True):
    return ImageFont.truetype(SEGOE_UI_BOLD if bold else SEGOE_UI, size)


def detect_cows(det_model, img, conf):
    r = det_model(img, verbose=False, conf=conf)[0]
    cows = []
    if r.boxes is not None:
        confs = r.boxes.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        for ci, (c, b) in enumerate(zip(confs, boxes)):
            if r.names[cls[ci]] == 'cow':
                cows.append([tuple(int(v) for v in b), None, float(c), None])
    return cows


def crop_pose_kpts(pose_model, img, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mw, mh = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
    cx1, cy1 = max(0, int(x1 - mw)), max(0, int(y1 - mh))
    cx2, cy2 = min(w, int(x2 + mw)), min(h, int(y2 + mh))
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    imgsz = 1280 if (y2 - y1) < 200 else 640
    r = pose_model(crop, verbose=False, conf=0.15, imgsz=imgsz)[0]
    if r.keypoints is None or len(r.keypoints.data) == 0:
        return None
    kds = r.keypoints.data.cpu().numpy()
    if r.boxes is not None and len(r.boxes.conf) > 0:
        confs = r.boxes.conf.cpu().numpy()
    else:
        confs = np.ones(len(kds))
    kp = kds[int(np.argmax(confs))]
    kp = kp.copy()
    kp[:, 0] = kp[:, 0] * (cx2 - cx1) / r.orig_shape[1] + cx1
    kp[:, 1] = kp[:, 1] * (cy2 - cy1) / r.orig_shape[0] + cy1
    return kp


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0


def full_pose_kpts(pose_model, img, cows):
    if not cows:
        return [None] * len(cows)
    r = pose_model(img, verbose=False, conf=0.25, imgsz=640)[0]
    out = [None] * len(cows)
    if r.keypoints is None or len(r.keypoints.data) == 0:
        return out
    dets = []
    if r.boxes is not None:
        for b, kd in zip(r.boxes.xyxy.cpu().numpy(),
                          r.keypoints.data.cpu().numpy()):
            if kd.shape[0] == 12:
                dets.append((b, kd))
    for ci, (box, _kpt, _cf, _) in enumerate(cows):
        best, best_iou = None, 0.3
        for b, kd in dets:
            iou = _iou(box, b)
            if iou >= best_iou:
                best, best_iou = kd.copy(), iou
        out[ci] = best
    return out


FULL_KPT_BAR = 0.30


def ensemble_kpts(crop_kp, full_kp):
    if crop_kp is None or crop_kp.shape != (12, 3):
        crop_kp = None
    if full_kp is None or full_kp.shape != (12, 3):
        full_kp = None
    if crop_kp is None:
        return full_kp if full_kp is not None else None
    if full_kp is None:
        return crop_kp
    out = crop_kp.copy()
    for j in range(12):
        cv_, fv = crop_kp[j, 2], full_kp[j, 2]
        if cv_ >= KPT_CONF and fv >= FULL_KPT_BAR:
            d = float(np.hypot(crop_kp[j, 0] - full_kp[j, 0],
                               crop_kp[j, 1] - full_kp[j, 1]))
            if d < 60:
                wc = cv_ / (cv_ + fv)
                out[j, :2] = wc * crop_kp[j, :2] + (1 - wc) * full_kp[j, :2]
                out[j, 2] = max(cv_, fv)
            else:
                out[j] = crop_kp[j] if cv_ >= fv else full_kp[j]
        elif cv_ >= KPT_CONF:
            out[j] = crop_kp[j]
        elif fv >= FULL_KPT_BAR:
            out[j] = full_kp[j]
        else:
            out[j, 2] = 0
    return out


def postprocess_kpts(kpts, box):
    interp = np.zeros(12, bool)
    if kpts is None:
        return None, interp
    k = kpts.copy()
    x1, y1, x2, y2 = box
    H = max(1, y2 - y1)
    vis = k[:, 2] >= KPT_CONF

    for chain in SPINE_CHAINS:
        for j in chain:
            if not vis[j]:
                nbrs = [n for n in chain if n != j and vis[n]]
                if len(nbrs) >= 2:
                    k[j, 0] = float(np.mean(k[nbrs, 0]))
                    k[j, 1] = float(np.mean(k[nbrs, 1]))
                    k[j, 2] = 0.12
                    interp[j] = True

    for h_, fx in HOOVES.items():
        if vis[h_] and k[h_, 1] < y2 - 0.5 * H:
            k[h_, 1] = y2 - 0.03 * H
            k[h_, 2] = 0.12
            interp[h_] = True
        elif not vis[h_]:
            k[h_, 0] = x1 + fx * (x2 - x1)
            k[h_, 1] = y2 - 0.03 * H
            k[h_, 2] = 0.12
            interp[h_] = True

    return k, interp


class TrackSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.tracks = {}
        self.next_id = 0

    def step(self, cows):
        for cow in cows:
            box = cow[0]
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            w, h = box[2] - box[0], box[3] - box[1]
            best_id, best_d = None, None
            for tid, t in self.tracks.items():
                d = np.hypot(t['cx'] - cx, t['cy'] - cy)
                if d < 0.5 * max(t['w'], t['h']) and (best_d is None or d < best_d):
                    best_id, best_d = tid, d
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
                self.tracks[best_id] = {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'kpts': None}
            t = self.tracks[best_id]
            t['cx'], t['cy'] = 0.7 * t['cx'] + 0.3 * cx, 0.7 * t['cy'] + 0.3 * cy
            t['w'], t['h'] = w, h
            kpts = cow[1]
            if kpts is not None and kpts.shape == (12, 3):
                if t['kpts'] is None:
                    t['kpts'] = kpts.copy()
                else:
                    mix = kpts[:, 2] >= 0.5
                    t['kpts'][mix] = (self.alpha * t['kpts'][mix] +
                                      (1 - self.alpha) * kpts[mix])
                cow[1] = t['kpts'].copy()
            cow[3] = best_id


def confidence_bar(dr, x, y, w, frac, color, h=7, radius=3):
    dr.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=(0, 0, 0, 120))
    fw = max(6, int(w * frac))
    dr.rounded_rectangle([x, y, x + fw, y + h], radius=radius, fill=color + (255,))


def head_anchor(kpts, vis, box):
    pts = [kpts[i] for i in (0, 1, 2) if vis[i]]
    if pts:
        return int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))
    if vis[3]:
        return int(kpts[3][0]), int(kpts[3][1])
    return int(box[0]) + int((box[2] - box[0]) / 2), int(box[1])


def annotate(img_bgr, cows, frame_name, track_id):
    h, w = img_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)

    dr.rectangle([0, 0, w, h], fill=(0, 0, 0, 18))

    f_main = font(30)
    f_sub = font(20, bold=False)
    f_head = font(34)

    for i, cow in enumerate(cows):
        box, kpts, cf, tid = cow
        col = PALETTE[i % len(PALETTE)]
        if kpts is not None and kpts.shape[0] == 12:
            kpts, interp = postprocess_kpts(kpts, box)
            vis = kpts[:, 2] >= KPT_CONF
        else:
            kpts, interp, vis = None, np.zeros(12, bool), [False] * 12

        if kpts is not None:
            for a, b in SKELETON:
                if vis[a] and vis[b]:
                    est = interp[a] or interp[b]
                    axy = tuple(int(v) for v in kpts[a, :2])
                    bxy = tuple(int(v) for v in kpts[b, :2])
                    dr.line([axy, bxy], fill=col + ((255,) if not est else (130,)),
                            width=5 if not est else 3, joint="curve")
                    dr.line([axy, bxy], fill=(255, 255, 255, 200), width=2, joint="curve")

            for j in range(12):
                if vis[j]:
                    x, y = int(kpts[j, 0]), int(kpts[j, 1])
                    if interp[j]:
                        dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=col + (90,),
                                   outline=(255, 255, 255, 255), width=2)
                    else:
                        dr.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 255, 255, 255))
                        dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col + (255,))

        anc = head_anchor(kpts, vis, box) if kpts is not None else None
        if anc is None:
            anc = (int(box[0]) + int((box[2] - box[0]) / 2), int(box[1]))
        ax, ay = anc

        label = f"Cow_{i+1:02d}"
        if tid is not None:
            label = f"Cow_{tid:02d}"
        tb = dr.textbbox((0, 0), label, font=f_main)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 16, 10
        pill_w = tw + pad_x * 2
        pill_h = th + pad_y * 2 + 16
        px1 = ax - pill_w // 2
        py1 = ay - pill_h - 16
        px1 = max(6, min(px1, w - pill_w - 6))
        px2, py2 = px1 + pill_w, py1 + pill_h

        lx = max(6, min(ax, w - 6))
        dr.line([(ax, ay - 4), (lx, py2)], fill=col + (200,), width=3)
        dr.line([(lx - 6, py2 + 8), (lx, py2), (lx + 6, py2 + 8)],
                fill=col + (255,), width=3, joint="curve")

        dr.rounded_rectangle([px1, py1, px2, py2], radius=14,
                             fill=(12, 16, 20, 215), outline=col + (255,), width=2)
        bx1, by1 = px1 + 8, py1 + (pill_h - 30) // 2
        dr.ellipse([bx1, by1, bx1 + 30, by1 + 30], fill=col + (255,))
        nb = dr.textbbox((0, 0), str(i + 1), font=font(20))
        dr.text((bx1 + 15 - (nb[2] - nb[0]) / 2, by1 + 15 - (nb[3] - nb[1]) / 2 - 2),
                str(i + 1), font=font(20), fill=(255, 255, 255, 255))
        tx, ty = bx1 + 42, py1 + 12
        dr.text((tx + 1, ty + 1), label, font=f_main, fill=(0, 0, 0, 140))
        dr.text((tx, ty), label, font=f_main, fill=(255, 255, 255, 255))
        confidence_bar(dr, tx, ty + th + 6, pill_w - (tx - px1) - 8, cf, col)

    n = len(cows)
    title = "CATTLE RE-ID"
    sub = (f"{n} cow{'s' if n != 1 else ''}  |  skeleton + head tag  |  "
           f"{frame_name}  |  {'track ' + str(track_id) if track_id is not None else ''}")
    tb = dr.textbbox((0, 0), title, font=f_head)
    sb = dr.textbbox((0, 0), sub, font=f_sub)
    panel_w = max(tb[2] - tb[0], sb[2] - sb[0]) + 48
    px0 = 20
    dr.rounded_rectangle([px0, 20, px0 + panel_w, 112], radius=18,
                         fill=(12, 16, 20, 175), outline=(255, 255, 255, 60), width=1)
    dr.text((px0 + 24, 36), title, font=f_head, fill=(255, 255, 255, 255))
    dr.text((px0 + 24, 78), sub, font=f_sub, fill=(200, 210, 220, 255))
    dr.rounded_rectangle([px0 + 24, 90, px0 + 24 + int((tb[2] - tb[0] + 20) * 0.4), 92],
                         radius=1, fill=(72, 205, 184, 255))

    merged = Image.alpha_composite(base, ov)
    return cv2.cvtColor(np.array(merged.convert("RGB")), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description='Cow skeleton + head tag annotator')
    ap.add_argument('--src', default=DEFAULT_SRC, help='folder with input images')
    ap.add_argument('--out', default=DEFAULT_OUT, help='output folder')
    ap.add_argument('--conf', type=float, default=CONF, help='detection confidence')
    ap.add_argument('--smooth', action='store_true',
                    help='temporal EMA + consistent tags over a frame sequence')
    args = ap.parse_args()

    det_model = YOLO(DET_MODEL)
    pose_model = YOLO(POSE_MODEL)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, '*.jpg')) +
                   glob.glob(os.path.join(args.src, '*.png')))
    smoother = TrackSmoother() if args.smooth else None

    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        cows = detect_cows(det_model, img, args.conf)
        full = full_pose_kpts(pose_model, img, cows)
        for cow, fk in zip(cows, full):
            ck = crop_pose_kpts(pose_model, img, cow[0])
            cow[1] = ensemble_kpts(ck, fk)
        if smoother is not None:
            smoother.step(cows)
        name = os.path.splitext(os.path.basename(p))[0]
        tid_str = None
        if smoother is not None:
            ids = sorted({c[3] for c in cows if c[3] is not None})
            if ids:
                tid_str = 'tracks ' + ','.join(str(i) for i in ids)
        vis = annotate(img, cows, name, tid_str)
        out = os.path.join(args.out, name + '_tagged.jpg')
        cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        dets = [c[1] for c in cows]
        tids = [c[3] for c in cows]
        print(f'{name}: {len(cows)} cows tids={tids} -> {os.path.basename(out)}')


if __name__ == '__main__':
    sys.exit(main())
