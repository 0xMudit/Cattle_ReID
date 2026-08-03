#!/usr/bin/env python3
"""
Annotate frames with cow skeletons + head tags (no bounding boxes).

Pipeline:
  1. Detect cows with YOLOv8n (COCO class 21) on each input image.
  2. Run the cow-pose model (models/cow_pose.pt, 12 keypoints) on each crop.
  3. Draw a color-coded skeleton and a tag pill over the cow's head.

Usage:
    python annotate.py [--src DIR] [--out DIR] [--conf 0.15]
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
KPT_CONF = 0.30
COW_CLS = 21

# 12 cow keypoints: 0 Nose 1 R.Eye 2 L.Eye 3 Neck 4 LFHoof 5 RFHoof 6 LBHoof 7 RBHoof 8 Backbone 9 TailRoot 10 BackPose 11 Stomach
KP_NAMES = ["Nose", "R_Eye", "L_Eye", "Neck", "LF_Hoof", "RF_Hoof",
            "LB_Hoof", "RB_Hoof", "Backbone", "TailRoot", "BackPose", "Stomach"]
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 3),       # head
    (3, 8), (8, 9),                        # spine
    (3, 4), (3, 5),                        # front legs
    (9, 6), (9, 7),                        # back legs
    (8, 10), (10, 11),                     # body underside
]

SEGOE_UI = r"C:\Windows\Fonts\segoeui.ttf"
SEGOE_UI_BOLD = r"C:\Windows\Fonts\segoeuib.ttf"

PALETTE = [
    (255, 90, 90), (72, 205, 184), (255, 201, 60), (120, 220, 120),
    (150, 110, 230), (255, 150, 100), (80, 170, 255), (230, 110, 200),
]


def font(size, bold=True):
    return ImageFont.truetype(SEGOE_UI_BOLD if bold else SEGOE_UI, size)


def cow_skeletons(pose_model, img, boxes):
    """Run the pose model per detected cow box; return keypoints mapped to full image."""
    h, w = img.shape[:2]
    results = []
    for (x1, y1, x2, y2) in boxes:
        m = 12
        cx1 = max(0, int(x1 - m)); cy1 = max(0, int(y1 - m))
        cx2 = min(w, int(x2 + m)); cy2 = min(h, int(y2 + m))
        crop = img[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        r = pose_model(crop, verbose=False, conf=0.05, imgsz=640)[0]
        if r.keypoints is None or len(r.keypoints.data) == 0:
            continue
        kp = r.keypoints.data.cpu().numpy()[0]
        kp[:, 0] = kp[:, 0] * (cx2 - cx1) / r.orig_shape[1] + cx1
        kp[:, 1] = kp[:, 1] * (cy2 - cy1) / r.orig_shape[0] + cy1
        results.append(kp)
    return results


def confidence_bar(dr, x, y, w, frac, color, h=7, radius=3):
    dr.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=(0, 0, 0, 120))
    fw = max(6, int(w * frac))
    dr.rounded_rectangle([x, y, x + fw, y + h], radius=radius, fill=color + (255,))


def head_anchor(kpts, vis):
    pts = [kpts[i] for i in (0, 1, 2) if vis[i]]
    if pts:
        return int(np.mean([p[0] for p in pts])), int(np.mean([p[1] for p in pts]))
    if vis[3]:
        return int(kpts[3][0]), int(kpts[3][1])
    return None


def annotate(img_bgr, cows, frame_name):
    h, w = img_bgr.shape[:2]
    base = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
    ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    dr = ImageDraw.Draw(ov)

    dr.rectangle([0, 0, w, h], fill=(0, 0, 0, 18))

    f_main = font(30)
    f_sub = font(20, bold=False)
    f_head = font(34)

    for i, cow in enumerate(cows):
        box, kpts, cf = cow
        col = PALETTE[i % len(PALETTE)]
        if kpts is not None and kpts.shape[0] == 12:
            vis = [kpts[j, 2] > KPT_CONF for j in range(12)]
        else:
            vis = [False] * 12

        for a, b in SKELETON:
            if vis[a] and vis[b]:
                axy = tuple(int(v) for v in kpts[a, :2])
                bxy = tuple(int(v) for v in kpts[b, :2])
                dr.line([axy, bxy], fill=col + (255,), width=5, joint="curve")
                dr.line([axy, bxy], fill=(255, 255, 255, 200), width=2, joint="curve")

        for j in range(12):
            if vis[j]:
                x, y = int(kpts[j, 0]), int(kpts[j, 1])
                dr.ellipse([x - 9, y - 9, x + 9, y + 9], fill=(255, 255, 255, 255))
                dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col + (255,))

        anc = head_anchor(kpts, vis)
        if anc is None:
            anc = (int(box[0]) + int((box[2] - box[0]) / 2), int(box[1]))
        ax, ay = anc

        label = f"Cow_{i+1:02d}"
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
    sub = f"{n} cow{'s' if n != 1 else ''} detected  |  skeleton + head tag  |  {frame_name}"
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
    args = ap.parse_args()

    det_model = YOLO(DET_MODEL)
    pose_model = YOLO(POSE_MODEL)
    os.makedirs(args.out, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.src, '*.jpg')) +
                   glob.glob(os.path.join(args.src, '*.png')))
    for p in files:
        img = cv2.imread(p)
        if img is None:
            continue
        r = det_model(img, verbose=False, conf=args.conf)[0]
        cows = []
        if r.boxes is not None:
            confs = r.boxes.conf.cpu().numpy()
            boxes = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            for ci, (conf, box) in enumerate(zip(confs, boxes)):
                if r.names[cls[ci]] != 'cow':
                    continue
                cows.append([box, None, float(conf)])
        kpt_list = cow_skeletons(pose_model, img, [c[0] for c in cows])
        for c, kp in zip(cows, kpt_list):
            c[1] = kp
        name = os.path.splitext(os.path.basename(p))[0]
        vis = annotate(img, cows, name)
        out = os.path.join(args.out, name + '_tagged.jpg')
        cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f'{name}: {len(cows)} cows -> {os.path.basename(out)}')


if __name__ == '__main__':
    sys.exit(main())
