#!/usr/bin/env python3
"""
Video ReID v5 — DeepSORT + Multi-Embedding Gallery + Appearance Re-ID
Improvements:
  1. DeepSORT tracker (appearance + motion) with correct Hungarian assignment
  2. Multi-embedding gallery (top-K per identity)
  3. Appearance-based re-identification recovery
  4. Confidence thresholding
  5. Kalman filter for smooth tracking

Usage:
    python training/video_reid_v5.py --video Dataset/A1.mp4
    python training/video_reid_v5.py --all
"""
import os
import sys
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reid_common import (
    MODELS_DIR,
    META_PATH,
    load_split,
    load_meta,
    make_heatmap,
    prep_crop,
    load_checkpoint,
)

YOLO_DET = REPO / "cattle_osnet" / "yolov8m.pt"
YOLO_POSE = REPO / "cattle_osnet" / "models" / "cow_pose.pt"
DATASET_DIR = REPO / "input"
DEFAULT_CKPT = MODELS_DIR / "hanwoo_reid_best.pth"
FALLBACK_CKPT = MODELS_DIR / "hanwoo_reid_final.pth"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

PALETTE = [
    (255, 90, 90), (72, 205, 184), (255, 201, 60), (120, 220, 120),
    (150, 110, 230), (255, 150, 100), (80, 170, 255), (230, 110, 200),
    (200, 200, 80), (100, 180, 220), (220, 130, 160), (160, 220, 100),
]


class KalmanBoxTracker:
    """Kalman filter for bounding box tracking."""

    count = 0

    def __init__(self, bbox):
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.kf = cv2.KalmanFilter(7, 4)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
        ], np.float32)

        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1],
        ], np.float32)

        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1

        self.time_since_update = 0
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.history = []

        self._init_state(bbox)

    def _init_state(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.kf.statePost = np.array([cx, cy, w, h, 0, 0, 0], np.float32).reshape(-1, 1)

    def predict(self):
        state = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self._state_to_bbox(state)

    def update(self, bbox):
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        measurement = np.array([cx, cy, w, h], np.float32).reshape(-1, 1)
        self.kf.correct(measurement)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.history = []

    def _state_to_bbox(self, state):
        cx, cy, w, h = state[:4].flatten()
        return [int(cx - w / 2), int(cy - h / 2), int(cx + w / 2), int(cy + h / 2)]

    def get_state(self):
        return self._state_to_bbox(self.kf.statePost)


class DeepSORTTracker:
    """DeepSORT-style tracker with appearance + motion.

    update() returns [(track_id, detection_idx), ...] — the direct output of
    the Hungarian assignment (gated by IoU), so each detection is matched to
    at most one tracker and no tracker is double-assigned.
    """

    def __init__(self, max_age=150, n_init=3, iou_threshold=0.3,
                 appearance_weight=0.7):
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.trackers = []
        self.reid = {}
        self.frame_count = 0

    def update(self, detections, embeddings=None):
        self.frame_count += 1

        # Predict existing trackers, keep only alive ones
        predicted = []  # (tracker, predicted_box)
        alive = []
        for tracker in self.trackers:
            pred_box = tracker.predict()
            if tracker.time_since_update <= self.max_age:
                predicted.append((tracker, pred_box))
                alive.append(tracker)
        self.trackers = alive

        # Drop reid state for expired trackers
        alive_ids = {t.id for t in self.trackers}
        for tid in list(self.reid.keys()):
            if tid not in alive_ids:
                del self.reid[tid]

        assignments = []
        if not detections:
            return assignments

        if predicted:
            n_tracks, n_dets = len(predicted), len(detections)
            cost = np.zeros((n_tracks, n_dets))

            for i, (tracker_obj, pred_box) in enumerate(predicted):
                for j, det_box in enumerate(detections):
                    iou = self._iou(pred_box, det_box)
                    motion_cost = 1 - iou
                    appearance_cost = 1.0
                    track_emb = self.reid.get(tracker_obj.id, {}).get("emb")
                    if (track_emb is not None and embeddings is not None
                            and j < len(embeddings) and embeddings[j] is not None):
                        appearance_cost = 1 - float(np.clip(
                            np.dot(track_emb, embeddings[j]), -1.0, 1.0))
                    cost[i, j] = ((1 - self.appearance_weight) * motion_cost
                                  + self.appearance_weight * appearance_cost)

            rows, cols = linear_sum_assignment(cost)
            matched_dets = set()
            for i, j in zip(rows, cols):
                # IoU gate: pure-motion matches need real overlap
                if self._iou(predicted[i][1], detections[j]) < self.iou_threshold:
                    continue
                tracker_obj = predicted[i][0]
                tracker_obj.update(detections[j])
                assignments.append((tracker_obj.id, j))
                matched_dets.add(j)

            unmatched = [j for j in range(n_dets) if j not in matched_dets]
        else:
            unmatched = list(range(len(detections)))

        for j in unmatched:
            self.trackers.append(KalmanBoxTracker(detections[j]))

        return assignments

    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0


def run_pose_on_crop(pose_model, img, box, margin=0.3):
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
    kp = kp[np.argmax((kp[:, :, 2] > 0.25).sum(axis=1))]
    kp_img = kp.copy()
    kp_img[:, 0] += cx1
    kp_img[:, 1] += cy1
    pw, ph = max(1, x2 - x1), max(1, y2 - y1)
    rel = np.column_stack([(kp_img[:, 0] - x1) / pw, (kp_img[:, 1] - y1) / ph,
                           kp_img[:, 2]])
    return rel.tolist()


def reid_embed(model, img, box, pose_model, device):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = img[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return None
    if pose_model is not None:
        kpts = run_pose_on_crop(pose_model, img, box)
    else:
        kpts = None
    hm = make_heatmap(kpts)
    with torch.no_grad():
        x = prep_crop(crop).unsqueeze(0).to(device)
        hm_t = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0).to(device)
        emb, _ = model(x, hm_t)
        emb = F.normalize(emb, dim=1).cpu().numpy()[0]
    return emb


def build_gallery_multi(model, meta_map, device, top_k=5):
    """Gallery with top-K embeddings per identity (most representative views)."""
    items = load_split("gallery")
    print(f"[gallery] Embedding {len(items)} images...")
    id_embs = defaultdict(list)
    bs = 128
    for i in range(0, len(items), bs):
        batch = items[i:i + bs]
        imgs, hms = [], []
        for fp, _ in batch:
            img = cv2.imread(fp)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(prep_crop(img).numpy())
            rel = os.path.relpath(fp, REPO)
            hms.append(make_heatmap(meta_map.get(rel)))
        if not imgs:
            continue
        x = torch.from_numpy(np.stack(imgs)).to(device)
        hm = torch.from_numpy(np.stack(hms)).unsqueeze(1).to(device)
        with torch.no_grad():
            emb, _ = model(x, hm)
        emb = F.normalize(emb, dim=1)
        for j, (_, pid) in enumerate(batch):
            id_embs[pid].append(emb[j:j + 1].cpu())

    gallery = {}
    for pid, embs in id_embs.items():
        embs_cat = torch.cat(embs, dim=0)
        mean_emb = embs_cat.mean(dim=0)
        sims = F.cosine_similarity(embs_cat, mean_emb.unsqueeze(0))
        top_indices = sims.argsort(descending=True)[:top_k]
        top_embs = embs_cat[top_indices]
        gallery[pid] = F.normalize(top_embs.mean(dim=0), dim=0).numpy()
        gallery[f"{pid}_embeddings"] = top_embs.numpy()

    n_ids = len([k for k in gallery if not str(k).endswith('_embeddings')])
    print(f"[gallery] {n_ids} identities, top-{top_k} embeddings per ID")
    return gallery


def multi_embed_match(emb, gallery, gallery_pids, top_k=5):
    """Match using multiple embeddings per identity (max similarity)."""
    scores = {}
    for pid in gallery_pids:
        key = f"{pid}_embeddings"
        if key in gallery:
            sim = (emb @ gallery[key].T).max()
        else:
            sim = float(emb @ gallery[pid])
        scores[pid] = sim
    sorted_pids = sorted(scores.keys(), key=lambda p: scores[p], reverse=True)
    return [(pid, scores[pid]) for pid in sorted_pids[:top_k]]


def ema_update(emb, prev_emb, alpha=0.85):
    e = alpha * prev_emb + (1 - alpha) * emb
    e = e / (np.linalg.norm(e) + 1e-9)
    return e


def filter_cow_boxes(boxes, names, classes, confs, min_dim=50, max_aspect=5.0):
    """Keep cow boxes passing size/aspect filters.

    Resolution-agnostic: absolute minimum dimension (not % of frame area,
    which wrongly rejected 91% of distant CCTV cows on 2880x1620 footage).
    """
    cows = []
    for bb, ci, cf in zip(boxes, classes, confs):
        if ci != 21 and names[ci] != 'cow':
            continue
        x1, y1, x2, y2 = [int(v) for v in bb]
        box_w, box_h = x2 - x1, y2 - y1
        if box_w < min_dim or box_h < min_dim:
            continue
        aspect = box_w / max(box_h, 1)
        if aspect > max_aspect or aspect < 0.1:
            continue
        cows.append((x1, y1, x2, y2))
    cows.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    return cows


def detect_cows(detector, frame, conf=0.15, imgsz=1280):
    """Detect cows with YOLO, NMS + post-processing filters."""
    r = detector(frame, verbose=False, conf=conf, imgsz=imgsz,
                 iou=0.5, max_det=40, agnostic_nms=True)[0]
    if r.boxes is None or not len(r.boxes):
        return []
    return filter_cow_boxes(r.boxes.xyxy.cpu().numpy(),
                            r.names,
                            r.boxes.cls.cpu().numpy().astype(int),
                            r.boxes.conf.cpu().numpy())


def process_video(model, gallery, detector, pose_model, video_path, device, args):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return [], ""

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / src_fps

    print(f"\n  Video: {Path(video_path).name}")
    print(f"  {width}x{height} | {src_fps:.0f}fps | {duration:.0f}s | "
          f"{total_frames} frames")

    start_frame = int(args.start * src_fps) if args.start else 0
    end_frame = int((args.start + (args.dur or 0)) * src_fps) if args.dur else total_frames
    end_frame = min(end_frame, total_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Date-stamped output (README convention): output/YYYY-MM-DD/{stem}_reid_v5_{ts}.mp4
    date_dir = REPO / "output" / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = date_dir / f"{Path(video_path).stem}_reid_v5_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Write at source fps: sampled frames play back in real time (skips, no fast-forward)
    out = cv2.VideoWriter(str(out_path), fourcc, src_fps, (width, height))

    tracker = DeepSORTTracker(
        max_age=150,
        n_init=3,
        iou_threshold=args.iou_threshold,
        appearance_weight=args.appearance_weight,
    )

    frame_idx = start_frame
    results = []
    n_processed = 0
    id_counts = Counter()
    auto_id_counter = [0]
    track_auto_ids = {}

    gallery_pids = [pid for pid in gallery if not str(pid).endswith('_embeddings')]

    while cap.isOpened() and frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1  # frame_idx tracks the real video frame position

        if (frame_idx - start_frame) % args.sample_rate != 0:
            continue

        n_processed += 1
        cows = detect_cows(detector, frame, conf=args.conf, imgsz=args.imgsz)

        embeddings = []
        for box in cows:
            emb = reid_embed(model, frame, box, pose_model, device)
            embeddings.append(emb)

        assignments = tracker.update(cows, embeddings)

        for tid, bi in assignments:
            if bi >= len(cows) or bi >= len(embeddings):
                continue
            box = cows[bi]
            emb = embeddings[bi]
            if emb is None:
                continue

            prev = tracker.reid.get(tid)
            if prev is not None and prev.get('emb') is not None:
                smoothed_emb = ema_update(emb, prev['emb'], alpha=args.ema_alpha)
                matches = multi_embed_match(smoothed_emb, gallery, gallery_pids)
                if matches and matches[0][1] >= args.reid_conf:
                    label, score = matches[0]
                else:
                    label, score = prev['label'], prev['score']
            else:
                smoothed_emb = emb.copy()
                matches = multi_embed_match(smoothed_emb, gallery, gallery_pids)
                if matches and matches[0][1] >= args.reid_conf:
                    label, score = matches[0]
                elif args.auto_id:
                    if tid not in track_auto_ids:
                        auto_id_counter[0] += 1
                        track_auto_ids[tid] = auto_id_counter[0]
                    label, score = track_auto_ids[tid], 0.0
                else:
                    label, score = -1, 0.0

            tracker.reid[tid] = {
                'emb': smoothed_emb,
                'label': label,
                'score': score,
            }

            results.append({
                "frame": frame_idx,
                "time": round(frame_idx / src_fps, 2),
                "track_id": tid,
                "reid_id": label,
                "sim": round(score, 4),
                "bbox": list(box),
            })

            if label != -1:
                id_counts[label] += 1

        draw_frame(frame, cows, tracker, width, height, frame_idx, src_fps, args)
        out.write(frame)

        if n_processed % 20 == 0:
            print(f"    Frame {frame_idx} ({frame_idx-start_frame}/{end_frame-start_frame})"
                  f" | {len(tracker.trackers)} active tracks | "
                  f"{len(id_counts)} IDs", flush=True)

    cap.release()
    out.release()

    print(f"\n  Output: {out_path}")
    print(f"  Processed: {n_processed} frames")
    print(f"  Detected IDs: {len(id_counts)}")
    for pid, count in sorted(id_counts.items()):
        avg_sim = np.mean([r["sim"] for r in results if r["reid_id"] == pid])
        print(f"    Cow_{pid}: {count} frames (avg sim: {avg_sim:.3f})")

    return results, str(out_path)


def draw_frame(frame, cows, tracker, width, height, frame_idx, src_fps, args):
    for tracker_obj in tracker.trackers:
        if tracker_obj.time_since_update > 0:
            continue
        box = tracker_obj.get_state()
        x1, y1, x2, y2 = box
        color = PALETTE[tracker_obj.id % len(PALETTE)]
        thickness = max(2, width // 800)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        info = tracker.reid.get(tracker_obj.id, {})
        label_str = info.get('label', -1)
        score = info.get('score', 0)
        if label_str != -1:
            if score > 0:
                label = f"Cow_{label_str} ({score:.2f})"
            else:
                label = f"Cow_{label_str}"
        else:
            label = "Cow_?"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y = max(y1 - 10, th + 10)
        cv2.rectangle(frame, (x1, label_y - th - 8), (x1 + tw + 8, label_y + 4),
                      color, -1)
        cv2.putText(frame, label, (x1 + 4, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 0), 2)

    n_active = sum(1 for t in tracker.trackers if t.time_since_update == 0)
    n_known = sum(1 for t in tracker.trackers
                  if t.time_since_update == 0
                  and tracker.reid.get(t.id, {}).get('label', -1) != -1)
    info = f"Cows: {n_active} | Identified: {n_known} | Frame {frame_idx} | " \
           f"{frame_idx/src_fps:.1f}s"
    cv2.putText(frame, info, (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Video ReID v5 — DeepSORT + Multi-Embedding Gallery")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--all", action="store_true", help="Process all Dataset/*.mp4")
    parser.add_argument("--start", type=float, default=0, help="Start offset in seconds")
    parser.add_argument("--dur", type=float, default=None, help="Duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Process every Nth frame (higher = faster)")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="YOLO detection confidence")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size")
    parser.add_argument("--ema-alpha", type=float, default=0.85,
                        help="EMA smoothing factor")
    parser.add_argument("--reid-conf", type=float, default=0.45,
                        help="Min cosine similarity to assign gallery ID")
    parser.add_argument("--no-pose", action="store_true",
                        help="Disable pose estimation")
    parser.add_argument("--use-train-gallery", action="store_true",
                        help="Include training images in gallery")
    parser.add_argument("--yolo", type=str, default=None, help="Path to YOLO model")
    parser.add_argument("--appearance-weight", type=float, default=0.7,
                        help="Weight for appearance in DeepSORT "
                             "(0=motion only, 1=appearance only)")
    parser.add_argument("--iou-threshold", type=float, default=0.3,
                        help="Min IoU to match a detection to a track")
    parser.add_argument("--gallery-top-k", type=int, default=5,
                        help="Number of embeddings to keep per identity")
    parser.add_argument("--auto-id", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Auto-assign IDs when no gallery match found")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt_path = DEFAULT_CKPT
    if not ckpt_path.exists() and FALLBACK_CKPT.exists():
        ckpt_path = FALLBACK_CKPT
    if not ckpt_path.exists():
        print(f"No ReID checkpoint found. Looked for:\n  {DEFAULT_CKPT}\n  "
              f"{FALLBACK_CKPT}\nTrain first: python training/train_v3.py")
        return
    model, _ = load_checkpoint(ckpt_path, device)
    print(f"[reid] Model loaded: {model.classifier.out_features} classes")

    meta_map = load_meta()
    gallery = build_gallery_multi(model, meta_map, device, top_k=args.gallery_top_k)

    from ultralytics import YOLO
    yolo_path = args.yolo or str(YOLO_DET)
    detector = YOLO(yolo_path).to(device)
    pose_model = None if args.no_pose else YOLO(str(YOLO_POSE)).to(device)
    print(f"[det] YOLO loaded: {os.path.basename(yolo_path)}"
          + (" + pose" if pose_model else ""))

    if args.all:
        videos = sorted(DATASET_DIR.glob("*.mp4"))
    elif args.video:
        videos = [Path(args.video)]
    else:
        videos = sorted(DATASET_DIR.glob("*.mp4"))[:1]
    videos = [v for v in videos
              if "_reid" not in v.name and "_annotated" not in v.name]

    print(f"\nProcessing {len(videos)} video(s)...")
    all_results = {}
    for vpath in videos:
        results, out = process_video(model, gallery, detector, pose_model, vpath,
                                     device, args)
        all_results[str(vpath)] = {"results": results, "output": out}

    report_path = REPO / "logs" / "video_reid_v5.json"
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {report_path}")


if __name__ == "__main__":
    main()