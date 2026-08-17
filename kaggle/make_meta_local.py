# Regenerate pose keypoints for the 02-extraction crops locally (CPU).
#
# The v6 Kaggle extraction run returned an empty meta.json, so the pose keypoints the
# PHE module / VCR need are missing. This walks the saved crops, runs cow_pose.pt on
# each, and writes meta.json keyed with the same relative paths
# ("crops/<video>/<track>/<file>") that notebooks 03/04 expect.
#
# Usage:
#   python kaggle/make_meta_local.py                 # default: .run/02_out crops
#   python kaggle/make_meta_local.py --root <dir>    # dir that CONTAINS a crops/ folder

import argparse, glob, json, os, time

import cv2
import numpy as np
from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROOT = os.path.join(REPO, "kaggle", ".run", "02_out")
POSE_WEIGHTS = os.path.join(REPO, "cattle_osnet", "models", "cow_pose.pt")
MIN_CONF = 0.25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT, help="dir that contains crops/ (relpaths are crops/...)")
    ap.add_argument("--pose", default=POSE_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default=os.path.join(DEFAULT_ROOT, "meta.json"))
    a = ap.parse_args()

    crops_dir = os.path.join(a.root, "crops")
    if not os.path.isdir(crops_dir):
        raise SystemExit(f"no crops/ under {a.root}")

    pose = YOLO(a.pose)
    crops = sorted(glob.glob(os.path.join(crops_dir, "**", "*.jpg"), recursive=True))
    print(f"{len(crops)} crops, weights {a.pose}")

    meta = []
    t0 = time.time()
    for i, p in enumerate(crops):
        rel = os.path.relpath(p, a.root).replace(os.sep, "/")
        parts = rel.split("/")
        img = cv2.imread(p)
        h, w = img.shape[:2]
        r = pose(img, verbose=False, conf=MIN_CONF, imgsz=a.imgsz)[0]
        kp = None
        if r.keypoints is not None and len(r.keypoints):
            arr = r.keypoints.data.cpu().numpy()
            arr = arr[np.argmax((arr[:, :, 2] > MIN_CONF).sum(1))]
            kp = [[round(float(x) / w, 4), round(float(y) / h, 4), round(float(c), 3)]
                  for x, y, c in arr]
        meta.append({"image": rel, "video": parts[1], "track": parts[2],
                     "keypoints": kp,
                     "kpts_visible": int((arr[:, 2] > MIN_CONF).sum()) if kp else 0})
        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{len(crops)}  {rate:.2f} crops/s  "
                  f"eta {(len(crops) - i - 1) / rate / 60:.0f} min", flush=True)

    def _js(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(meta, f, default=_js)

    vis = [m["kpts_visible"] for m in meta if m["keypoints"]]
    print(f"wrote {a.out}: {len(meta)} entries, {os.path.getsize(a.out)} bytes")
    if vis:
        print(f"kpts_visible: mean {np.mean(vis):.1f} median {np.median(vis):.0f} "
              f">=6: {sum(1 for v in vis if v >= 6)}/{len(vis)}")


if __name__ == "__main__":
    main()
