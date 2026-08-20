!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
!pip -q install ultralytics opencv-python-headless pillow numpy
!apt-get -qq install -y ffmpeg >/dev/null 2>&1 || true

import os, glob, json
import cv2, numpy as np
from ultralytics import YOLO

VIDEOS = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True))
def find_input(fname):
    m = glob.glob('/kaggle/input/**/' + fname, recursive=True)
    return m[0] if m else None

det = YOLO(find_input('yolov8n.pt') or 'yolov8n.pt')
pose = YOLO(find_input('cow_pose.pt') or 'cow_pose.pt')
print('videos:', [os.path.basename(v) for v in VIDEOS])

IMGSZ, CONF = 1920, 0.15
SAMPLE_EVERY = 25   # save one crop per track every 25 frames (1 crop/s @25fps)
OUTDIR = '/kaggle/working/crops'

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1); ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,ix2-ix1), max(0,iy2-iy1); inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

class Tracker:
    def __init__(self, thr=0.25, expire=150):
        self.thr, self.expire, self.tracks, self.next_id, self.fidx = thr, expire, {}, 1, 0
    def update(self, boxes):
        self.fidx += 1; used = set(); assign = []
        for tid, rec in sorted(self.tracks.items(), key=lambda kv: -kv[1][1]):
            if tid in used: continue
            bi, biou = None, 0.0
            for i, b in enumerate(boxes):
                if i in used: continue
                v = iou(rec[0], b)
                if v > biou: biou, bi = v, i
            if bi is not None and biou >= self.thr:
                used.add(bi); self.tracks[tid] = (boxes[bi], 0.0, self.fidx); assign.append((tid, bi))
        for i, b in enumerate(boxes):
            if i not in used:
                self.tracks[self.next_id] = (b, 0.0, self.fidx); assign.append((self.next_id, i)); self.next_id += 1
        for tid in list(self.tracks):
            if self.fidx - self.tracks[tid][2] > self.expire: del self.tracks[tid]
        return assign

def pose_on_crop(img, box, margin=0.3):
    h, w = img.shape[:2]; x1, y1, x2, y2 = [int(v) for v in box]
    mw, mh = margin*(x2-x1), margin*(y2-y1)
    cx1, cy1 = max(0, x1-int(mw)), max(0, y1-int(mh))
    cx2, cy2 = min(w, x2+int(mw)), min(h, y2+int(mh))
    c = img[cy1:cy2, cx1:cx2]
    if c.size == 0: return None
    r = pose(c, verbose=False, conf=0.25, imgsz=640)[0]
    if r.keypoints is None or not len(r.keypoints): return None
    kp = r.keypoints.data.cpu().numpy()
    kp = kp[np.argmax((kp[:,:,2]>0.25).sum(1))]
    kp[:,0] += cx1; kp[:,1] += cy1
    pw, ph = max(1, x2-x1), max(1, y2-y1)
    rel = np.column_stack([(kp[:,0]-x1)/pw, (kp[:,1]-y1)/ph, kp[:,2]])
    return rel.tolist()

meta = []
for v in VIDEOS:
    name = os.path.basename(v).replace('.mp4', '')
    cap = cv2.VideoCapture(v); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    tracker = Tracker(); fidx = 0; n_saved = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        if fidx % SAMPLE_EVERY != 0:
            fidx += 1; continue
        r = det(frame, verbose=False, conf=CONF, imgsz=IMGSZ)[0]
        cows = []
        if r.boxes is not None and len(r.boxes):
            for bb, ci in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                if r.names[ci] == 'cow': cows.append(tuple(int(x) for x in bb))
        if not cows:
            fidx += 1; continue
        assign = tracker.update(cows)
        ts = round(fidx / fps, 2)
        for tid, bi in assign:
            box = cows[bi]
            x1, y1, x2, y2 = box
            if (x2-x1) < 50 or (y2-y1) < 50: continue   # skip tiny crops
            pad = 0.15
            px1, py1 = max(0, int(x1-pad*(x2-x1))), max(0, int(y1-pad*(y2-y1)))
            px2, py2 = min(frame.shape[1], int(x2+pad*(x2-x1))), min(frame.shape[0], int(y2+pad*(y2-y1)))
            crop = frame[py1:py2, px1:px2]
            d = os.path.join(OUTDIR, name, f'Cow_{tid:03d}')
            os.makedirs(d, exist_ok=True)
            fn = f't{ts:08.2f}_f{fidx:07d}.jpg'
            cv2.imwrite(os.path.join(d, fn), crop)
            kp = pose_on_crop(frame, (px1, py1, px2, py2))
            meta.append({'image': f'{name}/Cow_{tid:03d}/{fn}', 'video': name, 'track': f'Cow_{tid:03d}',
                         'bbox': box, 'frame': int(fidx), 'time_s': float(ts), 'keypoints': kp,
                         'kpts_visible': int((np.array(kp)[:, 2] > 0.25).sum()) if kp else 0})
            n_saved += 1
        fidx += 1
        if fidx % (SAMPLE_EVERY*200) == 0: print(name, 'frame', fidx, 'saved', n_saved, flush=True)
    cap.release()
    print(name, 'DONE, crops saved:', n_saved)

def _js(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)
with open('/kaggle/working/meta.json', 'w') as f: json.dump(meta, f, default=_js)
print('TOTAL crops:', len(meta), '-> /kaggle/working/crops + /kaggle/working/meta.json')
print('Download both folders from the Output tab.')

# Quick stats: how many crops per track, pose quality
import json, numpy as np
meta = json.load(open('/kaggle/working/meta.json'))
print('crops:', len(meta))
print('tracks:', len(set(m['track'] + '@' + m['video'] for m in meta)))
vis = [m['kpts_visible'] for m in meta if m['keypoints']]
if vis: print(f'pose: {sum(1 for v in vis if v>=6)}/{len(vis)} crops with >=6 keypoints')
per = {}
for m in meta: per.setdefault(m['video'], {}).setdefault(m['track'], 0); per[m['video']][m['track']] += 1
for v, tr in per.items():
    n = list(tr.values())
    print(v, 'tracks:', len(n), 'crops/track mean', round(np.mean(n),1), 'max', max(n))
