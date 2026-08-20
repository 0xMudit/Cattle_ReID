# 01 — Annotated video (paper Fig.5/Fig.12 style)
# Runs on a Kaggle **GPU** notebook (T4 or better). Input: your CCTV videos in a Kaggle dataset.
# Output: full-resolution annotated mp4 + inline preview frames.
#
# ## Setup on Kaggle first (one-time, 2 min)
# 1. Create a **Kaggle Dataset** named e.g. `cattle-reid-raw-videos`.
# 2. Upload the 6 files from `Dataset/` (A1, A2, A3, ch07m_*, ch10m_*).
# 3. Upload these model files too (from this repo):
#    - `cattle_osnet/yolov8n.pt`
#    - `cattle_osnet/models/cow_pose.pt`
# 4. In this notebook: Notebook settings -> Add Input -> your dataset.
# 5. Run all cells. Download `annotated_*.mp4` from the **Output** tab when done.

# Cell 2: Install dependencies
# !pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
# !pip -q install ultralytics opencv-python-headless pillow numpy
# !apt-get -qq install -y ffmpeg >/dev/null 2>&1 || true

# Cell 3: Load models
import os, glob
import cv2, numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Find your video inputs (new Kaggle mount nests under /kaggle/input/datasets/<owner>/<ds>)
VIDEOS = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True))
print('dataset videos:', len(VIDEOS))
for v in VIDEOS: print('  ', os.path.basename(v), round(os.path.getsize(v)/1e6,1), 'MB')

def find_input(fname):
    m = glob.glob('/kaggle/input/**/' + fname, recursive=True)
    return m[0] if m else None

det = YOLO(find_input('yolov8n.pt') or 'yolov8n.pt')
pose = YOLO(find_input('cow_pose.pt') or 'cow_pose.pt')
print('models loaded')

# Cell 4: Config
# ---- config ----
IMGSZ  = 1920        # detection inference size (higher = finds small/far cows)
CONF   = 0.15        # detection confidence
POSE_MAX = 12        # skeletons on the N largest cows (costly)
OUTFPS = 25          # output frame rate (keep 25 for full video)
SKIP   = 1           # process every Nth frame (2 = half work, 25fps -> ~12fps)

KP_NAMES = ["Nose","R_Eye","L_Eye","Neck","LF_Hoof","RF_Hoof","LB_Hoof","RB_Hoof","Backbone","TailRoot","BackPose","Stomach"]
SKELETON = [(0,1),(0,2),(1,3),(2,3),(3,8),(8,9),(3,4),(3,5),(9,6),(9,7),(8,10),(10,11)]
PAL = [(255,90,90),(72,205,184),(255,201,60),(120,220,120),(150,110,230),(255,150,100),(80,170,255),(230,110,200)]

# Cell 5: Tracker + Pose + Annotation
def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1,bx1), max(ay1,by1); ix2, iy2 = min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0,ix2-ix1), max(0,iy2-iy1); inter = iw*ih
    ua = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
    return inter/ua if ua>0 else 0.0

class Tracker:
    def __init__(self, thr=0.25, expire=180):
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

def cow_pose(img, box, margin=0.3):
    h, w = img.shape[:2]; x1, y1, x2, y2 = [int(v) for v in box]
    mw, mh = margin*(x2-x1), margin*(y2-y1)
    c = img[max(0,y1-int(mh)):min(h,y2+int(mh)), max(0,x1-int(mw)):min(w,x2+int(mw))]
    if c.size == 0: return None
    r = pose(c, verbose=False, conf=0.25, imgsz=640)[0]
    if r.keypoints is None or not len(r.keypoints): return None
    kp = r.keypoints.data.cpu().numpy()
    kp = kp[np.argmax((kp[:,:,2]>0.25).sum(1))]
    vis = kp[:,2] > 0.25
    if vis.sum() < 4: return None
    kp[:,0] += max(0, x1-int(mw)); kp[:,1] += max(0, y1-int(mh))
    return kp, vis

def annotate(frame):
    h, w = frame.shape[:2]
    r = det(frame, verbose=False, conf=CONF, imgsz=IMGSZ)[0]
    cows = []
    if r.boxes is not None and len(r.boxes):
        for bb, ci in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
            if r.names[ci] == 'cow': cows.append(tuple(int(v) for v in bb))
    cows.sort(key=lambda bb: (bb[2]-bb[0])*(bb[3]-bb[1]), reverse=True)
    cows = cows[:40]
    assign = tracker.update(cows)
    pose_boxes = set(cows[:POSE_MAX])
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, 'RGBA'); fs = max(18, w//80)
    try: fnt = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', fs)
    except Exception: fnt = ImageFont.load_default()
    for tid, bi in assign:
        box = cows[bi]; col = PAL[tid % len(PAL)]
        d.rectangle(box, outline=col, width=max(2, w//900))
        lab = f"Cow_{tid:02d}"; tw, th = d.textbbox((0,0), lab, font=fnt)[2:]
        d.rectangle([box[0], box[1]-th-6, box[0]+tw+6, box[1]], fill=col)
        d.text((box[0]+3, box[1]-th-3), lab, fill=(15,15,15), font=fnt)
        if box in pose_boxes:
            pk = cow_pose(frame, box)
            if pk is not None:
                kp, vis = pk
                for a, b in SKELETON:
                    if vis[a] and vis[b]: d.line((kp[a][0],kp[a][1],kp[b][0],kp[b][1]), fill=(255,255,255,255), width=max(2, fs//8))
                for j in range(12):
                    if vis[j]:
                        rr = max(3, fs//6)
                        d.ellipse([kp[j][0]-rr, kp[j][1]-rr, kp[j][0]+rr, kp[j][1]+rr], fill=(255,255,255,255), outline=(0,0,0,255))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# Cell 6: Render annotated video
VIDEO_TO_RENDER = VIDEOS[0]   # pick the file you want (change index)
START, DUR = 0, 0              # seconds; 0 = whole video

cap = cv2.VideoCapture(VIDEO_TO_RENDER)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.set(cv2.CAP_PROP_POS_MSEC, START*1000)
out_name = os.path.basename(VIDEO_TO_RENDER).replace('.mp4','') + '_annotated.mp4'
writer = None; tracker = Tracker(); total = 0; t0 = __import__('time').time()

nframes = int(DUR*fps) if DUR > 0 else 10**12
fi = 0
while fi < nframes:
    ok, frame = cap.read()
    if not ok: break
    if fi % SKIP == 0:
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(out_name, cv2.VideoWriter_fourcc(*'mp4v'), float(OUTFPS), (w, h))
        ann = annotate(frame)
        writer.write(ann); total += 1
        if total % 25 == 0:
            el = __import__('time').time() - t0
            print(f'{total} frames, {el:.0f}s, {total/max(el,0.01):.2f} fps', flush=True)
    fi += 1
cap.release()
if writer:
    writer.release(); print('done:', out_name, total, 'frames')
    print('Download it from the Kaggle notebook Output tab / Output > Add to dataset.')

# Cell 7: Inline preview
# Inline preview: show 6 sample frames from the annotated video
cap = cv2.VideoCapture(out_name)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for ax in axes.ravel():
    cap.read(); ok, f = cap.read()
    if not ok: break
    ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); ax.axis('off')
cap.release(); plt.tight_layout(); plt.show()
