!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
!pip -q install ultralytics opencv-python-headless numpy
import os, glob, json, tarfile
import numpy as np, cv2
from ultralytics import YOLO

def find_input(fname):
    m = glob.glob('/kaggle/input/**/' + fname, recursive=True)
    return m[0] if m else None

pose = YOLO(find_input('cow_pose.pt') or 'cow_pose.pt')

# crops arrive as a real dir or as crops.tar (built by run_pipeline with a 'crops/...'
# layout). Extract to /kaggle/working so relpaths are crops/<video>/<track>/<file>.
ROOT = None
for d in glob.glob('/kaggle/input/**/', recursive=True):
    if os.path.isdir(os.path.join(d, 'crops')):
        ROOT = d; break
if ROOT is None:
    tar = next((t for t in glob.glob('/kaggle/input/**/crops.tar', recursive=True)), None)
    if tar:
        with tarfile.open(tar) as tf:
            tf.extractall('/kaggle/working')
        ROOT = '/kaggle/working'
print('crop root:', ROOT)
crops = sorted(glob.glob(os.path.join(ROOT, 'crops', '**', '*.jpg'), recursive=True))
print('crops found:', len(crops))

MIN_CONF = 0.25
meta = []
for i, p in enumerate(crops):
    rel = os.path.relpath(p, ROOT).replace(os.sep, '/')   # crops/<video>/<track>/<file>
    parts = rel.split('/')
    img = cv2.imread(p)
    h, w = img.shape[:2]
    r = pose(img, verbose=False, conf=MIN_CONF, imgsz=640)[0]
    kp = None
    if r.keypoints is not None and len(r.keypoints):
        arr = r.keypoints.data.cpu().numpy()
        arr = arr[np.argmax((arr[:, :, 2] > MIN_CONF).sum(1))]
        kp = [[round(float(x) / w, 4), round(float(y) / h, 4), round(float(c), 3)]
              for x, y, c in arr]
    meta.append({'image': rel, 'video': parts[1], 'track': parts[2],
                 'keypoints': kp,
                 'kpts_visible': int((arr[:, 2] > MIN_CONF).sum()) if kp else 0})
    if (i + 1) % 300 == 0:
        print(i + 1, '/', len(crops), flush=True)

def _js(o):
    if isinstance(o, np.integer): return int(o)
    if isinstance(o, np.floating): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    return str(o)

with open('/kaggle/working/meta.json', 'w') as f:
    json.dump(meta, f, default=_js)
print('wrote meta.json:', len(meta), 'entries ->', os.path.getsize('/kaggle/working/meta.json'), 'bytes')

meta = json.load(open('/kaggle/working/meta.json'))
vis = [m['kpts_visible'] for m in meta if m['keypoints']]
print('crops:', len(meta), '| with keypoints:', len(vis))
if vis:
    print('kpts_visible: mean %.1f  median %.0f  >=6: %d/%d' % (
        np.mean(vis), np.median(vis),
        sum(1 for v in vis if v >= 6), len(vis)))
per = {}
for m in meta: per.setdefault(m['video'], {}).setdefault(m['track'], 0); per[m['video']][m['track']] += 1
for v, tr in per.items():
    n = list(tr.values())
    print(v, 'tracks:', len(n), 'crops/track mean', round(float(np.mean(n)), 1), 'max', max(n))
print('Download meta.json from the Output tab (the pipeline folds it into the crops dataset).')
