# Install dependencies - already installed on remote
print('Dependencies already installed!')

import os
PROJ = '/content/cattle_reid' if os.path.exists('/content') else os.path.expanduser('~/0xmudit_cattle_reID')
for d in ['data/raw','data/processed/train','data/processed/query','data/processed/gallery',
          'data/gallery','models','logs']:
    os.makedirs(f'{PROJ}/{d}', exist_ok=True)
print('Folders created at', PROJ)

# Download CID (Cow Images Dataset) from S3 - skip if already exists
import os
for name, url in [('images.tar.gz', 'https://cid-21.s3.amazonaws.com/images.tar.gz'),
                  ('yt_images.tar.gz', 'https://cid-21.s3.amazonaws.com/yt_images.tar.gz'),
                  ('dataset.csv', 'https://cid-21.s3.amazonaws.com/dataset.csv')]:
    path = f'{PROJ}/data/raw/{name}'
    if not os.path.exists(path):
        !curl -L --progress-bar -o {path} {url}
    else:
        print(f'Skipping {name} (already exists)')
print('Download complete!')

import tarfile

for tar in ['images.tar.gz', 'yt_images.tar.gz']:
    path = f'{PROJ}/data/raw/{tar}'
    with tarfile.open(path, 'r:gz') as t:
        t.extractall(f'{PROJ}/data/raw/')
    print(f'Extracted {tar}')

RAW = f'{PROJ}/data/raw/images'
!ls {RAW} | head -20

import cv2
import numpy as np
import torch
import albumentations as A
from ultralytics import YOLO

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

# Load YOLOv8 from ultralytics (modern, actively maintained)
yolo = YOLO('yolov8n.pt').to(device)

# COCO class IDs (1-based from ultralytics):
#  17 = horse,  18 = sheep,  19 = cow
COW_CLS = 19

print('YOLOv8 loaded on', device)

class Prep:
    def __init__(self, model, h=256, w=192, letterbox=True):
        self.m = model
        self.h = h
        self.w = w
        self.letterbox = letterbox
        self.aug = A.Compose([
            A.GaussNoise(var_limit=(10,80), p=0.33),
            A.Blur(blur_limit=3, p=0.33),
            A.RandomBrightnessContrast(brightness_limit=(-0.3,0.2), contrast_limit=(-0.3,0.2), p=0.3),
            A.CLAHE(clip_limit=4, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1, sat=0.3, p=0.33),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, p=0.33),
            A.HorizontalFlip(p=0.5)])

    def detect(self, img):
        """Run YOLO detection, return Nx4 array of cow bounding boxes."""
        r = self.m(img, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 4), dtype=int)
        # Filter to cow class
        mask = boxes.cls.cpu().numpy() == COW_CLS
        if not mask.any():
            return np.empty((0, 4), dtype=int)
        xyxy = boxes.xyxy.cpu().numpy()[mask].astype(int)
        return xyxy

    def crop(self, img, bb):
        """Crop and resize with aspect-ratio-preserving letterbox."""
        x1, y1, x2, y2 = bb
        crop = img[y1:y2, x1:x2]
        if not self.letterbox:
            return cv2.resize(crop, (self.w, self.h))
        # Letterbox resize: preserve aspect ratio, pad to target size
        h0, w0 = crop.shape[:2]
        scale = min(self.h / h0, self.w / w0)
        new_w, new_h = int(w0 * scale), int(h0 * scale)
        resized = cv2.resize(crop, (new_w, new_h))
        out = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        pad_w = (self.w - new_w) // 2
        pad_h = (self.h - new_h) // 2
        out[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        return out

    def aug_img(self, img, n=5):
        """Return [img] + n augmented versions."""
        imgs = [img]
        for _ in range(n):
            imgs.append(self.aug(image=img)['image'])
        return imgs


prep = Prep(yolo)
print('Prep ready')

from glob import glob
from pathlib import Path
from tqdm import tqdm

RAW = f'{PROJ}/data/raw'
PROC = f'{PROJ}/data/processed'

# Collect all image paths, group by cow ID from directory structure
imgs = []
for e in ['*.jpg', '*.jpeg', '*.png']:
    imgs.extend(glob(f'{RAW}/**/{e}', recursive=True))

cows = {}  # cow_id -> list of image paths
for p in imgs:
    for part in Path(p).parts:
        if part.isdigit():
            cows.setdefault(int(part), []).append(p)
            break

print(f'Found {len(cows)} cows with {sum(len(v) for v in cows.values())} images total')

# --- Train / Gallery / Query split ---
# Reserve some cow IDs for gallery (enrollment) and query (probe).
# The rest are training identities.
cow_ids = sorted(cows.keys())
n = len(cow_ids)
n_train = int(n * 0.7)
n_gal   = int(n * 0.15)
n_qry   = n - n_train - n_gal

train_ids = set(cow_ids[:n_train])
gallery_ids = set(cow_ids[n_train:n_train + n_gal])
query_ids = set(cow_ids[n_train + n_gal:])

print(f'Split: {len(train_ids)} train, {len(gallery_ids)} gallery, {len(query_ids)} query identities')

# --- Process images ---
# For train cows: use up to 50 images each, augment 3x
# For gallery cows: use up to 10 images each (no augmentation)
# For query cows: use up to 5 images each (no augmentation)
# When YOLO misses a cow → skip the image entirely (no background saved!)

counts = {'train': 0, 'gallery': 0, 'query': 0}

for cid, paths in tqdm(cows.items(), desc='Processing cows'):
    if cid in train_ids:
        subset = paths[:50]
        dest = f'{PROC}/train'
        augment = True
    elif cid in gallery_ids:
        subset = paths[:10]
        dest = f'{PROC}/gallery'
        augment = False
    elif cid in query_ids:
        subset = paths[:5]
        dest = f'{PROC}/query'
        augment = False
    else:
        continue

    for p in subset:
        im = cv2.imread(p)
        if im is None:
            continue

        bbs = prep.detect(im)
        if len(bbs) == 0:
            # YOLO missed the cow — skip this image entirely
            # (original saved the whole resized frame as a cow — WRONG)
            continue

        # Crop the first detected cow
        cropped = prep.crop(im, bbs[0])

        if augment:
            images = prep.aug_img(cropped, 3)
        else:
            images = [cropped]

        for img_aug in images:
            counts[dest.split('/')[-1] if '/' in dest else dest] += 1
            name = f'c0_p{cid}_{counts[dest.split("/")[-1]]}.jpg'
            cv2.imwrite(f'{dest}/{name}', img_aug)

print(f'Created: {counts}')
print(f'Train dir: {len(os.listdir(f"{PROC}/train"))} files')
print(f'Gallery dir: {len(os.listdir(f"{PROC}/gallery"))} files')
print(f'Query dir: {len(os.listdir(f"{PROC}/query"))} files')

import os, glob, random, string, torch
import torchreid
from torchreid.data.datasets import ImageDataset

PROC = f'{PROJ}/data/processed'
TRAIN_DIR = f'{PROC}/train'
QUERY_DIR = f'{PROC}/query'
GALLERY_DIR = f'{PROC}/gallery'


class CattleDS(ImageDataset):
    """
    Custom dataset that reads from separate train / query / gallery directories.
    This ensures the model is evaluated on identities it has never seen during training.
    """
    def __init__(self, root='', **kw):
        super().__init__(
            self._pd(TRAIN_DIR, False),
            self._pd(QUERY_DIR, True),
            self._pd(GALLERY_DIR, False),
            **kw
        )

    def _pd(self, d, is_query):
        data = []
        if not os.path.isdir(d):
            return data
        for p in glob.glob(os.path.join(d, '*.jpg')):
            try:
                nm = os.path.basename(p).split('_')
                pid = int(nm[1][1:])
                camid = int(nm[0][1:])
                if is_query:
                    camid += 10
                data.append((p, pid, camid))
            except (IndexError, ValueError):
                pass
        return data


print(f'Train identities available: check dir contents')
print(f'Train: {len(os.listdir(TRAIN_DIR))} files, Query: {len(os.listdir(QUERY_DIR))} files, '
      f'Gallery: {len(os.listdir(GALLERY_DIR))} files')

from torchreid.engine import ImageTripletEngine

# Hyper-parameters
CFG = {
    'name': 'osnet_x1_0',
    'h': 256,
    'w': 192,
    'bs': 32,
    'lr': 0.003,
    'ep': 30,
    'eval': 5,
    'step': 10,   # LR scheduler step size
    'm': 0.3,     # triplet margin
    'wt': 1,      # triplet loss weight
    'wx': 50,     # softmax loss weight
}

# Register dataset under a random name (torchreid requirement)
dn = 'cattle_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
torchreid.data.register_image_dataset(dn, CattleDS)

dm = torchreid.data.ImageDataManager(
    sources=dn,
    height=CFG['h'],
    width=CFG['w'],
    batch_size_train=CFG['bs'],
    batch_size_test=100,
    transforms=['random_flip', 'random_crop']
)

model = torchreid.models.build_model(
    name=CFG['name'],
    num_classes=dm.num_train_pids,
    loss='triplet',
    pretrained=True
).to(device)

opt = torchreid.optim.build_optimizer(model, optim='adam', lr=CFG['lr'])
sch = torchreid.optim.build_lr_scheduler(opt, lr_scheduler='single_step', stepsize=CFG['step'])

engine = ImageTripletEngine(
    dm, model,
    optimizer=opt,
    scheduler=sch,
    margin=CFG['m'],
    weight_t=CFG['wt'],
    weight_x=CFG['wx'],
)

n_params = sum(p.numel() for p in model.parameters())
print(f'Model ready on {device}: {n_params:,} params, '
      f'{dm.num_train_pids} train identities, '
      f'{len(dm.query_dataset)} query, '
      f'{len(dm.gallery_dataset)} gallery')

engine.run(
    save_dir=f'{PROJ}/logs/{CFG["name"]}',
    max_epoch=CFG['ep'],
    eval_freq=CFG['eval'],
    print_freq=50
)
print('Training done!')

import pickle
from torchreid.utils import FeatureExtractor


class Registry:
    """Gallery of known cows with their embedding signatures."""
    def __init__(self, name='osnet_x1_0', path=None):
        self.gal = {}
        self.gf = f'{PROJ}/data/gallery/gal.pkl'
        kw = {
            'model_name': name,
            'device': device,
            'image_size': (CFG['h'], CFG['w']),
            'verbose': False,
        }
        if path:
            kw['model_path'] = path
        self.ext = FeatureExtractor(**kw)
        if os.path.exists(self.gf):
            with open(self.gf, 'rb') as f:
                self.gal = pickle.load(f)
            print(f'Loaded gallery: {len(self.gal)} cows')

    def register(self, name, imgs):
        embs = []
        for im in imgs:
            if isinstance(im, str):
                im = cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2RGB)
            e = self.ext([im]).cpu().detach().numpy().flatten()
            embs.append(e)
        self.gal[name] = {
            'embs': np.array(embs),
            'mean': np.mean(embs, axis=0),
            'n': len(imgs),
        }
        with open(self.gf, 'wb') as f:
            pickle.dump(self.gal, f)
        print(f'Registered {name}: {len(imgs)} images')

    def names(self):
        return list(self.gal.keys())

    def remove(self, name):
        if name in self.gal:
            del self.gal[name]
        with open(self.gf, 'wb') as f:
            pickle.dump(self.gal, f)


reg = Registry()
print('Registry ready')

class Recognizer:
    """Detect cows in an image and recognize them against the gallery."""
    def __init__(self, reg, yolo, thr=0.6):
        self.reg = reg
        self.yolo = yolo
        self.thr = thr

    def l2(self, a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    def run(self, img):
        """img: BGR numpy array (as read by cv2.imread)."""
        res = []
        r = self.yolo(img, verbose=False)[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return res
        # Filter cow class
        mask = boxes.cls.cpu().numpy() == COW_CLS
        if not mask.any():
            return res
        xyxy = boxes.xyxy.cpu().numpy()[mask].astype(int)
        confs = boxes.conf.cpu().numpy()[mask]

        if not self.reg.gal:
            return res

        for bb, det_conf in zip(xyxy, confs):
            x1, y1, x2, y2 = bb
            crop = cv2.cvtColor(
                cv2.resize(img[y1:y2, x1:x2], (CFG['w'], CFG['h'])),
                cv2.COLOR_BGR2RGB
            )
            emb = self.reg.ext([crop]).cpu().detach().numpy().flatten()

            best, bd = None, 1e9
            for nm, d in self.reg.gal.items():
                dist = self.l2(emb, d['mean'])
                if dist < bd:
                    bd = dist
                    best = nm

            known = bd < self.thr
            cid = best if known else 'Unknown'
            conf = 1.0 - bd / self.thr if known else 0.0
            res.append({
                'id': cid,
                'conf': conf,
                'dist': bd,
                'bbox': [x1, y1, x2, y2],
                'det_conf': float(det_conf),
            })
        return res

    def draw(self, img, res):
        v = img.copy()
        for r in res:
            x1, y1, x2, y2 = r['bbox']
            color = (0, 255, 0) if r['id'] != 'Unknown' else (0, 0, 255)
            cv2.rectangle(v, (x1, y1), (x2, y2), color, 3)
            label = f"{r['id']} {r['conf']:.2f}"
            cv2.putText(v, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return v


rec = Recognizer(reg, yolo)
print('Recognizer ready')

# Register gallery cows from the processed gallery directory
gallery_imgs = glob.glob(f'{PROC}/gallery/*.jpg')
gallery_cows = {}
for p in gallery_imgs:
    parts = os.path.basename(p).split('_')
    try:
        cid = int(parts[1][1:])
        gallery_cows.setdefault(cid, []).append(p)
    except (IndexError, ValueError):
        pass

for cid, ps in gallery_cows.items():
    reg.register(f'Cow_{cid:03d}', ps[:10])

print(f'Registered {len(reg.names())} cows: {reg.names()[:10]}...')
print(f'Unregistered query cows will appear as "Unknown" — that\'s correct!')

import matplotlib.pyplot as plt

# Test on a query image (cow not in gallery — should be "Unknown")
query_imgs = glob.glob(f'{PROC}/query/*.jpg')
if query_imgs:
    test = query_imgs[0]
    im = cv2.imread(test)
    res = rec.run(im)
    vis = rec.draw(im, res)

    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f'Query image: {os.path.basename(test)}')
    plt.show()

    for r in res:
        print(f"  {r['id']}: conf={r['conf']:.2f}  dist={r['dist']:.3f}  det_conf={r['det_conf']:.2f}")
else:
    print('No query images found — run the processing cell first')

import onnx

# Find best checkpoint
cks = glob.glob(f'{PROJ}/logs/{CFG["name"]}/model/model.pth.tar-*')
if cks:
    ck = max(cks, key=os.path.getctime)
    m = torchreid.models.build_model(
        name=CFG['name'], num_classes=dm.num_train_pids
    ).to(device)
    torchreid.utils.load_pretrained_weights(m, ck)
    m.eval()

    onnx_p = f'{PROJ}/models/cattle_reid.onnx'
    # Plain tensor — no deprecated Variable
    dummy = torch.randn(1, 3, CFG['h'], CFG['w']).to(device)
    torch.onnx.export(
        m,
        dummy,
        onnx_p,
        input_names=['input'],
        output_names=['output'],
    )
    onnx.checker.check_model(onnx.load(onnx_p))
    print(f'✅ ONNX exported and verified: {onnx_p}')
else:
    print('No checkpoint found yet — train the model first')
