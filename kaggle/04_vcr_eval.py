!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
!pip -q install timm pillow
import os, glob, json, math
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm

import tarfile, shutil
ROOT = None
for d in glob.glob('/kaggle/input/**/', recursive=True):
    if os.path.isdir(os.path.join(d, 'crops')) or os.path.isdir(os.path.join(d, 'labeled')):
        ROOT = d; break
if ROOT is None:
    tar = next((t for t in glob.glob('/kaggle/input/**/crops.tar', recursive=True)), None)
    if tar:
        with tarfile.open(tar) as tf: tf.extractall('/kaggle/working')
        ROOT = '/kaggle/working'
        mp = next((p for p in glob.glob('/kaggle/input/**/meta.json', recursive=True)), None)
        if mp: shutil.copy2(mp, os.path.join(ROOT, 'meta.json'))
MODEL_PATH = None
for p in glob.glob('/kaggle/input/**/hanwoo_reid.pth', recursive=True):
    MODEL_PATH = p; break
print('data root:', ROOT)
print('model:', MODEL_PATH)

META = {}
for mp in sorted(set(glob.glob('/kaggle/input/**/meta.json', recursive=True) + glob.glob('/kaggle/working/**/meta.json', recursive=True))):
    for m in json.load(open(mp)): META[m['image']] = m['keypoints']
    print('meta images:', len(META))

CAL = {}
cp = os.path.join(ROOT, 'calib.json')
if os.path.exists(cp):
    CAL = json.load(open(cp)); print('calib.json videos:', list(CAL))

ck = torch.load(MODEL_PATH, map_location='cpu')
ID2L = ck['id2l']; L2ID = ck['l2id']
ga, qu = ck['items']['gallery'], ck['items']['query']
print('gallery items:', len(ga), 'query items:', len(qu), 'ids:', len(ID2L))

HOOF_F, HOOF_R = [4, 5], [6, 7]   # LF/RF front, LB/RB rear

def view_id_image(kpts):
    # Fallback: orientation from hoof midpoints in image plane, 4 bins.
    if not kpts: return None
    k = np.array(kpts)
    if k.shape[0] < 8 or (k[:, 2] < 0.25).any(): return None
    front = k[HOOF_F, :2].mean(0); rear = k[HOOF_R, :2].mean(0)
    v = front - rear
    ang = math.degrees(math.atan2(v[1], v[0])) % 360
    return int(ang // 90)

def view_id_bev(kpts, cal):
    # Paper VCR: project hooves to BEV via homography K@[R[:,:2]|t], then angle.
    if not kpts or not cal: return None
    k = np.array(kpts)
    if k.shape[0] < 8 or (k[:, 2] < 0.25).any(): return None
    K = np.array(cal['K']); R = np.array(cal['R']); t = np.array(cal['t']).reshape(3, 1)
    w, h = cal.get('width', 2880), cal.get('height', 1620)
    H = K @ np.hstack([R[:, :2], t])                     # world(XY0) -> image px
    Hi = np.linalg.inv(H)
    pts = []
    for idx in HOOF_F + HOOF_R:
        u, v = k[idx, 0] * w, k[idx, 1] * h              # normalized -> pixels
        X = Hi @ np.array([u, v, 1.0])
        pts.append((X[0] / X[2], X[1] / X[2]))
    front = np.mean(pts[:2], 0); rear = np.mean(pts[2:], 0)
    vv = front - rear
    ang = math.degrees(math.atan2(vv[1], vv[0])) % 360
    return int(ang // 90)

def get_view(rel, use_cal=True):
    kpts = META.get(rel, None)
    parts = rel.split('/')
    vid = parts[1] if len(parts) > 1 and parts[0] == 'crops' else parts[0]
    if use_cal and vid in CAL:
        return view_id_bev(kpts, CAL[vid])
    return view_id_image(kpts)

# ---- Reuse the model from 03 (same architecture) ----
class PHE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, dim, 1), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 1))
    def forward(self, hm): return self.enc(hm)

class ReID(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=256,
                                          patch_size=16, num_classes=0, global_pool='', fc_norm=False)
        self.phe = PHE(self.backbone.embed_dim)
        self.bn = nn.BatchNorm1d(self.backbone.embed_dim)
        self.classifier = nn.Linear(self.backbone.embed_dim, ncls, bias=False)
    def forward(self, x, hm=None):
        feats = self.backbone.forward_features(x)
        if hm is not None:
            E = self.phe(hm).flatten(2).transpose(1, 2)
            feats[:, 1:] = feats[:, 1:] + E
        return self.bn(feats[:, 0]), self.classifier(self.bn(feats[:, 0]))

def make_heatmap(kpts):
    hm = np.zeros((16, 16), np.float32)
    if kpts:
        xx, yy = np.meshgrid(np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32))
        for (x, y, c) in kpts:
            if c < 0.25 or not (0 <= x <= 1 and 0 <= y <= 1): continue
            gx, gy = x * 15, y * 15
            hm = np.maximum(hm, np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * 1.2 ** 2))))
    return hm

class DS(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rel, ident = self.items[i]
        img = cv2.imread(os.path.join(ROOT, rel))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        hm = make_heatmap(META.get(rel, None))
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy()), torch.from_numpy(hm).unsqueeze(0), ID2L[ident]

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ReID(len(ID2L)).to(device)
model.load_state_dict(ck['state'])
model.eval()
print('model loaded')

@torch.no_grad()
def embed(ds):
    E = []; P = []
    for x, hm, pid in DataLoader(ds, batch_size=64, num_workers=4):
        x, hm = x.to(device), hm.to(device)
        emb, _ = model(x, hm)
        E.append(F.normalize(emb, dim=1).cpu().numpy()); P.append(pid.numpy())
    return np.vstack(E), np.concatenate(P)

qe, qp = embed(DS(qu))
ge, gp = embed(DS(ga))
sc = qe @ ge.T
qv_all = [get_view(rel) for rel, _ in qu]
gv_all = [get_view(rel) for rel, _ in ga]

def eval_vcr(qv_all, gv_all, relaxed=0):
    mAPs = []; r1 = r5 = 0; n = 0
    for i in range(len(qp)):
        qi = qp[i]; qv = qv_all[i]
        if qv is None:
            keep = np.arange(len(ga))
        else:
            keep = np.array([j for j, gv in enumerate(gv_all)
                             if gv is not None and abs(gv - qv) <= relaxed])
        if len(keep) == 0: continue
        gal = gp[keep][np.argsort(-sc[i][keep])]
        pos = np.where(gal == qi)[0]
        if len(pos) == 0: continue
        ap = 0.0; nc = 0
        for rnk, p in enumerate(pos):
            nc += 1; ap += nc / (rnk + 1)
        mAPs.append(ap / len(pos)); n += 1
        if pos[0] == 0: r1 += 1
        if (pos < 5).any(): r5 += 1
    return (np.mean(mAPs) if mAPs else 0), (r1 / n if n else 0), (r5 / n if n else 0)

print(f"{'setting':<20} {'mAP':>8} {'R1':>8} {'R5':>8}   queries-evaluated")
for name, relaxed in [('no VCR', 99), ('VCR same bin', 0), ('VCR +-1 bin', 1)]:
    m, r1_, r5_ = eval_vcr(qv_all, gv_all, relaxed)
    print(f"{name:<20} {m*100:7.1f}% {r1_*100:7.1f}% {r5_*100:7.1f}%")

# Visualize a few ViewIDs: show crops grouped by bin for one video
import matplotlib.pyplot as plt
video0 = ga[0][0].split('/')[0] if ga else None
bins = {}
for rel, _ in ga[:2000]:
    if rel.split('/')[0] != video0: continue
    v = get_view(rel)
    if v is not None: bins.setdefault(v, []).append(rel)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for b in range(4):
    rels = bins.get(b, [])[:6]
    ax = axes[b]
    if rels:
        imgs = [cv2.cvtColor(cv2.imread(os.path.join(ROOT, r)), cv2.COLOR_BGR2RGB) for r in rels[:3]]
        imgs = [im for im in imgs if im is not None]
        if imgs:
            h = min(im.shape[0] for im in imgs)
            imgs = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in imgs]
            ax.imshow(np.hstack(imgs))
    ax.set_title(f'ViewID {b}'); ax.axis('off')
plt.tight_layout(); plt.show()
print('If each row looks like a consistent viewpoint, VCR is filtering correctly.')
