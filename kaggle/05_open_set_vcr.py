!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
!pip -q install timm pillow
import os, glob, json, math, random
import numpy as np, cv2
from collections import defaultdict
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

ck = torch.load(MODEL_PATH, map_location='cpu')
ID2L, L2ID = ck['id2l'], ck['l2id']
ga, qu, tr = ck['items']['gallery'], ck['items']['query'], ck['items']['train']

# Unknown queries: balanced sample of the model's TRAIN identities (no gallery entry).
rng = random.Random(42)
byid = defaultdict(list)
for p, i in tr: byid[i].append(p)
unk = []
for i in sorted(byid):
    ims = byid[i][:]
    rng.shuffle(ims)
    unk += [(p, i) for p in ims[:20]]
print('gallery items:', len(ga), 'known queries:', len(qu), 'unknown queries:', len(unk))
print('known identities (in gallery):', len(set(i for _, i in ga)),
      '| unknown identities (no gallery entry):', len(set(i for _, i in unk)))

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

CAL = {}
cp = os.path.join(ROOT, 'calib.json')
if os.path.exists(cp):
    CAL = json.load(open(cp)); print('calib.json videos:', list(CAL))

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
print('model loaded on', device)

@torch.no_grad()
def embed(ds):
    E = []; P = []
    for x, hm, pid in DataLoader(ds, batch_size=64, num_workers=4):
        x, hm = x.to(device), hm.to(device)
        emb, _ = model(x, hm)
        E.append(F.normalize(emb, dim=1).cpu().numpy()); P.append(pid.numpy())
    return np.vstack(E), np.concatenate(P)

qe, qp = embed(DS(qu))      # known queries
ue, up = embed(DS(unk))     # unknown queries
ge, gp = embed(DS(ga))      # gallery
sc  = qe @ ge.T
usc = ue @ ge.T
qv_all = [get_view(rel) for rel, _ in qu]
gv_all = [get_view(rel) for rel, _ in ga]
uv_all = [get_view(rel) for rel, _ in unk]
print('embedded: known', qe.shape, 'unknown', ue.shape, 'gallery', ge.shape)
print('view coverage  gallery: %.0f%%  known-query: %.0f%%  unknown-query: %.0f%%' % (
    100 * np.mean([v is not None for v in gv_all]),
    100 * np.mean([v is not None for v in qv_all]),
    100 * np.mean([v is not None for v in uv_all])))

def filter_keep(scmat_i, qv, gv_all, relaxed):
    if qv is None:
        return np.arange(len(gv_all))
    return np.array([j for j, gv in enumerate(gv_all)
                     if gv is not None and abs(gv - qv) <= relaxed])

def eval_known(scmat, qp, qv_all, gv_all, relaxed):
    mAPs = []; r1 = r5 = 0; n = 0
    for i in range(len(qp)):
        qi = qp[i]
        keep = filter_keep(scmat[i], qv_all[i], gv_all, relaxed)
        if len(keep) == 0: continue
        gal = gp[keep][np.argsort(-scmat[i][keep])]
        pos = np.where(gal == qi)[0]
        if len(pos) == 0: continue
        ap = 0.0; nc = 0
        for rnk, p in enumerate(pos):
            nc += 1; ap += nc / (rnk + 1)
        mAPs.append(ap / len(pos)); n += 1
        if pos[0] == 0: r1 += 1
        if (pos < 5).any(): r5 += 1
    return (np.mean(mAPs) if mAPs else 0), (r1 / n if n else 0), (r5 / n if n else 0)

print(f"{'setting':<16} {'mAP':>8} {'R1':>8} {'R5':>8}   (closed-set, known queries)")
cs = {}
for name, relaxed in [('no VCR', 99), ('VCR same bin', 0), ('VCR +-1 bin', 1)]:
    m, r1_, r5_ = eval_known(sc, qp, qv_all, gv_all, relaxed)
    cs[name] = [round(m, 4), round(r1_, 4), round(r5_, 4)]
    print(f"{name:<16} {m*100:7.1f}% {r1_*100:7.1f}% {r5_*100:7.1f}%")

def top1_scores(scmat, qv_all, gv_all, relaxed):
    # top-1 similarity after view-constraining the gallery. No valid candidate -> -inf
    # (always rejected, never retrieved).
    out = np.full(scmat.shape[0], -np.inf)
    for i in range(scmat.shape[0]):
        keep = filter_keep(scmat[i], qv_all[i], gv_all, relaxed)
        if len(keep) == 0: continue
        out[i] = float(scmat[i][keep].max())
    return out

def auroc(pos, neg):
    # Exact: P(random known score > random unknown score), ties count 0.5.
    pos = np.asarray(pos, dtype=float)[:, None]
    neg = np.asarray(neg, dtype=float)[None, :]
    if pos.size == 0 or neg.size == 0: return 0.5
    gt = (pos > neg).sum()
    eq = (pos == neg).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))

def tpr_at_fpr(pos, neg, fpr_target=0.01):
    # best TPR (known accepted) reachable while FPR (unknown accepted) <= fpr_target.
    ths = np.unique(np.concatenate([pos, neg]))
    ths = ths[np.isfinite(ths)]
    best = 0.0
    for th in ths:
        fp = (neg >= th).mean()
        if fp <= fpr_target:
            best = max(best, (pos >= th).mean())
    return best

def open_mAP(sc_known, sc_unknown, qp, qv_all, uv_all, gv_all, relaxed):
    aps = []
    for i in range(sc_known.shape[0]):
        qi = qp[i]
        keep = filter_keep(sc_known[i], qv_all[i], gv_all, relaxed)
        if len(keep) == 0:
            aps.append(0.0); continue
        gal = gp[keep][np.argsort(-sc_known[i][keep])]
        pos = np.where(gal == qi)[0]
        if len(pos) == 0:
            aps.append(0.0); continue
        ap = 0.0; nc = 0
        for rnk, p in enumerate(pos):
            nc += 1; ap += nc / (rnk + 1)
        aps.append(ap / len(pos))
    aps += [0.0] * sc_unknown.shape[0]      # unknowns have no correct match
    return float(np.mean(aps))

print(f"{'setting':<16} {'open-mAP':>9} {'AUROC':>7} {'TPR@1%FPR':>9}")
rep = {'known_queries': int(len(qu)), 'unknown_queries': int(len(unk)), 'gallery': int(len(ga))}
for name, relaxed in [('no VCR', 99), ('VCR same bin', 0), ('VCR +-1 bin', 1)]:
    ks = top1_scores(sc,  qv_all, gv_all, relaxed)
    us = top1_scores(usc, uv_all, gv_all, relaxed)
    om = open_mAP(sc, usc, qp, qv_all, uv_all, gv_all, relaxed)
    au = auroc(ks, us)
    dr = tpr_at_fpr(ks, us, 0.01)
    print(f"{name:<16} {om*100:8.1f}% {au:7.3f} {dr*100:8.1f}%")
    rep[name.replace(' ','_')] = dict(open_mAP=round(om, 4), auroc=round(au, 4), tpr_at_1pct_fpr=round(dr, 4),
                                      mean_known_top1=float(ks[ks > -1e9].mean()) if np.any(ks > -1e9) else None,
                                      mean_unknown_top1=float(us[us > -1e9].mean()) if np.any(us > -1e9) else None)
for k in cs: rep['closed_' + k.replace(' ','_')] = cs[k]
json.dump(rep, open('/kaggle/working/open_set_report.json', 'w'), indent=2)
print('saved /kaggle/working/open_set_report.json')

# Interpretation
# - open-mAP drops below 100% because unknown queries contribute AP=0 (they must be rejected).
# - AUROC / TPR@1%FPR quantify rejection: how well a top-1-similarity threshold separates
#   known queries from unknown ones.
# - If VCR rows beat no-VCR, view-constraining the gallery is removing confusable candidates.
#   If they tie, the scene/identities are too easy for VCR to matter (like 04's saturated
#   closed-set numbers).
