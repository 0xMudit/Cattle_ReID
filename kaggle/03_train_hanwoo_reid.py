!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
!pip -q install timm pillow matplotlib
import os, glob, json, random, math
import numpy as np, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
import timm
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())

# New Kaggle mount nests inputs under /kaggle/input/datasets/<owner>/<ds>.
# ROOT = the dataset dir that contains crops/ or labeled/
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
print('dataset root:', ROOT)

# ---- load metadata (keypoints per image, normalized 0..1) if present ----
META = {}
for mp in sorted(set(glob.glob('/kaggle/input/**/meta.json', recursive=True) + glob.glob('/kaggle/working/**/meta.json', recursive=True))):
    for m in json.load(open(mp)):
        META[m['image']] = m['keypoints']
print('meta.json loaded, images with kpts:', len(META))

# ---- build items: (relpath, identity) with an identity split ----
def walk(root):
    out = []
    for d in sorted(glob.glob(os.path.join(root, '**'), recursive=True)):
        if not os.path.isdir(d): continue
        imgs = sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.png')))
        if imgs:
            ident = os.path.relpath(d, root).replace(os.sep, '/')
            out += [(os.path.relpath(p, ROOT).replace(os.sep, '/'), ident) for p in imgs]
    return out

lab = os.path.join(ROOT, 'labeled')
if os.path.isdir(lab):
    tr = walk(os.path.join(lab, 'train')); ga = walk(os.path.join(lab, 'gallery')); qu = walk(os.path.join(lab, 'query'))
    print('labeled split: train', len(tr), 'gallery', len(ga), 'query', len(qu))
else:
    items = walk(os.path.join(ROOT, 'crops'))
    ids = sorted(set(i for _, i in items))
    random.Random(0).shuffle(ids)
    n = len(ids); na = int(n * 0.7)
    train_ids, eval_ids = set(ids[:na]), ids[na:]
    tr = [(p, i) for p, i in items if i in train_ids]
    # Market-1501 style: eval identities appear in BOTH query and gallery,
    # split by image (same identity, different crops/tracks).
    per = defaultdict(list)
    for p, i in items:
        if i in eval_ids: per[i].append(p)
    ga = []; qu = []
    for i in eval_ids:
        ims = per[i]
        if len(ims) < 2: continue
        random.Random(hash(i) % 2 ** 32).shuffle(ims)
        half = len(ims) // 2
        ga += [(p, i) for p in ims[:half]]
        qu += [(p, i) for p in ims[half:]]
    print('auto-split from crops: train', len(tr), 'gallery', len(ga), 'query', len(qu), '| ids', n)

ID2L = {i: k for k, i in enumerate(sorted(set(i for _, i in items)))}
L2ID = {v: k for k, v in ID2L.items()}
print('all identities:', len(ID2L), '| train:', len(train_ids), 'eval:', len(eval_ids))

GRID, SIGMA = 16, 1.2
def make_heatmap(kpts):
    hm = np.zeros((GRID, GRID), np.float32)
    if not kpts: return hm
    xx, yy = np.meshgrid(np.arange(GRID, dtype=np.float32), np.arange(GRID, dtype=np.float32))
    for (x, y, c) in kpts:
        if c < 0.25 or not (0 <= x <= 1 and 0 <= y <= 1): continue
        gx, gy = x * (GRID - 1), y * (GRID - 1)
        hm = np.maximum(hm, np.exp(-(((xx - gx) ** 2 + (yy - gy) ** 2) / (2 * SIGMA ** 2))))
    return hm

def load_img(path):
    img = cv2.imread(path); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class DS(Dataset):
    def __init__(self, items, train):
        self.items = items; self.train = train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        rel, ident = self.items[i]
        img = load_img(os.path.join(ROOT, rel))
        h, w = img.shape[:2]
        if self.train:
            if max(h, w) > 0:
                s = 288 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            img = cv2.copyMakeBorder(img, 0, max(0, 288 - img.shape[0]), 0, max(0, 288 - img.shape[1]),
                                     cv2.BORDER_CONSTANT, value=(128, 128, 128))
            img = img[:288, :288]
            x = random.randint(0, 32); y = random.randint(0, 32)
            img = img[y:y + 256, x:x + 256]
            if random.random() < 0.5: img = img[:, ::-1]
        else:
            img = cv2.resize(img, (256, 256))
        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        hm = make_heatmap(META.get(rel, None))
        pid = ID2L[ident]
        return torch.from_numpy(img.copy()), torch.from_numpy(hm).unsqueeze(0), pid

class PK(Sampler):
    def __init__(self, items, p=8, k=8):
        self.idx = defaultdict(list)
        for i, (_, pid) in enumerate(items): self.idx[ID2L[pid]].append(i)
        self.pids = list(self.idx); self.p, self.k = p, k
        self.nbatches = max(20, min(200, len(items) // (p * k)))
    def __iter__(self):
        for _ in range(self.nbatches):
            for pid in np.random.choice(self.pids, self.p, replace=False):
                yield from np.random.choice(self.idx[pid], self.k, replace=True)
    def __len__(self): return self.nbatches * self.p * self.k

# ---- PHE + ViT-B/16 (paper architecture, simplified to run on T4) ----
class PHE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(1, dim, 1), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 1))
    def forward(self, hm):
        return self.enc(hm)

class ReID(nn.Module):
    def __init__(self, ncls):
        super().__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=True, img_size=256,
                                          patch_size=16, num_classes=0, global_pool='', fc_norm=False)
        self.phe = PHE(self.backbone.embed_dim)
        self.bn = nn.BatchNorm1d(self.backbone.embed_dim)
        self.classifier = nn.Linear(self.backbone.embed_dim, ncls, bias=False)
    def forward(self, x, hm=None):
        feats = self.backbone.forward_features(x)            # B, 1+256, D
        if hm is not None:
            E = self.phe(hm).flatten(2).transpose(1, 2)      # B,256,D
            feats[:, 1:] = feats[:, 1:] + E                  # attention prior
        emb = self.bn(feats[:, 0])
        return emb, self.classifier(emb)

def hard_triplet(emb, labels, margin=0.3):
    d = torch.cdist(emb, emb)
    pos = labels[:, None] == labels[None, :]
    ap = d.masked_fill(~pos, -1e9).max(1).values
    an = d.masked_fill(pos, 1e9).min(1).values
    return F.relu(ap - an + margin).mean()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = ReID(len(ID2L)).to(device)
print('trainable params:', sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---- Market-1501 style evaluation (used during training) ----
@torch.no_grad()
def embed(model, ds, use_hm=True):
    model.eval(); E = []; P = []
    ld = DataLoader(ds, batch_size=64, num_workers=4)
    for x, hm, pid in ld:
        x, hm = x.to(device), hm.to(device)
        emb, _ = model(x, hm if use_hm else None)
        E.append(F.normalize(emb, dim=1).cpu().numpy()); P.append(pid.numpy())
    return np.vstack(E), np.concatenate(P)

def evaluate(model, qu_ds, ga_ds, g_ids=None, use_hm=True):
    qe, qp = embed(model, qu_ds, use_hm)
    ge, gp = embed(model, ga_ds, use_hm)
    sc = qe @ ge.T
    mAPs = []; r1 = r5 = 0; n = 0
    for i in range(len(qp)):
        qi = qp[i]
        if g_ids is not None:                   # VCR: only consider allowed gallery indices
            keep = np.array(g_ids[i]); gal = gp[keep][np.argsort(-sc[i][keep])]
        else:
            order = np.argsort(-sc[i]); gal = gp[order]
        pos = np.where(gal == qi)[0]
        if len(pos) == 0: continue
        ap = 0.0; nc = 0
        for rnk, p in enumerate(pos):
            nc += 1; ap += nc / (rnk + 1)
        mAPs.append(ap / len(pos)); n += 1
        if pos[0] == 0: r1 += 1
        if (pos < 5).any(): r5 += 1
    return (np.mean(mAPs) if mAPs else 0.0), (r1 / n if n else 0.0), (r5 / n if n else 0.0)

EPOCHS, BS, LR = 60, 64, 1e-3
WARMUP, MARGIN, WEIGHT_CE = 5, 0.3, 1.0
tr_ds = DS(tr, True); ga_ds = DS(ga, False); qu_ds = DS(qu, False)
tr_ld = DataLoader(tr_ds, batch_size=BS, sampler=PK(tr, 8, 8), num_workers=4, drop_last=True)

opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[30, 45], gamma=0.1)
import time
for ep in range(1, EPOCHS + 1):
    if ep <= WARMUP: opt.param_groups[0]['lr'] = LR * ep / WARMUP
    model.train(); t0 = time.time(); tl = 0.0; nb = 0
    for x, hm, pid in tr_ld:
        x, hm, pid = x.to(device), hm.to(device), pid.to(device)
        emb, logits = model(x, hm)
        ce = F.cross_entropy(logits, pid)
        trip = hard_triplet(emb, pid, MARGIN)
        loss = WEIGHT_CE * ce + trip
        opt.zero_grad(); loss.backward(); opt.step()
        tl += loss.item(); nb += 1
    sched.step()
    if ep % 5 == 0 or ep == EPOCHS:
        mAP, r1, r5 = evaluate(model, qu_ds, ga_ds)
        print(f'ep {ep:3d} loss {tl/max(nb,1):.3f} lr {opt.param_groups[0]["lr"]:.5f} '
              f'| mAP {mAP*100:.1f}  R1 {r1*100:.1f}  R5 {r5*100:.1f}  ({time.time()-t0:.0f}s)', flush=True)
    else:
        print(f'ep {ep:3d} loss {tl/max(nb,1):.3f} ({time.time()-t0:.0f}s)')

# ---- Final evaluation + save ----
mAP, r1, r5 = evaluate(model, qu_ds, ga_ds)
print(f'FINAL (PHE):  mAP {mAP*100:.2f}  Rank-1 {r1*100:.2f}  Rank-5 {r5*100:.2f}')
mAP0, r10, r50 = evaluate(model, qu_ds, ga_ds, use_hm=False)
print(f'FINAL (no PHE): mAP {mAP0*100:.2f}  Rank-1 {r10*100:.2f}  Rank-5 {r50*100:.2f}')

torch.save({'state': model.state_dict(), 'id2l': ID2L, 'l2id': L2ID,
            'items': {'gallery': ga, 'query': qu, 'train': tr}},
           '/kaggle/working/hanwoo_reid.pth')
import json
json.dump({'phe_mAP': round(float(mAP), 4), 'phe_r1': round(float(r1), 4),
           'phe_r5': round(float(r5), 4), 'no_phe_mAP': round(float(mAP0), 4)},
          open('/kaggle/working/report.json', 'w'), indent=2)
print('saved /kaggle/working/hanwoo_reid.pth + report.json  (download from Output tab)')
