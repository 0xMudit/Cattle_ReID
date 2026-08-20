!pip -q install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
!pip -q install timm ultralytics opencv-python-headless pillow numpy matplotlib
import os, glob, json, math, time
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm, torch.nn.functional as F
from ultralytics import YOLO
import matplotlib.pyplot as plt
print('torch', torch.__version__, '| cuda', torch.cuda.is_available())

# ---- find inputs (new Kaggle mount nests under /kaggle/input/datasets/<owner>/<ds>) ----
VIDEOS = sorted(glob.glob('/kaggle/input/**/*.mp4', recursive=True))
print('videos:', len(VIDEOS))
for v in VIDEOS: print('  ', os.path.basename(v), round(os.path.getsize(v)/1e6,1), 'MB')

def find_input(fname, root=None):
    m = glob.glob(os.path.join(root, '**', fname) if root else '/kaggle/input/**/' + fname, recursive=True)
    return m[0] if m else None

det  = YOLO(find_input('yolov8n.pt')  or 'yolov8n.pt')
pose = YOLO(find_input('cow_pose.pt') or 'cow_pose.pt')
ckpt_path = find_input('hanwoo_reid.pth')

ROOT = None
for d in glob.glob('/kaggle/input/**/', recursive=True):
    if os.path.isdir(os.path.join(d, 'crops')): ROOT = d; break
print('crops root:', ROOT)

META = {}
for mp in glob.glob('/kaggle/input/**/meta.json', recursive=True):
    for m in json.load(open(mp)): META[m['image']] = m['keypoints']
print('meta keypoints:', len(META))

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
        emb = self.bn(feats[:, 0])
        return emb, self.classifier(emb)

def prep_crop(img):
    img = cv2.resize(img, (256, 256)).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ck = torch.load(ckpt_path, map_location='cpu')
model = ReID(len(ck['id2l'])).to(device).eval()
model.load_state_dict(ck['state'])
print('ReID loaded:', sum(p.numel() for p in model.parameters())/1e6, 'M params')
print('gallery ids:', len(set(i for _, i in ck['items']['gallery'])), '| crops:', len(ck['items']['gallery']))

# ALL 30 identities (train 21 + eval 9) -> every cow in the video has a real
# gallery entry, so matching gives a UNIQUE label instead of force-matching to
# one of the 6 eval IDs.
gal_items = ck['items']['train'] + ck['items']['gallery'] + ck['items']['query']
gallery_paths = [os.path.join(ROOT, r) for r, _ in gal_items]
gallery_ids = [i for _, i in gal_items]
gal_emb = embed_paths(gallery_paths)
print('gallery embeddings:', gal_emb.shape, '| identities:', len(set(gallery_ids)))
id_short = {i: i.split('/')[-1] for i in set(gallery_ids)}
print('known identities:', sorted(id_short.values()))

def reid_embed(img, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = img[max(0,y1):y2, max(0,x1):x2]
    if crop.size == 0: return None, None, None
    if POSE_ENABLED:
        kp, vis = cow_pose_hm(img, box)
        hm = make_heatmap([(kp[j,0]/img.shape[1], kp[j,1]/img.shape[0], float(vis[j])) for j in range(12)] if kp is not None else None)
    else:
        hm = make_heatmap(None)
    with torch.no_grad():
        x = prep_crop(crop).unsqueeze(0).to(device)
        hm = torch.from_numpy(hm).unsqueeze(0).unsqueeze(0).to(device)
        emb, _ = model(x, hm)
        emb = F.normalize(emb, dim=1).cpu().numpy()[0]
    sim = emb @ gal_emb.T
    j = int(np.argmax(sim))
    return emb, gallery_ids[j], float(sim[j])

def ema_match(emb, prev_emb, alpha=0.85):
    # temporal smoothing: blend track embedding, then re-match -> stable label
    e = alpha * prev_emb + (1 - alpha) * emb
    e = e / (np.linalg.norm(e) + 1e-9)
    sim = e @ gal_emb.T
    j = int(np.argmax(sim))
    return e, gallery_ids[j], float(sim[j])

        for tid, bi in assign:
            emb, ident, score = reid_embed(frame, cows[bi])
            prev = tracker.reid.get(tid)
            if emb is not None:
                if prev is None or prev.get('emb') is None:
                    tracker.reid[tid] = {'emb': emb, 'label': id_short[ident] if score >= REID_CONF else 'unknown', 'score': score}
                else:
                    e, ident, score = ema_match(emb, prev['emb'])
                    tracker.reid[tid] = {'emb': e, 'label': id_short[ident] if score >= REID_CONF else 'unknown', 'score': score}
            else:
                tracker.reid.setdefault(tid, {'emb': None, 'label': 'unknown', 'score': None})
            draw_track(frame, tid, bi, cows[bi])

# ---- save + show preview frames ----
cap = cv2.VideoCapture(out_name)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
for k, ax in enumerate(axes.ravel()):
    cap.read()
    ok, f = cap.read()
    if not ok: break
    ax.imshow(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)); ax.axis('off')
    cv2.imwrite(f'/kaggle/working/preview_{k+1}.png', f)
cap.release()
plt.tight_layout(); plt.savefig('/kaggle/working/preview_mosaic.png', dpi=100, bbox_inches='tight')
plt.show()
print('preview PNGs saved to /kaggle/working')
