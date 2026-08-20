#!/usr/bin/env python3
"""Shared building blocks for the HanwooReID ViT + PHE pipeline.

Single source of truth for model definition, preprocessing, heatmap
generation, evaluation and checkpoint I/O. Used by train_v3.py,
test_model_v3.py, video_reid_v5.py and export.py so that fixes land in
one place instead of six copies.
"""
import os
import re
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler
import timm

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data" / "processed"
MODELS_DIR = REPO / "models"
LOGS_DIR = REPO / "logs"
META_PATH = REPO / "data" / "meta.json"

DEFAULT_BACKBONE = "vit_base_patch16_224"
DEFAULT_IMG_SIZE = 256

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

for d in [MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def parse_identity(filename: str):
    m = re.match(r'c0_p(\d+)_', filename)
    return int(m.group(1)) if m else None


def load_split(split_dir: str):
    """Return [(abs_path, raw_pid), ...] for a train/query/gallery split."""
    items = []
    split_path = DATA_DIR / split_dir
    if not split_path.exists():
        return items
    for fname in sorted(os.listdir(split_path)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        pid = parse_identity(fname)
        if pid is not None:
            items.append((str(split_path / fname), pid))
    return items


def load_meta():
    if not META_PATH.exists():
        return {}
    with open(META_PATH) as f:
        meta = json.load(f)
    return {m["image"]: m.get("keypoints") for m in meta}


def count_identities() -> int:
    """Number of identities across all splits (train/query/gallery)."""
    return len(set(pid for split in ("train", "query", "gallery")
                   for _, pid in load_split(split)))


# ---------------------------------------------------------------------------
# Heatmaps (grid precomputed once — previously rebuilt per image)
# ---------------------------------------------------------------------------

GRID, SIGMA = 16, 1.2
_GRID_XX, _GRID_YY = np.meshgrid(
    np.arange(GRID, dtype=np.float32),
    np.arange(GRID, dtype=np.float32),
)


def make_heatmap(kpts):
    """Normalized keypoints (x, y, conf) -> 16x16 Gaussian heatmap."""
    hm = np.zeros((GRID, GRID), np.float32)
    if not kpts:
        return hm
    for (x, y, c) in kpts:
        if c < 0.25 or not (0 <= x <= 1 and 0 <= y <= 1):
            continue
        gx, gy = x * (GRID - 1), y * (GRID - 1)
        hm = np.maximum(hm, np.exp(-((( _GRID_XX - gx) ** 2 + (_GRID_YY - gy) ** 2) / (2 * SIGMA ** 2))))
    return hm


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def letterbox_crop(img_bgr, target_h=256, target_w=256):
    """Aspect-ratio preserving resize with gray (128) padding.

    Gray padding matches the train-time augmentation (scale to 288, pad 128),
    so inference preprocessing is consistent with training.
    """
    h0, w0 = img_bgr.shape[:2]
    if h0 == 0 or w0 == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = cv2.resize(img_bgr, (new_w, new_h))
    out = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    out[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return out


def prep_crop(img_bgr, target_h=256, target_w=256):
    """BGR crop -> normalized (mean-0.5) RGB CHW float32 tensor."""
    img = letterbox_crop(img_bgr, target_h, target_w)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    return torch.from_numpy(np.transpose(img, (2, 0, 1)).copy())


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class PHE(nn.Module):
    """Pose-Guided Heatmap Encoder: 16x16 heatmap -> per-patch attention prior."""

    def __init__(self, dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 1),
        )

    def forward(self, hm):
        return self.enc(hm)


class HanwooReID(nn.Module):
    """ViT backbone + PHE + BN embedding head + linear classifier."""

    def __init__(self, ncls, img_size=DEFAULT_IMG_SIZE,
                 backbone=DEFAULT_BACKBONE, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=16,
            num_classes=0,
            global_pool='',
            fc_norm=False,
        )
        self.phe = PHE(self.backbone.embed_dim)
        self.bn = nn.BatchNorm1d(self.backbone.embed_dim)
        self.classifier = nn.Linear(self.backbone.embed_dim, ncls, bias=False)

    def forward(self, x, hm=None):
        feats = self.backbone.forward_features(x)
        if hm is not None:
            E = self.phe(hm).flatten(2).transpose(1, 2)
            feats[:, 1:] = feats[:, 1:] + E
        emb = self.bn(feats[:, 0])
        logits = self.classifier(emb)
        return emb, logits


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def infer_ncls(state_dict) -> int:
    """Number of classes, read from the classifier head (never guess from data)."""
    for key in ("classifier.weight", "classifier.bias"):
        if key in state_dict:
            return int(state_dict[key].shape[0])
    raise ValueError("classifier weights not found in checkpoint state dict")


def save_checkpoint(model, args, epoch, metrics, tag):
    state = {k.replace("_orig_mod.", ""): v for k, v in model.state_dict().items()}
    path = MODELS_DIR / f"hanwoo_reid_{tag}.pth"
    torch.save({
        "state": state,
        "epoch": epoch,
        "metrics": metrics,
        "config": vars(args),
    }, path)
    return path


def load_checkpoint(path, device="cuda", img_size=DEFAULT_IMG_SIZE,
                    backbone=DEFAULT_BACKBONE):
    """Load a checkpoint; class count is inferred from the head weights.

    Returns (model, checkpoint_dict). Model is in eval mode.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "state" in ckpt:
        state = ckpt["state"]
    else:
        state = ckpt
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    ncls = infer_ncls(state)
    model = HanwooReID(ncls, img_size=img_size, backbone=backbone,
                       pretrained=False).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, ckpt


# ---------------------------------------------------------------------------
# Loaders & evaluation
# ---------------------------------------------------------------------------

def make_loader(ds, batch_size=128, shuffle=False, num_workers=4, drop_last=False,
                sampler=None):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, drop_last=drop_last,
                      sampler=sampler)


@torch.no_grad()
def embed_all(model, data, device, batch_size=128):
    """Embed a Dataset or DataLoader. Returns (embeddings, pids)."""
    ld = data if isinstance(data, DataLoader) \
        else make_loader(data, batch_size=batch_size)
    model.eval()
    embeddings, pids = [], []
    for x, hm, pid in ld:
        x, hm = x.to(device), hm.to(device)
        emb, _ = model(x, hm)
        embeddings.append(F.normalize(emb, dim=1).cpu().numpy())
        pids.append(pid.numpy())
    if not embeddings:
        return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64)
    return np.vstack(embeddings), np.concatenate(pids)


@torch.no_grad()
def embed_no_phe(model, data, device, batch_size=128):
    """Embed without the pose branch (PHE ablation)."""
    ld = data if isinstance(data, DataLoader) \
        else make_loader(data, batch_size=batch_size)
    model.eval()
    embeddings, pids = [], []
    for x, hm, pid in ld:
        emb, _ = model(x.to(device), None)
        embeddings.append(F.normalize(emb, dim=1).cpu().numpy())
        pids.append(pid.numpy())
    if not embeddings:
        return np.zeros((0, 0), np.float32), np.zeros((0,), np.int64)
    return np.vstack(embeddings), np.concatenate(pids)


def rank_metrics(qe, qp, ge, gp, ranks=(1, 5, 10)):
    """Market-1501 style mAP + Rank-k on cosine similarity."""
    n_total = len(qp)
    if n_total == 0 or ge.shape[0] == 0:
        out = {"mAP": 0.0, "n_eval": 0, "n_total": n_total}
        out.update({f"Rank-{r}": 0.0 for r in ranks})
        return out

    sc = qe @ ge.T
    mAPs, n = [], 0
    hits = {r: 0 for r in ranks}
    for i in range(n_total):
        qi = qp[i]
        order = np.argsort(-sc[i])
        gal = gp[order]
        pos = np.where(gal == qi)[0]
        if len(pos) == 0:
            continue
        n += 1
        ap, nc = 0.0, 0
        for rnk, p in enumerate(pos):
            nc += 1
            ap += nc / (rnk + 1)
        mAPs.append(ap / len(pos))
        for r in ranks:
            if (pos < r).any():
                hits[r] += 1

    return {
        "mAP": float(np.mean(mAPs)) if mAPs else 0.0,
        **{f"Rank-{r}": hits[r] / n if n else 0.0 for r in ranks},
        "n_eval": n,
        "n_total": n_total,
    }


def evaluate(model, query_ds, gallery_ds, device, batch_size=128):
    qe, qp = embed_all(model, query_ds, device, batch_size)
    ge, gp = embed_all(model, gallery_ds, device, batch_size)
    return rank_metrics(qe, qp, ge, gp)


def evaluate_no_phe(model, query_ds, gallery_ds, device, batch_size=128):
    qe, qp = embed_no_phe(model, query_ds, device, batch_size)
    ge, gp = embed_no_phe(model, gallery_ds, device, batch_size)
    return rank_metrics(qe, qp, ge, gp, ranks=(1, 5))


@torch.no_grad()
def compute_eval_loss(model, data, device, label_smoothing=0.0):
    model.eval()
    ld = data if isinstance(data, DataLoader) \
        else make_loader(data, batch_size=128)
    total_loss, n = 0.0, 0
    for x, hm, pid in ld:
        x, hm, pid = x.to(device), hm.to(device), pid.to(device)
        emb, logits = model(x, hm)
        loss = F.cross_entropy(logits, pid, label_smoothing=label_smoothing)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


# ---------------------------------------------------------------------------
# Losses & sampling
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """Focal Loss for hard examples."""

    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                  label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def hard_triplet_loss(emb, labels, margin=0.3, hard_neg_mining=True):
    """Triplet loss with optional semi-hard negative mining."""
    d = torch.cdist(emb, emb)
    pos = labels[:, None] == labels[None, :]
    ap = d.masked_fill(~pos, -1e9).max(1).values
    if hard_neg_mining:
        neg_mask = ~pos
        violations = d < ap.unsqueeze(1) + margin
        semi_hard = neg_mask & violations
        if semi_hard.any():
            an = torch.where(semi_hard, d, d.new_tensor(1e9)).min(1).values
        else:
            an = d.masked_fill(pos, 1e9).min(1).values
    else:
        an = d.masked_fill(pos, 1e9).min(1).values
    return F.relu(ap - an + margin).mean()


class PKSampler(Sampler):
    """P identities x K instances per batch (identity-balanced sampling)."""

    def __init__(self, items, id2label, p=16, k=4):
        self.idx_map = defaultdict(list)
        for i, (_, raw_pid) in enumerate(items):
            label = id2label[raw_pid]
            self.idx_map[label].append(i)
        self.labels = list(self.idx_map.keys())
        self.p, self.k = p, k
        self.nbatches = max(20, min(300, len(items) // (p * k)))

    def __iter__(self):
        for _ in range(self.nbatches):
            sampled = np.random.choice(self.labels, min(self.p, len(self.labels)),
                                       replace=False)
            for label in sampled:
                yield from np.random.choice(self.idx_map[label], self.k,
                                            replace=True)

    def __len__(self):
        return self.nbatches * self.p * self.k