"""
HanwooReID Training v3 — All 8 improvements, tuned for maximum GPU utilization
(RTX 5080 / 16 GB VRAM / Blackwell sm_120 / 32 cores):

1. Letterbox preprocessing match between train/inference
2. Hard negative mining (semi-hard triplet)
3. Multi-embedding gallery (in video_reid_v5.py)
4. Aggressive augmentation (ColorJitter, RandomErasing, Cutout, GridMask, Mixup)
5. DeepSORT tracking (in video_reid_v5.py)
6. ViT-Large backbone option
7. Focal Loss
8. Synthetic data augmentation

GPU-max settings:
  - AMP (fp16) + GradScaler
  - TF32 matmul/conv + cudnn.benchmark (set in training/reid_common.py)
  - torch.compile (inductor) — guarded, falls back if unsupported
  - Real batch size = PK p*k (16x8 = 128) via identity-balanced PKSampler
  - 16 dataloader workers + pin_memory
  - Optional gradient accumulation
"""
import os
import sys
import json
import time
import random
import math
import argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reid_common import (
    DATA_DIR,
    MODELS_DIR,
    LOGS_DIR,
    DEFAULT_BACKBONE,
    DEFAULT_IMG_SIZE,
    set_seed,
    load_split,
    load_meta,
    make_heatmap,
    letterbox_crop,
    HanwooReID,
    save_checkpoint,
    load_checkpoint,
    evaluate,
    evaluate_no_phe,
    compute_eval_loss,
    hard_triplet_loss,
    FocalLoss,
    PKSampler,
    make_loader,
)

_NUM_WORKERS = min(16, os.cpu_count() or 4)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train HanwooReID v3 (all improvements)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128,
                   help="Nominal batch size (must equal --pk-p * --pk-k; the PK sampler "
                        "defines the real per-step batch)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--margin", type=float, default=0.3)
    p.add_argument("--weight-ce", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-freq", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=_NUM_WORKERS)
    p.add_argument("--pk-p", type=int, default=16, help="Identities per batch")
    p.add_argument("--pk-k", type=int, default=8, help="Instances per identity")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--keep-top", type=int, default=5)
    p.add_argument("--skip-meta", action="store_true")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--backbone", type=str, default=DEFAULT_BACKBONE,
                   choices=["vit_base_patch16_224", "vit_large_patch16_224"],
                   help="ViT backbone size")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile (inductor) for Blackwell GPUs")
    p.add_argument("--focal-loss", action=argparse.BooleanOptionalAction, default=True,
                   help="Use Focal Loss instead of CE")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--hard-neg-mining", action=argparse.BooleanOptionalAction, default=True,
                   help="Semi-hard negative mining in triplet loss")
    p.add_argument("--mixup-alpha", type=float, default=0.4,
                   help="Mixup alpha (0=disabled)")
    p.add_argument("--synth-augment", action=argparse.BooleanOptionalAction, default=True,
                   help="Synthetic augmentation (cutout, gridmask)")
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps (effective batch = bs * accum)")
    return p.parse_args(argv)


class CattleReID(torch.utils.data.Dataset):
    def __init__(self, items, meta_map, id2label, train=True,
                 img_size=DEFAULT_IMG_SIZE, synth_augment=True):
        self.items = items
        self.meta = meta_map
        self.id2label = id2label
        self.train = train
        self.img_size = img_size
        self.synth_augment = synth_augment

    def __len__(self):
        return len(self.items)

    def _strong_aug(self, img):
        # Color jitter (brightness, contrast, saturation)
        if random.random() < 0.5:
            img = img.astype(np.float32)
            b = random.uniform(0.8, 1.2)
            c = random.uniform(0.8, 1.2)
            s = random.uniform(0.8, 1.2)
            img = np.clip(img * c * s + (b - 1) * 128, 0, 255).astype(np.uint8)
        # Random grayscale
        if random.random() < 0.1:
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        # Random erasing (simulate occlusion)
        if random.random() < 0.4:
            h, w = img.shape[:2]
            for _ in range(random.randint(1, 4)):
                eh = random.randint(h // 8, h // 3)
                ew = random.randint(w // 8, w // 3)
                ey = random.randint(0, h - eh)
                ex = random.randint(0, w - ew)
                img[ey:ey + eh, ex:ex + ew] = random.randint(0, 255)
        return img

    def _cutout(self, img, n_holes=1, length=40):
        h, w = img.shape[:2]
        mask = np.ones((h, w), np.float32)
        for _ in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        return (img * mask[:, :, None]).astype(np.uint8)

    def _gridmask(self, img, d=64, ratio=0.5):
        h, w = img.shape[:2]
        mask = np.ones((h, w), np.float32)
        for y in range(0, h, d):
            for x in range(0, w, d):
                if random.random() < ratio:
                    mask[y:y + d // 2, x:x + d // 2] = 0.0
        return (img * mask[:, :, None]).astype(np.uint8)

    def __getitem__(self, i):
        fpath, raw_pid = self.items[i]
        img = cv2.imread(fpath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if self.train:
            # Scale to 288, gray-pad, random crop to 256, random flip
            if max(h, w) > 0:
                s = 288 / max(h, w)
                img = cv2.resize(img, (int(w * s), int(h * s)))
            pad_h = max(0, 288 - img.shape[0])
            pad_w = max(0, 288 - img.shape[1])
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                     cv2.BORDER_CONSTANT, value=(128, 128, 128))
            img = img[:288, :288]
            x = random.randint(0, 32)
            y = random.randint(0, 32)
            img = img[y:y + 256, x:x + 256]
            if random.random() < 0.5:
                img = img[:, ::-1].copy()
            img = self._strong_aug(img)
            if self.synth_augment:
                if random.random() < 0.3:
                    img = self._cutout(img, n_holes=random.randint(1, 3),
                                       length=random.randint(20, 60))
                if random.random() < 0.2:
                    img = self._gridmask(img, d=random.choice([32, 48, 64]),
                                         ratio=random.uniform(0.3, 0.6))
        else:
            img = letterbox_crop(img, self.img_size, self.img_size)

        img = img.astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))

        rel = os.path.relpath(fpath, REPO)
        kpts = self.meta.get(rel)
        hm = make_heatmap(kpts)

        label = self.id2label.get(raw_pid, -1)

        return (torch.from_numpy(img.copy()),
                torch.from_numpy(hm).unsqueeze(0),
                label)


def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam
    return x, y, y, 1.0


def train(args=None):
    if args is None:
        args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"[train] GPU: {torch.cuda.get_device_name(0)} | "
              f"{props.total_memory / 1e9:.1f} GB | sm_{props.major}{props.minor}")

    set_seed(args.seed)

    train_items = load_split("train")
    query_items = load_split("query")
    gallery_items = load_split("gallery")
    print(f"[data] Train: {len(train_items)} | Query: {len(query_items)} | "
          f"Gallery: {len(gallery_items)}")

    meta_map = load_meta()
    print(f"[meta] Loaded {len(meta_map)} entries")

    all_raw_ids = sorted(set(pid for _, pid in train_items + query_items + gallery_items))
    ncls = len(all_raw_ids)
    id2label = {raw_id: label for label, raw_id in enumerate(all_raw_ids)}
    print(f"[data] Total identities: {ncls} (mapped to 0..{ncls-1})")

    tr_ds = CattleReID(train_items, meta_map, id2label, train=True,
                       img_size=args.img_size, synth_augment=args.synth_augment)
    qu_ds = CattleReID(query_items, meta_map, id2label, train=False,
                       img_size=args.img_size)
    ga_ds = CattleReID(gallery_items, meta_map, id2label, train=False,
                       img_size=args.img_size)

    # The PK sampler defines the real per-step batch size (p*k); the DataLoader
    # batch_size must match it so one sampler iteration == one optimizer step.
    eff_bs = args.pk_p * args.pk_k
    if eff_bs != args.batch_size:
        print(f"[train] NOTE: batch_size={args.batch_size} != pk_p*pk_k={eff_bs}; "
              f"using {eff_bs} (effective batch)")
    sampler = PKSampler(train_items, id2label, p=args.pk_p, k=args.pk_k)
    tr_ld = make_loader(tr_ds, batch_size=eff_bs, num_workers=args.num_workers,
                        drop_last=True, sampler=sampler)
    qu_ld = make_loader(qu_ds, batch_size=128, num_workers=min(4, args.num_workers))
    ga_ld = make_loader(ga_ds, batch_size=128, num_workers=min(4, args.num_workers))

    model = HanwooReID(ncls, img_size=args.img_size, backbone=args.backbone).to(device)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {args.backbone} + PHE — {nparams / 1e6:.1f}M trainable params")

    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        model, ckpt_info = load_checkpoint(args.resume, device,
                                           img_size=args.img_size,
                                           backbone=args.backbone)
        start_epoch = ckpt_info.get("epoch", 0) if isinstance(ckpt_info, dict) else 0
        print(f"[train] Resumed from {args.resume} (epoch {start_epoch})")

    use_compile = args.compile and device.type == "cuda"
    if use_compile:
        try:
            model = torch.compile(model)
            print("[train] torch.compile: ON (inductor)")
        except Exception as e:  # pragma: no cover
            use_compile = False
            print(f"[train] torch.compile failed, continuing without: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def cosine_lr(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / args.warmup
        progress = (epoch - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lr)

    use_amp = (not args.no_amp) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    focal_fn = FocalLoss(gamma=args.focal_gamma,
                         label_smoothing=args.label_smoothing) if args.focal_loss else None

    print(f"[train] AMP: {'ON' if use_amp else 'OFF'} | LR: {args.lr} | "
          f"WD: {args.weight_decay} | LabelSmooth: {args.label_smoothing}")
    print(f"[train] Focal Loss: {'ON' if args.focal_loss else 'OFF'} (gamma={args.focal_gamma})")
    print(f"[train] Hard Neg Mining: {'ON' if args.hard_neg_mining else 'OFF'}")
    print(f"[train] Mixup: {'ON' if args.mixup_alpha > 0 else 'OFF'} (alpha={args.mixup_alpha})")
    print(f"[train] Synth Augment: {'ON' if args.synth_augment else 'OFF'}")
    print(f"[train] Grad Accum: {args.grad_accum} | Workers: {args.num_workers} | "
          f"Compile: {'ON' if use_compile else 'OFF'}")

    best_mAP = 0.0
    top_checkpoints = []
    epochs_no_improve = 0

    print(f"\n{'='*70}")
    print(f" Training v3: {args.epochs} epochs | eff BS={eff_bs} | LR={args.lr} | "
          f"PK=({args.pk_p},{args.pk_k}) | EarlyStop={args.early_stop_patience}")
    print(f"{'='*70}\n")

    history = []

    for ep in range(start_epoch + 1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_ce, total_trip, n_steps = 0.0, 0.0, 0.0, 0

        opt.zero_grad(set_to_none=True)
        for step, (x, hm, pid) in enumerate(tr_ld):
            x, hm, pid = x.to(device), hm.to(device), pid.to(device)

            if args.mixup_alpha > 0 and random.random() < 0.5:
                x_mixed, y_a, y_b, lam = mixup_data(x, pid, args.mixup_alpha)
                with autocast("cuda", enabled=use_amp):
                    emb, logits = model(x_mixed, hm)
                    if focal_fn is not None:
                        ce = lam * focal_fn(logits, y_a) + (1 - lam) * focal_fn(logits, y_b)
                    else:
                        ce = lam * F.cross_entropy(logits, y_a, label_smoothing=args.label_smoothing) + \
                             (1 - lam) * F.cross_entropy(logits, y_b, label_smoothing=args.label_smoothing)
                    trip = hard_triplet_loss(emb, pid, args.margin, args.hard_neg_mining)
                    loss = args.weight_ce * ce + trip
            else:
                with autocast("cuda", enabled=use_amp):
                    emb, logits = model(x, hm)
                    if focal_fn is not None:
                        ce = focal_fn(logits, pid)
                    else:
                        ce = F.cross_entropy(logits, pid, label_smoothing=args.label_smoothing)
                    trip = hard_triplet_loss(emb, pid, args.margin, args.hard_neg_mining)
                    loss = args.weight_ce * ce + trip

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            total_loss += loss.item()
            total_ce += ce.item()
            total_trip += trip.item()
            n_steps += 1

        # flush any leftover accumulated grads
        if n_steps % args.grad_accum != 0:
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

        sched.step()
        avg_loss = total_loss / max(n_steps, 1)
        avg_ce = total_ce / max(n_steps, 1)
        avg_trip = total_trip / max(n_steps, 1)
        elapsed = time.time() - t0
        lr_now = opt.param_groups[0]["lr"]

        eval_loss = compute_eval_loss(model, qu_ld, device, args.label_smoothing)

        if ep % args.eval_freq == 0 or ep == args.epochs:
            metrics = evaluate(model, qu_ld, ga_ld, device)
            mAP = metrics["mAP"]
            r1, r5, r10 = metrics["Rank-1"], metrics["Rank-5"], metrics["Rank-10"]

            gap = avg_loss - eval_loss
            record = {
                "epoch": ep,
                "train_loss": round(avg_loss, 4),
                "train_ce": round(avg_ce, 4),
                "train_trip": round(avg_trip, 4),
                "eval_loss": round(eval_loss, 4),
                "loss_gap": round(gap, 4),
                "mAP": round(mAP, 4),
                "r1": round(r1, 4),
                "r5": round(r5, 4),
                "r10": round(r10, 4),
                "lr": lr_now,
            }
            history.append(record)

            tag = "BEST" if mAP > best_mAP else "     "
            print(f"  ep {ep:3d}/{args.epochs} | trL {avg_loss:.4f} evL {eval_loss:.4f} "
                  f"gap {gap:+.4f} | mAP {mAP*100:5.1f}% R1 {r1*100:5.1f}% "
                  f"R5 {r5*100:5.1f}% R10 {r10*100:5.1f}% | lr {lr_now:.6f} "
                  f"({elapsed:.0f}s) {tag}")

            if mAP > best_mAP:
                best_mAP = mAP
                epochs_no_improve = 0
                path = save_checkpoint(model, args, ep, metrics, "best")
                print(f"  [save] {path}")
                top_checkpoints.append((mAP, ep, path))
                top_checkpoints.sort(reverse=True)
                if len(top_checkpoints) > args.keep_top:
                    old = top_checkpoints.pop()
                    if os.path.exists(old[2]) and "best" not in str(old[2]):
                        os.remove(old[2])
            else:
                epochs_no_improve += 1

            if ep % 10 == 0:
                path = save_checkpoint(model, args, ep, metrics, f"ep{ep}")
                print(f"  [save] {path}")

            if epochs_no_improve >= args.early_stop_patience:
                print(f"\n  [early-stop] No improvement for {args.early_stop_patience} "
                      f"epochs. Stopping.")
                break
        else:
            print(f"  ep {ep:3d}/{args.epochs} | trL {avg_loss:.4f} evL {eval_loss:.4f} "
                  f"gap {gap:+.4f} | lr {lr_now:.6f} ({elapsed:.0f}s)")

    final_metrics = evaluate(model, qu_ld, ga_ld, device)
    save_checkpoint(model, args, ep, final_metrics, "final")

    phe_results = final_metrics
    no_phe_results = evaluate_no_phe(model, qu_ld, ga_ld, device)

    print(f"\n{'='*70}")
    print(f" FINAL RESULTS (v3 — All Improvements)")
    print(f"  With PHE:    mAP {phe_results['mAP']*100:.2f}%  "
          f"R1 {phe_results['Rank-1']*100:.2f}%  R5 {phe_results['Rank-5']*100:.2f}%  "
          f"R10 {phe_results['Rank-10']*100:.2f}%")
    print(f"  Without PHE: mAP {no_phe_results['mAP']*100:.2f}%  "
          f"R1 {no_phe_results['Rank-1']*100:.2f}%  R5 {no_phe_results['Rank-5']*100:.2f}%")
    print(f"  Best mAP during training: {best_mAP*100:.2f}%")
    print(f"{'='*70}")

    if history:
        gaps = [h["loss_gap"] for h in history]
        print(f"\n  Train-Eval loss gap: mean={np.mean(gaps):.4f}, last={gaps[-1]:.4f}")
        if np.mean(gaps) > 2.0:
            print("  WARNING: Large train-eval gap suggests OVERFITTING")
        elif np.mean(gaps) < 0.1:
            print("  NOTE: Very small gap — eval set may be too easy")

    report = {
        "phe_results": {k: round(float(v), 4) for k, v in phe_results.items()},
        "no_phe_results": {k: round(float(v), 4) for k, v in no_phe_results.items()},
        "best_mAP": round(float(best_mAP), 4),
        "config": vars(args),
        "history": history,
    }
    report_path = LOGS_DIR / "hanwoo_report_v3.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    train()