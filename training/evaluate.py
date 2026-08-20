#!/usr/bin/env python3
"""Evaluate trained model — GPU-accelerated batch evaluation."""
import os
import glob
import pickle
import numpy as np
import torch
import torchreid
from torchreid.utils import FeatureExtractor
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm

from .config import CFG
from .dataset import register_cattle_dataset


def find_best_checkpoint(save_dir):
    cks = glob.glob(os.path.join(save_dir, "model", "model.pth.tar-*"))
    if not cks:
        return None
    return max(cks, key=os.path.getctime)


def letterbox_resize(img, target_h, target_w):
    """Aspect-ratio preserving resize with zero-padding (matches training Prep.crop)."""
    w0, h0 = img.size
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = int(w0 * scale), int(h0 * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    out = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    out.paste(resized, (pad_w, pad_h))
    return out


def evaluate(save_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = save_dir or os.path.join(CFG["logs_dir"], CFG["model_name"])

    ck = find_best_checkpoint(save_dir)
    if ck is None:
        print("No checkpoint found. Train first.")
        return

    print(f"Checkpoint: {ck}")

    dn = register_cattle_dataset()
    dm = torchreid.data.ImageDataManager(
        sources=dn,
        height=CFG["h"],
        width=CFG["w"],
        batch_size_train=CFG["bs"],
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = torchreid.models.build_model(
        name=CFG["model_name"], num_classes=dm.num_train_pids
    ).to(device)
    torchreid.utils.load_pretrained_weights(model, ck)
    model.eval()

    # Use letterbox resize to match training preprocessing (process.py Prep.crop)
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    proc = CFG["data_proc"]
    gal_dir = os.path.join(proc, "gallery")
    query_dir = os.path.join(proc, "query")

    print("Building gallery embeddings...")
    gal = {}
    for p in glob.glob(os.path.join(gal_dir, "*.jpg")):
        try:
            pid = int(os.path.basename(p).split("_")[1][1:])
        except (IndexError, ValueError):
            continue
        img = Image.open(p).convert("RGB")
        img = letterbox_resize(img, CFG["h"], CFG["w"])
        x = np.array(img).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)
        key = f"Cow_{pid:03d}"
        gal.setdefault(key, []).append(emb)

    for k in gal:
        gal[k] = np.mean(gal[k], axis=0)

    print(f"Gallery: {len(gal)} cows")

    correct = 0
    total = 0
    results = []

    for p in tqdm(glob.glob(os.path.join(query_dir, "*.jpg")), desc="Evaluating"):
        try:
            true_pid = int(os.path.basename(p).split("_")[1][1:])
        except (IndexError, ValueError):
            continue

        img = Image.open(p).convert("RGB")
        img = letterbox_resize(img, CFG["h"], CFG["w"])
        x = np.array(img).astype(np.float32) / 255.0
        x = (x - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        best, best_dist = None, 1e9
        for k, v in gal.items():
            dist = np.sqrt(np.mean((emb - v) ** 2))
            if dist < best_dist:
                best_dist = dist
                best = k

        pred_pid = int(best.split("_")[1]) if best else -1
        hit = pred_pid == true_pid
        if hit:
            correct += 1
        total += 1
        results.append({
            "file": os.path.basename(p),
            "true": true_pid,
            "pred": pred_pid,
            "dist": best_dist,
            "correct": hit,
        })

    if total > 0:
        acc = correct / total * 100
        print(f"\nAccuracy: {correct}/{total} = {acc:.1f}%")

        wrong = [r for r in results if not r["correct"]]
        if wrong:
            print(f"\nMisclassified ({len(wrong)}):")
            for r in wrong[:10]:
                print(f"  {r['file']}: true={r['true']}, pred={r['pred']}, dist={r['dist']:.3f}")

        gal_pkl = os.path.join(CFG["gallery_dir"], "gal.pkl")
        with open(gal_pkl, "wb") as f:
            pickle.dump(gal, f)
        print(f"\nGallery saved: {gal_pkl}")
    else:
        print("No query images found")


if __name__ == "__main__":
    evaluate()
