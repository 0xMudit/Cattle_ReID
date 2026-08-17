#!/usr/bin/env python3
"""Evaluate trained model on query/gallery split."""
import os
import glob
import pickle
import numpy as np
import torch
import torchreid
from torchreid.utils import FeatureExtractor

from .config import CFG
from .dataset import register_cattle_dataset


def find_best_checkpoint(save_dir):
    cks = glob.glob(os.path.join(save_dir, "model", "model.pth.tar-*"))
    if not cks:
        return None
    return max(cks, key=os.path.getctime)


def load_model(save_dir, num_classes, device):
    ck = find_best_checkpoint(save_dir)
    if ck is None:
        print("No checkpoint found")
        return None
    print(f"Loading checkpoint: {ck}")
    m = torchreid.models.build_model(
        name=CFG["model_name"], num_classes=num_classes
    ).to(device)
    torchreid.utils.load_pretrained_weights(m, ck)
    m.eval()
    return m


def extract_gallery(model, proc_dir, device):
    """Build gallery embeddings from processed gallery images."""
    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((CFG["h"], CFG["w"])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    gal = {}
    gal_dir = os.path.join(proc_dir, "gallery")
    for p in glob.glob(os.path.join(gal_dir, "*.jpg")):
        try:
            nm = os.path.basename(p).split("_")
            pid = int(nm[1][1:])
        except (IndexError, ValueError):
            continue

        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model(x).cpu().numpy().flatten()
        emb = emb / (np.linalg.norm(emb) + 1e-12)

        key = f"Cow_{pid:03d}"
        if key not in gal:
            gal[key] = []
        gal[key].append(emb)

    for k in gal:
        gal[k] = np.mean(gal[k], axis=0)
    return gal


def evaluate(save_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = save_dir or os.path.join(CFG["logs_dir"], CFG["model_name"])

    dn = register_cattle_dataset()
    dm = torchreid.data.ImageDataManager(
        sources=dn,
        height=CFG["h"],
        width=CFG["w"],
        batch_size_train=CFG["bs"],
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    model = load_model(save_dir, dm.num_train_pids, device)
    if model is None:
        return

    gal = extract_gallery(model, CFG["data_proc"], device)
    print(f"Gallery: {len(gal)} cows")

    from PIL import Image
    import torchvision.transforms as T

    transform = T.Compose([
        T.Resize((CFG["h"], CFG["w"])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    correct = 0
    total = 0
    proc = CFG["data_proc"]
    query_dir = os.path.join(proc, "query")

    for p in glob.glob(os.path.join(query_dir, "*.jpg")):
        try:
            nm = os.path.basename(p).split("_")
            true_pid = int(nm[1][1:])
        except (IndexError, ValueError):
            continue

        img = Image.open(p).convert("RGB")
        x = transform(img).unsqueeze(0).to(device)
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
        if pred_pid == true_pid:
            correct += 1
        total += 1

    if total > 0:
        acc = correct / total * 100
        print(f"Accuracy: {correct}/{total} = {acc:.1f}%")
    else:
        print("No query images found")


if __name__ == "__main__":
    evaluate()
