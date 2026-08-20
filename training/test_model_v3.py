#!/usr/bin/env python3
"""
Test HanwooReID model — letterbox preprocessing, multi-embedding gallery.
Loads the trained ViT-B/16 + PHE checkpoint (class count inferred from the
checkpoint head, never guessed from data) and evaluates query vs gallery.
"""
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reid_common import (
    LOGS_DIR,
    MODELS_DIR,
    load_split,
    load_meta,
    make_heatmap,
    letterbox_crop,
    prep_crop,
    load_checkpoint,
)

DEFAULT_CKPT = MODELS_DIR / "hanwoo_reid_best.pth"
FALLBACK_CKPT = MODELS_DIR / "hanwoo_reid_final.pth"


@torch.no_grad()
def embed_images(model, file_list, meta_map, device, batch_size=64):
    embeddings, file_ids, file_names = [], [], []
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i + batch_size]
        imgs, hms = [], []
        for fpath, _ in batch:
            img = cv2.imread(fpath)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(prep_crop(img, 256, 256).numpy())

            rel = os.path.relpath(fpath, REPO)
            kpts = meta_map.get(rel)
            hms.append(make_heatmap(kpts))

        x = torch.from_numpy(np.stack(imgs)).to(device)
        hm = torch.from_numpy(np.stack(hms)).unsqueeze(1).to(device)
        emb, _ = model(x, hm)
        emb = F.normalize(emb, dim=1)
        embeddings.append(emb.cpu().numpy())
        file_ids.extend([pid for _, pid in batch])
        file_names.extend([os.path.basename(f) for f, _ in batch])

    return np.vstack(embeddings), np.array(file_ids), file_names


def evaluate_retrieval(qe, qp, ge, gp, gallery_names):
    sc = qe @ ge.T
    nq = len(qp)

    r1_correct, r5_correct, r10_correct = 0, 0, 0
    total_with_match = 0
    ap_list = []
    per_query = []

    for i in range(nq):
        qi = qp[i]
        order = np.argsort(-sc[i])
        gal_pids = gp[order]
        gal_names = [gallery_names[j] for j in order]
        sims = sc[i][order]

        pos_mask = gal_pids == qi
        pos_idx = np.where(pos_mask)[0]

        if len(pos_idx) == 0:
            per_query.append({
                "query_id": int(qi), "query_name": f"p{qi}",
                "status": "NO_GALLERY_MATCH",
                "top1_sim": float(sims[0]), "top1_id": int(gal_pids[0]),
                "top1_name": gal_names[0],
            })
            continue

        total_with_match += 1
        ap = 0.0
        for rnk, p in enumerate(pos_idx):
            ap += (rnk + 1) / (p + 1)
        ap /= len(pos_idx)
        ap_list.append(ap)

        hit1 = pos_idx[0] == 0
        hit5 = (pos_idx < 5).any()
        hit10 = (pos_idx < 10).any()
        r1_correct += hit1
        r5_correct += hit5
        r10_correct += hit10

        per_query.append({
            "query_id": int(qi),
            "status": "HIT" if hit1 else "MISS",
            "rank1_correct": hit1,
            "rank5_correct": hit5,
            "rank10_correct": hit10,
            "ap": float(ap),
            "top1_sim": float(sims[0]),
            "top1_id": int(gal_pids[0]),
            "top1_name": gal_names[0],
            "true_rank": int(np.where(pos_idx == 0)[0][0]) + 1 if 0 in pos_idx else -1,
            "num_gallery_matches": int(len(pos_idx)),
        })

    mAP = np.mean(ap_list) if ap_list else 0.0
    return {
        "mAP": float(mAP),
        "Rank-1": r1_correct / total_with_match if total_with_match else 0,
        "Rank-5": r5_correct / total_with_match if total_with_match else 0,
        "Rank-10": r10_correct / total_with_match if total_with_match else 0,
        "total_queries": nq,
        "queries_with_gallery_match": total_with_match,
        "queries_no_gallery_match": nq - total_with_match,
        "per_query": per_query,
    }


def compute_distance_stats(qe, qp, ge, gp):
    sc = qe @ ge.T
    pos_scores, neg_scores = [], []
    for i in range(len(qp)):
        same = gp == qp[i]
        diff = ~same
        pos_scores.extend(sc[i][same].tolist())
        neg_scores.extend(sc[i][diff].tolist())
    pos_scores = np.array(pos_scores)
    neg_scores = np.array(neg_scores)
    return {
        "positive_mean": float(pos_scores.mean()),
        "positive_std": float(pos_scores.std()),
        "negative_mean": float(neg_scores.mean()),
        "negative_std": float(neg_scores.std()),
        "separation": float(neg_scores.mean() - pos_scores.mean()),
        "threshold_eer": float(np.median(np.concatenate([pos_scores, neg_scores]))),
    }


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Evaluate HanwooReID query vs gallery")
    p.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt_path = Path(args.ckpt) if args.ckpt else DEFAULT_CKPT
    if not ckpt_path.exists() and FALLBACK_CKPT.exists():
        ckpt_path = FALLBACK_CKPT
    if not ckpt_path.exists():
        print(f"No checkpoint found. Looked for:\n  {DEFAULT_CKPT}\n  {FALLBACK_CKPT}")
        return
    print(f"Checkpoint: {ckpt_path}")

    model, ckpt_info = load_checkpoint(ckpt_path, device)
    ncls = model.classifier.out_features
    print(f"Model classes: {ncls} (inferred from checkpoint)")

    meta_map = load_meta()
    print(f"Meta entries: {len(meta_map)}")

    gallery_items = load_split("gallery")
    query_items = load_split("query")
    print(f"Gallery: {len(gallery_items)} images | Query: {len(query_items)} images")

    print("\nEmbedding gallery...")
    t0 = time.time()
    ge, gp, gn = embed_images(model, gallery_items, meta_map, device,
                              args.batch_size)
    print(f"  Gallery embeddings: {ge.shape} ({time.time()-t0:.1f}s)")

    print("Embedding queries...")
    t0 = time.time()
    qe, qp, qn = embed_images(model, query_items, meta_map, device,
                              args.batch_size)
    print(f"  Query embeddings: {qe.shape} ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 70)
    print("  RETRIEVAL EVALUATION (letterbox preprocessing)")
    print("=" * 70)
    results = evaluate_retrieval(qe, qp, ge, gp, gn)
    print(f"  mAP:       {results['mAP']*100:.2f}%")
    print(f"  Rank-1:    {results['Rank-1']*100:.2f}%")
    print(f"  Rank-5:    {results['Rank-5']*100:.2f}%")
    print(f"  Rank-10:   {results['Rank-10']*100:.2f}%")
    print(f"  Queries:   {results['total_queries']} "
          f"({results['queries_with_gallery_match']} with match, "
          f"{results['queries_no_gallery_match']} without)")

    print("\n" + "=" * 70)
    print("  DISTANCE STATISTICS")
    print("=" * 70)
    dist_stats = compute_distance_stats(qe, qp, ge, gp)
    print(f"  Positive mean: {dist_stats['positive_mean']:.4f} ± "
          f"{dist_stats['positive_std']:.4f}")
    print(f"  Negative mean: {dist_stats['negative_mean']:.4f} ± "
          f"{dist_stats['negative_std']:.4f}")
    print(f"  Separation:    {dist_stats['separation']:.4f}")

    print("\n" + "=" * 70)
    print("  SAMPLE RETRIEVAL RESULTS (first 10)")
    print("=" * 70)
    sc = qe @ ge.T
    for i in range(min(10, len(qp))):
        order = np.argsort(-sc[i])
        top5_pids = gp[order[:5]]
        top5_sims = sc[i][order[:5]]
        true_pid = qp[i]
        hit = top5_pids[0] == true_pid
        print(f"\n  Query p{true_pid} ({qn[i]}): {'CORRECT' if hit else 'WRONG'}")
        print(f"    True ID: {true_pid}")
        for j in range(min(5, len(order))):
            marker = " *" if top5_pids[j] == true_pid else ""
            print(f"    Rank {j+1}: p{top5_pids[j]} (sim={top5_sims[j]:.4f}){marker}")

    report = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "epoch": ckpt_info.get("epoch") if isinstance(ckpt_info, dict) else None,
        "retrieval": {k: v for k, v in results.items() if k != "per_query"},
        "distance_stats": dist_stats,
        "per_query": results["per_query"],
    }
    report_path = LOGS_DIR / "test_results.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()