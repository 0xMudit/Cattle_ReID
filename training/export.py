#!/usr/bin/env python3
"""Export the trained HanwooReID model (ViT + PHE) to ONNX.

Outputs a single "embedding" tensor (normalized Re-ID features), suitable for
gallery matching and edge deployment.
"""
import os
import sys
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import onnx

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.reid_common import MODELS_DIR, DEFAULT_IMG_SIZE, load_checkpoint

DEFAULT_CKPT = MODELS_DIR / "hanwoo_reid_best.pth"
FALLBACK_CKPT = MODELS_DIR / "hanwoo_reid_final.pth"


class EmbeddingWrapper(nn.Module):
    """Expose only the embedding output (drops the classifier head)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, hm):
        emb, _ = self.model(x, hm)
        return emb


def find_checkpoint(ckpt_path=None):
    ckpt_path = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not ckpt_path.exists() and FALLBACK_CKPT.exists():
        ckpt_path = FALLBACK_CKPT
    if not ckpt_path.exists():
        print(f"No checkpoint found. Looked for:\n  {DEFAULT_CKPT}\n  {FALLBACK_CKPT}")
        return None
    return ckpt_path


def export_onnx(ckpt_path=None, out_path=None, img_size=DEFAULT_IMG_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = find_checkpoint(ckpt_path)
    if ckpt_path is None:
        return None

    model, _ = load_checkpoint(ckpt_path, device, img_size=img_size)
    model.eval()
    ncls = model.classifier.out_features

    out_path = Path(out_path or (MODELS_DIR / "hanwoo_reid.onnx"))
    wrapped = EmbeddingWrapper(model).to(device).eval()

    dummy_img = torch.randn(1, 3, img_size, img_size, device=device)
    dummy_hm = torch.zeros(1, 1, 16, 16, device=device)

    torch.onnx.export(
        wrapped, (dummy_img, dummy_hm), str(out_path),
        input_names=["image", "heatmap"],
        output_names=["embedding"],
        opset_version=14,
        dynamic_axes={
            "image": {0: "batch"},
            "heatmap": {0: "batch"},
            "embedding": {0: "batch"},
        },
    )
    onnx.checker.check_model(onnx.load(str(out_path)))
    print(f"ONNX exported: {out_path} (checkpoint: {ckpt_path.name}, "
          f"{ncls} classes, embedding dim {model.backbone.embed_dim})")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description="Export HanwooReID to ONNX")
    p.add_argument("--ckpt", type=str, default=None, help="Path to .pth checkpoint")
    p.add_argument("--out", type=str, default=None, help="Output .onnx path")
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    args = p.parse_args(argv)
    export_onnx(args.ckpt, args.out, args.img_size)


if __name__ == "__main__":
    main()