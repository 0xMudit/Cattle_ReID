#!/usr/bin/env python3
"""Export trained model to ONNX format."""
import os
import glob
import torch
import torchreid
import onnx

from .config import CFG


def export_onnx(save_dir=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = save_dir or os.path.join(CFG["logs_dir"], CFG["model_name"])

    cks = glob.glob(os.path.join(save_dir, "model", "model.pth.tar-*"))
    if not cks:
        print("No checkpoint found. Train first.")
        return

    ck = max(cks, key=os.path.getctime)
    print(f"Using checkpoint: {ck}")

    model = torchreid.models.build_model(
        name=CFG["model_name"], num_classes=100
    ).to(device)
    torchreid.utils.load_pretrained_weights(model, ck)
    model.eval()

    onnx_path = os.path.join(CFG["models_dir"], "cattle_reid.onnx")
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    dummy = torch.randn(1, 3, CFG["h"], CFG["w"]).to(device)
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["input"],
        output_names=["output"],
    )
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"ONNX exported: {onnx_path}")


if __name__ == "__main__":
    export_onnx()
