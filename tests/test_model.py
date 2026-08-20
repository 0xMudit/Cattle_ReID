"""Model tests: forward pass, checkpoint round-trip with inferred class count."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.reid_common import (
    HanwooReID,
    load_checkpoint,
    infer_ncls,
)

TINY = "vit_tiny_patch16_224"


def test_hanwoo_forward_with_and_without_pose():
    model = HanwooReID(ncls=5, img_size=256, backbone=TINY, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    hm = torch.zeros(2, 1, 16, 16)

    with torch.no_grad():
        emb, logits = model(x, hm)
        assert emb.shape == (2, model.backbone.embed_dim)
        assert logits.shape == (2, 5)
        emb_no, _ = model(x, None)
        assert emb_no.shape == emb.shape


def test_checkpoint_roundtrip_infers_ncls(tmp_path):
    model = HanwooReID(ncls=17, img_size=256, backbone=TINY, pretrained=False)
    ckpt_path = tmp_path / "model.pth"
    torch.save({"state": model.state_dict(), "epoch": 3, "metrics": {}},
               ckpt_path)

    loaded, info = load_checkpoint(str(ckpt_path), device="cpu",
                                   backbone=TINY)
    assert loaded.classifier.out_features == 17
    assert info["epoch"] == 3
    with torch.no_grad():
        emb, _ = loaded(torch.randn(1, 3, 256, 256),
                        torch.zeros(1, 1, 16, 16))
        assert emb.shape[1] == model.backbone.embed_dim


def test_infer_ncls_from_head():
    model = HanwooReID(ncls=9, img_size=256, backbone=TINY, pretrained=False)
    assert infer_ncls({k.replace("_orig_mod.", ""): v
                       for k, v in model.state_dict().items()}) == 9