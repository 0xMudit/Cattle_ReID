"""Smoke tests for shared preprocessing / heatmap / split logic."""
import os
import sys

import numpy as np
import cv2
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.reid_common import (
    parse_identity,
    make_heatmap,
    letterbox_crop,
    prep_crop,
    PKSampler,
    GRID,
)
from training.process import split_paths


def test_parse_identity():
    assert parse_identity("c0_p227_36569.jpg") == 227
    assert parse_identity("c0_p0_12373.jpg") == 0
    assert parse_identity("something_else.png") is None


def test_make_heatmap_empty():
    hm = make_heatmap(None)
    assert hm.shape == (GRID, GRID)
    assert hm.dtype == np.float32
    assert hm.max() == 0.0


def test_make_heatmap_peak_at_keypoint():
    hm = make_heatmap([(0.5, 0.5, 0.9)])
    assert hm.shape == (GRID, GRID)
    assert hm.max() > 0.5
    cy, cx = np.unravel_index(np.argmax(hm), hm.shape)
    assert abs(cx / (GRID - 1) - 0.5) < 0.2
    assert abs(cy / (GRID - 1) - 0.5) < 0.2


def test_make_heatmap_drops_low_conf():
    hm = make_heatmap([(0.5, 0.5, 0.1)])
    assert hm.max() == 0.0


def test_letterbox_preserves_aspect_and_pads_gray():
    img = np.full((200, 100, 3), 40, np.uint8)
    out = letterbox_crop(img, 256, 256)
    assert out.shape == (256, 256, 3)
    assert out[0, 0, 0] == 128  # gray padding
    assert (out[128:130, 110:120] == 40).all()  # content preserved (scale 1.28)


def test_prep_crop_tensor():
    img = np.full((100, 100, 3), 128, np.uint8)
    t = prep_crop(img, 256, 256)
    assert t.shape == (3, 256, 256)
    assert t.dtype == torch.float32
    assert t.min() >= -1.01 and t.max() <= 1.01


def test_split_paths_no_leakage_deterministic():
    paths = [f"img_{i}.jpg" for i in range(10)]
    tr, ga, qu = split_paths(paths, seed=42)
    assert len(tr) == 6 and len(ga) == 2 and len(qu) == 2
    assert len(set(tr) | set(ga) | set(qu)) == 10  # no overlap, all used
    tr2, _, _ = split_paths(paths, seed=42)
    assert tr == tr2


def test_split_paths_small_identity():
    tr, ga, qu = split_paths(["a.jpg", "b.jpg"])
    assert set(tr) == {"a.jpg", "b.jpg"} and ga == [] and qu == []


def test_pk_sampler_shape():
    items = [(f"x_{i}.jpg", i % 3) for i in range(30)]
    id2label = {i: i % 3 for i in range(30)}
    s = PKSampler(items, id2label, p=3, k=2)
    it = iter(s)
    batch = [next(it) for _ in range(3 * 2)]  # one PK batch = p*k samples
    assert len(batch) == 3 * 2
    labels = [id2label[items[i][1]] for i in batch]
    assert len(set(labels)) == 3  # every identity present in the batch