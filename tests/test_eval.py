"""Evaluation metric tests: mAP / Rank-k math."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.reid_common import rank_metrics


def _norm(vec):
    return np.asarray(vec, np.float32) / (np.linalg.norm(vec) + 1e-9)


def test_perfect_separation():
    qe = np.stack([_norm([1, 0]), _norm([1, 0]), _norm([0, 1])])   # ids 0,0,1
    qp = np.array([0, 0, 1])
    ge = np.stack([_norm([1, 0]), _norm([0.9, 0.1]), _norm([0, 1]),
                   _norm([0.1, 0.9])])
    gp = np.array([0, 0, 1, 1])

    res = rank_metrics(qe, qp, ge, gp)
    assert res["mAP"] == 1.0
    assert res["Rank-1"] == 1.0
    assert res["Rank-5"] == 1.0


def test_no_positive_gallery_matches():
    qe = np.stack([_norm([1, 0])])
    qp = np.array([9])  # id 9 not in gallery
    ge = np.stack([_norm([1, 0])])
    gp = np.array([0])

    res = rank_metrics(qe, qp, ge, gp)
    assert res["mAP"] == 0.0
    assert res["n_eval"] == 0
    assert res["n_total"] == 1


def test_empty_inputs_are_safe():
    qe = np.zeros((0, 0), np.float32)
    qp = np.zeros((0,), np.int64)
    ge = np.zeros((0, 0), np.float32)
    gp = np.zeros((0,), np.int64)
    res = rank_metrics(qe, qp, ge, gp)
    assert res["mAP"] == 0.0
    assert res["n_total"] == 0