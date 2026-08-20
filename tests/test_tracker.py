"""Tracker tests: correct Hungarian assignment, no double-matching, IoU gating."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.video_reid_v5 import DeepSORTTracker, filter_cow_boxes


def _tracker(**kw):
    kw.setdefault("iou_threshold", 0.1)
    kw.setdefault("max_age", 150)
    return DeepSORTTracker(**kw)


def test_new_tracks_created_when_no_predictions():
    t = _tracker()
    dets = [(0, 0, 50, 50), (100, 100, 150, 150)]
    assignments = t.update(dets)
    assert len(t.trackers) == 2
    assert assignments == []  # first frame: no tracks to match yet


def test_detections_match_existing_tracks_once_each():
    t = _tracker()
    dets1 = [(0, 0, 50, 50), (100, 100, 150, 150)]
    t.update(dets1)
    ids1 = {tr.id for tr in t.trackers}

    dets2 = [(5, 5, 55, 55), (105, 105, 155, 155)]
    assignments = t.update(dets2)

    assert len(assignments) == 2
    assigned_ids = {tid for tid, _ in assignments}
    assigned_dets = {j for _, j in assignments}
    assert len(assigned_ids) == 2 and len(assigned_dets) == 2  # no dupes
    assert assigned_ids == ids1
    assert {tr.id for tr in t.trackers} == ids1  # no new tracks leaked


def test_far_away_detection_creates_new_track():
    t = _tracker()
    t.update([(0, 0, 50, 50)])
    ids_before = {tr.id for tr in t.trackers}

    dets2 = [(500, 500, 560, 560)]  # IoU ~0 with any prediction
    assignments = t.update(dets2)

    assert assignments == []  # unmatched -> new track
    assert len(t.trackers) == 2
    new_id = {tr.id for tr in t.trackers} - ids_before
    assert len(new_id) == 1


def test_iou_threshold_gate():
    t = _tracker(iou_threshold=0.9)  # strict gate
    t.update([(0, 0, 50, 50)])
    assignments = t.update([(20, 20, 70, 70)])  # IoU < 0.9
    assert assignments == []
    assert len(t.trackers) == 2  # new track created


def test_expired_tracks_are_pruned():
    t = _tracker(max_age=3)
    t.update([(0, 0, 50, 50)])
    tid = t.trackers[0].id
    t.reid[tid] = {"emb": np.zeros(4, np.float32)}
    for _ in range(4):
        t.update([])
    assert len(t.trackers) == 0
    assert t.reid == {}  # reid state cleaned up with expired trackers


def test_filter_cow_boxes_keeps_distant_cctv_cows():
    # Regression: relative 1%-of-frame-area filter rejected 91% of detections
    # on 2880x1620 CCTV (median cow box was ~0.13% of frame). Filter must be
    # resolution-agnostic (absolute min dimension only).
    names = {0: "person", 21: "cow"}
    boxes = [
        [1200.0, 700.0, 1260.0, 780.0],  # 60x80 distant cow: tiny area vs frame
        [10.0, 10.0, 200.0, 100.0],  # 190x90 cow
        [0.0, 0.0, 500.0, 1000.0],  # wide tall cow
        [100.0, 100.0, 150.0, 200.0],  # 50x100 cow (boundary: kept)
        [1000.0, 1000.0, 1030.0, 1030.0],  # 30x30 too small: dropped
    ]
    classes = [0, 21, 21, 21, 21]
    confs = [0.9, 0.7, 0.8, 0.6, 0.4]
    kept = filter_cow_boxes(boxes, names, classes, confs)
    assert kept == [
        (0, 0, 500, 1000),  # 500k area, first (largest)
        (10, 10, 200, 100),  # 17.1k area
        (100, 100, 150, 200),  # 5k area (distant cow survives the filter)
    ]


def test_filter_cow_boxes_rejects_extreme_aspect_and_person():
    names = {0: "person", 21: "cow"}
    boxes = [
        [0.0, 0.0, 1000.0, 100.0],  # cow, aspect 10 > 5: dropped
        [0.0, 0.0, 100.0, 1000.0],  # cow, aspect 0.1: kept (boundary)
        [500.0, 0.0, 600.0, 100.0],  # person: dropped
    ]
    classes = [21, 21, 0]
    confs = [0.9, 0.9, 0.9]
    kept = filter_cow_boxes(boxes, names, classes, confs)
    assert kept == [(0, 0, 100, 1000)]