# HanwooReID — Paper Study & Implementation Plan

> **Paper:** "HanwooReID: Multi-view cattle re-identification with pose-aware transformer enhancements"
> **Authors:** Jiaqi Liu, Alvaro Fuentes, Shujie Han, Yongchae Jeong, Sook Yoon, Dong Sun Park
> (Jeonbuk National University / Mokpo National University)
> **Venue:** Computers and Electronics in Agriculture, vol. 239 (2025), article 111117
> **Local PDF:** `cattle-researchpaper.pdf` (repo root)
> **Status:** 📖 studied — 🧪 data assessed — 🚧 implementation planned

---

## 1. TL;DR

Hanwoo cattle (a Korean beef breed) have **no distinctive coat markings**, so
appearance-only Re-ID fails. This paper builds a **Transformer-based Re-ID
framework** on a new multi-view farm dataset and adds two novel modules:

1. **PHE — Pose-Guided Heatmap Encoder:** feeds 2D pose keypoints (from a
   pretrained animal pose model) into the ViT as an **attention prior**, so the
   model focuses on identity-relevant anatomy.
2. **VCR — Viewpoint-Constrained Retrieval:** projects the 4 **hoof**
   keypoints onto a bird's-eye-view (BEV) plane using camera calibration,
   estimates the cow's heading relative to the camera, and **filters out
   gallery images with mismatched orientation** before matching.

Results: **+7.4% mAP (closed-set)** and **+6.8% mAP (open-set)** over the
TransReID baseline; ~94–95% mAP on the largest subsets; ~70 FPS single-image,
~130 FPS batched, on one H100.

---

## 2. Why this matters for our repo

Our current pipeline is **zero-shot OSNet** (ImageNet-pretrained embeddings,
`cattle_osnet/run.py`) — it works but was never trained on cattle and does no
training-based learning. The paper's recipe gives us a clear **supervised,
state-of-the-art upgrade path** that reuses almost everything we already have:

| We already have | The paper needs |
|-----------------|-----------------|
| `yolov8n.pt` cow detection | Detection ✓ |
| `cow_pose.pt` (12 cow keypoints incl. 4 hooves) | Pose → heatmaps ✓ (paper uses 17 kps; ours has the 4 hooves VCR needs) |
| `annotate.py` pose overlay pipeline | Visualization / sanity checks ✓ |
| `prep_videos.py` video→crops | Dataset building ✓ |
| OSNet gallery/query matching (`run.py`) | Retrieval scaffold ✓ |
| Colab notebook (ID loss + triplet loss training) | Training recipe ✓ (swap in ViT + PHE) |
| ✗ Transformer backbone + PHE | ViT-B/16 (TransReID) + heatmap encoder |
| ✗ Camera calibration | Intrinsics K + extrinsics (R, t) per camera (for VCR) |
| ✗ Labeled multi-view cattle data | Identity labels + multiple simultaneous views |

---

## 3. The HanwooReID dataset (from the paper)

- **Scale:** 134,071 raw images captured by multiple **synchronized cameras**;
  **9,929 manually verified, cropped cattle instances** of **31 individual
  cattle** annotated with identity labels.
- **Capture:** Hikvision cameras at **3840×2160 @ 60 Hz**; RGB by day, auto-IR
  (grayscale) at night; 24/7 recording.
- **Labeling:** CVAT-based tool; trained annotators labeled 12 predefined
  **action classes** + consistent identity labels.
- **Calibration:** barn-friendly method — **DLT (Direct Linear Transform) +
  Levenberg–Marquardt nonlinear optimization** on manually selected structural
  landmarks (no chessboard needed). Produces intrinsic K, distortion D, and
  extrinsics (R, t) per camera.

### Subset breakdown (Table 1)

| Subset | Cameras | Cattle | Images | Duration |
|--------|---------|--------|--------|----------|
| Imsil-day1 | 2 | 19 | 895 | 24 h |
| Imsil-day2 | 2 | 19 | 1,456 | 24 h |
| Imsil-day3 | 2 | 19 | 2,015 | 24 h |
| Imsil-day4 | 2 | 19 | 2,724 | 24 h |
| Namwon-cam4 | 1 | 3 | 850 | 12 h |
| Namwon-cam5 | 1 | 9 | 1,989 | 12 h |

- **Imsil farm:** 17 adults + 2 calves in a 36 m × 12 m barn; **two cameras
  facing each other** with overlapping FOVs (multi-view).
- **Namwon farm:** two 10 m × 10 m zones; one camera each, similar orientation.
- **Protocols:** Imsil = **cross-camera** evaluation; Namwon = **same-camera**.
  Open-set subsets (Imsil10/13/16, Namwon9) vary the train/test identity ratio
  (TTR).

> **Key takeaway:** 9,929 images of 31 cows is *not* huge. With our own videos
> we can realistically build a comparable mini-dataset.

---

## 4. Method

### 4.1 Baseline: TransReID (ViT-B/16)

- Input **256 × 256**, ImageNet-pretrained ViT-B/16.
- **Softmax (ID) loss + triplet loss** (8 instances per identity), batch size 64.
- SGD lr **0.001**, weight decay **1×10⁻⁴**, linear warmup, **up to 720 epochs**.
- Augmentation: random horizontal flip, random crop with padding, random erasing.
- Normalization mean/std **0.5**. Single **H100**.

### 4.2 PHE — Pose-Guided Heatmap Encoder

1. Detect **J = 17 anatomical keypoints** (eyes, shoulders, knees, hooves) with a
   pretrained animal pose model (RTMPose / ViTPose, trained on AP-10K / Animal-Pose).
2. Convert each keypoint into a **Gaussian heatmap** → multi-channel tensor
   **H_pose ∈ R^(J×H×W)**; missing keypoints = zero channels.
3. **Max-pool across channels** (proven best: max > mean > concat):
   **H_agg = max over j of H_j ∈ R^(1×H×W)**.
4. Encode with **two 1×1 conv layers (+ReLU)** → 768-dim pose feature map E.
5. Flatten E and **add it to the patch embeddings as an attention prior**.

Why it works: adds anatomical structure without a hard dependency on pose
quality — even with ~9 of 17 keypoints removed, mAP stays stable.

### 4.3 VCR — Viewpoint-Constrained Retrieval

**Estimation (per image):**
1. Take 4 hoof keypoints (LF, RF, LR, RR); assume they lie on the ground plane
   (height h = 0).
2. **Back-project** each to the BEV plane by intersecting the camera ray (K, R, t)
   with the ground: q = c + λ·(ray), where c = −Rᵀt is the camera center, λ chosen
   so the point lands on h = 0.
3. Body orientation **d = (midpoint of rear hooves) → (midpoint of front hooves)**
   in BEV coordinates.
4. Relative angle θ = atan2(v_y, v_x) − atan2(d_y, d_x) vs. camera forward.
5. Discretize θ into **4 angular bins → ViewID**.

**Retrieval (at evaluation):**
- Only compare the query against gallery images whose **ViewID matches** (or
  differs within tolerance); return Top-N from the constrained set.
- If any hoof is missing → **skip VCR for that image** (fall back to unconstrained).
- VCR costs only **0.00013 s / cow** (~7,000 FPS equivalent).

---

## 5. Results

### Closed-set (Table 4) — mAP / Rank-1

| Method | Imsil-day1 | Imsil-day4 | Namwon-cam5 |
|--------|-----------|-----------|-------------|
| AGW (ResNet-50) | 61.5 / 72.1 | 80.4 / 86.8 | 72.9 / — |
| PFD | 83.3 / 88.5 | 93.2 / 93.9 | 78.6 / 83.1 |
| RotTrans | 84.3 / 86.9 | 94.6 / 94.8 | 76.2 / 82.5 |
| PHA | 83.2 / 85.2 | 92.3 / 94.3 | 62.8 / 77.3 |
| **Ours (PHE only)** | 88.3 / 91.8 | 94.0 / 95.3 | 80.6 / 84.4 |
| **Ours (PHE + VCR)** | **94.0 / 92.7** | **95.3 / 96.8** | best mAP |

### Open-set (Table 5) — ViT-B/16 backbone, mAP / Rank-1 / Rank-5

| Method | Imsil10 (TTR 0.47) | Imsil13 (TTR 0.32) | Imsil16 (TTR 0.16) | Namwon9 (TTR 0.25) |
|--------|--------------------|--------------------|--------------------|--------------------|
| TransReID | 53.3 / 84.8 / 93.7 | 60.6 / 81.9 / 95.4 | 70.7 / 85.8 / 96.7 | 62.8 / 82.3 / 90.6 |
| RotTrans | 54.3 / 85.1 / 95.5 | 60.2 / 83.5 / 94.9 | 71.0 / 86.7 / 96.7 | 63.6 / 83.3 / 93.8 |
| PHA | 43.1 / 80.3 / 92.9 | 59.0 / 82.3 / 93.2 | 67.4 / 86.7 / 94.2 | 63.3 / 83.3 / 92.7 |
| **Ours (PHE only)** | 54.9 / 85.1 / 97.0 | 61.5 / 85.2 / 96.2 | 71.4 / 87.5 / 97.5 | 64.5 / 89.6 / 95.8 |
| **Ours (PHE + VCR)** | **60.1 / 85.7 / 96.1** | 62.8 / 81.0 / 93.7 | **74.8 / 86.7 / 95.0** | **68.7 / 86.5 / 94.8** |

### Ablations (Table 6 / Table 7)

- **VID (learnable camera ViewID embedding) hurts:** e.g. mAP 53.3 → 51.8 on
  Imsil10. Rejected as a design choice — VCR's explicit orientation filtering is
  superior for cattle.
- **PHE + VCR is the best mAP combination** (60.1 / 62.8 / 74.8 / 68.7).
- **Heatmap aggregation:** max pooling beats mean and concat everywhere
  (e.g. Imsil-day1 mAP 94.0 vs 91.7 vs 91.7; Imsil10 mAP 60.1 vs 59.9 vs 43.3).
- **Robustness:** removing up to 9 of 17 keypoints barely dents mAP; heatmaps
  degrade gracefully.
- **Speed (H100):** 69.7 FPS @batch 1, 129.9 FPS @batch 32, 133.7 FPS @batch 64.

### Limitations (from the paper)

- Daytime RGB only — IR/night footage not used (future work, incl. cross-temporal ReID).
- Small identity count (31 cattle).
- Dataset is **"available on request" — not public**.
- Failures cluster around fine-grained cues (horns vs no horns) and occlusions.

---

## 6. Our data — can we use the local `Dataset/` videos?

### Inventory (probed 2026-08-13 with ffprobe + our `cow_pose.pt`)

| File | Size | Codec | Res | FPS | Dur | Audio | Cows found* |
|------|------|-------|-----|-----|-----|-------|-------------|
| A1.mp4 | 75 MB | HEVC | 1920×1080 | 25 | 5 min | AAC | 3 detections, last minute only (1 cow, ~9/12 kpts) |
| A2.mp4 | 75 MB | HEVC | 1920×1080 | 25 | 5 min | AAC | **0** across all samples |
| A3.mp4 | 258 MB | HEVC | 1920×1080 | 25 | 17 min | AAC | 14 detections, sporadic (1–2 cows, kpts poor ~5.4/12) |
| ch07m_20260804063451t_n01.mp4 | 601 MB | HEVC | **2880×1620** | 25 | 26 min | PCM | **0** (53 samples) — morning 06:34 |
| ch10m_20260803175648t_n02.mp4 | 230 MB | HEVC | **2880×1620** | 25 | 10 min | PCM | **0** (20 samples) — evening 17:56 |
| ch10m_20260803180646t_n02.mp4 | 582 MB | HEVC | **2880×1620** | 25 | 25 min | PCM | **0** (51 samples) — evening 18:06 (contiguous) |

\* samples every 30 s, `cow_pose.pt` @ conf 0.25. Brightness is daytime-OK (mean 105–136).

### What the probe tells us

- **A1–A3** = handheld phone-style footage (has AAC audio). Real cows appear but
  **sparsely**: A2 appears to have none, A1 only in the final minute, A3 in
  scattered bursts. 1–2 cows per frame max. No identity labels, no calibration.
- **ch07 / ch10** = fixed 2.7K barn CCTV ("m" = main stream, "n01/n02" = camera
  channel). **No cows detected at any sampled frame** — either the pen is empty
  at those times or cows are too small/far for the detector after downscaling.
  ch10_175648 + ch10_180646 are **contiguous** (600 s + 1514 s ≈ one evening).
  ch07 is the **next morning** — so there is **no same-time overlap between the
  two cameras** yet.
- **Stream corruption:** all files emit `Could not find ref with POC` HEVC
  warnings (missing reference frames) — they decode, but re-encoding
  (HEVC → H.264) is still advisable (as our `docs.md` already documents).

### Verdict

> ⚠️ **Not usable as-is for supervised multi-view Re-ID training** — but usable
> as **raw material**. What's missing: (1) per-cow identity labels, (2) camera
> calibration, (3) the *same* cattle under *overlapping* time in *both* cameras,
> (4) enough confirmed cow frames (A2 looks cowless; ch feeds need a full scan,
> not just 30 s samples).

To become a HanwooReID-style dataset we must:
1. **Full-frame scan** the ch* feeds (all ~2 h at higher pose resolution /
   imgsz) to find when cattle are actually present.
2. Record/tag **which cow is which** (manual labeling assisted by tracking —
   `prep_videos.py` already does detection + tracking + crop export).
3. **Calibrate each camera** (DLT + LM, per the paper) using measured barn
   landmarks, so VCR's hoof→BEV projection is possible.
4. Keep **daytime RGB clips**; skip night/IR for v1 (paper does too).

---

## 7. Kaggle plan

Why Kaggle: free **T4/P100 (30 h/week)** + dataset hosting (20 GB/dataset,
20 GB/file) — our ~1.8 GB of video fits easily.

### Phase 0 — Data on Kaggle
- Create Kaggle dataset **`cattle-reid-raw-videos`**; upload `Dataset/*.mp4`
  (or re-encoded H.264 to avoid HEVC POC issues).
- Keep versioning notes (A-files = phone footage; ch07/ch10 = barn CCTV).

### Phase 1 — Extraction notebook (`extract_crops.ipynb`)
1. Load video via OpenCV (or ffmpeg-python).
2. Run `yolov8n.pt` (cow class 21) → crop → run `cow_pose.pt` → save
   `crops/{video}/{cow_track}/{frame}.jpg` + `meta.json` (bbox, 12 kpts, view).
3. **Output:** a first untrained "identity-free" gallery per track.

### Phase 2 — Labeling notebook (`label_and_split.ipynb`)
- Merge tracks into identities (tracklets spanning cuts / cameras = one cow).
- Build `train/query/gallery` split by **identity** (never by image).
- Mirror the paper's protocol: cross-camera split if ch07 & ch10 ever overlap;
  otherwise same-camera.

### Phase 3 — Training notebook (`train_hanwoo_reid.ipynb`)
- ViT-B/16 + PHE (2× 1×1 conv → 768), softmax + triplet, 256×256.
- T4-friendly: **shorter epochs** (paper uses 720 on H100; we target ~50–150),
  mixed precision, smaller batch. Track mAP / Rank-1 / Rank-5.
- Add VCR at evaluation using a `calib.json` per camera.

### Phase 4 — Evaluation / deployment
- Compare: OSNet zero-shot (our current baseline) vs ViT+PHE vs ViT+PHE+VCR.
- Export best to ONNX for on-farm edge inference.

---

## 8. Implementation roadmap (in this repo)

1. **`cattle_osnet/transformer_reid/`** — new module: TransReID-style ViT-B/16 +
   PHE (heatmap encoder) + ID/triplet training loop (adapted from our notebook).
2. **`cattle_osnet/pose_heatmaps.py`** — kpts → Gaussian heatmaps → max-pool →
   1×1 conv encoder (reuses `cow_pose.pt` outputs; extend to 17 kps if desired).
3. **`cattle_osnet/calibrate.py`** — DLT + Levenberg–Marquardt from manual 3D-2D
   landmark correspondences → per-camera `calib.json`.
4. **`cattle_osnet/vcr.py`** — hoof back-projection to BEV, ViewID estimation,
   constrained retrieval (hooks into `run.py`).
5. **Kaggle notebooks** under `notebooks/` or `experiments/` for Phases 1–4.
6. **README/doc updates** as each piece lands.

---

## 9. Open questions (brainstorm)

- [ ] Are the ch07/ch10 cameras viewing the **same pen**? (need to check FOVs /
      overlapping landmarks) — decides whether cross-camera Re-ID is possible.
- [ ] Which camera naming is which (ch07 vs ch10 = physical cameras 7 & 10)?
- [ ] Do we have a way to confirm identities (ear tags / farm record) for labeling?
- [ ] T4 memory: ViT-B/16 @ 256×256 + batch 64 needs H100 — we'll shrink batch
      (~16–32) and use AMP on T4/P100.
- [ ] Extend `cow_pose.pt` to 17 keypoints (AP-10K/RTMPose) for full PHE parity,
      or keep 12 (VCR only needs the 4 hooves anyway)?
- [ ] Priority: same-camera Re-ID (v1) before cross-camera (needs both cams
      synchronized).

---

## 10. Sources

- Paper: `cattle-researchpaper.pdf` (repo root)
- Baseline code & docs: `cattle_osnet/` (run.py, prep_videos.py, annotate.py, docs.md)
