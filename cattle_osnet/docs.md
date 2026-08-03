# Cattle Re-ID & Pose Pipeline — End-to-End Documentation

Goal: identify **individual cows** from camera footage and visualize them with
per-cow tags and skeleton overlays.

This doc covers the full journey: the source footage problem, the zero-shot
re-identification (Re-ID) approach with OSNet, the detection + tracking +
pose/visualization layers, what works, what doesn't, and how to run it all.

---

## 1. High-level architecture

```
Source video (A1/A2/A3.mp4)
        │  ffmpeg re-encode HEVC→H.264  (source streams are corrupt)
        ▼
H.264 videos  ──prep_videos.py──▶  frame sampling + YOLO cow detection
        │                              + centroid tracking + crop saving
        ▼
cow crops per identity (video_cow_<id>/)
        │  OSNet x1.0 (ImageNet-pretrained, zero-shot)
        ▼
512-dim embeddings ── run.py ──▶  gallery mean per cow, cosine match
                                 against queries → Known / Unknown

Visualization (separate path):
frame ──▶ YOLOv8n (detect cows) ──▶ cow crops ──▶ cow_pose.pt (12 keypoints)
        ▶ draw skeleton + head tag (no bounding box) → output/annotated/
```

Two separate concerns:

1. **Re-ID** — "is this cow the same as the one in the gallery?" (embeddings).
2. **Visualization** — "show me the cows and where their joints are" (boxes/skeleton/tags).

---

## 2. Directory layout

```
cattle_osnet/
├── run.py                  # zero-shot OSNet gallery/query matching (main Re-ID script)
├── prep_videos.py          # frames → YOLO detection → tracking → cow crops
├── models/
│   ├── osnet.py            # standalone OSNet architecture (KaiyangZhou deep-person-reid)
│   ├── osnet_x1_0_imagenet.pth   # ImageNet-pretrained weights (HF kaiyangzhou/osnet)
│   └── cow_pose.pt         # YOLOv8m-pose fine-tuned for 12 cow keypoints (HF)
├── yolov8n.pt              # COCO detection model (cow class id = 21)
├── gallery/<cow_id>/*.jpg  # known identities (one folder per cow)
├── queries/*.jpg           # unknown images to match
├── output/
│   ├── gallery.pkl         # cached per-cow mean embeddings
│   ├── frames/             # extracted test frames (A1_t030.jpg, ...)
│   ├── annotated/          # tagged + skeleton overlay images
│   ├── reencoded/          # H.264 versions of source videos
│   └── vidcrops/           # per-track cow crops + meta.json
└── docs.md                 # this file
```

---

## 3. Environment

- **OS**: Windows (win32), PowerShell 5.1
- **Python**: 3.12 at
  `C:\Users\0xmud\AppData\Local\Programs\Python\Python312\python.exe`
  ⚠️ *Always use this interpreter.* The default `python` on PATH is a 3.11 venv
  without pip/torch.
- **Key packages**:
  - `torch==2.5.1+cpu`, `torchvision==0.20.1+cpu`
  - `ultralytics==8.3.60` (YOLOv8 detection + pose)
  - `opencv-python` (frame/crop handling)
  - `Pillow==11.1.0` (beautiful overlay rendering, TTF fonts)
  - `huggingface_hub` (model downloads)

---

## 4. Source footage & the video problem

Source files: `C:\Users\0xmud\OneDrive\Pictures\Cattle Repo\Dataset\`

| File | Duration | Frames | Note |
|------|----------|--------|------|
| A1.mp4 | 300 s | 7,494 | HEVC Main, 1920×1080 @ 25 fps |
| A2.mp4 | 300 s | 7,499 | HEVC Main, 1920×1080 @ 25 fps |
| A3.mp4 | 1,020 s | 25,500 | HEVC Main, 1920×1080 @ 25 fps |

### The corruption problem

The original files are **HEVC (H.265)** and the streams are **broken**:

- ffmpeg/OpenCV report:
  `Could not find ref with POC N` / `Error constructing the frame RPS`
- Sequential decode collapses to a nearly **static field frame**
  (pixel variance ≈ 6,300 across the whole video) → YOLO finds **0 cows**.

So no amount of tuning the pipeline helps until the video itself is fixed.

### The fix: re-encode to H.264

```
ffmpeg -i A1.mp4 -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p A1_h264.mp4
```

Re-encoding produces decodable H.264 files where real cows are visible and
detectable.

| Output | Size | Status |
|--------|------|--------|
| A1_h264.mp4 | 258 MB | ✅ OK |
| A2_h264.mp4 | 303 MB | ✅ OK |
| A3_h264.mp4 |  90 MB | ⚠️ **truncated** ("moov atom not found") — needs re-encode before use |

---

## 5. Re-identification (Re-ID) with OSNet

`run.py` implements a **zero-shot** Re-ID pipeline — no training, no labelled
identity dataset.

### Why not `torchreid`?

PyPI's `torchreid==0.2.5` is a **fake/stub package** that doesn't install the
real library. Instead we use the standalone `models/osnet.py` (from
`KaiyangZhou/deep-person-reid`) with the ImageNet-pretrained
`osnet_x1_0_imagenet.pth`.

### How it works

1. **Gallery build**: every image in `gallery/<cow_id>/` is embedded; each cow
   becomes a **mean embedding** of its photos. Result cached to
   `output/gallery.pkl` (use `--rebuild` to force).
2. **Query embed**: each image in `queries/` (or `--image`) is embedded the
   same way.
3. **Match**: nearest gallery mean by **cosine similarity** (+ L2 distance
   reported). Best cosine below `--threshold` (default **0.6**) → **Unknown**.

Key parameters (`run.py`):

| Constant | Value | Meaning |
|----------|-------|---------|
| `IMG_H / IMG_W` | 256 × 128 | input size for OSNet |
| `COS_THRESHOLD` | 0.6 | below this a query is "Unknown" |
| embedding dim | 512 | normalized L2 vector |

### Embedding recipe

```python
BGR → RGB → resize 256×128 → /255 → ImageNet normalize
(mean 0.485/0.456/0.406, std 0.229/0.224/0.225)
→ forward through osnet_x1_0 → flatten → L2-normalize
```

### Results so far

- **Smoke test (sample photos)**: self-match cos = **1.000**; brown vs black
  cow cross-match cos = **0.425** → cleanly separated.
- **Real footage test** (8 cow crops from re-encoded frames):
  - same-video A1 pairs: cos **0.78–0.92**
  - A1 vs A3 (different camera/video): cos **0.66–0.76**
- Interpretation: the ImageNet-pretrained OSNet gives reasonable *within-camera*
  clustering but weaker *cross-camera* discrimination. Fine-tuning on cattle
  data would be the next step.

---

## 6. Video prep: `prep_videos.py`

Builds per-cow crop sets from video for the gallery:

1. Sample every `--step` frames (default **50** = 2 s @ 25 fps).
2. **YOLOv8n detection** filtered to class 21 (`COW_CLS`), `--conf` default 0.25.
3. **Centroid tracker** (`Tracker` class) assigns a stable id per cow across
   samples (greedy nearest-centroid match, threshold `0.5 × max(w,h)`, EMA
   smoothing `0.7·old + 0.3·new`).
4. **Crop** each detection with 5% margin (`expand_box`), save to
   `output/vidcrops/<video>_cow_<id>/f<frame>.jpg`.
5. Only tracks with ≥ 2 crops are kept; summary written to `output/vidcrops/meta.json`.

**Current status**: `meta.json` shows `tracks: {}` — the prep was run on the
corrupt HEVC files (0 detections). It has **not yet been re-run** on the H.264
re-encodes. This is the immediate next step.

```bash
python prep_videos.py --videos output/reencoded --out output/vidcrops --step 50
```

---

## 7. Detection + skeleton + tag visualization

Final visual output: **no bounding boxes** — a color-coded **skeleton** per cow
plus a **tag pill over the cow's head**, plus a header panel.

### Models used

| Model | File | Purpose |
|-------|------|---------|
| YOLOv8n | `yolov8n.pt` | cow detection (class 21), conf 0.15 in tests |
| Cow pose (YOLOv8m-pose fine-tune) | `models/cow_pose.pt` | 12 cow keypoints |
| OSNet | `models/osnet_x1_0_imagenet.pth` | Re-ID embeddings |

### Cow pose model

- Source: HuggingFace `luciayen/yolov8-cow_pose-model` (`best.pt`)
- Base: YOLOv8m-pose pretrained on COCO, fine-tuned 200 epochs
- Trained on only **341 images** (Merged Cow Pose Estimation Dataset, Kaggle)
  → works, but with limited recall/quality on our field footage.
- **12 keypoints** (order matters):

| Idx | Keypoint | Idx | Keypoint |
|-----|----------|-----|----------|
| 0 | Nose | 6 | LB_Hoof |
| 1 | R_Eye | 7 | RB_Hoof |
| 2 | L_Eye | 8 | Backbone |
| 3 | Neck | 9 | TailRoot |
| 4 | LF_Hoof | 10 | BackPose |
| 5 | RF_Hoof | 11 | Stomach |

- **Skeleton edges** drawn: head (nose–eyes–neck), spine (neck–backbone–tail),
  legs (neck→front hooves, tail→back hooves), body (backbone–backpose–stomach).

### Why hybrid detection?

The pose model alone detects far fewer cows than YOLOv8n (341-image training
set). So we:
1. Detect cows with YOLOv8n (high recall).
2. Crop each detected cow (12 px margin) and run `cow_pose.pt` on the crop.
3. Scale keypoints back to full-frame coordinates.
4. Draw skeleton + tag; fall back to the box top for the tag if no head
   keypoints are visible.

### Rendering details (PIL)

- **Skeleton**: colored bones (5 px + white 2 px core) + white-ringed keypoint
  dots (keypoints below `KPT_CONF = 0.30` are skipped).
- **Tag pill**: dark glass rounded rect with colored border, numbered badge
  circle, `Cow_XX` label (Segoe UI Bold), confidence bar, leader line to the
  head anchor (mean of nose/eyes, else neck, else box top).
- **Header panel**: "CATTLE RE-ID • N cows detected • skeleton + head tag •
  <frame>".
- Per-cow color from an 8-color palette.

### Frame detection results

| Frame | Cows | Visible keypoints per cow |
|-------|------|---------------------------|
| A1_frame300 | 4 | 3, 1, 2, 6 |
| A1_t030 | 2 | 1, 0 |
| A1_t150 | 3 | 10, 1, 0 |
| A1_t270 | 2 | 3, 4 |
| A2_frame300 | 3 | 3, 4, 4 |
| A2_t060 | 0 | — |
| A3_frame300 | 1 | 10 |
| A3_t120 | 1 | 10 |

Best skeletons: `A1_t150` (10/12 kpts) and `A3` frames (10/12). Several cows get
0–1 keypoints → skeleton invisible, tag still shown.

### A known rendering bug (fixed)

Early "beautiful tag" output looked like the plain image. Cause:
`ov = Image.alpha_composite(ov, vg)` **reassigned** the overlay while the
`ImageDraw` object still referenced the old one — every box/tag/text was drawn
on a discarded layer. Fix: draw directly on the single `ov` layer. Verify by
pixel-diffing output vs. source (expect max diff 255 and tens of thousands of
changed pixels).

---

## 8. How to run

```powershell
# Re-ID (gallery + queries from folders)
$py = "C:\Users\0xmud\AppData\Local\Programs\Python\Python312\python.exe"
& $py run.py --rebuild --threshold 0.6

# Re-ID a single image
& $py run.py -i queries/some_cow.jpg

# Video prep (on re-encoded H.264!)
& $py prep_videos.py --videos output/reencoded --out output/vidcrops
```

The visualization scripts currently live in
`%TEMP%\opencode\` (`tag_frames.py`, `beauty_tag.py`, `skeleton_tag.py`).
**Recommendation:** move `skeleton_tag.py` into the repo (e.g.
`annotate.py`) so the visualization is reproducible — otherwise a fresh session
loses it.

---

## 9. Known limitations & next steps

1. **A3_h264.mp4 is truncated** — re-encode A3 properly before using it.
2. **`prep_videos.py` not yet validated on clean H.264** — re-run it on
   `output/reencoded/` to populate `vidcrops/`.
3. **Cow pose model is weak** (341 training images) — many partial/empty
   skeletons. Retraining on cattle footage or a stronger dataset (e.g. AP-10K
   based) would help.
4. **OSNet is zero-shot (ImageNet)**, not cattle-tuned — cross-camera
   discrimination (0.66–0.76) is soft. Fine-tune OSNet on collected cow tracks
   once a good gallery exists.
5. Cross-frame **consistent IDs** (tracking the same tagged cow through the
   video) are not yet wired into the visualization; `prep_videos.py`'s tracker
   is the building block.
6. The corrupted HEVC source means **full-video** processing is only possible
   on the H.264 re-encodes (or after re-ripping the source).
