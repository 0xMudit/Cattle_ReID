# Cattle Re-Identification System

Identify individual cattle from images using AI — facial recognition, but for cows.

Two complementary pipelines: a **zero-shot OSNet** approach (no training needed) and a **supervised training** pipeline (Colab/Kaggle). A HanwooReID paper study provides the state-of-the-art upgrade path.

---

## Model Weights

All model weights are hosted on HuggingFace and downloaded automatically by `scripts/download_weights.py`.

| Model | HuggingFace | Size | Purpose |
|-------|------------|------|---------|
| OSNet x1.0 | [0xmudit/cattle-reid-weights](https://huggingface.co/0xmudit/cattle-reid-weights/blob/main/osnet_x1_0_imagenet.pth) | 10.4 MB | Re-ID backbone (ImageNet-pretrained) |
| Cow Pose (YOLOv8m-pose) | [0xmudit/cattle-reid-weights](https://huggingface.co/0xmudit/cattle-reid-weights/blob/main/cow_pose.pt) | 50.7 MB | 12 cow keypoint detection |
| YOLOv8n | [0xmudit/cattle-reid-weights](https://huggingface.co/0xmudit/cattle-reid-weights/blob/main/yolov8n.pt) | 6.2 MB | Cow detection (COCO class 21) |

**Repository:** [huggingface.co/0xmudit/cattle-reid-weights](https://huggingface.co/0xmudit/cattle-reid-weights)

```bash
# Download all weights (~67 MB total)
python scripts/download_weights.py

# Or download individually
python scripts/download_weights.py --osnet
python scripts/download_weights.py --pose
python scripts/download_weights.py --yolo
```

---

## Quick Start

### Zero-shot (local, no training)

```bash
pip install -r requirements.txt
python scripts/download_weights.py   # ~67 MB from HuggingFace

cd cattle_osnet
python run.py --rebuild --threshold 0.6   # gallery/query matching
python run.py -i queries/some_cow.jpg     # match a single image
python annotate.py --src output/frames --out output/annotated   # skeleton + tag overlays
python prep_videos.py --videos ../Dataset --out output/vidcrops # cow crops from videos
```

### Supervised training (Colab, T4 GPU)

Open `cattle_reid_colab_fixed.ipynb` in [Google Colab](https://colab.research.google.com/), switch runtime to T4 GPU, run all cells. ~30-45 min end-to-end.

### Kaggle pipeline (GPU, T4 or better)

```bash
pip install kaggle
kaggle auth login
python kaggle/run_pipeline.py --user <your-kaggle-username> run
```

> **Note:** Model weights (`*.pth`, `*.pt`) are not committed. Download with `python scripts/download_weights.py` or they'll be fetched automatically on first run. Source footage (`Dataset/*.mp4`) is also gitignored.

---

## Repository Layout

```
Cattle_ReID/
├── README.md                         # this file
├── LICENSE                           # MIT
├── requirements.txt                  # Python dependencies
├── scripts/
│   └── download_weights.py           # download model weights from HuggingFace
├── Assets/                           # pipeline diagrams
├── cattle_osnet/                     # zero-shot OSNet Re-ID + pose visualization
│   ├── run.py                        # gallery/query matching with OSNet embeddings
│   ├── prep_videos.py                # videos → frame sampling → YOLO detection → cow crops
│   ├── annotate.py                   # skeleton + head-tag overlay (no bounding boxes)
│   ├── annotate_video.py             # video annotation
│   ├── models/
│   │   ├── osnet.py                  # OSNet architecture (KaiyangZhou/deep-person-reid)
│   │   ├── osnet_x1_0_imagenet.pth   # ImageNet-pretrained weights (auto-downloaded)
│   │   └── cow_pose.pt              # YOLOv8m-pose for cow keypoints (auto-downloaded)
│   ├── yolov8n.pt                    # COCO detection (cow class id = 21, auto-downloaded)
│   ├── gallery/<cow_id>/*.jpg        # known identities (one folder per cow)
│   ├── queries/*.jpg                 # unknown images to match
│   └── output/                       # annotated results
├── experiments/                      # ResNet / contrastive / multi-backbone research
│   ├── cattle_resnet.py              # ResNet18/34/50 backbone + embedding extractor
│   ├── contrastive_pretrain.py       # self-supervised NTXent pre-training
│   ├── multi_backbone.py             # swappable backbones + benchmark (11 architectures)
│   ├── kfold_eval.py                 # k-fold cross-validation
│   └── knn_matcher.py                # k-NN matching vs. mean-embedding matching
├── kaggle/                           # Kaggle pipeline (6 notebooks + CLI runner)
│   ├── 01_annotate_video.ipynb       # GPU video annotation demo
│   ├── 02_extract_crops.ipynb        # YOLO detection + crop extraction
│   ├── 02b_pose_meta.ipynb           # pose keypoint metadata generation
│   ├── 03_train_hanwoo_reid.ipynb    # supervised ReID training (ViT + PHE)
│   ├── 04_vcr_eval.ipynb             # viewpoint-constrained retrieval eval
│   ├── 05_open_set_vcr.ipynb         # open-set VCR evaluation
│   ├── 06_reid_video_demo.ipynb      # annotated demo video generation
│   ├── run_pipeline.py               # CLI: push datasets, run notebooks, chain steps
│   ├── upload_kaggle.py              # dataset upload helper
│   └── make_meta_local.py            # local meta.json regeneration
├── cattle_reid_colab_fixed.ipynb     # supervised OSNet training notebook (Colab)
└── cattle_reid_master.ipynb          # legacy notebook (Kaggle/Colab compatible)
```

---

## Pipeline Architecture

```
Source video / images
        │
        ▼
YOLOv8n detection (cow class 21) ──▶ cow crops
        │
        ├──▶ OSNet x1.0 embedding (512-dim) ──▶ gallery match ──▶ identity
        │
        └──▶ cow_pose.pt (12 keypoints) ──▶ skeleton + head tag overlay
```

Two separate concerns:
1. **Re-ID** — "is this cow the same as the one in the gallery?" (embeddings)
2. **Visualization** — "show me the cows and where their joints are" (skeleton/tags)

---

## Zero-Shot OSNet (Local Pipeline)

The `cattle_osnet/` directory implements a **zero-shot** Re-ID pipeline — no training, no labelled identity dataset.

### How it works

1. **Gallery build**: every image in `gallery/<cow_id>/` is embedded; each cow becomes a **mean embedding** of its photos. Cached to `output/gallery.pkl`.
2. **Query embed**: each image in `queries/` (or `--image`) is embedded the same way.
3. **Match**: nearest gallery mean by **cosine similarity** (+ L2 distance reported). Below `--threshold` (default 0.6) → **Unknown**.

### Embedding recipe

```
BGR → RGB → resize 256×128 → /255 → ImageNet normalize
→ forward through osnet_x1_0 → flatten → L2-normalize
```

### Why not `torchreid`?

PyPI's `torchreid==0.2.5` is a fake/stub package. We use the standalone `models/osnet.py` (from `KaiyangZhou/deep-person-reid`) with `osnet_x1_0_imagenet.pth`.

### Results

- **Smoke test**: self-match cos = 1.000; brown vs black cow cross-match cos = 0.425 → cleanly separated.
- **Real footage** (8 cow crops from re-encoded frames):
  - Same-video A1 pairs: cos 0.78–0.92
  - A1 vs A3 (different camera/video): cos 0.66–0.76

The ImageNet-pretrained OSNet gives reasonable within-camera clustering but weaker cross-camera discrimination. Fine-tuning on cattle data is the next step.

### Source footage notes

The `Dataset/` directory contains 6 CCTV videos (~1.8 GB total). All are HEVC (H.265) and some streams are corrupted — re-encode to H.264 before use:

```bash
ffmpeg -i A1.mp4 -c:v libx264 -crf 23 -preset fast -pix_fmt yuv420p A1_h264.mp4
```

| File | Duration | Resolution | Status |
|------|----------|------------|--------|
| A1.mp4 | 5 min | 1920×1080 | OK after re-encode |
| A2.mp4 | 5 min | 1920×1080 | 0 cows detected |
| A3.mp4 | 17 min | 1920×1080 | Truncated, needs re-encode |
| ch07m_*.mp4 | 26 min | 2880×1620 | CCTV, empty pen at sampled times |
| ch10m_*.mp4 (×2) | 10+25 min | 2880×1620 | CCTV, empty pen at sampled times |

---

## Supervised Training (Colab Notebook)

The `cattle_reid_colab_fixed.ipynb` notebook fixes 8 critical bugs in the original implementation and trains an OSNet model on the CID (Cow Images Dataset).

### What the notebook does

| Cell | Step | What Happens |
|------|------|-------------|
| 1-2 | Setup | Install packages, verify GPU |
| 3 | Data download | Download CID images from S3 |
| 4 | Model loading | YOLOv8n (cow class 21) + OSNet |
| 5 | Prep class | YOLO detection → crop → letterbox resize → augmentation |
| 6 | Split & process | 70% train / 15% gallery / 15% query (identity-based split) |
| 7 | CattleDS | Custom torchreid dataset |
| 8 | Model setup | OSNet with triplet + softmax loss |
| 9 | Training | 30 epochs, ~15-20 min on T4 |
| 10 | Gallery registration | Store known cow embeddings |
| 11 | Recognizer | End-to-end detection + embedding + matching |
| 12 | Test | Run on a query image |
| 13 | ONNX export | Export for deployment |

### Training configuration

```python
CFG = {
    'name': 'osnet_x1_0', 'h': 256, 'w': 192,
    'bs': 32, 'lr': 0.003, 'ep': 30, 'eval': 5,
    'step': 10, 'm': 0.3, 'wt': 1, 'wx': 50,
}
```

| Parameter | Meaning |
|-----------|---------|
| Learning rate (0.003) | Step size for weight updates |
| Batch size (32) | Images per training step |
| Epochs (30) | Full passes through the dataset |
| Triplet margin (0.3) | Min distance between different cows' embeddings |
| Loss weights (wt=1, wx=50) | Softmax weight higher for faster class discrimination |

### Expected results (30 epochs, T4 GPU)

| Metric | Range | Interpretation |
|--------|-------|----------------|
| mAP | 0.75-0.85 | Overall ranking quality |
| Rank-1 | 0.70-0.80 | Top match is correct |
| Rank-5 | 0.88-0.95 | Correct cow in top 5 |

### Key fixes applied

| # | Bug | Fix |
|---|-----|-----|
| 1 | `COW_CLS = 19` (horse, not cow) | `COW_CLS = 21` |
| 2 | No train/gallery/query split | Identity-based 70/15/15 split |
| 3 | Background images saved as cows | Skip images where YOLO finds no cow |
| 4 | Plain resize (stretched) | Letterbox resize (aspect-ratio preserving) |
| 5 | Deprecated YOLOv5 via torch.hub | YOLOv8 via ultralytics |
| 6 | Deprecated `Variable` for ONNX | Plain tensor |
| 7 | Duplicate torchreid installs | Single source: GitHub `deep-person-reid` |
| 8 | No version pinning | `torch==2.1.0`, `torchvision==0.16.0` |

---

## Kaggle Pipeline

The `kaggle/` directory implements a HanwooReID-style pipeline on CCTV footage using Kaggle's free GPU tier.

### Pipeline steps

| Step | Notebook | What it does |
|------|----------|-------------|
| 02 | `02_extract_crops.ipynb` | YOLO detection + crop extraction from videos |
| 02b | `02b_pose_meta.ipynb` | Pose keypoint metadata via `cow_pose.pt` |
| 03 | `03_train_hanwoo_reid.ipynb` | Supervised ReID training (ViT + PHE) |
| 04 | `04_vcr_eval.ipynb` | Viewpoint-constrained retrieval evaluation |
| 05 | `05_open_set_vcr.ipynb` | Open-set VCR evaluation |
| 06 | `06_reid_video_demo.ipynb` | Annotated demo video generation |

### Results

Training (60 epochs, P100, 21 train / 9 eval identities):

| Metric | with PHE | no PHE |
|--------|----------|--------|
| mAP | 100.00 | 100.00 |
| Rank-1 | 96.25 | 96.25 |
| Rank-5 | 98.12 | 98.12 |

Results saturate because the eval set is small (9 identities) and the ViT-B model easily separates them. The PHE/VCR deltas are the point — they show on larger, harder sets.

### CLI usage

```bash
# one-time setup
pip install kaggle
kaggle auth login

# upload videos + run the full pipeline
python kaggle/run_pipeline.py --user <username> push-videos
python kaggle/run_pipeline.py --user <username> run

# or run individual steps
python kaggle/run_pipeline.py --user <username> run --only 03
python kaggle/run_pipeline.py --user <username> status
python kaggle/run_pipeline.py --user <username> logs 03
```

### Bugs fixed along the way

- **Input mounting** — Kaggle's new mount nests inputs at `/kaggle/input/datasets/<owner>/<ds>/`. All notebooks now glob recursively.
- **CUDA arch** — Tesla P100 (sm_60) needs `cu118` wheels. Cell 1 pins `torch==2.5.1 torchvision==0.20.1 --index-url .../whl/cu118`.
- **JSON serialization** — `np.int64` is not JSON serializable. Cast to `int` and added a `default` handler.
- **Heatmap dtype** — `np.arange(GRID)` produced int64, broke float arithmetic. Fixed with `dtype=np.float32`.
- **Resumability** — `run_pipeline.py run --resume` reuses an existing kernel instead of pushing a new one.

---

## HanwooReID Paper Study

**Paper:** "HanwooReID: Multi-view cattle re-identification with pose-aware transformer enhancements" (Liu et al., Computers and Electronics in Agriculture, 2025)
**PDF:** `cattle-researchpaper.pdf` (repo root, gitignored)

Hanwoo cattle have **no distinctive coat markings**, so appearance-only Re-ID fails. The paper builds a Transformer-based framework with two novel modules:

### PHE — Pose-Guided Heatmap Encoder

Converts 2D pose keypoints into Gaussian heatmaps, encodes them with 1×1 conv layers, and adds the result to ViT patch embeddings as an **attention prior**. The model focuses on identity-relevant anatomy without hard dependency on pose quality.

### VCR — Viewpoint-Constrained Retrieval

Projects hoof keypoints onto a bird's-eye-view plane using camera calibration, estimates the cow's heading, and **filters out gallery images with mismatched orientation** before matching. Costs only 0.00013 s/cow.

### Results from the paper

| Method | Imsil-day1 mAP | Imsil-day4 mAP | Namwon-cam5 mAP |
|--------|---------------|---------------|-----------------|
| AGW (ResNet-50) | 61.5 | 80.4 | 72.9 |
| TransReID baseline | 83.2 | 92.3 | 62.8 |
| **Ours (PHE + VCR)** | **94.0** | **95.3** | 80.6 |

### Our data assessment

| File | Codec | Cows found | Notes |
|------|-------|------------|-------|
| A1.mp4 | HEVC | 3 detections (last minute only) | Phone footage |
| A2.mp4 | HEVC | **0** | Phone footage, appears cowless |
| A3.mp4 | HEVC | 14 detections (sporadic) | Phone footage |
| ch07m_*.mp4 | HEVC | **0** (53 samples) | CCTV, morning, empty pen |
| ch10m_*.mp4 (×2) | HEVC | **0** (71 samples) | CCTV, evening, empty pen |

Not usable as-is for supervised multi-view Re-ID training — needs identity labels, camera calibration, and confirmed cow frames.

### Implementation roadmap

1. `cattle_osnet/transformer_reid/` — TransReID-style ViT-B/16 + PHE
2. `cattle_osnet/pose_heatmaps.py` — kpts → Gaussian heatmaps → encoder
3. `cattle_osnet/calibrate.py` — DLT camera calibration
4. `cattle_osnet/vcr.py` — hoof back-projection + constrained retrieval

---

## Experiments & Research Code

The `experiments/` directory contains research code adapted from the [CowIDentifier](https://github.com/Phoenix4582/CowIDentifier) project:

| File | Lines | Purpose |
|------|-------|---------|
| `cattle_resnet.py` | 253 | ResNet18/34/50 backbone with cattle fine-tuning |
| `contrastive_pretrain.py` | 405 | Self-supervised NTXent pre-training with hard negative mining |
| `multi_backbone.py` | 380 | 11 swappable backbones + benchmark |
| `kfold_eval.py` | 302 | K-fold cross-validation for reliable metrics |
| `knn_matcher.py` | 239 | KNN majority voting (replaces simple mean-embedding matching) |

### Supported backbones

| Backbone | Params | Embedding Dim | Best For |
|----------|--------|---------------|----------|
| resnet18 | 11.7M | 512 | Baseline, well-tested |
| resnet34 | 21.3M | 512 | Slightly better than resnet18 |
| resnet50 | 23.5M | 2048 | Richer features |
| efficientnet_b0 | 5.3M | 1280 | **Best accuracy/speed tradeoff** |
| efficientnet_b2 | 9.1M | 1408 | Better accuracy than b0 |
| mobilenet_v3_small | 2.5M | 576 | **Fastest** — edge deployment, CCTV |
| mobilenet_v3_large | 5.4M | 960 | Edge with better accuracy |
| convnext_tiny | 28.6M | 768 | **Maximum accuracy** |
| convnext_small | 50.2M | 768 | When accuracy is everything |
| swin_tiny | 28.3M | 768 | Global body features |
| swin_small | 50.0M | 768 | Best global understanding |

---

## Key Concepts

| Term | Definition |
|------|-----------|
| **Embedding** | A 512-dimensional numerical vector representing a cow's visual features — a "fingerprint" |
| **Gallery** | Database of known cows with their embeddings |
| **Query** | An unknown cow image we want to identify |
| **Re-ID** | Re-identification — recognizing the same individual across different images |
| **Threshold** | Cutoff for matching (distance < 0.6 = same cow, ≥ 0.6 = Unknown) |
| **Transfer Learning** | Reusing a model trained on one task for another (person ReID → cattle) |
| **PHE** | Pose-Guided Heatmap Encoder — adds anatomical structure as attention prior |
| **VCR** | Viewpoint-Constrained Retrieval — filters by camera heading before matching |
| **ONNX** | Universal model format for deployment on edge devices |
| **YOLO** | You Only Look Once — fast object detection (cow class = 21 in COCO) |
| **OSNet** | Omni-Scale Network — learns features at multiple scales simultaneously |
| **CID** | Cow Images Dataset — the training dataset hosted on Amazon S3 |
| **NTXentLoss** | Normalized Temperature-scaled Cross Entropy — self-supervised contrastive loss |

---

## Dependencies

```bash
# Core
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git

# Computer vision + utilities
pip install albumentations onnx onnxruntime ultralytics matplotlib tqdm huggingface_hub

# For Kaggle pipeline
pip install kaggle

# For contrastive pre-training (experiments/)
pip install pytorch_metric_learning
```

> Do NOT install `torchreid` from PyPI — it's a stub. Use the GitHub `deep-person-reid` package.

---

## FAQ

**Q: YOLO doesn't detect any cows?**
Check `COW_CLS = 21` (not 19 = horse). Try `YOLO('yolov8m.pt')` for better detection. Lower confidence: `model(img, conf=0.1)`.

**Q: All results show "Unknown"?**
Gallery may be empty — re-run gallery registration. Threshold may be too strict — try `thr=0.8`. Model may need more training epochs.

**Q: Training is too slow?**
Ensure you're using a GPU (`Runtime → Change runtime type → T4 GPU`). Reduce `CFG['ep']` to 15, or use fewer images per cow.

**Q: Can I run without GPU?**
Yes for inference (acceptably fast). Training on CPU takes 6-12 hours vs 15-20 min on T4.

**Q: How do I add my own cow photos?**
Organize by cow ID folders, upload to Colab, copy into `data/raw/images/`, re-run processing cells.

**Q: Can I resume training?**
Yes — checkpoints are saved automatically. Find them in `logs/osnet_x1_0/model/`.

**Q: Installation fails with `ModuleNotFoundError: No module named 'torchreid'`?**
You installed the fake PyPI `torchreid`. Uninstall it and use `pip install git+https://github.com/KaiyangZhou/deep-person-reid.git` instead. Restart runtime after install.

---

## Known Limitations

| Limitation | Impact |
|------------|--------|
| Train/inference resize mismatch | Training uses letterbox, inference uses plain resize |
| YOLOv8n is smallest variant | May miss cows in cluttered scenes |
| Mean embedding matching | May not capture multi-modal appearances (front vs side view) |
| No temporal smoothing | Each video frame processed independently |
| 30 epochs may be insufficient | More epochs needed for >200 cows |
| Cow pose model is weak | Trained on only 341 images — many partial skeletons |
| Cross-camera discrimination is soft | OSNet zero-shot gives cos 0.66–0.76 across cameras |

The system fails most with: poor lighting, extreme occlusion, very similar-looking cows (solid colors), low-resolution images, unusual angles.

---

## Future Work

**Quick wins:** Try YOLOv8m/l, increase epochs to 50-100, add more gallery images per cow.

**Medium-term:** k-NN gallery matching, fine-tune YOLOv8 on cattle, ONNX → TensorRT conversion, Kalman filtering for video tracking.

**Advanced:** Video-level tracking, multi-view fusion, active learning, Siamese networks, edge deployment on Jetson Nano.

**HanwooReID upgrade:** Implement ViT-B/16 + PHE + VCR as described in the paper study above.

---

## License

MIT License — see [LICENSE](LICENSE).

---

> **Last Updated:** August 2026
> **Questions?** Open an issue or discussion on the repository.
