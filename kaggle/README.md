# Kaggle pipeline (run on Kaggle GPU, T4 or better)

## Current status (2026-08-14)

| Step | Status | Notes |
|---|---|---|
| `cattle-cctv-videos` dataset | READY | 6 mp4s, 1.9 GB |
| `cattle-cctv-weights` dataset | READY | `yolov8n.pt` + `cow_pose.pt` |
| `cattle-cctv-crops` dataset | READY | 234 MB, `crops/A1..A3/` + `meta.json` (705 KB, 2,088 crops) |
| 02 `cattle-02-extract-crops` | COMPLETE (v6) | 2,088 crops (A1: 1,363 / A2: 234 / A3: 491); CCTV `ch*` feeds had 0 cows |
| `make_meta_local.py` | DONE | regenerated `meta.json` locally on CPU (poses via `cow_pose.pt`); 1,297/1,781 crops with >=6 keypoints |
| 03 `cattle-03-train-reid` | COMPLETE (v2) | 60 epochs, ~47 min; mAP 100.0 / R1 96.25 / R5 98.12; `hanwoo_reid.pth` + `report.json` pulled |
| `cattle-reid-model` dataset | READY | `hanwoo_reid.pth` (346 MB) + `report.json` |
| 04 `cattle-04-vcr-eval` | COMPLETE (v2) | VCR eval done — see table below |
| 05 `cattle-05-open-set-vcr` | COMPLETE (v2) | open-set VCR eval done — see table below |
| 06 `cattle-06-reid-video-demo` | COMPLETE (v3) | annotated demo video — see below |

## Final results

Training (03) — 60 epochs on a P100 (~47 min), 21 train / 9 eval identities,
1,768 train / 157 gallery / 160 query crops:

| metric | with PHE | no PHE |
|---|---|---|
| mAP | 100.00 | 100.00 |
| Rank-1 | 96.25 | 96.25 |
| Rank-5 | 98.12 | 98.12 |

VCR eval (04) on the 9 eval identities (closed set, single-camera scene):

| setting | mAP | R1 | R5 |
|---|---|---|---|
| no VCR | 100.0% | 96.2% | 98.1% |
| VCR same bin | 100.0% | 96.2% | 98.1% |
| VCR ±1 bin | 100.0% | 96.2% | 98.1% |

Results saturate because the eval set is small (9 identities, ~160 query crops)
and the ViT-B model easily separates them; the PHE / VCR *deltas* are the
point of the paper and would show on a larger, harder set. PHE vs no-PHE and
VCR vs no-VCR are identical here — expected when accuracy is already ~100%.

Open-set VCR eval (05) — gallery = 157 crops / 9 eval identities, known
queries = 160 (eval ids), unknown queries = 271 (train ids, no gallery entry):

| setting | closed mAP | closed R1 | closed R5 | open mAP | AUROC | TPR@1%FPR |
|---|---|---|---|---|---|---|
| no VCR | 100.0% | 96.2% | 98.1% | 37.12% | 0.766 | 7.5% |
| VCR same bin | 100.0% | 96.2% | 98.1% | 37.12% | 0.758 | 7.5% |
| VCR ±1 bin | 100.0% | 96.2% | 98.1% | 37.12% | 0.765 | 7.5% |

Open-mAP = known queries scored with gallery retrieval + unknown queries
scored 0 (unknowns can't match). AUROC = separability of known vs unknown
top-1 similarity (0.5 = random, 1.0 = perfect); the model keeps 98.9% vs
72.9% mean top-1 similarity, so rejection is feasible, but VCR's viewpoint
filtering does not improve rejection on this small single-camera set.

## Demo video (06)

`A1_reid_demo.mp4` — first 30 s of A1, 375 frames at ~12.5 fps effective
(SKIP=2, 8.6 fps render on P100). Each cow gets a track box; labels are the
trained ReID's gallery identity (`Cow_0XX`) when the top-1 similarity is
>= `REID_CONF=0.72`, else `unknown` (open-set rejection). Pose heatmaps (PHE)
are computed live with `cow_pose.pt`. Preview frames in `.run/06_out/preview_*.png`.

## Total footage

~88 min (A1/A2 5 min, A3 17 min, ch07 26 min, ch10a 10 min, ch10b 25 min).
At `IMGSZ=1920`, `SAMPLE_EVERY=25` the extraction sampled ~5,300 frames and
took ~45 min on a P100.

### Bugs fixed along the way
- **Input mounting** — Kaggle's new mount nests inputs at
  `/kaggle/input/datasets/<owner>/<ds>/`, so `glob('/kaggle/input/*')` only
  returns `/kaggle/input/datasets`. All notebooks now glob recursively
  (`/kaggle/input/**/*.mp4` / `hanwoo_reid.pth`, `crops/` dir discovery).
- **CUDA arch** — Kaggle assigns a Tesla P100 (sm_60) for this account;
  both the preinstalled torch and the default `cu124` wheels dropped Pascal
  support ("no kernel image is available for execution on the device").
  Cell 1 pins `torch==2.5.1 torchvision==0.20.1 --index-url .../whl/cu118`,
  which ships sm_60→sm_90 kernels.
- **JSON serialization** — 02 died at the very end of extraction with
  `TypeError: Object of type int64 is not JSON serializable`: the
  `kpts_visible` count is a `np.int64`. Cast to `int` and added a `default`
  handler to `json.dump` so no numpy scalar can break the dump again.
- **Resumability** — `run_pipeline.py run --resume` reuses an existing /
  still-running kernel (waiting on it instead of pushing a new version), so a
  local crash or timeout never wastes a completed extraction or training run.
- **Heatmap dtype** — 03 v1 crashed at the first training step with
  `RuntimeError: Input type (double) and bias type (float)`. `make_heatmap`
  built the Gaussian grid with `np.arange(GRID)` (int64), so the exp arithmetic
  promoted the heatmap to float64. Fixed with
  `np.arange(GRID, dtype=np.float32)` in both 03 and 04.
- **04 viz cell** — 04 v1 ERROR'd in the ViewID visualization cell
  (`np.hstack` on crops of differing heights, 790 vs 792 px). The VCR metrics
  had already printed; the fix resizes crops to a common height before stacking.
  Note the v1 run still counts toward the hourly GPU quota even though only the
  debug plot crashed.
- **Local meta regeneration** — Kaggle CLI skips real subdirectories and
  auto-extracts `.tar` files, so re-uploading crops was unreliable. Instead,
  `make_meta_local.py` regenerates `meta.json` locally on CPU (same
  `cow_pose.pt`), and `build_crops_folder` packs `crops.tar` with entries
  `A1/...`, `A2/...`, `A3/...` + `meta.json` so the extracted layout on Kaggle
  is exactly `crops/A1/...` + `crops/meta.json` (matches the relpaths 03/04
  compute).

The four notebooks replicate the HanwooReID paper pipeline on the CCTV
footage in `Dataset/`. Two ways to run them:

1. **Web UI** — upload the `.ipynb` files by hand (steps below).
2. **Fully from the CLI** — `run_pipeline.py` pushes the dataset + notebooks,
   polls for completion, downloads outputs, and chains 02→03→04 automatically.

## CLI path (recommended, no manual steps)

```bash
# one-time setup
pip install kaggle
kaggle auth login   # browser OAuth

# check auth + upload the Dataset/ videos as a Kaggle dataset
python kaggle/run_pipeline.py --user <your-kaggle-username> check
python kaggle/run_pipeline.py --user <your-kaggle-username> push-videos

# run the whole chain: 02 extract -> 03 train -> 04 VCR eval (GPU, auto-chained)
python kaggle/run_pipeline.py --user <your-kaggle-username> run

# or only a single notebook, and check status/logs
python kaggle/run_pipeline.py --user <your-kaggle-username> run --only 03
python kaggle/run_pipeline.py --user <your-kaggle-username> status
python kaggle/run_pipeline.py --user <your-kaggle-username> logs 03
```

What the CLI does per step:

| Step | Command it runs | Creates |
|---|---|---|
| `push-videos` | `kaggle datasets create` | `cattle-cctv-videos` (Dataset/*.mp4) |
| `push-crops` | local re-pack + `kaggle datasets version` | `cattle-cctv-crops` (`crops.tar` + `meta.json`) |
| notebook 02 | `kaggle kernels push` → poll `kernels status` → `kernels output` | crops dataset `cattle-cctv-crops` |
| notebook 03 | same, input = crops dataset | `cattle-reid-model` dataset (`.pth` + report) |
| notebook 04 | same, inputs = crops + model | VCR comparison (printed / output files) |

Downloaded outputs land in `kaggle/.run/`. Note: 01 (annotate demo) is optional
and not part of the default chain.

## Web UI path

1. **Upload videos.** In Kaggle, create a new **Dataset** (or use the
   "Datasets" sidebar → New Dataset) and add your `.mp4` files from
   `Dataset/` (the `ch07m_*` / `ch10m_*` CCTV clips — the `A1/A2/A3.mp4`
   phone videos are older and have fewer usable far-field views).
2. **Create a notebook.** New Notebook → **Add Input** → select that dataset
   (and the output datasets from earlier steps as you go).
3. Each notebook's first cells auto-discover inputs recursively under
   `/kaggle/input/` (new Kaggle mount nests under `datasets/<owner>/<ds>/`),
   so no path edits are needed.
4. **02 note:** adjust `IMGSZ` (default 1920 — required for the 2880x1620
   CCTV feeds) and `CONF` (default 0.15). Runs on GPU; the 6-video run takes
   roughly 30-60 min at `IMGSZ=1920`, `SAMPLE_EVERY=25`.
5. **03 note:** after training, download `hanwoo_reid.pth` + `report.json`
   from the Output tab. Upload them as an input dataset for notebook 04.

## Machine notes (this PC)

- 6 GB RAM, CPU only → do everything heavy on Kaggle.
- Local `annotate_video.py` (in `cattle_osnet/`) exists for quick CPU demos;
  notebook 01 is the GPU equivalent.

## Expected output

- `meta.json`: per-crop keypoints normalized to 0..1 (paper: pose → PHE).
- `tracks.json`: per-track identity + frame ranges (paper: temporal
  grouping via tracking).
- `report.json` from 03: mAP / Rank-1 / Rank-5 with and without PHE.
- 04's table: mAP / Rank-1 / Rank-5 with and without VCR gallery filtering.

## Paper targets (for reference)

Closed-set mAP up to ~94–95%, Rank-1 up to ~93–97%; open-set +6.8 mAP via
VCR; 69.7 FPS (batch 1) / 129.9 FPS (batch 32) on one H100. Our CCTV set is
a single camera scene with ~20-25 visible cows per frame, so results will be
lower but the *deltas* (PHE vs no-PHE, VCR vs no-VCR) are the story.
