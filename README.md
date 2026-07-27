# Cattle Re-Identification System

AI-powered individual cattle identification from images — like facial recognition, but for cows.

## Pipeline

```
Input Image → YOLOv8 (detect cows) → OSNet/ResNet (extract embedding) → Gallery Match → Identify
```

## Quick Start (Kaggle)

1. Open `cattle_reid_master.ipynb` in [Kaggle](https://www.kaggle.com/)
2. Enable GPU (Settings → Accelerator → GPU)
3. Run all cells
4. Model trains automatically (~20 min on T4 GPU)

## Quick Start (Google Colab)

1. Open `cattle_reid_master.ipynb` in [Colab](https://colab.research.google.com/)
2. Enable GPU (Runtime → Change runtime type → T4 GPU)
3. Run all cells

## Features

- **YOLOv8** cow detection (COCO class 21)
- **OSNet** cattle-pretrained re-identification
- **Multiple backbones** — ResNet, EfficientNet, MobileNet, ConvNeXt, Swin Transformer
- **Contrastive pre-training** (self-supervised, no labels needed)
- **K-Fold cross-validation** for reliable metrics
- **KNN matching** (more robust than simple L2)
- **ONNX export** for edge deployment

## Project Structure

```
Cattle_ReID/
├── cattle_reid_master.ipynb    # Master notebook (Kaggle/Colab)
├── cattle_reid_colab_fixed.ipynb  # Original notebook
├── multi_backbone.py           # Swappable backbone architectures
├── contrastive_pretrain.py     # Self-supervised pre-training
├── cattle_resnet.py            # ResNet backbone
├── knn_matcher.py              # KNN-based matching
├── kfold_eval.py               # K-Fold evaluation
├── cattle_reid_colab_fixed_docs.md  # Full documentation
└── README.md
```

## Backbones

| Model | Params | Speed | Best For |
|-------|--------|-------|----------|
| `osnet_x1_0` | 2.2M | Fast | Default (cattle-pretrained) |
| `efficientnet_b0` | 5.3M | Fast | Best accuracy/speed |
| `mobilenet_v3_small` | 2.5M | Fastest | Edge/CCTV deployment |
| `convnext_tiny` | 28.6M | Medium | Maximum accuracy |
| `swin_tiny` | 28.3M | Medium | Global body features |

## Dataset

Uses the [CID (Cow Images Dataset)](https://cid-21.s3.amazonaws.com/) — auto-downloaded in the notebook.

## Citations

```bibtex
@misc{yu2024multicamcows2024multiviewimage,
      title={MultiCamCows2024 -- A Multi-view Image Dataset for AI-driven 
             Holstein-Friesian Cattle Re-Identification on a Working Farm}, 
      author={Phoenix Yu and Tilo Burghardt and Andrew W Dowsey and Neill W Campbell},
      year={2024},
      eprint={2410.12695},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```

## License

MIT

## Authors

- **Khushbu** — Original notebook, documentation
- **Mudit** — Pipeline architecture, enhanced components
