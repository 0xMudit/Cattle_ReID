#!/bin/bash
set -e

echo "=== Setting up Cattle ReID training environment ==="

# Clone repo
cd ~
if [ ! -d "Cattle_ReID" ]; then
    git clone https://github.com/0xmudit/Cattle_ReID.git
fi
cd Cattle_ReID

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
pip install albumentations ultralytics tqdm huggingface_hub pytorch_metric_learning jupyter onnx onnxruntime matplotlib

# Download weights
python scripts/download_weights.py

# Create Dataset directory
mkdir -p Dataset

echo "=== Setup complete! ==="
echo "Next: Transfer videos with:"
echo '  scp "C:\path\to\Dataset\*.mp4" techteam@100.67.41.64:~/Cattle_ReID/Dataset/'
echo "Then run training:"
echo "  cd ~/Cattle_ReID && jupyter nbconvert --to notebook --execute cattle_reid_colab_fixed.ipynb"
