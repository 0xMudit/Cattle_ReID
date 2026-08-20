#!/bin/bash
source /home/techteam/miniconda3/etc/profile.d/conda.sh
conda activate cattle
cd /home/techteam/0xmudit_cattle_reID

echo "=== Starting HanwooReID Training (GPU-max) ==="
echo "Time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

python training/train_v3.py \
    --epochs 60 \
    --batch-size 128 \
    --eval-freq 2 \
    --num-workers 16 \
    --pk-p 16 \
    --pk-k 8 \
    2>&1 | tee training_run.log

echo "=== Training Complete at $(date) ==="