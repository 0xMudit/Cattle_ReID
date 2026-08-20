#!/usr/bin/env python3
"""Train OSNet on cattle data — optimized for RTX 5080 (16GB VRAM).

Features:
  - AMP (mixed precision) for 2x faster training
  - torch.compile for kernel fusion (RTX 5080 Blackwell optimized)
  - Cosine annealing with warmup
  - Large batch size (256)充分利用 16GB VRAM
  - Multi-worker data loading (12 workers)
  - Identity-level sampling (4 instances per identity)
  - TF32 for matrix multiplications
"""
import os
import time
import torch
import torchreid
from torchreid.engine import ImageTripletEngine

from .config import CFG
from .dataset import register_cattle_dataset


class WarmupCosineScheduler:
    """Linear warmup then cosine annealing to lr_min."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, lr_min=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.base_lr = optimizer.param_groups[0]["lr"]
        self._epoch = -1

    def state_dict(self):
        return {"epoch": self._epoch}

    def load_state_dict(self, d):
        self._epoch = d.get("epoch", -1)

    def step(self, epoch=None):
        import math
        if epoch is not None:
            self._epoch = epoch
        else:
            self._epoch += 1
        epoch = self._epoch
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            import math
            lr = self.lr_min + 0.5 * (self.base_lr - self.lr_min) * (1 + math.cos(math.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"VRAM: {vram:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        print(f"TF32: {torch.backends.cuda.matmul.allow_tf32}")
        print(f"cuDNN benchmark: {torch.backends.cudnn.benchmark}")

    dn = register_cattle_dataset()

    dm = torchreid.data.ImageDataManager(
        sources=dn,
        height=CFG["h"],
        width=CFG["w"],
        batch_size_train=CFG["bs"],
        batch_size_test=100,
        transforms=["random_flip", "random_crop", "random_erase"],
        num_instances=CFG["num_instances"],
        workers=CFG["workers"],
        use_gpu=CFG["amp"],
    )

    print(f"Dataset: {dm.num_train_pids} train identities, "
          f"{len(dm.test_dataset)} test images")
    print(f"Config: bs={CFG['bs']}, workers={CFG['workers']}, "
          f"epochs={CFG['ep']}, lr={CFG['lr']}, amp={CFG['amp']}")

    model = torchreid.models.build_model(
        name=CFG["model_name"],
        num_classes=dm.num_train_pids,
        loss="triplet",
        pretrained=True,
    ).to(device)

    # torch.compile disabled — RTX 5080 Blackwell causes OOM with max-autotune
    # for OSNet's architecture at batch 256
    print("torch.compile: disabled (OOM with max-autotune on this hardware)")

    opt = torchreid.optim.build_optimizer(model, optim="adam", lr=CFG["lr"])

    if CFG["cosine_anneal"]:
        scheduler = WarmupCosineScheduler(
            opt, CFG["warmup_epochs"], CFG["ep"], CFG["lr_min"]
        )
    else:
        scheduler = torchreid.optim.build_lr_scheduler(
            opt, lr_scheduler="single_step", stepsize=CFG["step"]
        )

    engine = ImageTripletEngine(
        dm, model,
        optimizer=opt,
        scheduler=scheduler,
        margin=CFG["margin"],
        weight_t=CFG["weight_t"],
        weight_x=CFG["weight_x"],
        use_gpu=CFG["amp"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    save_dir = os.path.join(CFG["logs_dir"], CFG["model_name"])
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()
    engine.run(
        save_dir=save_dir,
        max_epoch=CFG["ep"],
        eval_freq=CFG["eval_freq"],
        print_freq=100,
    )
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed / 60:.1f} minutes")
    print(f"Checkpoints: {save_dir}")
    return save_dir


if __name__ == "__main__":
    train()
