#!/usr/bin/env python3
"""Train OSNet on cattle data using torchreid ImageTripletEngine."""
import os
import torch
import torchreid
from torchreid.engine import ImageTripletEngine

from .config import CFG
from .dataset import register_cattle_dataset


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    dn = register_cattle_dataset()

    dm = torchreid.data.ImageDataManager(
        sources=dn,
        height=CFG["h"],
        width=CFG["w"],
        batch_size_train=CFG["bs"],
        batch_size_test=100,
        transforms=["random_flip", "random_crop"],
    )

    print(f"Dataset: {dm.num_train_pids} train identities, "
          f"{len(dm.query_dataset)} query, {len(dm.gallery_dataset)} gallery")

    model = torchreid.models.build_model(
        name=CFG["model_name"],
        num_classes=dm.num_train_pids,
        loss="triplet",
        pretrained=True,
    ).to(device)

    opt = torchreid.optim.build_optimizer(model, optim="adam", lr=CFG["lr"])
    sch = torchreid.optim.build_lr_scheduler(
        opt, lr_scheduler="single_step", stepsize=CFG["step"]
    )

    engine = ImageTripletEngine(
        dm, model,
        optimizer=opt,
        scheduler=sch,
        margin=CFG["margin"],
        weight_t=CFG["weight_t"],
        weight_x=CFG["weight_x"],
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")

    save_dir = os.path.join(CFG["logs_dir"], CFG["model_name"])
    engine.run(
        save_dir=save_dir,
        max_epoch=CFG["ep"],
        eval_freq=CFG["eval_freq"],
        print_freq=50,
    )
    print("Training complete!")
    return save_dir


if __name__ == "__main__":
    train()
