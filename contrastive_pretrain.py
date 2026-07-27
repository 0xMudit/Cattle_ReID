"""
Contrastive Self-Supervised Pre-Training for Cattle ReID
=========================================================
Stolen from: https://github.com/Phoenix4582/CowIDentifier (Self-Supervised folder)
Adapted for cattle ReID without requiring labeled data.

How it works:
  1. Takes augmented views of the same cow image
  2. Pulls their embeddings closer (positive pair)
  3. Pushes embeddings of different cows apart (negative pairs)
  4. Uses NTXentLoss (Normalized Temperature-scaled Cross Entropy)
  5. MultiSimilarityMiner finds hard negatives for stronger training signal

Usage:
  python contrastive_pretrain.py --data_dir data/processed/train --epochs 50
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import resnet18, resnet34, resnet50
import numpy as np
from pathlib import Path
from glob import glob
import cv2
import argparse

try:
    from pytorch_metric_learning import miners, losses
    HAS_METRIC_LEARNING = True
except ImportError:
    HAS_METRIC_LEARNING = False
    print("WARNING: pytorch_metric_learning not installed. Install with:")
    print("  pip install pytorch_metric_learning")


# =============================================================================
# Augmentation Pipeline (from CowIDentifier, adapted)
# =============================================================================

class ContrastiveAugmentation:
    """
    Creates two different augmented views of the same image for contrastive learning.
    Each call produces a different random augmentation.
    """
    def __init__(self, imsize=128):
        self.transform = T.Compose([
            T.RandomResizedCrop(imsize, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
            T.RandomGrayscale(p=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class CattleContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning. Loads cow images from the processed directory.
    Each image is a crop of a detected cow (from YOLO preprocessing).
    """
    def __init__(self, data_dir, imsize=128):
        self.data_dir = data_dir
        self.imsize = imsize
        self.images = sorted(glob(os.path.join(data_dir, '*.jpg')))

        # Extract cow IDs from filenames (format: c0_p<COW_ID>_N.jpg)
        self.cow_ids = []
        for p in self.images:
            try:
                name = os.path.basename(p).split('_')
                cow_id = int(name[1][1:])  # p<COW_ID> -> <COW_ID>
                self.cow_ids.append(cow_id)
            except (IndexError, ValueError):
                self.cow_ids.append(-1)

        self.cow_ids = np.array(self.cow_ids)
        self.unique_cows = np.unique(self.cow_ids)
        self.augment = ContrastiveAugmentation(imsize)

        # Pre-load all images into memory (fast training)
        self.cache = {}
        for i, path in enumerate(self.images):
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imsize, imsize))
                self.cache[i] = img

        print(f"Loaded {len(self.cache)} images from {data_dir}")

    def __len__(self):
        return len(self.cache)

    def __getitem__(self, idx):
        img = self.cache[idx]
        # Convert to tensor [0, 1]
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        label = self.cow_ids[idx]

        # Apply contrastive augmentation (two different views)
        view1, view2 = self.augment(tensor)
        return view1, view2, label


# =============================================================================
# Model Architecture (from CowIDentifier)
# =============================================================================

class CattleContrastiveModel(nn.Module):
    """
    ResNet backbone with projection head for contrastive learning.
    After pre-training, remove the projection head and use the backbone
    for feature extraction.

    Based on CowIDentifier's MultiCamModel architecture.
    """
    def __init__(self, backbone='resnet18', hidden_dims=64, pretrained=True):
        super().__init__()
        # Load backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(weights='DEFAULT' if pretrained else None)
            feat_dim = 512
        elif backbone == 'resnet34':
            self.backbone = resnet34(weights='DEFAULT' if pretrained else None)
            feat_dim = 512
        elif backbone == 'resnet50':
            self.backbone = resnet50(weights='DEFAULT' if pretrained else None)
            feat_dim = 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # Remove original FC layer
        self.backbone.fc = nn.Identity()

        # Projection head (maps to embedding space)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, hidden_dims),
        )

        # For inference: just the backbone (no projector)
        self.embed_dim = hidden_dims

    def forward(self, x):
        """Forward pass through backbone + projection head."""
        features = self.backbone(x)
        embeddings = self.projector(features)
        return embeddings

    def extract_embedding(self, x):
        """Extract embedding without projection head (for inference)."""
        return self.backbone(x)


# =============================================================================
# NTXent Loss (from CowIDentifier)
# =============================================================================

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss.
    This is the core contrastive loss used in CowIDentifier.

    For each batch of N images, it creates 2N augmented views.
    For each anchor, the positive is its other view, and all other 2(N-1)
    images are negatives.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i: embeddings from view 1 [batch_size, embed_dim]
        z_j: embeddings from view 2 [batch_size, embed_dim]
        """
        batch_size = z_i.shape[0]

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate both views
        z = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, embed_dim]

        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]

        # Mask out self-similarity (diagonal)
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=sim.device)
        sim = sim.masked_select(mask).view(2 * batch_size, -1)

        # Positive pair indices
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(sim.device)

        # For each anchor, positive is at index (anchor + batch_size) % (2*batch_size)
        # After masking, adjust indices
        pos_sim = torch.cat([
            sim[torch.arange(batch_size), batch_size - 1:],
            sim[torch.arange(batch_size, 2 * batch_size), :batch_size]
        ], dim=0)

        # Actually, let's compute it more carefully
        # After masking diagonal, sim has shape [2N, 2N-1]
        # We need to find the positive for each anchor

        # Simpler approach: compute InfoNCE loss directly
        pos_sim = torch.sum(z_i * z_j, dim=1) / self.temperature  # [batch_size]

        # All similarities between view1 and view2
        all_sim = torch.mm(z_i, z_j.t()) / self.temperature  # [batch_size, batch_size]

        # InfoNCE: for each anchor in view1, the positive is the matching index in view2
        labels = torch.arange(batch_size, device=sim.device)
        loss_i2j = F.cross_entropy(all_sim, labels)
        loss_j2i = F.cross_entropy(all_sim.t(), labels)

        return (loss_i2j + loss_j2i) / 2


# =============================================================================
# Hard Negative Mining (from CowIDentifier)
# =============================================================================

class HardNegativeMiner:
    """
    Multi-Similarity Miner from CowIDentifier.
    Finds hard negatives and semi-hard positives for stronger training signal.
    """
    def __init__(self):
        if HAS_METRIC_LEARNING:
            self.miner = miners.MultiSimilarityMiner(epsilon=0.1)
        else:
            self.miner = None

    def mine(self, embeddings, labels):
        if self.miner is None:
            return None
        return self.miner(embeddings, labels)


# =============================================================================
# Contrastive Training Loop
# =============================================================================

def train_contrastive(model, dataloader, epochs=50, lr=0.001, device='cuda',
                      save_dir='models/contrastive', use_miner=True):
    """
    Train the model using contrastive learning.

    Based on CowIDentifier's training approach with NTXentLoss + hard mining.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = NTXentLoss(temperature=0.07)
    miner = HardNegativeMiner() if use_miner else None

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for view1, view2, labels in dataloader:
            view1 = view1.to(device)
            view2 = view2.to(device)
            labels = labels.to(device)

            # Get embeddings from both views
            emb1 = model(view1)
            emb2 = model(view2)

            # Concatenate embeddings and labels from both views
            embeddings = torch.cat([emb1, emb2], dim=0)
            all_labels = torch.cat([labels, labels], dim=0)

            # Compute contrastive loss
            loss = criterion(emb1, emb2)

            # Add hard negative mining loss if available
            if miner is not None and HAS_METRIC_LEARNING:
                hard_pairs = miner.mine(embeddings, all_labels)
                if hard_pairs is not None:
                    try:
                        mining_loss = losses.NTXentLoss(temperature=0.07)(
                            embeddings, all_labels, hard_pairs
                        )
                        loss = loss + 0.5 * mining_loss
                    except Exception:
                        pass  # Skip mining loss if it fails

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'embed_dim': model.embed_dim,
            }, os.path.join(save_dir, 'best_contrastive.pth'))
            print(f"  -> Saved best model (loss: {best_loss:.4f})")

    # Save final model
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'embed_dim': model.embed_dim,
    }, os.path.join(save_dir, 'final_contrastive.pth'))

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Models saved to {save_dir}/")
    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Contrastive Pre-Training for Cattle ReID')
    parser.add_argument('--data_dir', type=str, default='data/processed/train',
                        help='Directory with processed cow images')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50'],
                        help='CNN backbone architecture')
    parser.add_argument('--hidden_dims', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--imsize', type=int, default=128,
                        help='Image size (resized to imsize x imsize)')
    parser.add_argument('--no_miner', action='store_true',
                        help='Disable hard negative mining')
    parser.add_argument('--save_dir', type=str, default='models/contrastive',
                        help='Directory to save models')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataset and dataloader
    dataset = CattleContrastiveDataset(args.data_dir, imsize=args.imsize)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # Create model
    model = CattleContrastiveModel(
        backbone=args.backbone,
        hidden_dims=args.hidden_dims,
        pretrained=True,
    )
    print(f"Model: {args.backbone} | Embed dim: {model.embed_dim} | "
          f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train_contrastive(
        model=model,
        dataloader=dataloader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        save_dir=args.save_dir,
        use_miner=not args.no_miner,
    )
