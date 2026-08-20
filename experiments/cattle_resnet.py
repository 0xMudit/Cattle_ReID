"""
Cattle ReID ResNet Backbone
============================
Stolen from: https://github.com/Phoenix4582/CowIDentifier (lightning_supervised_model.py)
Adapted as a drop-in replacement for OSNet in the main pipeline.

Why ResNet instead of OSNet?
  - OSNet is cattle-pretrained, but ResNet18 offers a lighter alternative
  - Both work well for cattle ReID — ResNet is smaller and faster
  - CowIDentifier uses ResNet18 as their primary backbone
  - Easier to export to ONNX

Usage:
  from cattle_resnet import CattleResNet
  model = CattleResNet(num_classes=90, backbone='resnet18')
  embedding = model.extract_embedding(cow_image)  # 512-dim vector
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50


class CattleResNet(nn.Module):
    """
    ResNet-based model for cattle re-identification.

    Based on CowIDentifier's MultiCamSupervisedModel architecture:
      - Backbone: ResNet18/34/50 with ImageNet pre-training
      - FC head: ReLU -> Linear -> ReLU -> Dropout -> Linear(num_classes)
      - For inference: remove classification head, use features directly

    Reference: CowIDentifier/Supervised/lightning_supervised_model.py
      self.net.fc = nn.Sequential(
          self.net.fc, nn.ReLU(),
          nn.Linear(1000, self.hparams.hidden_dims), nn.ReLU(),
          nn.Dropout(),
          nn.Linear(self.hparams.hidden_dims, self.num_classes)
      )
    """
    def __init__(self, backbone='resnet18', hidden_dims=64,
                 num_classes=90, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes

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

        # Store original FC
        orig_fc = self.backbone.fc

        # Classification head (for training)
        self.classifier = nn.Sequential(
            orig_fc,              # Linear(512/2048, 1000)
            nn.ReLU(),
            nn.Linear(1000, hidden_dims),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dims, num_classes),
        )

        # Embedding head (for inference)
        self.embedder = nn.Sequential(
            orig_fc,              # Linear(512/2048, 1000)
            nn.ReLU(),
            nn.Linear(1000, hidden_dims),
        )

        # Remove original FC from backbone
        self.backbone.fc = nn.Identity()

        self.embed_dim = hidden_dims

    def forward(self, x):
        """Forward pass for training (returns class logits)."""
        features = self.backbone(x)
        return self.classifier(features)

    def extract_embedding(self, x):
        """Extract embedding for inference (returns feature vector)."""
        features = self.backbone(x)
        return self.embedder(features)

    def load_from_contrastive(self, checkpoint_path):
        """
        Load backbone weights from a contrastive pre-training checkpoint.
        This is the key integration point with contrastive_pretrain.py.

        Reference: CowIDentifier/Supervised/lightning_supervised_model.py
          checkpoint = torch.load(pretrained_weights)
          state_dict = checkpoint['state_dict']
          self.net.load_state_dict(state_dict, strict=False)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Remove 'backbone.' prefix if present (from contrastive model)
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace('backbone.', '').replace('projector.', '')
            cleaned[new_key] = v

        # Load into backbone only (ignore classifier/embedder mismatch)
        missing, unexpected = self.backbone.load_state_dict(cleaned, strict=False)
        print(f"Loaded contrastive weights from {checkpoint_path}")
        if missing:
            print(f"  Missing keys: {missing}")
        return checkpoint


class CattleEmbeddingExtractor:
    """
    Convenience wrapper for extracting embeddings from cattle images.
    Combines YOLO detection + ResNet embedding in one call.

    Usage:
      extractor = CattleEmbeddingExtractor('models/cattle_resnet.pth')
      results = extractor.extract(image_bgr)
      # results = [{'bbox': [x1,y1,x2,y2], 'embedding': [512-dim array], 'conf': 0.95}, ...]
    """
    def __init__(self, model_path=None, backbone='resnet18', hidden_dims=64,
                 device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = CattleResNet(
            backbone=backbone,
            hidden_dims=hidden_dims,
            pretrained=(model_path is None),
        ).to(self.device)

        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            print(f"Loaded model from {model_path}")

        self.model.eval()

    @torch.no_grad()
    def extract(self, image_bgr, yolo_model=None, conf_threshold=0.25):
        """
        Extract embeddings from all cows in an image.

        Args:
            image_bgr: BGR numpy array (from cv2.imread)
            yolo_model: YOLO model for detection (if None, use whole image)
            conf_threshold: minimum detection confidence

        Returns:
            list of dicts with 'bbox', 'embedding', 'conf'
        """
        import cv2
        import numpy as np

        results = []

        if yolo_model is not None:
            # Detect cows
            r = yolo_model(image_bgr, verbose=False)[0]
            boxes = r.boxes
            if boxes is None or len(boxes) == 0:
                return results

            # Filter to cow class (COCO class 19)
            mask = boxes.cls.cpu().numpy() == 19
            if not mask.any():
                return results

            xyxy = boxes.xyxy.cpu().numpy()[mask].astype(int)
            confs = boxes.conf.cpu().numpy()[mask]

            for bb, conf in zip(xyxy, confs):
                if conf < conf_threshold:
                    continue
                x1, y1, x2, y2 = bb
                crop = image_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                embedding = self._process_crop(crop)
                results.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'embedding': embedding,
                    'conf': float(conf),
                })
        else:
            # Use whole image
            embedding = self._process_crop(image_bgr)
            results.append({
                'bbox': [0, 0, image_bgr.shape[1], image_bgr.shape[0]],
                'embedding': embedding,
                'conf': 1.0,
            })

        return results

    def _process_crop(self, crop_bgr):
        """Convert crop to tensor and extract embedding."""
        import cv2
        import numpy as np

        # Resize to 128x128 (standard for ResNet cattle models)
        resized = cv2.resize(crop_bgr, (128, 128))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize with ImageNet stats
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        tensor = tensor.unsqueeze(0).to(self.device)

        # Extract embedding
        embedding = self.model.extract_embedding(tensor)
        return embedding.cpu().numpy().flatten()


# Convenience function to create a model matching the notebook's interface
def create_cattle_model(pretrained_path=None, num_classes=90, backbone='resnet18'):
    """
    Create a cattle ReID model that can be used as a drop-in replacement for OSNet.

    Usage in notebook:
      from cattle_resnet import create_cattle_model
      model = create_cattle_model(num_classes=dm.num_train_pids)
      embedding = model.extract_embedding(cow_tensor)
    """
    model = CattleResNet(
        backbone=backbone,
        hidden_dims=64,
        num_classes=num_classes,
        pretrained=(pretrained_path is None),
    )
    if pretrained_path:
        model.load_from_contrastive(pretrained_path)
    return model


import os
