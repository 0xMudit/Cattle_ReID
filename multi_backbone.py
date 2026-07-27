"""
Multi-Backbone Module for Cattle ReID
=======================================
Swappable backbone architectures for cattle re-identification.
Compare and choose the best one for your use case.

Supported backbones:
  - resnet18        (11.7M params, 512-dim)
  - resnet34        (21.3M params, 512-dim)
  - resnet50        (23.5M params, 2048-dim)
  - efficientnet_b0 (5.3M params, 1280-dim)
  - efficientnet_b2 (9.1M params, 1408-dim)
  - mobilenet_v3_small (2.5M params, 576-dim)
  - mobilenet_v3_large (5.4M params, 960-dim)
  - convnext_tiny   (28.6M params, 768-dim)
  - convnext_small  (50.2M params, 768-dim)
  - swin_tiny       (28.3M params, 768-dim)
  - swin_small      (50.0M params, 768-dim)

Usage:
  model = CattleBackbone(backbone='efficientnet_b0', embed_dim=64)
  embedding = model.extract_embedding(cow_tensor)

  # Quick benchmark
  python multi_backbone.py --benchmark
"""

import torch
import torch.nn as nn
import torchvision.models as models
import time
import os


# =============================================================================
# Backbone Registry
# =============================================================================

BACKBONE_REGISTRY = {
    # ResNet family
    'resnet18': {
        'builder': lambda pretrained: models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None),
        'feat_dim': 512,
        'family': 'resnet',
    },
    'resnet34': {
        'builder': lambda pretrained: models.resnet34(
            weights=models.ResNet34_Weights.DEFAULT if pretrained else None),
        'feat_dim': 512,
        'family': 'resnet',
    },
    'resnet50': {
        'builder': lambda pretrained: models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None),
        'feat_dim': 2048,
        'family': 'resnet',
    },

    # EfficientNet family
    'efficientnet_b0': {
        'builder': lambda pretrained: models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None),
        'feat_dim': 1280,
        'family': 'efficientnet',
    },
    'efficientnet_b2': {
        'builder': lambda pretrained: models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.DEFAULT if pretrained else None),
        'feat_dim': 1408,
        'family': 'efficientnet',
    },

    # MobileNet family
    'mobilenet_v3_small': {
        'builder': lambda pretrained: models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None),
        'feat_dim': 576,
        'family': 'mobilenet',
    },
    'mobilenet_v3_large': {
        'builder': lambda pretrained: models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None),
        'feat_dim': 960,
        'family': 'mobilenet',
    },

    # ConvNeXt family
    'convnext_tiny': {
        'builder': lambda pretrained: models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None),
        'feat_dim': 768,
        'family': 'convnext',
    },
    'convnext_small': {
        'builder': lambda pretrained: models.convnext_small(
            weights=models.ConvNeXt_Small_Weights.DEFAULT if pretrained else None),
        'feat_dim': 768,
        'family': 'convnext',
    },

    # Swin Transformer family
    'swin_tiny': {
        'builder': lambda pretrained: models.swin_t(
            weights=models.Swin_T_Weights.DEFAULT if pretrained else None),
        'feat_dim': 768,
        'family': 'swin',
    },
    'swin_small': {
        'builder': lambda pretrained: models.swin_s(
            weights=models.Swin_S_Weights.DEFAULT if pretrained else None),
        'feat_dim': 768,
        'family': 'swin',
    },
}


# =============================================================================
# Feature Extractor (removes classification head)
# =============================================================================

def get_feature_extractor(backbone_name, pretrained=True):
    """
    Load a backbone and return (feature_extractor, feat_dim).
    The feature extractor outputs a flat feature vector.
    """
    if backbone_name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone: {backbone_name}\n"
            f"Available: {list(BACKBONE_REGISTRY.keys())}"
        )

    info = BACKBONE_REGISTRY[backbone_name]
    model = info['builder'](pretrained)
    feat_dim = info['feat_dim']
    family = info['family']

    # Remove classification head based on family
    if family == 'resnet':
        # ResNet: replace fc with Identity
        model.fc = nn.Identity()

    elif family == 'efficientnet':
        # EfficientNet: classifier is model.classifier
        # Output is batch_size x feat_dim after adaptive_avg_pool
        model.classifier = nn.Identity()
        # Need to add pooling
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            model,
        )

    elif family == 'mobilenet':
        # MobileNetV3: classifier is model.classifier
        model.classifier = nn.Identity()
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            model,
        )

    elif family == 'convnext':
        # ConvNeXt: classifier is model.classifier
        model.classifier = nn.Identity()
        model = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            model,
        )

    elif family == 'swin':
        # Swin Transformer: head is model.head
        model.head = nn.Identity()
        # Swin outputs (batch, feat_dim) already after avgpool

    return model, feat_dim


# =============================================================================
# Cattle Backbone (full model with projection head)
# =============================================================================

class CattleBackbone(nn.Module):
    """
    Swappable backbone for cattle ReID.

    Architecture:
      Input → Backbone → Linear(feat_dim, embed_dim) → Embedding

    The projection head maps backbone features to a compact embedding space
    optimized for cattle re-identification.
    """
    def __init__(self, backbone='resnet18', embed_dim=64, pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        self.embed_dim = embed_dim

        # Get feature extractor
        self.feature_extractor, self.feat_dim = get_feature_extractor(
            backbone, pretrained
        )

        # Projection head: backbone features → embedding
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim),
        )

        # For training with classification head
        self.classifier = None  # Built lazily when num_classes is known

        self._print_info()

    def _print_info(self):
        info = BACKBONE_REGISTRY[self.backbone_name]
        n_params = sum(p.numel() for p in self.parameters())
        print(f"Backbone: {self.backbone_name} | "
              f"Feature dim: {self.feat_dim} | "
              f"Embed dim: {self.embed_dim} | "
              f"Params: {n_params:,}")

    def forward(self, x):
        """Forward pass: returns embedding."""
        features = self.feature_extractor(x)
        if features.dim() > 2:
            features = features.flatten(1)
        embedding = self.projector(features)
        return embedding

    def extract_embedding(self, x):
        """Extract embedding (same as forward, for API compatibility)."""
        return self.forward(x)

    def build_classifier(self, num_classes):
        """Build classification head for supervised training."""
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, num_classes),
        )

    def forward_classify(self, x):
        """Forward pass with classification head."""
        embedding = self.forward(x)
        if self.classifier is not None:
            return self.classifier(embedding)
        return embedding

    def load_from_contrastive(self, checkpoint_path):
        """Load backbone weights from contrastive pre-training."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict', checkpoint)

        # Clean keys
        cleaned = {}
        for k, v in state_dict.items():
            new_key = k.replace('backbone.', '').replace('projector.', '')
            cleaned[new_key] = v

        missing, unexpected = self.feature_extractor.load_state_dict(cleaned, strict=False)
        print(f"Loaded contrastive weights from {checkpoint_path}")
        if missing:
            print(f"  Missing keys (may be expected): {len(missing)}")
        return checkpoint


# =============================================================================
# Benchmark
# =============================================================================

def benchmark_backbones(input_size=128, device='cuda'):
    """
    Benchmark all backbones: inference time + parameter count.
    """
    if not torch.cuda.is_available():
        device = 'cpu'

    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    results = []

    print(f"\n{'Backbone':<22} {'Params':>10} {'Feat Dim':>9} {'Time (ms)':>10} {'Family':<12}")
    print("-" * 70)

    for name, info in BACKBONE_REGISTRY.items():
        try:
            model, feat_dim = get_feature_extractor(name, pretrained=False)
            model = model.to(device)
            model.eval()

            # Warmup
            with torch.no_grad():
                for _ in range(5):
                    _ = model(dummy)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(20):
                    start = time.perf_counter()
                    _ = model(dummy)
                    if device == 'cuda':
                        torch.cuda.sink()
                    end = time.perf_counter()
                    times.append((end - start) * 1000)

            n_params = sum(p.numel() for p in model.parameters())
            avg_time = sum(times) / len(times)

            results.append({
                'name': name,
                'params': n_params,
                'feat_dim': feat_dim,
                'time_ms': avg_time,
                'family': info['family'],
            })

            print(f"{name:<22} {n_params:>10,} {feat_dim:>9} {avg_time:>10.2f} {info['family']:<12}")

            del model
            torch.cuda.empty_cache() if device == 'cuda' else None

        except Exception as e:
            print(f"{name:<22} ERROR: {e}")

    # Summary
    if results:
        fastest = min(results, key=lambda x: x['time_ms'])
        smallest = min(results, key=lambda x: x['params'])
        print(f"\nFastest: {fastest['name']} ({fastest['time_ms']:.1f}ms)")
        print(f"Smallest: {smallest['name']} ({smallest['params']:,} params)")

    return results


# =============================================================================
# Quick model creation (drop-in for cattle_resnet.py)
# =============================================================================

def create_model(backbone='resnet18', embed_dim=64, num_classes=None, pretrained=True):
    """
    Create a cattle ReID model with specified backbone.

    Usage:
      model = create_model('efficientnet_b0', embed_dim=64, num_classes=90)
      embedding = model.extract_embedding(cow_tensor)
    """
    model = CattleBackbone(backbone=backbone, embed_dim=embed_dim, pretrained=pretrained)
    if num_classes is not None:
        model.build_classifier(num_classes)
    return model


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Backbone for Cattle ReID')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmark of all backbones')
    parser.add_argument('--backbone', type=str, default='resnet18',
                        help='Backbone to use')
    parser.add_argument('--embed_dim', type=int, default=64,
                        help='Embedding dimension')
    parser.add_argument('--input_size', type=int, default=128,
                        help='Input image size')
    args = parser.parse_args()

    if args.benchmark:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
        benchmark_backbones(input_size=args.input_size, device=device)
    else:
        model = create_model(args.backbone, args.embed_dim)
        dummy = torch.randn(1, 3, args.input_size, args.input_size)
        emb = model.extract_embedding(dummy)
        print(f"Output shape: {emb.shape}")
