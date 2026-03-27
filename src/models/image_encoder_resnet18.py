# src/models/image_encoder_resnet18.py

import torch
import torch.nn as nn
import torchvision.models as tv_models


class ImageEncoder_ResNet18(nn.Module):
    """ResNet-18 backbone + projection head for CLIP-style training."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        backbone = tv_models.resnet18(weights=None)  # no pretrained weights
        # The original last layer: backbone.fc = Linear(512, 1000)
        # We record 512 and replace fc with nn.Identity and then add projection head
        in_features: int = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Same structure as SmallCNN's projection for fair comparison.
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 images"""
        h = self.backbone(x)          # (B, 512)
        return self.projection(h)     # (B, embed_dim)
