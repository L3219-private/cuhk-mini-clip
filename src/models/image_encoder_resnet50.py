# src/models/image_encoder_resnet50.py

import torch
import torch.nn as nn
import torchvision.models as tv_models

class ImageEncoder_ResNet50(nn.Module):

    def __init__(self, embed_dim: int = 128):
        super().__init__()

        backbone = tv_models.resnet50(weights=None)  # no pretrained weights
        in_features: int = backbone.fc.in_features    # 2048
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Same projection structure as SmallCNN and ResNet-18
        self.projection = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) float32 images"""
        h = self.backbone(x)          # (B, 2048)
        return self.projection(h)     # (B, embed_dim)
