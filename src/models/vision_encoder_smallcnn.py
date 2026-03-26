# src/models/vision_encoder_smallcnn.py

import torch
import torch.nn as nn

class VisionEncoder_SmallCNN(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        # channel: 32->64->128->256 
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 118*118

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 56*56

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 28*28

            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # (B, 256, 1, 1)
        )

        # Projection head maps visual feature into shared CLIP space.
        self.projection = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)              # (B, 256, 1, 1)
        h = h.flatten(start_dim=1)       # (B, 256)
        return self.projection(h)        # (B, embed_dim)