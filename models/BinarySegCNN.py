import torch
import torch.nn as nn


class BinarySegCNN(nn.Module):
    """
    Lightweight encoder-decoder CNN for binary segmentation.

    Architecture
    ------------
    Encoder     : 2x (Conv2d → ReLU → MaxPool2d)  – halves spatial dims twice
    Bottleneck  : Conv2d → ReLU                    – deepens features
    Decoder     : 2x (Upsample → Conv2d → ReLU)   – restores original resolution
    Head        : Conv2d(16, 1, 1) → Sigmoid       – outputs per-pixel probability

    Input  : [B, 3,  H, W]  (RGB, any size divisible by 4)
    Output : [B, 1,  H, W]  (float32 in [0, 1])

    Parameters
    ----------
    in_channels : int
        Number of input image channels (default: 3 for RGB).
    base_filters : int
        Number of filters in the first Conv layer.  Subsequent layers
        are multiples of this value (×2, ×4).  Default: 16.
    """

    def __init__(self, in_channels: int = 3, base_filters: int = 16):
        super().__init__()

        f = base_filters  # shorthand: f=16, 2f=32, 4f=64

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # H/2,  W/2
            nn.Conv2d(f, f * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # H/4,  W/4
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(f * 2, f * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # H/2, W/2
            nn.Conv2d(f * 4, f * 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),  # H,   W
            nn.Conv2d(f * 2, f, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(f, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return self.head(x)