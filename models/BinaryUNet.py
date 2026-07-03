import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building block
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    """Two consecutive Conv2d -> BatchNorm -> ReLU blocks.

    This is the standard UNet building block. Two convolutions per level
    give the network more capacity to learn features before/after each
    pooling or upsampling step, without adding extra pooling overhead.
    BatchNorm is added (the original UNet didn't have it) for more stable
    training and faster convergence.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BinaryUNet(nn.Module):
    """Lightweight UNet-style binary segmentation model with skip connections.

    Architecture
    ------------
    The encoder progressively halves the spatial resolution while doubling
    the number of feature maps. At each encoder level the feature map is
    saved as a skip connection before pooling.

    The decoder mirrors the encoder: it upsamples back to the previous
    resolution, then concatenates the corresponding encoder skip, and runs
    a DoubleConv to fuse the two streams. This is the key difference from
    the previous flat CNN -- the decoder now has direct access to the
    high-resolution feature maps from the encoder, so thin structures like
    stems that would otherwise be lost during downsampling are preserved.

    Diagram (base_filters=32, depth=2)
    -----------------------------------

        Input [B, 3, 512, 512]
            |
        enc1: DoubleConv(3, 32)    -> skip1 [B, 32,  512, 512]
            | MaxPool2d(2)
        enc2: DoubleConv(32, 64)   -> skip2 [B, 64,  256, 256]
            | MaxPool2d(2)
        bottleneck: DoubleConv(64, 128)    [B, 128, 128, 128]
            | Upsample x2
            | cat(skip2)           -> [B, 192, 256, 256]
        dec2: DoubleConv(192, 64)          [B, 64,  256, 256]
            | Upsample x2
            | cat(skip1)           -> [B, 96,  512, 512]
        dec1: DoubleConv(96, 32)           [B, 32,  512, 512]
            |
        head: Conv2d(32, 1, 1) -> Sigmoid  [B, 1,   512, 512]

    The concatenation doubles the channel count at each decoder step
    (hence the DoubleConv input channels are in_ch + skip_ch).

    Parameters
    ----------
    in_channels : int
        Input image channels (default 3 for RGB).
    base_filters : int
        Filters in the first encoder block. Each subsequent encoder level
        doubles this. Default 32.
    depth : int
        Number of encoder/decoder levels (i.e. number of MaxPool steps).
        More depth = more context but more downsampling. Default 2 matches
        the original flat CNN. Try 3 for harder cases.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int = 32,
        depth: int = 2,
    ):
        super().__init__()
        self.depth = depth

        # Encoder: one DoubleConv block per level
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        self.enc_channels = []  # track output channels for skip connections
        for i in range(depth):
            out_ch = base_filters * (2 ** i)  # 32, 64, 128, ...
            self.encoders.append(DoubleConv(ch, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            self.enc_channels.append(out_ch)
            ch = out_ch

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)  # 128 for depth=2
        self.bottleneck = DoubleConv(ch, bottleneck_ch)
        ch = bottleneck_ch

        # Decoder: upsample, concatenate skip, then DoubleConv
        self.upsamples = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            skip_ch = self.enc_channels[i]
            out_ch = skip_ch  # decode back down to the skip channel count
            self.upsamples.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
            self.decoders.append(DoubleConv(ch + skip_ch, out_ch))
            ch = out_ch

        # Segmentation head
        self.head = nn.Sequential(
            nn.Conv2d(ch, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder -- save feature maps before pooling as skip connections
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)   # save before spatial reduction
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder -- upsample, concatenate skip, fuse
        for up, dec, skip in zip(self.upsamples, self.decoders, reversed(skips)):
            x = up(x)
            # Guard against off-by-one pixel sizes (e.g. odd input dimensions)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)  # channel-wise concat
            x = dec(x)

        return self.head(x)