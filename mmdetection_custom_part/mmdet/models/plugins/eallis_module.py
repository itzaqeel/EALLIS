import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

        self.spatial_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        c_att = self.channel_mlp(self.avg_pool(x))
        s_att = self.spatial_conv(x)
        return x * c_att * s_att


class FeatureRectifier(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        residual = F.relu(self.conv1(x))
        residual = self.conv2(residual)

        gate = self.gate(x)

        return x + gate * residual


class EdgeBranch(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)


class EALLISBlock(nn.Module):
    # Per-stage block: illumination attention + feature rectifier + edge branch.

    def __init__(self, channels, use_edge=True):
        super().__init__()

        self.use_edge = use_edge
        self.illum = IlluminationAttention(channels)
        self.rectifier = FeatureRectifier(channels)

        if use_edge:
            self.edge = EdgeBranch(channels)

        # Unused layer kept for checkpoint compatibility.
        self.fuse = nn.Conv2d(channels + 1, channels, 1)

    def forward(self, x):
        x2 = self.rectifier(self.illum(x))

        if not self.use_edge:
            return x2, None

        edge_map = self.edge(x2)
        out = x2 * (1 + torch.sigmoid(edge_map))

        return out, edge_map
