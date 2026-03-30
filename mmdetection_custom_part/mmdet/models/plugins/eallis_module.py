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
    def __init__(self, channels, illum_only=False):
        super().__init__()
        
        self.illum_only = illum_only
        self.illum = IlluminationAttention(channels)
        self.rectifier = FeatureRectifier(channels)
        
        if not self.illum_only:
            self.edge = EdgeBranch(channels)
            self.fuse = nn.Conv2d(channels + 1, channels, 1)

    def forward(self, x):
        x1 = self.illum(x)
        x2 = self.rectifier(x1)
        
        if self.illum_only:
            return x2, None

        edge_map = self.edge(x2)
        edge_sigmoid = torch.sigmoid(edge_map)

        out = x2 * (1 + edge_sigmoid)

        return out, edge_map
