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
    """Per-stage EALLIS block: illumination attention + feature rectifier + edge branch.

    Note on the ``eallis_c3.fuse`` / ``eallis_c4.fuse`` "unexpected key" warning
    when loading older checkpoints (e.g. best_bbox_mAP_epoch_12.pth, trained
    Jun-22 before commit 6c8ff8e): the previous version of this block defined a
    ``self.fuse = nn.Conv2d(channels + 1, channels, 1)`` layer that was registered
    (so its weights were saved) but **never called in forward** — the fusion has
    always been ``out = x2 * (1 + sigmoid(edge_map))``. The current forward is
    therefore identical to the one that produced those checkpoints, and the dropped
    ``fuse.*`` weights have no effect on inference. The warning is benign; the dead
    layer is intentionally not restored.
    """

    def __init__(self, channels, use_edge=True):
        super().__init__()

        self.use_edge = use_edge
        self.illum = IlluminationAttention(channels)
        self.rectifier = FeatureRectifier(channels)

        if use_edge:
            self.edge = EdgeBranch(channels)
            
        # Re-added dummy fuse layer to fix optimizer load_state_dict crash 
        # when resuming from old checkpoints like best_bbox_mAP_epoch_12.pth.
        # This layer is intentionally not used in forward().
        self.fuse = nn.Conv2d(channels + 1, channels, 1)

    def forward(self, x):
        x2 = self.rectifier(self.illum(x))

        if not self.use_edge:
            return x2, None

        edge_map = self.edge(x2)
        out = x2 * (1 + torch.sigmoid(edge_map))

        return out, edge_map
