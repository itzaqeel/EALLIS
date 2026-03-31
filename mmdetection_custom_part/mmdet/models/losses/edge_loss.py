import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1).float()

        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)

        return bce + dice
