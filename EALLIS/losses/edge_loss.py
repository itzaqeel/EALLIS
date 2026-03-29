import torch
import torch.nn.functional as F


def edge_loss(pred, target):
    # BCE
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = torch.sigmoid(pred)

    # Dice
    intersection = (pred * target).sum()
    dice = 1 - (2 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)

    return bce + dice
