import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeLoss(nn.Module):
    """Class-balanced edge loss (BCE + Dice).

    Edge pixels are only ~1-2% of the image, so a plain BCE collapses to the
    trivial "predict no edge" solution (loss parks near its floor and never
    learns). We re-weight the positive (edge) class by the neg/pos pixel ratio
    so the branch actually receives gradient on the thin boundary regions.
    """

    def __init__(self, max_pos_weight=20.0):
        super().__init__()
        self.max_pos_weight = max_pos_weight

    def forward(self, pred, target):
        pred = pred.squeeze(1)
        target = target.squeeze(1).float()

        # Balance the heavy edge/background pixel imbalance.
        num_pos = target.sum()
        num_neg = target.numel() - num_pos
        pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=self.max_pos_weight)
        bce = F.binary_cross_entropy_with_logits(pred, target, pos_weight=pos_weight)

        pred_sig = torch.sigmoid(pred)
        intersection = (pred_sig * target).sum()
        dice = 1 - (2 * intersection + 1) / (pred_sig.sum() + target.sum() + 1)

        return bce + dice
