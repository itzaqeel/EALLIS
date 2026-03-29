import numpy as np
import cv2


def boundary(mask):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)


def boundary_f1(pred, gt):
    pb = boundary(pred)
    gb = boundary(gt)

    tp = np.sum((pb == 1) & (gb == 1))
    fp = np.sum((pb == 1) & (gb == 0))
    fn = np.sum((pb == 0) & (gb == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return 2 * precision * recall / (precision + recall + 1e-6)
