import cv2
import numpy as np

def boundary_iou(pred_mask, gt_mask):
    \"\"\"
    Compute the Boundary IoU for predicted and ground truth masks.
    \"\"\"
    def get_boundary(mask):
        kernel = np.ones((3, 3), np.uint8)
        return cv2.dilate(mask, kernel) - cv2.erode(mask, kernel)

    pred_b = get_boundary(pred_mask)
    gt_b = get_boundary(gt_mask)

    intersection = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()

    return intersection / (union + 1e-6)
