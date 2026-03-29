import cv2
import numpy as np


def mask_to_edge(mask):
    kernel = np.ones((3, 3), np.uint8)
    dil = cv2.dilate(mask, kernel, iterations=1)
    ero = cv2.erode(mask, kernel, iterations=1)
    edge = (dil - ero) > 0
    return edge.astype(np.float32)
