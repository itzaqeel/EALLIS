import numpy as np
import cv2


class GenerateEdgeTargets:
    def __call__(self, results):
        masks = results['gt_masks'].masks

        edge_maps = []

        for mask in masks:
            mask = mask.astype(np.uint8)

            dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            ero = cv2.erode(mask, np.ones((3, 3), np.uint8))

            edge = (dil - ero) > 0
            edge_maps.append(edge.astype(np.uint8))

        results['gt_edges'] = edge_maps

        return results
