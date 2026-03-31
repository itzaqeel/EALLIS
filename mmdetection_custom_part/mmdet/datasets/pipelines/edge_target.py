import numpy as np
import cv2
from mmdet.datasets.builder import PIPELINES
from mmdet.core.mask.structures import BitmapMasks
from mmcv.parallel import DataContainer as DC

@PIPELINES.register_module()
class GenerateEdgeTargets:
    def __call__(self, results):
        masks = results['gt_masks'].to_ndarray()

        edge_maps = []

        for mask in masks:
            mask = mask.astype(np.uint8)

            dil = cv2.dilate(mask, np.ones((3, 3), np.uint8))
            ero = cv2.erode(mask, np.ones((3, 3), np.uint8))

            edge = (dil - ero) > 0
            edge_maps.append(edge.astype(np.uint8))

        if len(edge_maps) > 0:
            edge_maps = np.stack(edge_maps)
        else:
            edge_maps = np.empty((0, masks.shape[1], masks.shape[2]), dtype=np.uint8)
            
        gt_edges = BitmapMasks(edge_maps, results['gt_masks'].height, results['gt_masks'].width)
        results['gt_edges'] = DC(gt_edges, cpu_only=True)

        return results
