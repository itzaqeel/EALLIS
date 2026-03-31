# Re-export MaskRCNN and MaskRCNNNoiseInv from the installed mmdet package.
# The real custom implementations (AWD/SCB/DSL) live in
# mmdetection/mmdet/models/detectors/mask_rcnn.py which is installed via pip -e.
from mmdet.models.detectors.mask_rcnn import MaskRCNN, MaskRCNNNoiseInv

__all__ = ['MaskRCNN', 'MaskRCNNNoiseInv']
