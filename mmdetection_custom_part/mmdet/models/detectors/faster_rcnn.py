# Re-export FasterRCNN from the installed mmdet package.
# The custom_part overrides only two_stage.py; FasterRCNN has no custom logic.
from mmdet.models.detectors.faster_rcnn import FasterRCNN

__all__ = ['FasterRCNN']
