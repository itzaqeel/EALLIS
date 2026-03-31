# Re-export SwinTransformer variants from the installed mmdet package.
from mmdet.models.backbones.swin import SwinTransformer, SwinTransformerAdaD

__all__ = ['SwinTransformer', 'SwinTransformerAdaD']
