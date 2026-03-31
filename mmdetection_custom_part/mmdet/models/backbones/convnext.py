# Re-export ConvNeXt variants from the installed mmdet package.
from mmdet.models.backbones.convnext import ConvNeXt, ConvNeXtAdaD

__all__ = ['ConvNeXt', 'ConvNeXtAdaD']
