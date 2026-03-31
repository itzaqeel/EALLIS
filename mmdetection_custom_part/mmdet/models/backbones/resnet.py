from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResNet as MMDetResNet
from mmdet.models.backbones.resnet import ResNetV1d, ResNetAdaD, ResNetAdaDSmoothPrior
from mmdetection_custom_part.mmdet.models.plugins.eallis_module import EALLISBlock

@BACKBONES.register_module(force=True)
class ResNet(MMDetResNet):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # EALLIS blocks
        self.eallis_c3 = EALLISBlock(512)
        self.eallis_c4 = EALLISBlock(1024)

    def forward(self, x):
        # Reset edge outputs every forward
        self.edge_outputs = []

        # Get original outputs
        outs = super().forward(x)

        new_outs = []

        for i, feat in enumerate(outs):

            if i == 1:
                feat, edge = self.eallis_c3(feat)
                self.edge_outputs.append(edge)

            elif i == 2:
                feat, edge = self.eallis_c4(feat)
                self.edge_outputs.append(edge)

            new_outs.append(feat)

        return tuple(new_outs)
