from mmdet.models.builder import BACKBONES
from mmdet.models.backbones.resnet import ResNet as MMDetResNet
from mmdet.models.backbones.resnet import ResNetV1d, ResNetAdaD, ResNetAdaDSmoothPrior
from mmdetection_custom_part.mmdet.models.plugins.eallis_module import EALLISBlock

@BACKBONES.register_module(force=True)
class ResNet(MMDetResNet):

    def __init__(self, *args, use_eallis=True, use_edge=True, **kwargs):
        super().__init__(*args, **kwargs)

        # Ablation switches: use_eallis toggles the whole block (baseline when
        # False); use_edge toggles the edge branch + edge loss (illum-only when
        # False). The detector's edge loss no-ops when no edge outputs exist.
        self.use_eallis = use_eallis
        self.use_edge = use_edge
        self.edge_outputs = []

        if use_eallis:
            self.eallis_c3 = EALLISBlock(512, use_edge=use_edge)
            self.eallis_c4 = EALLISBlock(1024, use_edge=use_edge)

    def forward(self, x):
        # Reset edge outputs every forward
        self.edge_outputs = []

        # Get original outputs
        outs = super().forward(x)

        if not self.use_eallis:
            return outs

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
