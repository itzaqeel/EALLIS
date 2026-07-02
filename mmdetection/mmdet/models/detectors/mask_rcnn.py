from genericpath import exists
from math import fabs, pi
import os
from time import time
from unicodedata import name

from cv2 import decolor
from torch.nn.modules import loss
from .aux_modules import *
from ..builder import DETECTORS
from .two_stage import TwoStageDetector
import torch
import copy
import torch.nn.functional as F
from mmcv.runner import load_checkpoint
from PIL import Image
import matplotlib.pyplot as plt
from .aux_modules import *
from .multiscale_discriminator import *
from .lsid import LSID
import time

colors = plt.cm.jet(np.linspace(0.0, 1.00, 256))

@DETECTORS.register_module()
class MaskRCNN(TwoStageDetector):
    # Mask R-CNN (He et al., 2017).

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        """Test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)


@DETECTORS.register_module()
class MaskRCNNNoiseInv(MaskRCNN):
    # Mask R-CNN with disturbance-suppression (noise-invariant) training.

    def __init__(self, **args):
        super().__init__(**args)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      return_proposal=False,
                      **kwargs):
        # Single-branch forward: RPN + RoI heads plus the auxiliary edge loss.
        backbone_x, x = self.extract_feat(img, return_backbone=True)

        losses = dict()

        # RPN forward and loss.
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        gt_edges = kwargs.pop('gt_edges', None)

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)

        if gt_edges is not None and hasattr(self.backbone, 'edge_outputs'):
            valid_edges = [e for e in getattr(self.backbone, 'edge_outputs', []) if e is not None]
            if len(valid_edges) > 0:
                import sys
                from mmdetection_custom_part.mmdet.models.losses.edge_loss import EdgeLoss
                import numpy as np
                edge_loss_fn = EdgeLoss().to(x[0].device)
                edge_loss_val = 0.0
                for edge_pred in valid_edges:
                    b, _, h, w = edge_pred.shape
                    targets = []
                    for i in range(b):
                        masks = gt_edges[i].to_ndarray()
                        if len(masks) > 0:
                            mask = masks.max(axis=0)
                        else:
                            mask = np.zeros((gt_edges[i].height, gt_edges[i].width), dtype=np.uint8)
                        
                        mask = torch.from_numpy(mask).float().to(edge_pred.device).unsqueeze(0).unsqueeze(0)
                        mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=False)
                        targets.append(mask.squeeze(0))
                    
                    targets = torch.stack(targets)
                    edge_loss_val = edge_loss_val + edge_loss_fn(edge_pred, targets)
                # Down-weight the auxiliary edge loss (was 0.5, ~51% of total loss; 0.1 keeps it ~10-15%).
                EDGE_LOSS_WEIGHT = 0.1
                edge_loss_val = edge_loss_val / max(len(valid_edges), 1)
                losses['edge_loss'] = EDGE_LOSS_WEIGHT * edge_loss_val

        if return_proposal:
            return losses, backbone_x, x, proposal_list
        else:
            return losses, backbone_x, x

    def forward(self, img, img_metas, return_loss=True, **kwargs):
        # Dispatch to forward_train or forward_test based on return_loss.
        if torch.onnx.is_in_onnx_export():
            assert len(img_metas) == 1
            return self.onnx_export(img[0], img_metas[0])

        if return_loss:
            losses = dict()
            noisy_img = kwargs.pop('noisy_img')
            clean_losses, clean_backbone_x, clean_x = self.forward_train(img, img_metas, **kwargs)
            noisy_losses, noisy_backbone_x, noisy_x = self.forward_train(noisy_img, img_metas, **kwargs)
            losses['noise_inv_loss'] = 0
            for idx, (_noisy, _clean) in enumerate(zip(noisy_backbone_x[:], clean_backbone_x[:])):
                losses['noise_inv_loss'] += 0.01 * torch.clamp(F.mse_loss(input=_noisy, target=_clean, reduction='none'), max=0.1, min=0).mean()
            for k in clean_losses: losses[f'clean_{k}'] = clean_losses[k]
            for k in noisy_losses: losses[f'{k}'] = noisy_losses[k]
            for k in losses: 
                if isinstance(losses[k], int): 
                    losses.pop(k)
            return losses
        else:
            return super().forward_test(img, img_metas, **kwargs)

    def extract_feat(self, img, return_backbone=True):
        # Extract features from backbone (and neck).
        x = self.backbone(img)
        if self.with_neck:
            if return_backbone:
                backbone_x, fpn_x = self.neck(x, return_backbone)
                return backbone_x, fpn_x
            else:
                fpn_x = self.neck(x, return_backbone)
                return fpn_x
        else:
            return x
        

    def forward_dummy(self, img):
        # Used for computing network FLOPs.
        outs = ()
        x = self.extract_feat(img, return_backbone=False)
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def simple_test(self, img, img_metas, proposals=None, rescale=False, **kwargs):
        # Test without augmentation.
        assert self.with_bbox, 'Bbox head must be implemented.'
        def feature2image(feat, save_dir, name, normalize=False):
            _, _, h, w = feat.shape
            feat = feat.mean(dim=1).repeat(3, 1, 1)
            feat = F.interpolate(feat[None, ], size=(400, 600), mode='bilinear')[0]
            save_img = feat.permute(1, 2, 0).cpu().numpy()
            if normalize:
                save_img = (save_img - save_img.min()) / (save_img.max() - save_img.min()) * 255
            else:
                save_img = save_img * 255
                save_img = np.clip(save_img, 0, 255)
            save_img = save_img.astype(np.uint8)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            colored = np.zeros(shape=(save_img.shape[0], save_img.shape[1], 3)).astype(np.uint8)
            for i in range(3):
                for num in np.unique(save_img):
                    colored[:,:,i][save_img[:, :, i] == int(num)] = colors[int(num)][i] * 255.
            Image.fromarray(colored).save(f'{save_dir}/{name}.png')

        def tensor2img(t, save_dir, name):
            norm = torch.tensor([103.530, 116.280, 123.675])[None, :, None, None].to(t.device)
            t += norm
            img = t[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            Image.fromarray(img).save(f'{save_dir}/{name}.png')

        backbone_x, x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
