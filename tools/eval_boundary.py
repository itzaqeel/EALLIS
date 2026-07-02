# Evaluate mean Boundary IoU (mask-edge quality) on the EALLIS test set.
# Usage: python tools/eval_boundary.py --checkpoint path/to/ckpt.pth [--score-thr 0.3 --dilation 3]
import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'mmdetection'))

# Register custom EALLIS modules.
import mmdetection_custom_part.mmdet.models.detectors          # noqa
import mmdetection_custom_part.mmdet.models.backbones           # noqa
import mmdetection_custom_part.mmdet.models.plugins             # noqa
import mmdetection_custom_part.mmdet.models.seg_heads           # noqa
import mmdetection_custom_part.mmdet.models.losses              # noqa
import mmdetection_custom_part.mmdet.datasets.pipelines.edge_target  # noqa

import cv2
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from pycocotools import mask as mask_util
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test


def get_boundary(mask, dilation=3):
    """Boundary band of a binary mask via morphological gradient."""
    mask = mask.astype(np.uint8)
    k = np.ones((dilation, dilation), np.uint8)
    return ((cv2.dilate(mask, k) - cv2.erode(mask, k)) > 0)


def boundary_iou(pred_mask, gt_mask, dilation=3):
    pb = get_boundary(pred_mask, dilation)
    gb = get_boundary(gt_mask, dilation)
    inter = np.logical_and(pb, gb).sum()
    union = np.logical_or(pb, gb).sum()
    return inter / (union + 1e-6)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return inter / (union + 1e-6)


def decode_mask(m):
    """mmdet segm entries are RLE dicts or ndarrays — return a 0/1 ndarray."""
    if isinstance(m, dict):
        return mask_util.decode(m)
    return np.asarray(m).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--score-thr', type=float, default=0.3)
    ap.add_argument('--match-iou', type=float, default=0.5,
                    help='mask IoU needed to count a prediction as a match to a GT instance')
    ap.add_argument('--dilation', type=int, default=3,
                    help='boundary band width in pixels')
    ap.add_argument('--use-eallis', type=int, default=None,
                    help='override backbone use_eallis (1=on, 0=off) for the ablation arms')
    ap.add_argument('--use-edge', type=int, default=None,
                    help='override backbone use_edge (1=on, 0=off) for the ablation arms')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)

    # Ablation-arm overrides: build the custom EALLIS ResNet with the requested flags so a
    # single config evaluates every arm and each checkpoint loads without random weights.
    if args.use_eallis is not None or args.use_edge is not None:
        cfg.model.backbone.type = 'ResNet'
        if args.use_eallis is not None:
            cfg.model.backbone.use_eallis = bool(args.use_eallis)
        if args.use_edge is not None:
            cfg.model.backbone.use_edge = bool(args.use_edge)
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0,
                              dist=False, shuffle=False)

    cfg.model.pretrained = None
    if hasattr(cfg.model.backbone, 'init_cfg'):
        cfg.model.backbone.init_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    print(f'Running inference on {len(dataset)} images ...')
    results = single_gpu_test(model, loader, show=False)

    coco = dataset.coco
    cat_ids = dataset.cat_ids  # segm class index -> COCO category id

    biou_scores = []
    n_gt = 0
    n_matched = 0

    for idx, img_id in enumerate(dataset.img_ids):
        res = results[idx]
        bbox_res, segm_res = res if isinstance(res, tuple) else (res, None)
        if segm_res is None:
            continue

        # Ground-truth masks grouped by category.
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gts_by_cat = {}
        for a in anns:
            if a.get('iscrowd', 0):
                continue
            gts_by_cat.setdefault(a['category_id'], []).append(coco.annToMask(a).astype(np.uint8))
        n_gt += sum(len(v) for v in gts_by_cat.values())

        # Greedily match predictions (high score first) to unused GT, same class.
        for c, cat_id in enumerate(cat_ids):
            gts = gts_by_cat.get(cat_id, [])
            if not gts:
                continue
            used = [False] * len(gts)
            bboxes = bbox_res[c]
            masks = segm_res[c]
            order = np.argsort(-bboxes[:, 4]) if len(bboxes) else []
            for k in order:
                if bboxes[k, 4] < args.score_thr:
                    continue
                pm = decode_mask(masks[k])
                best, best_iou = -1, args.match_iou
                for gi, gm in enumerate(gts):
                    if used[gi]:
                        continue
                    iou = mask_iou(pm, gm)
                    if iou >= best_iou:
                        best, best_iou = gi, iou
                if best >= 0:
                    used[best] = True
                    n_matched += 1
                    biou_scores.append(boundary_iou(pm, gts[best], args.dilation))

    mean_biou = float(np.mean(biou_scores)) if biou_scores else 0.0
    recall = n_matched / (n_gt + 1e-6)

    print('\n' + '=' * 60)
    print('BOUNDARY IoU EVALUATION')
    print('=' * 60)
    print(f'  checkpoint        : {args.checkpoint}')
    print(f'  score_thr         : {args.score_thr}   match_iou: {args.match_iou}   dilation: {args.dilation}px')
    print(f'  GT instances      : {n_gt}')
    print(f'  matched instances : {n_matched}  (recall @IoU{args.match_iou}: {recall:.3f})')
    print(f'  mean Boundary IoU : {mean_biou:.4f}   (over matched instances)')
    print('=' * 60)


if __name__ == '__main__':
    main()
