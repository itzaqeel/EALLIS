"""
Evaluate mAP (bbox + segm) on the EALLIS test set using the provided config and checkpoint.
Usage:
    python tools/eval_map.py
"""
import sys
import os
import traceback

# Ensure project root is on the path
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.join(os.path.abspath('.'), 'mmdetection'))

# ---- Import custom modules to trigger registration ----
try:
    import mmdetection_custom_part.mmdet.models.detectors
    import mmdetection_custom_part.mmdet.models.backbones
    import mmdetection_custom_part.mmdet.models.dense_heads
    import mmdetection_custom_part.mmdet.models.roi_heads
    import mmdetection_custom_part.mmdet.models.plugins
    import mmdetection_custom_part.mmdet.models.seg_heads
    import mmdetection_custom_part.mmdet.models.losses
    print("[OK] Custom modules imported successfully.")
except Exception as e:
    print(f"[WARN] Error importing custom modules: {e}")
    traceback.print_exc()

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.apis import single_gpu_test

def main():
    config_path = 'Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py'
    checkpoint_path = 'Checkpoints/Checkpoint1.pth'

    print(f"Config:     {config_path}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load config
    cfg = Config.fromfile(config_path)

    # Build test dataset
    print("\n--- Building test dataset ---")
    dataset = build_dataset(cfg.data.test)
    print(f"Test dataset: {len(dataset)} images")

    # Build dataloader
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,  # Use 0 workers on Windows to avoid multiprocessing issues
        dist=False,
        shuffle=False
    )

    # Build model
    print("\n--- Building model ---")
    cfg.model.pretrained = None  # Don't download pretrained backbone
    cfg.model.backbone.init_cfg = None  # Don't initialize from pretrained
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')

    # Set classes from checkpoint or dataset
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    print(f"Classes: {model.CLASSES}")

    # Wrap model with MMDataParallel for proper DataContainer handling
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    print(f"Device: cuda (MMDataParallel)")

    # Run inference
    print(f"\n--- Running inference on {len(dataset)} images ---")
    results = single_gpu_test(model, data_loader, show=False)
    print(f"Inference complete. Got {len(results)} results.")

    # Evaluate
    print("\n--- Computing mAP ---")
    eval_kwargs = dict(metric=['bbox', 'segm'])
    eval_results = dataset.evaluate(results, **eval_kwargs)

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for key, val in eval_results.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")
    print("=" * 60)

if __name__ == '__main__':
    main()
