# =============================================================================
# In-domain fine-tuning config for EALLIS
# -----------------------------------------------------------------------------
# Purpose: continue from the COCO-trained checkpoint (best_bbox_mAP_epoch_12.pth)
# and fine-tune on the REAL low-light EALLIS/LIS training set to close the
# synthetic-to-real domain gap. This is the single highest-impact change for
# pushing past the ~0.32 bbox mAP plateau.
#
# Key differences vs the base SynCOCO config:
#   * train data  -> real EALLIS train set (NOT synthetic COCO)
#                    (this also aligns the class order with the test set)
#   * load_from   -> the epoch-12 checkpoint (weights only; NOT resume_from,
#                    because this is a fresh fine-tune with a new schedule)
#   * lr          -> lowered to 0.0025 for fine-tuning from a strong checkpoint
#   * schedule    -> shorter (8 epochs) with an aligned step schedule
#   * grad_clip   -> enabled (the base had grad_clip=None, which risks NaNs)
#
# NOTE: AWD/SCB are intentionally NOT enabled here — they change the backbone
# architecture and are incompatible with loading the epoch-12 weights. Use a
# from-scratch run for that ablation.
#
# Train with:
#   python mmdetection/tools/train.py \
#       Configs/mask_rcnn_finetune_EALLIS_from_ep12.py
# =============================================================================

_base_ = ['./mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py']

# ---- Train on the REAL in-domain EALLIS/LIS train set --------------------------
# Same class order as the test set -> no class-order mismatch.
# Reuses the base train_pipeline (keeps DSL noisy_img + edge targets).
data = dict(
    train=[dict(
        classes=('bicycle', 'car', 'motorbike', 'bus',
                 'bottle', 'chair', 'diningtable', 'tvmonitor'),
        type='CocoDataset',
        ann_file='data/eallis/annotations/eallis_coco_JPG_train+1.json',
        img_prefix='data/eallis/',
        pipeline={{_base_.train_pipeline}},
    )],
    # val / test inherited from base (the real EALLIS test set).
)

# ---- Continue from the epoch-12 weights (fine-tune, not resume) ---------------
# Set this to wherever the checkpoint actually lives on your training machine
# (e.g. '/content/drive/MyDrive/EALLIS/best_bbox_mAP_epoch_12.pth' on Colab).
load_from = 'best_bbox_mAP_epoch_12.pth'
resume_from = None

# ---- Lower LR + gradient clipping for stable fine-tuning ----------------------
optimizer = dict(lr=0.0025)                                  # ~1/4 of the base 0.01
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# ---- Schedule: longer budget with an ALIGNED step so the LR stays live --------
# (LR drops at epochs 8 and 11, not before; avoids the "extra epochs at a dead
#  LR" trap. With ~1561 train images the run is still short.)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# ---- Keep checkpoints from this run separate from the COCO run ----------------
work_dir = './work_dir_finetune_eallis'

# NOTE: the base config still sets val == test with save_best='bbox_mAP'
# (model selection on the test set). That is a known leakage caveat; for a
# leakage-free final number, evaluate the chosen checkpoint on a held-out split.
