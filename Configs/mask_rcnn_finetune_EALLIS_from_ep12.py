# In-domain fine-tune: continue from the COCO checkpoint, train on the real EALLIS/LIS train set.
# Run: python mmdetection/tools/train.py Configs/mask_rcnn_finetune_EALLIS_from_ep12.py

_base_ = ['./mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py']

# Real EALLIS train set (same class order as test); reuses base train_pipeline (DSL + edge targets).
data = dict(
    train=[dict(
        classes=('bicycle', 'car', 'motorbike', 'bus',
                 'bottle', 'chair', 'diningtable', 'tvmonitor'),
        type='CocoDataset',
        ann_file='data/eallis/annotations/eallis_coco_JPG_train+1.json',
        img_prefix='data/eallis/',
        pipeline={{_base_.train_pipeline}},
    )],
)

# Weights-only start (fine-tune, not resume); set to the actual checkpoint path on your machine.
load_from = 'best_bbox_mAP_epoch_12.pth'
resume_from = None

# Lower LR + gradient clipping for stable fine-tuning.
optimizer = dict(lr=0.0025)
optimizer_config = dict(_delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# Aligned step schedule so the LR stays live through the run.
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)

# Keep this run's checkpoints separate from the COCO run.
work_dir = './work_dir_finetune_eallis'
