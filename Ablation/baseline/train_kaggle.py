dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[103.53, 116.28, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=True,
        with_mask=True,
        poly2mask=False),
    dict(type='GenerateEdgeTargets'),
    dict(
        type='Resize',
        img_scale=(600, 400),
        keep_ratio=True,
        interpolate_mode='nearest'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='AddNoisyImg',
        model='PGRU',
        camera='CanonEOS5D4',
        cfa='bayer',
        use_255=True,
        pre_adjust_brightness=False,
        mode='unprocess_addnoise',
        dark_ratio=(1.0, 1.0),
        noise_ratio=(10, 100)),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[1.0, 1.0, 1.0],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks',
            'gt_edges'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(600, 400),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True, interpolate_mode='nearest'),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[103.53, 116.28, 123.675],
                std=[1.0, 1.0, 1.0],
                to_rgb=False),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
test_lod_coco = dict(
    classes=('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair',
             'diningtable', 'tvmonitor'),
    type='CocoDataset',
    ann_file='data/eallis/annotations/eallis_coco_JPG_test+1.json',
    img_prefix='data/eallis/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(600, 400),
            flip=False,
            transforms=[
                dict(
                    type='Resize', keep_ratio=True,
                    interpolate_mode='nearest'),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
train_eallis = dict(
    classes=('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair',
             'diningtable', 'tvmonitor'),
    type='CocoDataset',
    ann_file='data/eallis/annotations/eallis_coco_JPG_train+1.json',
    img_prefix='data/eallis/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            poly2mask=False),
        dict(type='GenerateEdgeTargets'),
        dict(
            type='Resize',
            img_scale=(600, 400),
            keep_ratio=True,
            interpolate_mode='nearest'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='AddNoisyImg',
            model='PGRU',
            camera='CanonEOS5D4',
            cfa='bayer',
            use_255=True,
            pre_adjust_brightness=False,
            mode='unprocess_addnoise',
            dark_ratio=(1.0, 1.0),
            noise_ratio=(10, 100)),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=[
                'img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                'gt_edges'
            ])
    ])
coco = dict(
    classes=('bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car',
             'tv', 'bus'),
    type='CocoDataset',
    ann_file='data/coco_subset/instances_train2017_subset.json',
    img_prefix='data/coco/train2017/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='LoadAnnotations',
            with_bbox=True,
            with_mask=True,
            poly2mask=False),
        dict(type='GenerateEdgeTargets'),
        dict(
            type='Resize',
            img_scale=(600, 400),
            keep_ratio=True,
            interpolate_mode='nearest'),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(
            type='AddNoisyImg',
            model='PGRU',
            camera='CanonEOS5D4',
            cfa='bayer',
            use_255=True,
            pre_adjust_brightness=False,
            mode='unprocess_addnoise',
            dark_ratio=(1.0, 1.0),
            noise_ratio=(10, 100)),
        dict(
            type='Normalize',
            mean=[103.53, 116.28, 123.675],
            std=[1.0, 1.0, 1.0],
            to_rgb=False),
        dict(type='Pad', size_divisor=32),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=[
                'img', 'noisy_img', 'gt_bboxes', 'gt_labels', 'gt_masks',
                'gt_edges'
            ])
    ])
coco_val = dict(
    classes=('bicycle', 'chair', 'dining table', 'bottle', 'motorcycle', 'car',
             'tv', 'bus'),
    type='CocoDataset',
    ann_file='data/coco/annotations/instances_val2017.json',
    img_prefix='data/coco/val2017/',
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(600, 400),
            flip=False,
            transforms=[
                dict(
                    type='Resize', keep_ratio=True,
                    interpolate_mode='nearest'),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
BATCHSIZE = 8
GPU = 1
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=[
        dict(
            classes=('bicycle', 'chair', 'dining table', 'bottle',
                     'motorcycle', 'car', 'tv', 'bus'),
            type='CocoDataset',
            ann_file='data/coco_subset/instances_train2017_subset.json',
            img_prefix='data/coco/train2017/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='LoadAnnotations',
                    with_bbox=True,
                    with_mask=True,
                    poly2mask=False),
                dict(type='GenerateEdgeTargets'),
                dict(
                    type='Resize',
                    img_scale=(600, 400),
                    keep_ratio=True,
                    interpolate_mode='nearest'),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='AddNoisyImg',
                    model='PGRU',
                    camera='CanonEOS5D4',
                    cfa='bayer',
                    use_255=True,
                    pre_adjust_brightness=False,
                    mode='unprocess_addnoise',
                    dark_ratio=(1.0, 1.0),
                    noise_ratio=(10, 100)),
                dict(
                    type='Normalize',
                    mean=[103.53, 116.28, 123.675],
                    std=[1.0, 1.0, 1.0],
                    to_rgb=False),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'noisy_img', 'gt_bboxes', 'gt_labels',
                        'gt_masks', 'gt_edges'
                    ])
            ])
    ],
    val=dict(
        classes=('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair',
                 'diningtable', 'tvmonitor'),
        type='CocoDataset',
        ann_file='data/eallis/annotations/eallis_coco_JPG_test+1.json',
        img_prefix='data/eallis/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(600, 400),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        keep_ratio=True,
                        interpolate_mode='nearest'),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        classes=('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair',
                 'diningtable', 'tvmonitor'),
        type='CocoDataset',
        ann_file='data/eallis/annotations/eallis_coco_JPG_test+1.json',
        img_prefix='data/eallis/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(600, 400),
                flip=False,
                transforms=[
                    dict(
                        type='Resize',
                        keep_ratio=True,
                        interpolate_mode='nearest'),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[103.53, 116.28, 123.675],
                        std=[1.0, 1.0, 1.0],
                        to_rgb=False),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric=['bbox', 'segm'], save_best='bbox_mAP')
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            llpf=dict(lr_mult=2.0),
            learnable_conv1=dict(lr_mult=2.0),
            AdaD=dict(lr_mult=2.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[9, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=1, max_keep_ckpts=1)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '/content/drive/MyDrive/Ablation/baseline'
load_from = '/content/drive/MyDrive/EALLIS_checkpoints/Checkpoint1.pth'
resume_from = None
gpu_ids = range(0, 1)
seed = 42
workflow = [('train', 1)]
find_unused_parameters = True
model = dict(
    type='MaskRCNNNoiseInv',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='caffe',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://detectron2/resnet50_caffe'),
        use_eallis=False,
        use_edge=False),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=8,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=8,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)))
edge_loss = dict(type='EdgeLoss', loss_weight=1.0)
custom_imports = dict(
    imports=[
        'mmdetection_custom_part.mmdet.datasets.pipelines.edge_target',
        'mmdetection_custom_part.mmdet.models.backbones.resnet'
    ],
    allow_failed_imports=False)
