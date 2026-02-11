# EALLIS: Enhanced Adaptive Low-Light Instance Segmentation

Instance segmentation in extremely dark environments using noise-invariant deep learning techniques.

## Overview

EALLIS implements a **Mask R-CNN** architecture enhanced with three key innovations to handle the challenges of low-light instance segmentation:

- **AWD (Adaptive Weighted Downsampling)** — Reduces high-frequency noise disturbances during feature downsampling
- **SCB (Smooth-oriented Convolutional Block)** — Suppresses feature noise during convolution operations
- **DSL (Disturbance Suppression Learning)** — Enables the model to learn disturbance-invariant features

The model trains on **SynCOCO** (COCO images with synthetically added low-light noise) and evaluates on real-world dark images from the EALLIS dataset.

## Architecture

```
Clean COCO Image → AddNoisyImg (synthetic noise) → Model receives img + noisy_img
                                                        ↓
                                            ResNetAdaDSmoothPrior (backbone)
                                                        ↓
                                                  FPN (neck)
                                                        ↓
                                              MaskRCNNNoiseInv
                                             (bbox + segm heads)
                                                        ↓
                                              DSL Loss (clean vs noisy)
```

## Results

Evaluated on the EALLIS test set (669 images, 8 classes):

| Metric | mAP | mAP@50 | mAP@75 |
|---|---|---|---|
| **Bbox** | **0.357** | 0.566 | 0.386 |
| **Segm** | **0.292** | 0.512 | 0.282 |

**Classes**: bicycle, car, motorbike, bus, bottle, chair, dining table, TV monitor

## Project Structure

```
EALLIS/
├── Configs/                              # Training configurations
│   └── mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py
├── Checkpoints/                          # Pre-trained model weights
│   └── Checkpoint1.pth
├── mmdetection/                          # Base MMDetection framework
├── mmdetection_custom_part/              # Custom modules (AWD, SCB, DSL, etc.)
│   └── mmdet/models/
│       ├── backbones/                    # ResNetAdaDSmoothPrior, CBAM, etc.
│       ├── detectors/                    # MaskRCNNNoiseInv
│       ├── losses/                       # Custom loss functions
│       └── plugins/                      # SCB and other plugins
├── data/
│   ├── coco/                             # COCO training data
│   └── eallis/                           # EALLIS evaluation data
│       ├── annotations/                  # COCO-format JSON annotations
│       └── images/                       # Dark scene images
├── tools/                                # Evaluation & utility scripts
│   ├── eval_map.py                       # Run mAP evaluation
│   └── verify_checkpoint.py              # Validate checkpoint loading
└── notebooks/
    └── EALLIS_Training_Colab.ipynb       # Google Colab training notebook
```

## Installation

### Requirements
- Python 3.10+
- PyTorch 1.13+ with CUDA
- mmcv-full 1.7.2
- MMDetection (included in repo)

### Setup

```bash
# Clone the repository
git clone https://github.com/itzaqeel/EALLIS.git
cd EALLIS

# Install mmcv-full (match your CUDA version)
pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html

# Install mmdetection
cd mmdetection && pip install -e . --no-deps && cd ..

# Install other dependencies
pip install pycocotools scikit-learn terminaltables
```

## Training

### Option 1: Google Colab (Recommended)

Use the ready-made notebook: [`notebooks/EALLIS_Training_Colab.ipynb`](notebooks/EALLIS_Training_Colab.ipynb)

It handles all setup, fixes, dataset preparation, training, and evaluation automatically.

### Option 2: Local Training

```bash
# Ensure COCO is at data/coco/ and EALLIS at data/eallis/
python mmdetection/tools/train.py Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py
```

### Training Pipeline (SynCOCO)

The training pipeline automatically converts clean COCO images into synthetic low-light data:

1. Load clean COCO image
2. `AddNoisyImg` — simulates camera RAW pipeline + realistic low-light noise (PGRU model, Canon EOS 5D4)
3. Model receives both clean (`img`) and noisy (`noisy_img`) versions
4. DSL loss encourages noise-invariant feature learning

## Evaluation

```bash
# Evaluate on EALLIS test set
python tools/eval_map.py
```

This computes both bbox and segm mAP using the EALLIS test annotations.

## Dataset

The EALLIS dataset contains real-world low-light images with instance-level pixel-wise annotations across 8 object classes.

| Split | Images | Annotations |
|---|---|---|
| Train | ~1,500 | ~6,500 |
| Test | 669 | 2,934 |

**Annotation format**: COCO-style JSON with polygon segmentation masks.

## Key Components

| Component | Location | Description |
|---|---|---|
| AWD | `mmdetection/mmdet/models/backbones/CustomConv.py` | Adaptive weighted downsampling |
| SConv | `mmdetection/mmdet/models/backbones/CustomConv.py` | Smooth convolution |
| DSL | `mmdetection_custom_part/mmdet/models/detectors/mask_rcnn.py` | Disturbance suppression loss |
| Noise Pipeline | `mmdetection/mmdet/datasets/pipelines/noisemodel/dark_noising.py` | SynCOCO noise synthesis |

## Acknowledgements

This work builds upon the research presented in:

- Chen et al., "Instance Segmentation in the Dark", IJCV 2023
- Hong et al., "Crafting Object Detection in Very Low Light", BMVC 2021
- [MMDetection](https://github.com/open-mmlab/mmdetection) framework by OpenMMLab