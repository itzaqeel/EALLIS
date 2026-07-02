# EALLIS: Enhanced Adaptive Low-Light Instance Segmentation

Instance segmentation in extremely dark environments using noise-invariant deep learning techniques.

## Overview

EALLIS implements a **Mask R-CNN** architecture with the following components **active in the current configuration** ([`Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py`](Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py)):

- **DSL (Disturbance Suppression Learning)** — dual clean/noisy forward pass with a noise-invariance loss (`MaskRCNNNoiseInv`)
- **EALLIS block** — our custom per-stage module (illumination attention + feature rectifier + edge branch) injected at the C3/C4 backbone stages

> **Note on AWD / SCB.** The AWD (`ResNetAdaD`) and SCB (`ResNetAdaDSmoothPrior`) backbone variants from *Instance Segmentation in the Dark* are present in the codebase but are **not enabled** in the current config (the backbone is `type='ResNet'`, i.e. a standard ResNet-50 + EALLIS blocks). The `AWD_SCB` in the config filename is historical. To actually use them, set the backbone `type` accordingly and re-train.

The model trains on **SynCOCO** (COCO images with synthetically added low-light noise) and evaluates on real-world dark images from the EALLIS dataset.

## Architecture

```
Clean COCO Image → AddNoisyImg (synthetic noise) → Model receives img + noisy_img
                                                        ↓
                                  ResNet-50 (caffe) + EALLIS blocks @ C3/C4
                                  (illum attention + rectifier + edge branch)
                                                        ↓
                                                  FPN (neck)
                                                        ↓
                                              MaskRCNNNoiseInv
                                             (bbox + segm heads)
                                                        ↓
                              DSL noise-invariance loss (clean vs noisy) + edge loss
```

## Results

Ablation on the LIS (EALLIS) test set (669 images, 8 classes), all variants trained for
30 epochs under identical settings. Values are COCO mAP (%).

| Model variant | bbox mAP | segm mAP | segm AP75 |
|---|---|---|---|
| Baseline (Mask R-CNN, no EALLIS block) | 38.6 | 31.8 | 30.2 |
| + Illumination attention | 40.3 | 33.2 | 32.5 |
| **+ Illumination + Edge (full EALLIS)** | **42.3** | **34.5** | **34.0** |

The full model improves the baseline by +3.7 bbox mAP and +2.7 segm mAP, with the largest
gain on the strict boundary metric segm AP75 (+3.8), which is the direct evidence for the
edge-aware component.

**Classes**: bicycle, car, motorbike, bus, bottle, chair, dining table, TV monitor

> **Model selection.** Model selection uses a held-out validation split (the real
> low-light LIS train set, disjoint from the test set); the LIS test set is evaluated
> once for the numbers above and is never used to pick the best checkpoint. These are the
> current experimental values and will be refreshed once the final training run completes.

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
- Python 3.12
- PyTorch 2.11.0 + torchvision 0.26.0 (CUDA 12.8)
- mmcv-full 1.7.2
- MMDetection 2.15.1 (included in repo)
- NumPy 2.0.2, OpenCV 4.13, pycocotools 2.0.11, Pillow 11.3, Matplotlib 3.10, Flask 3.1.3

> The Colab notebook applies compatibility patches so that mmcv-full 1.7.2 and MMDetection
> 2.15.1 run on Python 3.12 / NumPy 2.x / PyTorch 2.11. Use the notebook for a clean setup.

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

This computes both bbox and segm mAP using the EALLIS test annotations. Boundary quality
can be measured separately with `tools/eval_boundary.py` (Boundary IoU), and the full
three-arm ablation can be run with `tools/run_ablation.py`.

## Web Application

A Flask demo application ([`webapp/app.py`](webapp/app.py)) serves the trained model
through a browser interface. The user uploads a low-light image and sets a confidence
threshold; the app returns the instance masks and boxes, and also visualises the internal
EALLIS maps (the illumination attention map and the predicted edge map).

```bash
# Real model
python webapp/app.py --checkpoint Checkpoints/best_bbox_mAP_epoch_13.pth --device cuda:0
# UI-only demo (no model)
python webapp/app.py --demo
```

## Dataset

The EALLIS dataset contains real-world low-light images with instance-level pixel-wise annotations across 8 object classes.

| Split | Images | Annotations |
|---|---|---|
| Train (held out for validation) | 1,561 | 7,455 |
| Test (evaluation) | 669 | 2,934 |

**Annotation format**: COCO-style JSON with polygon segmentation masks.

## Key Components

| Component | Location | Description | Enabled in current config? |
|---|---|---|---|
| EALLIS block | `mmdetection_custom_part/mmdet/models/plugins/eallis_module.py` | Illumination attention + feature rectifier + edge branch (C3/C4) | ✅ Yes |
| DSL | `mmdetection_custom_part/mmdet/models/detectors/mask_rcnn.py` | Disturbance suppression (noise-invariance) loss | ✅ Yes |
| Edge loss | `mmdetection_custom_part/mmdet/models/losses/edge_loss.py` | Edge-aware boundary loss | ✅ Yes |
| Noise Pipeline | `mmdetection/mmdet/datasets/pipelines/noisemodel/dark_noising.py` | SynCOCO noise synthesis | ✅ Yes |
| AWD | `mmdetection/mmdet/models/backbones/CustomConv.py` | Adaptive weighted downsampling | ❌ Available, not wired into this config |
| SConv | `mmdetection/mmdet/models/backbones/CustomConv.py` | Smooth convolution | ❌ Available, not wired into this config |

## Acknowledgements

This work builds upon the research presented in:

- Chen et al., "Instance Segmentation in the Dark", IJCV 2023
- Hong et al., "Crafting Object Detection in Very Low Light", BMVC 2021
- [MMDetection](https://github.com/open-mmlab/mmdetection) framework by OpenMMLab