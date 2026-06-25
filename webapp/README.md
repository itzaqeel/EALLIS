# EALLIS Web App

A dark-themed web interface for the EALLIS low-light instance segmentation model.

---

## Quick Start

### 1. Install webapp dependencies

```bash
cd webapp
pip install -r requirements.txt
```

> `torch`, `mmcv`, `mmdetection`, and other heavy deps are assumed to already be
> installed in your EALLIS project environment.

---

### 2. Run with your trained checkpoint

```bash
python app.py --checkpoint /path/to/best_bbox_mAP_epoch_12.pth
```

| Flag | Default | Description |
|---|---|---|
| `--checkpoint` | *(required)* | Path to your `.pth` checkpoint |
| `--config` | `Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py` | mmdet config |
| `--device` | `cuda:0` | `cuda:0` or `cpu` |
| `--port` | `5000` | Web server port |
| `--demo` | off | Run without model (UI test only) |

---

### 3. Open in browser

```
http://localhost:5000
```

---

### Run in Demo Mode (no model needed)

```bash
python app.py --demo
```

Useful for testing the UI before the model is set up locally.

---

## What It Does

1. **Upload** a dark/low-light image (drag & drop or browse)
2. **Adjust** the confidence threshold slider (default 0.30)
3. Click **Analyse Image**
4. See the **original vs annotated** result side by side
5. Browse the **detection cards** — class, confidence bar, bounding box coords

---

## Detectable Classes

| Icon | Class | Color |
|---|---|---|
| 🚲 | bicycle | Red |
| 🚗 | car | Yellow |
| 🏍️ | motorbike | Green |
| 🚌 | bus | Blue |
| 🍾 | bottle | Orange |
| 🪑 | chair | Purple |
| 🍽️ | diningtable | Teal |
| 📺 | tvmonitor | Pink |
