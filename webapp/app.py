"""
EALLIS Web App — Flask Backend
Serves the inference API, training data API, and multi-page research dashboard.

Usage:
    python app.py --checkpoint /path/to/best_bbox_mAP_epoch_12.pth
    python app.py --checkpoint /path/to/checkpoint.pth --device cpu
    python app.py --demo   (run without model, for UI testing)
"""

import os
import sys
import io
import json
import base64
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# ─── Fix Windows console encoding (emoji-safe output) ────────────────────────
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ─── Resolve project paths ────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent.parent.resolve()
CONFIG_FILE = str(BASE_DIR / 'Configs' / 'mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py')
LOG_FILE    = str(BASE_DIR / 'None.log.json')

# ─── EALLIS 8-class definitions ───────────────────────────────────────────────
CLASSES = ('bicycle', 'car', 'motorbike', 'bus', 'bottle', 'chair', 'diningtable', 'tvmonitor')

CLASS_COLORS = {
    'bicycle':      (255, 107, 107),   # red
    'car':          (255, 209, 61),    # yellow
    'motorbike':    (107, 203, 119),   # green
    'bus':          (77,  150, 255),   # blue
    'bottle':       (255, 146, 43),    # orange
    'chair':        (204, 93,  232),   # purple
    'diningtable':  (32,  201, 151),   # teal
    'tvmonitor':    (240, 101, 149),   # pink
}

CLASS_ICONS = {
    'bicycle': '🚲', 'car': '🚗', 'motorbike': '🏍️', 'bus': '🚌',
    'bottle': '🍾', 'chair': '🪑', 'diningtable': '🍽️', 'tvmonitor': '📺',
}

app  = Flask(__name__)
model     = None
DEMO_MODE = False


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(config_file: str, checkpoint_file: str, device: str = 'cuda:0'):
    global model

    # Register EALLIS custom modules so mmdet can build the model
    for p in [str(BASE_DIR), str(BASE_DIR / 'mmdetection')]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import mmdetection_custom_part.mmdet.models.detectors          # noqa
    import mmdetection_custom_part.mmdet.datasets.pipelines.edge_target  # noqa
    import mmdetection_custom_part.mmdet.models.backbones           # noqa
    import mmdetection_custom_part.mmdet.models.plugins             # noqa
    import mmdetection_custom_part.mmdet.models.seg_heads           # noqa
    import mmdetection_custom_part.mmdet.models.losses              # noqa

    from mmdet.apis import init_detector
    from mmcv import Config

    cfg = Config.fromfile(config_file)
    cfg.model.pretrained = None
    if hasattr(cfg.model.backbone, 'init_cfg'):
        cfg.model.backbone.init_cfg = None

    model = init_detector(cfg, checkpoint_file, device=device)
    model.CLASSES = CLASSES
    print(f'✅ Model loaded  |  device={device}  |  checkpoint={checkpoint_file}')
    return model


# ─── Visualisation ────────────────────────────────────────────────────────────

def draw_results(img_bgr: np.ndarray, bbox_result, segm_result, score_thr: float = 0.3):
    """
    Draw bounding boxes + masks on the image.
    Returns (PIL.Image, list[dict]) — annotated image and detection metadata.
    """
    from PIL import Image, ImageDraw, ImageFont

    img_rgb = img_bgr[..., ::-1].copy()
    pil_img  = Image.fromarray(img_rgb).convert('RGBA')
    overlay  = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(pil_img)

    # Try to load a nicer font; fall back to default
    try:
        font_label = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 14)
    except Exception:
        font_label = ImageFont.load_default()

    detections = []

    for cls_idx, (bboxes, segms) in enumerate(zip(bbox_result, segm_result)):
        cls_name = CLASSES[cls_idx]
        color    = CLASS_COLORS[cls_name]

        for bbox, mask in zip(bboxes, segms):
            score = float(bbox[4])
            if score < score_thr:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            # ── Mask overlay ─────────────────────────────────────────────────
            if mask is not None:
                m = mask.astype(np.uint8)
                alpha_arr = (m * 110).astype(np.uint8)          # 110/255 opacity
                mask_layer = Image.new('RGBA', pil_img.size, (*color, 0))
                mask_layer.putalpha(Image.fromarray(alpha_arr, 'L'))
                overlay = Image.alpha_composite(overlay, mask_layer)

            # ── Bounding box ──────────────────────────────────────────────────
            for thickness in range(2):
                draw.rectangle(
                    [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                    outline=(*color, 255), width=1)

            # ── Label pill ────────────────────────────────────────────────────
            label = f'{CLASS_ICONS.get(cls_name, "")} {cls_name}  {score:.0%}'
            try:
                tw = draw.textlength(label, font=font_label)
            except AttributeError:
                tw = len(label) * 8
            th = 18
            pad = 4
            draw.rounded_rectangle(
                [x1, y1 - th - pad * 2, x1 + tw + pad * 2, y1],
                radius=4, fill=(*color, 230))
            draw.text((x1 + pad, y1 - th - pad), label, fill=(255, 255, 255, 255), font=font_label)

            detections.append({
                'class': cls_name,
                'icon':  CLASS_ICONS.get(cls_name, ''),
                'score': round(score, 4),
                'score_pct': f'{score:.1%}',
                'bbox':  [x1, y1, x2, y2],
                'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            })

    # Composite overlay onto image
    result = Image.alpha_composite(pil_img, overlay).convert('RGB')
    return result, sorted(detections, key=lambda d: -d['score'])


def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# ─── Training Data Parsing ────────────────────────────────────────────────────

_training_data_cache = None

def parse_training_log():
    """Parse None.log.json and return sampled training metrics for charts."""
    global _training_data_cache
    if _training_data_cache is not None:
        return _training_data_cache

    if not os.path.exists(LOG_FILE):
        return None

    iterations = []
    total_loss = []
    edge_loss  = []
    mask_loss  = []
    cls_loss   = []
    bbox_loss  = []
    accuracy   = []
    lr_vals    = []

    with open(LOG_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if entry.get('mode') != 'train':
                continue

            it = entry.get('iter')
            if it is None:
                continue

            iterations.append(it)
            total_loss.append(round(entry.get('loss', 0), 4))
            edge_loss.append(round(entry.get('edge_loss', 0), 4))
            mask_loss.append(round(entry.get('loss_mask', 0), 4))
            cls_loss.append(round(entry.get('loss_cls', 0), 4))
            bbox_loss.append(round(entry.get('loss_bbox', 0), 4))
            accuracy.append(round(entry.get('acc', 0), 2))
            lr_vals.append(entry.get('lr', 0))

    # Sample every 5th entry to keep chart data manageable
    step = max(1, len(iterations) // 350)
    _training_data_cache = {
        'iterations': iterations[::step],
        'total_loss':  total_loss[::step],
        'edge_loss':   edge_loss[::step],
        'mask_loss':   mask_loss[::step],
        'cls_loss':    cls_loss[::step],
        'bbox_loss':   bbox_loss[::step],
        'accuracy':    accuracy[::step],
        'lr':          lr_vals[::step],
    }
    return _training_data_cache


# ─── Page Routes ──────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('dashboard.html', active_page='dashboard')


@app.route('/predict')
def predict():
    return render_template('predict.html', active_page='predict',
                           demo_mode=DEMO_MODE, classes=CLASSES,
                           class_colors=CLASS_COLORS)


@app.route('/metrics')
def metrics():
    return render_template('metrics.html', active_page='metrics')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html', active_page='analytics')


@app.route('/comparison')
def comparison():
    return render_template('comparison.html', active_page='comparison')


@app.route('/research')
def research():
    return render_template('research.html', active_page='research')


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'demo': DEMO_MODE,
                    'model_loaded': model is not None})


@app.route('/api/training-data')
def api_training_data():
    data = parse_training_log()
    if data is None:
        return jsonify({'error': 'Training log not found'}), 404
    return jsonify(data)


@app.route('/infer', methods=['POST'])
def infer():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file       = request.files['image']
    score_thr  = float(request.form.get('score_thr', 0.3))

    img_bytes = file.read()

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if DEMO_MODE:
        from PIL import Image
        import random
        orig = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = orig.size
        demo_detections = []
        for cls_name in random.sample(CLASSES, min(4, len(CLASSES))):
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + 50, min(x1 + 200, w))
            y2 = random.randint(y1 + 50, min(y1 + 200, h))
            score = round(random.uniform(0.4, 0.95), 4)
            color = CLASS_COLORS[cls_name]
            demo_detections.append({
                'class': cls_name, 'icon': CLASS_ICONS[cls_name],
                'score': score, 'score_pct': f'{score:.1%}',
                'bbox': [x1, y1, x2, y2],
                'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            })
        from PIL import ImageDraw
        vis = orig.copy()
        d = ImageDraw.Draw(vis)
        for det in demo_detections:
            c = tuple(int(det['color'][1:][i*2:i*2+2], 16) for i in range(3))
            b = det['bbox']
            d.rectangle(b, outline=c, width=3)
        return jsonify({
            'success': True, 'demo': True,
            'inference_time': round(0.15, 3),
            'num_detections': len(demo_detections),
            'detections': sorted(demo_detections, key=lambda x: -x['score']),
            'result_image':   pil_to_b64(vis),
            'original_image': pil_to_b64(orig),
        })

    # ── Real inference ─────────────────────────────────────────────────────────
    try:
        import cv2
        from mmdet.apis import inference_detector
        from PIL import Image

        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Could not decode image'}), 400

        orig_rgb = Image.fromarray(img[..., ::-1])

        t0 = time.time()
        result = inference_detector(model, img)
        elapsed = round(time.time() - t0, 3)

        bbox_result, segm_result = result

        vis_img, detections = draw_results(img, bbox_result, segm_result, score_thr)

        return jsonify({
            'success':        True,
            'demo':           False,
            'inference_time': elapsed,
            'num_detections': len(detections),
            'detections':     detections,
            'result_image':   pil_to_b64(vis_img),
            'original_image': pil_to_b64(orig_rgb),
        })

    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EALLIS Web App')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--config', type=str, default=CONFIG_FILE,
                        help='Path to mmdet config file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device: cuda:0 | cpu')
    parser.add_argument('--demo', action='store_true',
                        help='Run in demo mode (no model required)')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--host', type=str, default='0.0.0.0')
    args = parser.parse_args()

    if args.demo:
        DEMO_MODE = True
        print('[DEMO] Running in DEMO MODE -- no model loaded, results are illustrative.')
    elif args.checkpoint:
        load_model(args.config, args.checkpoint, args.device)
    else:
        print('[INFO] No --checkpoint provided. Running in DEMO MODE.')
        DEMO_MODE = True

    print(f'\n[EALLIS] Web App running at  http://localhost:{args.port}\n')
    app.run(host=args.host, port=args.port, debug=False)
