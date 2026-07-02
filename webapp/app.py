# EALLIS web app (Flask): serves the demo UI and the /infer API.
# Run: python app.py --checkpoint <ckpt.pth> [--device cuda:0|cpu]  |  --demo for UI-only.

import sys
import io
import base64
import time
import argparse
import traceback
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, render_template

# Fix Windows console encoding (emoji-safe output)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Resolve project paths
BASE_DIR    = Path(__file__).parent.parent.resolve()
CONFIG_FILE = str(BASE_DIR / 'Configs' / 'mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py')

# EALLIS 8-class definitions
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

app  = Flask(__name__)
model     = None
DEMO_MODE = False


# Model loading
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
    print(f'[OK] Model loaded  |  device={device}  |  checkpoint={checkpoint_file}')
    return model


def clean_inference(model, img):
    # Clean inference (no noise injection); returns (result, attention/edge maps).
    import torch
    from mmcv.parallel import collate, scatter
    from mmdet.datasets import replace_ImageToTensor
    from mmdet.datasets.pipelines import Compose

    cfg = model.cfg.copy()
    device = next(model.parameters()).device
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    data = test_pipeline(dict(img=img))
    data = collate([data], samples_per_gpu=1)
    data['img_metas'] = [m.data[0] for m in data['img_metas']]
    data['img']       = [im.data[0] for im in data['img']]
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]

    # Capture the EALLIS illumination spatial-attention map (C3 stage)
    bb = getattr(model, 'backbone', None)
    cap, handles = {}, []
    if bb is not None and getattr(bb, 'use_eallis', False) and hasattr(bb, 'eallis_c3'):
        def _hook(m, i, o):
            cap['attention'] = o.detach().float().cpu()[0, 0].numpy()
        handles.append(bb.eallis_c3.illum.spatial_conv.register_forward_hook(_hook))

    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)[0]

    for h in handles:
        h.remove()

    # Edge map (C3 stage) is stashed on the backbone after the forward
    if bb is not None and getattr(bb, 'edge_outputs', None):
        e = bb.edge_outputs[0]
        if e is not None:
            cap['edge'] = e.detach().float().cpu().sigmoid()[0, 0].numpy()

    return result, cap


def render_heatmap(arr, base_rgb, mode='jet', overlay=True):
    # Turn a 2-D activation map into a display image sized to base_rgb.
    import cv2
    from PIL import Image

    a = arr.astype(np.float32)
    a -= a.min()
    if a.max() > 1e-8:
        a /= a.max()
    a8 = (a * 255).astype(np.uint8)
    W, H = base_rgb.size
    a8 = cv2.resize(a8, (W, H), interpolation=cv2.INTER_CUBIC)

    if mode == 'gray':                      # white-on-black edge map
        return Image.fromarray(np.stack([a8] * 3, axis=-1))

    cm = cv2.applyColorMap(a8, cv2.COLORMAP_JET)[..., ::-1]   # BGR -> RGB
    heat = Image.fromarray(cm)
    if overlay:
        heat = Image.blend(base_rgb.convert('RGB'), heat, 0.55)
    return heat


# Visualisation
def draw_results(img_bgr: np.ndarray, bbox_result, segm_result, score_thr: float = 0.3):
    # Returns (overlay_img, mask_only_img, detections) for an image.
    from PIL import Image, ImageDraw, ImageFont

    img_rgb = img_bgr[..., ::-1].copy()
    pil_img  = Image.fromarray(img_rgb).convert('RGBA')
    overlay  = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw     = ImageDraw.Draw(pil_img)

    # Separate canvas for the mask-only output (masks on black)
    mask_canvas = Image.new('RGB', pil_img.size, (0, 0, 0))

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

            # Mask overlay + mask-only canvas
            if mask is not None:
                m = mask.astype(bool)
                # translucent coloured mask on top of the original
                alpha_arr  = (m.astype(np.uint8) * 120)
                mask_layer = Image.new('RGBA', pil_img.size, (*color, 0))
                mask_layer.putalpha(Image.fromarray(alpha_arr, 'L'))
                overlay = Image.alpha_composite(overlay, mask_layer)
                # solid coloured mask on the black canvas
                solid = np.array(mask_canvas)
                solid[m] = color
                mask_canvas = Image.fromarray(solid)

            # Bounding box
            for thickness in range(2):
                draw.rectangle(
                    [x1 - thickness, y1 - thickness, x2 + thickness, y2 + thickness],
                    outline=(*color, 255), width=1)

            # Label pill
            label = f'{cls_name}  {score:.0%}'
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
                'score': round(score, 4),
                'score_pct': f'{score:.1%}',
                'bbox':  [x1, y1, x2, y2],
                'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            })

    # Composite overlay onto image
    result = Image.alpha_composite(pil_img, overlay).convert('RGB')
    return result, mask_canvas, sorted(detections, key=lambda d: -d['score'])


def pil_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return base64.b64encode(buf.getvalue()).decode()


# Page Routes
@app.route('/')
def index():
    return render_template('dashboard.html', active_page='dashboard')


@app.route('/segment')
def segment():
    return render_template('segmentation.html', active_page='segment',
                           demo_mode=DEMO_MODE, classes=CLASSES,
                           class_colors=CLASS_COLORS)


# API Routes
@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'demo': DEMO_MODE,
                    'model_loaded': model is not None})


def _predict(img_bytes, score_thr):
    # Run the model (or demo) on raw image bytes; returns a result dict.
    from PIL import Image, ImageDraw

    # Demo mode
    if DEMO_MODE:
        import random
        orig = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = orig.size
        detections, boxes = [], []
        for cls_name in random.sample(CLASSES, min(4, len(CLASSES))):
            x1 = random.randint(0, w // 2)
            y1 = random.randint(0, h // 2)
            x2 = random.randint(x1 + 50, min(x1 + 200, w))
            y2 = random.randint(y1 + 50, min(y1 + 200, h))
            score = round(random.uniform(0.4, 0.95), 4)
            color = CLASS_COLORS[cls_name]
            detections.append({
                'class': cls_name, 'score': score, 'score_pct': f'{score:.1%}',
                'bbox': [x1, y1, x2, y2],
                'color': f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}',
            })
            boxes.append(([x1, y1, x2, y2], color))

        vis  = orig.copy()
        mask = Image.new('RGB', (w, h), (0, 0, 0))
        dv, dm = ImageDraw.Draw(vis), ImageDraw.Draw(mask)
        for b, c in boxes:
            dv.rectangle(b, outline=c, width=3)
            dm.rectangle(b, fill=c)
        return {
            'success': True, 'demo': True,
            'inference_time': round(0.15, 3),
            'num_detections': len(detections),
            'detections': sorted(detections, key=lambda x: -x['score']),
            'result_image':   pil_to_b64(vis),
            'mask_image':     pil_to_b64(mask),
            'original_image': pil_to_b64(orig),
        }

    # Real inference
    import cv2

    nparr = np.frombuffer(img_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {'error': 'Could not decode image'}

    orig_rgb = Image.fromarray(img[..., ::-1])
    t0 = time.time()
    (bbox_result, segm_result), maps = clean_inference(model, img)
    elapsed = round(time.time() - t0, 3)

    vis_img, mask_img, detections = draw_results(img, bbox_result, segm_result, score_thr)
    resp = {
        'success': True, 'demo': False,
        'inference_time': elapsed,
        'num_detections': len(detections),
        'detections': detections,
        'result_image':   pil_to_b64(vis_img),
        'mask_image':     pil_to_b64(mask_img),
        'original_image': pil_to_b64(orig_rgb),
    }

    # EALLIS module visualisations
    if 'attention' in maps:
        resp['attention_image'] = pil_to_b64(render_heatmap(maps['attention'], orig_rgb, 'jet', overlay=True))
    if 'edge' in maps:
        resp['edge_image'] = pil_to_b64(render_heatmap(maps['edge'], orig_rgb, 'gray', overlay=False))

    return resp


@app.route('/infer', methods=['POST'])
def infer():
    score_thr = float(request.form.get('score_thr', 0.3))

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    img_bytes = request.files['image'].read()

    try:
        resp = _predict(img_bytes, score_thr)
    except Exception as e:
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500
    if 'error' in resp:
        return jsonify(resp), 400

    return jsonify(resp)


# Entry point
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
