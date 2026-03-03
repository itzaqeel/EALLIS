# --- Cell 2 ---
import torch, sys, os
print(f'PyTorch: {torch.__version__}')
print(f'CUDA:    {torch.version.cuda}')
print(f'Python:  {sys.version}')
print(f'GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE — enable GPU!"}')

# Kaggle paths
KAGGLE_INPUT  = '/kaggle/input'
KAGGLE_OUTPUT = '/kaggle/working'
REPO_DIR      = os.path.join(KAGGLE_OUTPUT, 'EALLIS')
SAVE_DIR      = os.path.join(KAGGLE_OUTPUT, 'outputs')   # wheels + checkpoints saved here
os.makedirs(SAVE_DIR, exist_ok=True)

# Auto-detect COCO dataset path from Kaggle input
COCO_ROOT = None
for item in os.listdir(KAGGLE_INPUT):
    candidate = os.path.join(KAGGLE_INPUT, item)
    # Check for coco2017 subfolder or direct train2017
    if os.path.isdir(os.path.join(candidate, 'coco2017')):
        COCO_ROOT = os.path.join(candidate, 'coco2017')
        break
    elif os.path.isdir(os.path.join(candidate, 'train2017')):
        COCO_ROOT = candidate
        break

if COCO_ROOT:
    print(f'\n✅ COCO root: {COCO_ROOT}')
    print(f'  contents: {os.listdir(COCO_ROOT)}')
else:
    print('\n❌ COCO dataset not found! Add it as Kaggle input.')
    print(f'  Available inputs: {os.listdir(KAGGLE_INPUT)}')

# --- Cell 4 ---
import os, glob, sys
import torch

torch_ver = torch.__version__.split('+')[0]
cuda_ver = torch.version.cuda.replace('.', '')

# Check if mmcv is already installed
try:
    import mmcv
    print(f'✅ mmcv-full {mmcv.__version__} already installed!')
except ImportError:
    # Check for a previously saved wheel in output
    cached_wheels = glob.glob(os.path.join(SAVE_DIR, 'mmcv_full-1.7.2*.whl'))
    
    if cached_wheels:
        print(f'📦 Found cached wheel, installing...')
        os.system(f'pip install -q {cached_wheels[0]}')
    else:
        print(f'⏳ Building mmcv-full from source (~15-25 min)...')
        print(f'   PyTorch {torch_ver}, CUDA {cuda_ver}')
        
        # Build wheel
        os.system(f'pip wheel mmcv-full==1.7.2 -w /tmp/mmcv_wheels '
                  f'-f https://download.openmmlab.com/mmcv/dist/cu{cuda_ver}/torch{torch_ver}/index.html '
                  f'2>&1 | tail -5')
        
        built = glob.glob('/tmp/mmcv_wheels/mmcv_full-1.7.2*.whl')
        if built:
            os.system(f'pip install -q {built[0]}')
            # Save wheel to output for download / reuse
            import shutil
            shutil.copy2(built[0], SAVE_DIR)
            print(f'💾 Wheel saved to {SAVE_DIR}')
        else:
            os.system(f'pip install mmcv-full==1.7.2 '
                      f'-f https://download.openmmlab.com/mmcv/dist/cu{cuda_ver}/torch{torch_ver}/index.html')
    
    import mmcv
    print(f'✅ mmcv-full {mmcv.__version__} installed!')

!pip install -q pycocotools scikit-learn terminaltables pretrainedmodels
print('✅ All dependencies ready!')

# --- Cell 6 ---
os.chdir(KAGGLE_OUTPUT)

if not os.path.exists(REPO_DIR):
    !git clone https://github.com/itzaqeel/EALLIS.git {REPO_DIR}
else:
    os.chdir(REPO_DIR)
    !git pull

os.chdir(REPO_DIR)
print(f'Working directory: {os.getcwd()}')

# --- Cell 8 ---
os.chdir(os.path.join(REPO_DIR, 'mmdetection'))
!pip install -q -e . --no-deps
os.chdir(REPO_DIR)

sys.path.insert(0, os.path.join(REPO_DIR, 'mmdetection'))

import mmdet
print(f'mmdet version: {mmdet.__version__}')

# --- Cell 10 ---
import glob, re
os.chdir(REPO_DIR)

# --- Fix 1: Bump mmcv version cap ---
init_file = 'mmdetection/mmdet/__init__.py'
with open(init_file, 'r') as f:
    content = f.read()
content = re.sub(r"mmcv_maximum_version\s*=\s*'[^']*'", "mmcv_maximum_version = '3.0.0'", content)
with open(init_file, 'w') as f:
    f.write(content)
print('[Fix 1] mmcv version cap updated.')

# --- Fix 2: Deprecated/removed imports ---
deprecated_imports = {
    'import imp': '# import imp  # removed in Python 3.12',
    'from os import pread': '# from os import pread',
    'from tokenize import group': '# from tokenize import group',
    'from numpy.core.fromnumeric import size': '# from numpy.core.fromnumeric import size',
    'from numpy.core.numeric import outer': '# from numpy.core.numeric import outer',
    'from numpy.lib.npyio import load': '# from numpy.lib.npyio import load',
    'from numpy.lib.arraypad import pad': 'from numpy import pad',
    'from numpy.lib.type_check import common_type': 'from numpy import common_type',
    'from torch.functional import _index_tensor_with_indices_list': '# from torch.functional import _index_tensor_with_indices_list',
    'from numpy.testing._private.utils import print_assert_equal': '# from numpy.testing._private.utils import print_assert_equal',
}
fix2_files = glob.glob('mmdetection/**/*.py', recursive=True) + \
             glob.glob('mmdetection_custom_part/**/*.py', recursive=True) + \
             glob.glob('utils/**/*.py', recursive=True)
fix2_count = 0
for py_file in fix2_files:
    with open(py_file, 'r') as f:
        content = f.read()
    new_content = content
    for old_imp, new_imp in deprecated_imports.items():
        new_content = new_content.replace(old_imp, new_imp)
    if new_content != content:
        with open(py_file, 'w') as f:
            f.write(new_content)
        fix2_count += 1
print(f'[Fix 2] Fixed deprecated imports in {fix2_count} files.')

# --- Fix 3: Relative imports in custom_part ---
import_fixes = [
    ('mmdetection_custom_part/mmdet/models/backbones/resnet.py', 'from ..utils', 'from mmdet.models.utils'),
    ('mmdetection_custom_part/mmdet/models/backbones/resnet.py', 'from .cbam', 'from mmdet.models.backbones.cbam'),
    ('mmdetection_custom_part/mmdet/models/backbones/resnext.py', 'from ..utils', 'from mmdet.models.utils'),
    ('mmdetection_custom_part/mmdet/models/backbones/swin.py', 'from ...utils', 'from mmdet.utils'),
    ('mmdetection_custom_part/mmdet/models/backbones/swin.py', 'from ..utils.ckpt_convert', 'from mmdet.models.utils.ckpt_convert'),
    ('mmdetection_custom_part/mmdet/models/backbones/swin.py', 'from ..utils.transformer', 'from mmdet.models.utils.transformer'),
    ('mmdetection_custom_part/mmdet/models/dense_heads/maskformer_head.py', 'from .anchor_free_head', 'from mmdet.models.dense_heads.anchor_free_head'),
    ('mmdetection_custom_part/mmdet/models/dense_heads/mask2former_head.py', 'from .anchor_free_head', 'from mmdet.models.dense_heads.anchor_free_head'),
    ('mmdetection_custom_part/mmdet/models/detectors/two_stage.py', 'from .base', 'from mmdet.models.detectors.base'),
    ('mmdetection_custom_part/mmdet/models/detectors/maskformer.py', 'from .single_stage', 'from mmdet.models.detectors.single_stage'),
    ('mmdetection_custom_part/mmdet/models/seg_heads/base_semantic_head.py', 'from ..utils import interpolate_as', 'from mmdet.models.utils import interpolate_as'),
    ('mmdetection_custom_part/mmdet/models/seg_heads/panoptic_fpn_head.py', 'from ..utils import ConvUpsample', 'from mmdet.models.utils import ConvUpsample'),
]
for filepath, old_imp, new_imp in import_fixes:
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            content = f.read()
        if old_imp in content:
            content = content.replace(old_imp, new_imp)
            with open(filepath, 'w') as f:
                f.write(content)
print('[Fix 3] Relative imports fixed.')

# --- Fix 4: Detector auxiliary imports ---
for det_file in [
    'mmdetection_custom_part/mmdet/models/detectors/mask_rcnn.py',
    'mmdetection_custom_part/mmdet/models/detectors/faster_rcnn_noise_inv.py'
]:
    if os.path.exists(det_file):
        with open(det_file, 'r') as f:
            content = f.read()
        content = content.replace('from ..backbones.aux_modules', 'from mmdet.models.backbones.aux_modules')
        content = content.replace('from ..backbones.multiscale_discriminator', 'from mmdet.models.backbones.multiscale_discriminator')
        content = content.replace('from ..backbones.lsid', 'from mmdet.models.backbones.lsid')
        with open(det_file, 'w') as f:
            f.write(content)
print('[Fix 4] Detector auxiliary imports fixed.')

# --- Fix 5: Force register_module() ---
count = 0
for py_file in glob.glob('mmdetection_custom_part/**/*.py', recursive=True):
    with open(py_file, 'r') as f:
        content = f.read()
    if '.register_module()' in content:
        with open(py_file, 'w') as f:
            f.write(content.replace('.register_module()', '.register_module(force=True)'))
        count += 1
print(f'[Fix 5] Forced registration in {count} files.')

# --- Fix 6: Rewrite __init__.py files ---
with open('mmdetection_custom_part/mmdet/models/backbones/__init__.py', 'w') as f:
    f.write("""from .resnet import ResNet, ResNetV1d, ResNetAdaD, ResNetAdaDSmoothPrior
from .resnext import ResNeXt
from .swin import SwinTransformer, SwinTransformerAdaD
from .convnext import ConvNeXt, ConvNeXtAdaD
__all__ = ['ResNet', 'ResNetV1d', 'ResNetAdaD', 'ResNetAdaDSmoothPrior',
           'ResNeXt', 'SwinTransformer', 'SwinTransformerAdaD',
           'ConvNeXt', 'ConvNeXtAdaD']
""")
with open('mmdetection_custom_part/mmdet/models/dense_heads/__init__.py', 'w') as f:
    f.write("""from .maskformer_head import MaskFormerHead
from .mask2former_head import Mask2FormerHead
__all__ = ['MaskFormerHead', 'Mask2FormerHead']
""")
with open('mmdetection_custom_part/mmdet/models/detectors/__init__.py', 'w') as f:
    f.write("""from .two_stage import TwoStageDetector
from .faster_rcnn import FasterRCNN
from .faster_rcnn_noise_inv import FasterRCNNNoiseInv
from .mask_rcnn import MaskRCNN, MaskRCNNNoiseInv as MaskRCNNNoiseInvDet
from .maskformer import MaskFormer
from .mask2former import Mask2Former
__all__ = ['TwoStageDetector', 'FasterRCNN', 'FasterRCNNNoiseInv',
           'MaskRCNN', 'MaskRCNNNoiseInvDet', 'MaskFormer', 'Mask2Former']
""")
models_init = 'mmdetection_custom_part/mmdet/models/__init__.py'
if os.path.exists(models_init):
    with open(models_init, 'r') as f:
        content = f.read()
    content = content.replace('from .necks import *', '# from .necks import *')
    with open(models_init, 'w') as f:
        f.write(content)
print('[Fix 6] __init__.py files rewritten.')

# --- Fix 7: NoiseModel camera_params path ---
noise_file = 'mmdetection/mmdet/datasets/pipelines/noisemodel/dark_noising.py'
with open(noise_file, 'r') as f:
    content = f.read()
if '~/code/mmdetection' in content:
    if 'import os\n' not in content:
        content = content.replace('import os.path as osp', 'import os\nimport os.path as osp')
    content = content.replace(
        """        if param_dir is None:
            try:
                self.param_dir = '~/code/mmdetection/mmdet/datasets/pipelines/noisemodel/camera_params'
            except:
                print('please specify the location of camera parameters, e.g., ~/code/mmdetection/mmdet/datasets/pipelines/noisemodel/camera_params')
                raise Exception""",
        """        if param_dir is None:
            self.param_dir = os.path.join(os.path.dirname(__file__), 'camera_params')""")
    with open(noise_file, 'w') as f:
        f.write(content)
    print('[Fix 7] NoiseModel camera_params path fixed.')
else:
    print('[Fix 7] NoiseModel path already fixed.')

# --- Fix 8: NumPy 2.x _sample_params inhomogeneous array ---
with open(noise_file, 'r') as f:
    content = f.read()
old_return = 'return np.array([K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio])'
new_return = 'return (K, color_bias, g_scale, G_scale, G_shape, R_scale, Q_step, saturation_level, ratio)'
if old_return in content:
    content = content.replace(old_return, new_return)
    with open(noise_file, 'w') as f:
        f.write(content)
    print('[Fix 8] NumPy 2.x _sample_params array fix applied.')
else:
    print('[Fix 8] _sample_params already fixed.')

print('\n✅ All fixes applied!')

# --- Cell 12 ---
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'mmdetection'))

try:
    import mmdetection_custom_part.mmdet.models.detectors
    import mmdetection_custom_part.mmdet.models.backbones
    import mmdetection_custom_part.mmdet.models.dense_heads
    import mmdetection_custom_part.mmdet.models.roi_heads
    import mmdetection_custom_part.mmdet.models.plugins
    import mmdetection_custom_part.mmdet.models.seg_heads
    import mmdetection_custom_part.mmdet.models.losses
    print('✅ All custom modules imported successfully!')
except Exception as e:
    import traceback
    traceback.print_exc()
    raise RuntimeError(f'Custom module import failed: {e}')

# --- Cell 14 ---
import shutil
import os
os.chdir(REPO_DIR)

# --- COCO: Symlink from Kaggle input (read-only, no copy needed) ---
os.makedirs('data/coco', exist_ok=True)

for folder in ['annotations', 'train2017', 'val2017', 'test2017']:
    src = os.path.join(COCO_ROOT, folder)
    dst = os.path.join(REPO_DIR, 'data', 'coco', folder)
    if os.path.islink(dst):
        os.unlink(dst)
    elif os.path.isdir(dst):
        shutil.rmtree(dst)
    if os.path.exists(src):
        os.symlink(src, dst)
        print(f'  🔗 {folder}/ → {src}')
    else:
        print(f'  ⚠️  {src} not found')

train_count = len(os.listdir('data/coco/train2017'))
ann_files = os.listdir('data/coco/annotations')
print(f'\n✅ COCO ready: {train_count:,} train images, {len(ann_files)} annotation files')

# --- EALLIS dataset ---
os.makedirs('data/eallis/annotations', exist_ok=True)
os.makedirs('data/eallis/images', exist_ok=True)

# Check if EALLIS is available as a Kaggle dataset input
eallis_kaggle = '/kaggle/input/datasets/aqeelaslam8870/eallis'
if os.path.exists(eallis_kaggle):
    for sub in ['annotations', 'images', 'JPEGImages']:
        src = os.path.join(eallis_kaggle, sub)
        if os.path.exists(src):
            target = 'data/eallis/images' if sub == 'JPEGImages' else f'data/eallis/{sub}'
            !cp -rn {src}/* {target}/
    print(f'EALLIS images: {len(os.listdir("data/eallis/images"))} files')
else:
    print(f'ℹ️  EALLIS dataset not found at {eallis_kaggle}.')
    print(f'   Add it as a Kaggle dataset input, or upload to data/eallis/ manually.')

# JPEGImages symlink for EALLIS
jpeg_link = 'data/eallis/JPEGImages'
if not os.path.exists(jpeg_link):
    os.symlink(os.path.abspath('data/eallis/images'), jpeg_link)

# --- Cell 16 ---
import json, random
os.chdir(REPO_DIR)

USE_SUBSET = True       # Set False for full COCO
SUBSET_RATIO = 0.10     # 10%

if USE_SUBSET:
    ann_path = 'data/coco/annotations/instances_train2017.json'
    # Write subset to writable location (Kaggle input is read-only)
    os.makedirs('data/coco_subset', exist_ok=True)
    subset_path = 'data/coco_subset/instances_train2017_subset.json'

    print('Loading full COCO annotations...')
    with open(ann_path, 'r') as f:
        coco_data = json.load(f)

    all_images = coco_data['images']
    num_subset = int(len(all_images) * SUBSET_RATIO)
    random.seed(42)
    subset_images = random.sample(all_images, num_subset)
    subset_img_ids = set(img['id'] for img in subset_images)
    subset_annotations = [a for a in coco_data['annotations'] if a['image_id'] in subset_img_ids]

    subset_data = {
        'info': coco_data.get('info', {}),
        'licenses': coco_data.get('licenses', []),
        'images': subset_images,
        'annotations': subset_annotations,
        'categories': coco_data['categories']
    }
    with open(subset_path, 'w') as f:
        json.dump(subset_data, f)

    print(f'✅ {SUBSET_RATIO*100:.0f}% subset: {len(subset_images):,} images, {len(subset_annotations):,} annotations')
else:
    print('Using full COCO dataset.')

# --- Cell 18 ---
os.chdir(REPO_DIR)

CONFIG_FILE = 'Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py'

with open(CONFIG_FILE, 'r') as f:
    config_content = f.read()

# Batch size for Kaggle GPU (T4 = 16GB, P100 = 16GB)
config_content = config_content.replace('BATCHSIZE = 8', 'BATCHSIZE = 4')

# No pretrained checkpoint on first run
config_content = re.sub(
    r"load_from\s*=\s*'[^']*'",
    "load_from = None",
    config_content)

# Use subset annotation file (writable location)
if USE_SUBSET:
    config_content = config_content.replace(
        "ann_file='data/coco/annotations/instances_train2017.json'",
        "ann_file='data/coco_subset/instances_train2017_subset.json'")

# Set work_dir to Kaggle output
config_content = config_content.replace(
    "work_dir = './work_dir'",
    f"work_dir = '{SAVE_DIR}'")

TRAIN_CONFIG = 'Configs/train_kaggle.py'
with open(TRAIN_CONFIG, 'w') as f:
    f.write(config_content)

subset_label = f'{SUBSET_RATIO*100:.0f}% subset' if USE_SUBSET else 'full'
print(f'Config saved: {TRAIN_CONFIG}')
print(f'  Train: COCO ({subset_label}) | Batch: 4 | Epochs: 12')
print(f'  Checkpoints → {SAVE_DIR}  (downloadable from Kaggle Output)')

# --- Cell 20 ---
os.chdir(REPO_DIR)

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, 'mmdetection'))

import mmdetection_custom_part.mmdet.models.detectors
import mmdetection_custom_part.mmdet.models.backbones
import mmdetection_custom_part.mmdet.models.dense_heads
import mmdetection_custom_part.mmdet.models.roi_heads
import mmdetection_custom_part.mmdet.models.plugins
import mmdetection_custom_part.mmdet.models.seg_heads
import mmdetection_custom_part.mmdet.models.losses

import shutil, torch, glob
import os
from mmcv import Config
from mmcv.runner import HOOKS, Hook
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

cfg = Config.fromfile('Configs/train_kaggle.py')

print('Building datasets...')
datasets = [build_dataset(cfg.data.train)]
print(f'Train dataset: {len(datasets[0])} images')

print('Building model...')
model = build_detector(cfg.model)
model.init_weights()
model.CLASSES = datasets[0].CLASSES

print(f'Model: {cfg.model.type} | Backbone: {cfg.model.backbone.type}')
print(f'Training for {cfg.runner.max_epochs} epochs')
print(f'Checkpoints saved to: {cfg.work_dir}')

train_detector(model, datasets, cfg, distributed=False, validate=True, meta=dict())

print('\n✅ Training complete!')
print(f'\n📁 Output files (downloadable from Kaggle Output tab):')
for f in sorted(glob.glob(os.path.join(SAVE_DIR, '*'))):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f'  {os.path.basename(f):40s} {size_mb:8.1f} MB')

# --- Cell 22 ---
os.chdir(REPO_DIR)

from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet.datasets import build_dataloader
from mmdet.apis import single_gpu_test

# Find best checkpoint
best_ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, 'best_*.pth')))
if best_ckpts:
    ckpt_path = best_ckpts[-1]
else:
    epoch_ckpts = sorted(glob.glob(os.path.join(SAVE_DIR, 'epoch_*.pth')))
    ckpt_path = epoch_ckpts[-1] if epoch_ckpts else os.path.join(SAVE_DIR, 'latest.pth')
print(f'Using checkpoint: {ckpt_path}')

test_dataset = build_dataset(cfg.data.test)
test_loader = build_dataloader(test_dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

cfg.model.pretrained = None
cfg.model.backbone.init_cfg = None
eval_model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
checkpoint = load_checkpoint(eval_model, ckpt_path, map_location='cpu')
eval_model.CLASSES = test_dataset.CLASSES
eval_model = MMDataParallel(eval_model, device_ids=[0])
eval_model.eval()

print(f'Running inference on {len(test_dataset)} images...')
results = single_gpu_test(eval_model, test_loader, show=False)

eval_results = test_dataset.evaluate(results, metric=['bbox', 'segm'])
print('\n' + '='*60)
print('EVALUATION RESULTS')
print('='*60)
for key, val in eval_results.items():
    print(f'  {key}: {val:.4f}' if isinstance(val, float) else f'  {key}: {val}')
print('='*60)

