import os
import sys
import glob
import torch

# ── CUDA debug helpers ────────────────────────────────────────────────────────
# If cudaErrorNoKernelImageForDevice occurs inside the training loop, these
# make CUDA report errors synchronously so the traceback points at the real
# offending line (not some unrelated async call).
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA']   = '1'

REPO_DIR = "/kaggle/working/EALLIS"
os.chdir(REPO_DIR)

sys.path.insert(0, REPO_DIR)
sys.path.insert(0, os.path.join(REPO_DIR, "mmdetection"))

print("Paths configured")

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# Patch scatter bug
import mmcv.parallel.scatter_gather as scatter_gather

if not hasattr(scatter_gather, "_original_scatter"):
    scatter_gather._original_scatter = scatter_gather.scatter

def patched_scatter(inputs, target_gpus, dim=0):
    import torch
    if isinstance(target_gpus[0], int):
        target_gpus = [torch.device(f"cuda:{i}") for i in target_gpus]
    return scatter_gather._original_scatter(inputs, target_gpus, dim)

scatter_gather.scatter = patched_scatter

# Load config
cfg = Config.fromfile("Configs/train_kaggle.py")

cfg.device = "cuda"
cfg.gpu_ids = [0]

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    sm = torch.cuda.get_device_capability(0)
    print(f"SM : sm_{sm[0]}{sm[1]}")

# Dataset
print("\nBuilding dataset...")
datasets = [build_dataset(cfg.data.train)]
print(f"Train dataset size: {len(datasets[0])}")

# Model
print("\nBuilding model...")
model = build_detector(cfg.model)
model.init_weights()

model.CLASSES = datasets[0].CLASSES

# Verify EALLIS
found = False
for name, module in model.named_modules():
    if "eallis" in name.lower():
        print("FOUND:", name)
        found = True

if not found:
    raise RuntimeError("EALLIS not detected")

# Train
train_detector(
    model,
    datasets,
    cfg,
    distributed=False,
    validate=True,
    meta=dict()
)

print("\nTraining complete!")

# Outputs
print("\nSaved files:")
for f in sorted(glob.glob(os.path.join(cfg.work_dir, "*"))):
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"{os.path.basename(f):40s} {size_mb:8.1f} MB")
