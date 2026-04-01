import os, glob, sys, shutil
import torch

torch_ver = torch.__version__.split('+')[0]
cuda_ver  = torch.version.cuda.replace('.', '')  # e.g. '121'

# ── mmcv-full install ─────────────────────────────────────────────────────────
# The cached wheel is only valid for the same torch+cuda combo.
# If the GPU changed (and torch was reinstalled above), purge the old wheel.
def _wheel_matches_env(whl_path):
    """Check that a cached mmcv wheel was built for the current torch+CUDA."""
    name = os.path.basename(whl_path)
    return f'torch{torch_ver}' in name and f'cu{cuda_ver}' in name

try:
    import mmcv
    print(f'✅ mmcv-full {mmcv.__version__} already installed!')
except ImportError:
    cached_wheels = glob.glob(os.path.join(SAVE_DIR, 'mmcv_full-1.7.2*.whl'))
    valid_wheels  = [w for w in cached_wheels if _wheel_matches_env(w)]
    stale_wheels  = [w for w in cached_wheels if not _wheel_matches_env(w)]

    for stale in stale_wheels:
        os.remove(stale)
        print(f'🗑️  Removed stale wheel (wrong torch/CUDA): {os.path.basename(stale)}')

    if valid_wheels:
        print(f'📦 Found compatible cached wheel, installing...')
        os.system(f'pip install -q {valid_wheels[0]}')
    else:
        print(f'⏳ Building mmcv-full from source for torch{torch_ver}+cu{cuda_ver} (~15-25 min)...')
        os.makedirs('/tmp/mmcv_wheels', exist_ok=True)
        ret = os.system(
            f'pip wheel mmcv-full==1.7.2 -w /tmp/mmcv_wheels '
            f'-f https://download.openmmlab.com/mmcv/dist/cu{cuda_ver}/torch{torch_ver}/index.html '
            f'2>&1 | tail -10'
        )
        built = glob.glob('/tmp/mmcv_wheels/mmcv_full-1.7.2*.whl')
        if built:
            os.system(f'pip install -q {built[0]}')
            shutil.copy2(built[0], SAVE_DIR)
            print(f'💾 Wheel saved to {SAVE_DIR}')
        else:
            print('⚠️  pip wheel failed, trying direct install...')
            os.system(
                f'pip install mmcv-full==1.7.2 '
                f'-f https://download.openmmlab.com/mmcv/dist/cu{cuda_ver}/torch{torch_ver}/index.html'
            )

    import mmcv
    print(f'✅ mmcv-full {mmcv.__version__} installed!')

!pip install -q pycocotools scikit-learn terminaltables pretrainedmodels
print('✅ All dependencies ready!')
