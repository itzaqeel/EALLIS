# ============================================================
# CELL 0 — GPU Architecture Compatibility Check
# Paste this as the VERY FIRST code cell in your Kaggle notebook.
# It detects cudaErrorNoKernelImageForDevice BEFORE training starts
# and auto-installs the correct PyTorch for the assigned GPU.
# ============================================================

import torch, sys, os, subprocess

if not torch.cuda.is_available():
    raise RuntimeError("❌ No GPU found! Enable GPU: Settings → Accelerator → T4/P100.")

gpu_name = torch.cuda.get_device_name(0)
sm_major, sm_minor = torch.cuda.get_device_capability(0)
torch_ver = torch.__version__.split('+')[0]
cuda_ver  = torch.version.cuda           # e.g. '12.1'
cuda_tag  = cuda_ver.replace('.', '')    # e.g. '121'
sm_int    = sm_major * 10 + sm_minor

print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {cuda_ver}")
print(f"Python  : {sys.version}")
print(f"GPU     : {gpu_name}  (sm_{sm_major}{sm_minor})")

# ── Quick functional test ──────────────────────────────────────────────────────
# cudaErrorNoKernelImageForDevice manifests here, not at import time.
def _check_torch_compat():
    """Return True if torch can actually dispatch a kernel on this GPU."""
    try:
        t = torch.zeros(1).cuda()
        _ = torch.relu(t)
        return True
    except Exception:
        return False

if _check_torch_compat():
    print(f"\n✅ PyTorch {torch_ver} is compatible with {gpu_name} (sm_{sm_int}).")
else:
    # ── Pick a known-good PyTorch for this SM ────────────────────────────────
    # sm_60 = P100 | sm_70 = V100 | sm_75 = T4 | sm_80 = A100 | sm_89 = L4 | sm_90 = H100
    print(f"\n⚠️  cudaErrorNoKernelImageForDevice — PyTorch {torch_ver} built for a "
          f"different arch than sm_{sm_int}.")

    if sm_int >= 90:       # H100
        pkgs = "torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121"
    elif sm_int >= 89:     # L4
        pkgs = "torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121"
    elif sm_int >= 80:     # A100
        pkgs = "torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121"
    elif sm_int >= 75:     # T4
        pkgs = "torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118"
    elif sm_int >= 70:     # V100
        pkgs = "torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117"
    else:                  # P100 / older
        pkgs = "torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cu116"

    cmd = f"pip install -q {pkgs}"
    print(f"    Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

    print("\n⚠️  PyTorch reinstalled.")
    print("    ⚡ RESTART THE KERNEL NOW: Runtime → Restart, then re-run ALL cells from Cell 0.")
    raise SystemExit("Kernel restart required after torch reinstall.")
