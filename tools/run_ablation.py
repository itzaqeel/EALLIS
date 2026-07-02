# Automated EALLIS ablation: trains baseline / illum / full variants and writes a summary table.
# Variants via backbone flags: baseline (use_eallis=False), illum (use_edge=False), full (both on).
# Usage: python tools/run_ablation.py --config <cfg>.py --work-root <dir> --load-from <ckpt>.pth --epochs 5 --lr-step 3
import os
import sys
import re
import json
import glob
import argparse
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VARIANTS = {
    'baseline': dict(use_eallis='False', use_edge='False'),
    'illum':    dict(use_eallis='True',  use_edge='False'),
    'full':     dict(use_eallis='True',  use_edge='True'),
}


def train_variant(name, flags, args):
    work_dir = os.path.join(args.work_root, name)
    os.makedirs(work_dir, exist_ok=True)
    if os.path.exists(os.path.join(work_dir, f'epoch_{args.epochs}.pth')):
        print(f'[skip] {name}: already trained')
        return work_dir
    cfg_opts = [
        f'model.backbone.use_eallis={flags["use_eallis"]}',
        f'model.backbone.use_edge={flags["use_edge"]}',
        f'runner.max_epochs={args.epochs}',
        f'lr_config.step=[{args.lr_step}]',
        'evaluation.interval=1',
        f'checkpoint_config.interval={args.epochs}',
        f'load_from={args.load_from}',
        'resume_from=None',
    ]
    cmd = [sys.executable, os.path.join(REPO, 'mmdetection/tools/train.py'),
           args.config, '--work-dir', work_dir, '--seed', '42',
           '--cfg-options'] + cfg_opts
    print(f'[train] {name} -> {work_dir}')
    
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{os.path.join(REPO, 'mmdetection')}:{env.get('PYTHONPATH', '')}"
    
    subprocess.run(cmd, check=True, cwd=REPO, env=env)
    return work_dir


def last_val_metrics(work_dir):
    logs = sorted(glob.glob(os.path.join(work_dir, '*.log.json')))
    if not logs:
        return {}
    vals = []
    for line in open(logs[-1]):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get('mode') == 'val':
            vals.append(r)
    return vals[-1] if vals else {}


def boundary_iou(work_dir, args):
    ckpts = sorted(glob.glob(os.path.join(work_dir, 'epoch_*.pth')))
    if not ckpts:
        return None
    # Use the variant's own dumped config so the model is rebuilt with the
    # exact flags it was trained with (otherwise a baseline checkpoint would be
    # evaluated with random edge layers attached).
    dumped = glob.glob(os.path.join(work_dir, '*.py'))
    cfg = dumped[0] if dumped else args.config
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{REPO}:{os.path.join(REPO, 'mmdetection')}:{env.get('PYTHONPATH', '')}"
    
    out = subprocess.run(
        [sys.executable, os.path.join(REPO, 'tools/eval_boundary.py'),
         '--config', cfg, '--checkpoint', ckpts[-1]],
        capture_output=True, text=True, cwd=REPO, env=env)
    m = re.search(r'mean Boundary IoU\s*:\s*([0-9.]+)', out.stdout)
    if not m:
        print(out.stdout[-500:])  # surface errors if parsing failed
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--work-root', required=True)
    ap.add_argument('--load-from',
                    default='/content/drive/MyDrive/EALLIS_Output/Checkpoint1.pth')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--lr-step', type=int, default=3)
    args = ap.parse_args()

    results = {}
    for name, flags in VARIANTS.items():
        wd = train_variant(name, flags, args)
        v = last_val_metrics(wd)
        results[name] = dict(
            bbox=v.get('bbox_mAP'), segm=v.get('segm_mAP'),
            segm75=v.get('segm_mAP_75'), biou=boundary_iou(wd, args))

    os.makedirs(args.work_root, exist_ok=True)
    csv_path = os.path.join(args.work_root, 'ablation_results.csv')
    with open(csv_path, 'w') as f:
        f.write('variant,bbox_mAP,segm_mAP,segm_AP75,boundary_iou\n')
        for n, r in results.items():
            f.write(f"{n},{r['bbox']},{r['segm']},{r['segm75']},{r['biou']}\n")

    print('\n==== ABLATION RESULTS ====')
    print(f"{'variant':<10}{'bbox':>8}{'segm':>8}{'AP75':>8}{'BIoU':>8}")
    for n, r in results.items():
        print(f"{n:<10}{r['bbox']!s:>8}{r['segm']!s:>8}{r['segm75']!s:>8}{r['biou']!s:>8}")
    print('saved ->', csv_path)


if __name__ == '__main__':
    main()
