import sys
import os
import torch

# Add mmdetection and project root to the path.
sys.path.append(os.path.abspath('mmdetection'))
sys.path.append(os.path.abspath('.'))

import traceback
try:
    from mmcv import Config
    from mmdet.models import build_detector
    from mmcv.runner import load_checkpoint
except Exception:
    traceback.print_exc()
    sys.exit(1)

# Import custom modules so the detector registers.
try:
    import mmdetection_custom_part.mmdet.models.detectors.mask_rcnn
    print("Successfully imported custom MaskRCNN module.")
except Exception:
    traceback.print_exc()
    sys.exit(1)

def main():
    config_file = 'Configs/mask_rcnn_r50_fpn_caffe_AWD_SCB_DSL_SynCOCO2EALLIS.py'
    checkpoint_file = 'Checkpoints/Checkpoint1.pth'

    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        sys.exit(1)
        
    if not os.path.exists(checkpoint_file):
        print(f"Checkpoint file not found: {checkpoint_file}")
        sys.exit(1)

    print(f"Loading config: {config_file}")
    try:
        cfg = Config.fromfile(config_file)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
    print("Building model...")
    try:
        model = build_detector(cfg.model)
    except Exception as e:
        print(f"Error building model: {e}")
        print("Did you forget to import the custom module defining the model type?")
        sys.exit(1)
    
    print(f"Loading checkpoint: {checkpoint_file}")
    try:
        checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    
    print("Model built and checkpoint loaded successfully!")

if __name__ == '__main__':
    main()
