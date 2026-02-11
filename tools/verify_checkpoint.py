import sys
import os
import torch

# Add mmdetection to path so 'mmdet' works
sys.path.append(os.path.abspath('mmdetection'))
# Add current dir so 'mmdetection_custom_part' works
sys.path.append(os.path.abspath('.'))

import traceback
try:
    from mmcv import Config
    from mmdet.models import build_detector
    from mmcv.runner import load_checkpoint
except Exception:
    traceback.print_exc()
    sys.exit(1)

# Import custom modules to register them
# We need to import the module where MaskRCNNNoiseInv is defined.
# Based on grep, it's in mmdetection_custom_part/mmdet/models/detectors/mask_rcnn.py
try:
    # We might need to make sure the internal imports in that file work.
    # If it does 'from ..registry import DETECTORS', it expects to be in a package.
    # Let's try importing it as a top-level module if possible or via the package structure.
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
    
    # Optional: Dummy Forward Pass
    # model.eval()
    # input = torch.randn(1, 3, 400, 600)
    # print("Attempting dummy forward pass (this might fail if input format is specific)...")
    # try:
    #     # simple_test API: img (list of tensor), img_metas (list of list of dict)
    #     # model.simple_test([input], [[{'img_shape':(400,600,3), 'scale_factor':1.0, 'filename':'dummy', 'ori_shape':(400,600,3)}]])
    #     print("Dummy forward pass passed (or skipped).")
    # except Exception as e:
    #     print(f"Dummy forward pass failed: {e}")

if __name__ == '__main__':
    main()
