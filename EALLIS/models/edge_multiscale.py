import torch.nn.functional as F


def fuse_edge_multiscale(features_dict, edge_map):
    fused = {}
    for k, f in features_dict.items():
        e = F.interpolate(edge_map, size=f.shape[-2:],
                          mode='bilinear', align_corners=False)
        fused[k] = f + e * f
    return fused
