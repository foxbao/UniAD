# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner import force_fp32
from torch import nn

from ..builder import VOXEL_ENCODERS


@VOXEL_ENCODERS.register_module()
class HardSimpleVFE(nn.Module):
    """Simple voxel feature encoder used in SECOND.

    It averages the point features inside each hard voxel. This is the only
    voxel encoder required by the KL LiDAR BEVFormer config.
    """

    def __init__(self, num_features=4):
        super().__init__()
        self.num_features = num_features
        self.fp16_enabled = False

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        points_mean = features[:, :, :self.num_features].sum(
            dim=1, keepdim=False) / num_points.type_as(features).view(-1, 1)
        return points_mean.contiguous()
