#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module(force=True)
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        return torch.sum(err * mask)/(torch.sum(mask) + 1e-5)


@LOSSES.register_module(force=True)
class CollisionLoss(nn.Module):
    def __init__(self,
                 delta=0.5,
                 weight=1.0,
                 ego_width=1.85,
                 ego_length=4.084,
                 coordinate_frame='rfu'):
        super(CollisionLoss, self).__init__()
        self.w = ego_width + delta
        self.h = ego_length + delta
        self.weight = weight
        self.coordinate_frame = coordinate_frame.lower()
        assert self.coordinate_frame in ['rfu', 'flu']
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask: [1, 6] or [1, 6, 3]
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        if sdc_planning_gt_mask is not None:
            step_valid = sdc_planning_gt_mask[..., 0] \
                if sdc_planning_gt_mask.shape[-1] == 3 \
                else sdc_planning_gt_mask
            step_valid = step_valid.reshape(-1) > 0.5
        else:
            step_valid = None
        for i in range(n_futures):
            if step_valid is not None and i < step_valid.numel() and not bool(step_valid[i]):
                continue
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_bbox(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        
        xi1, yi1 = torch.minimum(xa1, xb1), torch.minimum(ya1, yb1)
        xi2, yi2 = torch.maximum(xa2, xb2), torch.maximum(ya2, yb2)
        intersect = (xi1 - xi2).clamp_min(0) * (yi1 - yi2).clamp_min(0)
        return intersect

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        if self.coordinate_frame == 'flu':
            corners = x.new_tensor([
                [l/2, w/2], [l/2, -w/2], [-l/2, -w/2], [-l/2, w/2]
            ])
            rot_mat = torch.stack(
                (torch.stack((torch.cos(theta), -torch.sin(theta))),
                 torch.stack((torch.sin(theta), torch.cos(theta))))
            ).to(x.device)
        else:
            corners = x.new_tensor([
                [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2, -l/2]
            ])
            rot_mat = torch.stack(
                (torch.stack((torch.cos(theta), torch.sin(theta))),
                 torch.stack((-torch.sin(theta), torch.cos(theta))))
            ).to(x.device)
        center = torch.stack((x, y))[:, None]
        new_corners = rot_mat @ corners.T + center
        return new_corners.T
