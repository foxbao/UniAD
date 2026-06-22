#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from mmdet.models import LOSSES

@LOSSES.register_module(force=True)
class TrajLoss(nn.Module):
    """
    MTP loss modified to include variances. Uses MSE for mode selection.
    Can also be used with
    Multipath outputs, with residuals added to anchors.
    """

    def __init__(self,
                 use_variance=False,
                 cls_loss_weight=1.,
                 nll_loss_weight=1.,
                 loss_weight_minade=0.,
                 loss_weight_minfde=1.,
                 loss_weight_mr=1.,
                 best_mode_metric='minade',
                 fde_weight=0.5,
                 turn_loss_weights=None,
                 static_path_thr=2.0,
                 straight_deg=15.0,
                 mild_deg=45.0,
                 straight_lateral_ratio=0.15,
                 mild_lateral_ratio=0.35,
                 normalize_turn_weights=True):
        """
        Initialize MTP loss
        :param args: Dictionary with the following (optional) keys
            use_variance: bool, whether or not to use variances for computing
            regression component of loss,
                default: False
            alpha: float, relative weight assigned to classification component,
            compared to regression component
                of loss, default: 1
        """
        super(TrajLoss, self).__init__()
        self.use_variance = use_variance
        self.cls_loss_weight = cls_loss_weight
        self.nll_loss_weight = nll_loss_weight
        self.loss_weight_minade = loss_weight_minade
        self.loss_weight_minfde = loss_weight_minfde
        if best_mode_metric not in ('minade', 'minfde', 'ade_fde'):
            raise ValueError(
                'best_mode_metric must be one of minade, minfde, ade_fde, '
                f'got {best_mode_metric}')
        self.best_mode_metric = best_mode_metric
        self.fde_weight = float(fde_weight)
        self.turn_loss_weights = turn_loss_weights
        self.static_path_thr = float(static_path_thr)
        self.straight_deg = float(straight_deg)
        self.mild_deg = float(mild_deg)
        self.straight_lateral_ratio = float(straight_lateral_ratio)
        self.mild_lateral_ratio = float(mild_lateral_ratio)
        self.normalize_turn_weights = bool(normalize_turn_weights)

    def forward(self,
                traj_prob, 
                traj_preds, 
                gt_future_traj, 
                gt_future_traj_valid_mask):
        """
        Compute MTP loss
        :param predictions: Dictionary with 'traj': predicted trajectories
        and 'probs': mode (log) probabilities
        :param ground_truth: Either a tensor with ground truth trajectories
        or a dictionary
        :return:
        """
        # Unpack arguments
        traj = traj_preds # (b, nmodes, seq, 5)
        log_probs = traj_prob
        traj_gt = gt_future_traj

        # Useful variables
        batch_size = traj.shape[0]
        sequence_length = traj.shape[2]
        pred_params = 5 if self.use_variance else 2

        # Masks for variable length ground truth trajectories
        masks = 1 - gt_future_traj_valid_mask.to(traj.dtype)

        per_mode_ade, per_mode_fde = mode_ade_fde(traj, traj_gt, masks)
        l_minade, inds_ade = torch.min(per_mode_ade, dim=1)
        l_minfde, inds_fde = torch.min(per_mode_fde, dim=1)
        try:
            l_mr = miss_rate(traj, traj_gt, masks)
        except:
            l_mr = torch.zeros_like(l_minfde)
        if self.best_mode_metric == 'minade':
            inds = inds_ade
            selected_ade = l_minade
        elif self.best_mode_metric == 'minfde':
            inds = inds_fde
            selected_ade = torch.gather(
                per_mode_ade, 1, inds.unsqueeze(1)).squeeze(1)
        else:
            best_score = per_mode_ade + self.fde_weight * per_mode_fde
            _, inds = torch.min(best_score, dim=1)
            selected_ade = torch.gather(
                per_mode_ade, 1, inds.unsqueeze(1)).squeeze(1)

        gather_idx = inds[:, None, None, None].expand(
            -1, 1, sequence_length, traj.size(-1))

        # Calculate MSE or NLL loss for trajectories corresponding to selected
        # outputs:
        traj_best = traj.gather(1, gather_idx).squeeze(dim=1)

        if self.use_variance:
            l_reg = traj_nll(traj_best, traj_gt, masks)
        else:
            l_reg = selected_ade

        # Compute classification loss
        l_class = -log_probs.gather(1, inds.unsqueeze(1)).squeeze(1)

        sample_weights = self._sample_weights(traj_gt, masks)
        l_reg = torch.sum(l_reg * sample_weights)/(batch_size + 1e-5)
        l_class = torch.sum(l_class * sample_weights)/(batch_size + 1e-5)
        l_minade = torch.sum(l_minade * sample_weights)/(batch_size + 1e-5)
        l_minfde = torch.sum(l_minfde * sample_weights)/(batch_size + 1e-5)

        loss = l_class * self.cls_loss_weight + l_reg * self.nll_loss_weight + l_minade * self.loss_weight_minade + l_minfde * self.loss_weight_minfde
        return loss, l_class, l_reg, l_minade, l_minfde, l_mr

    def _sample_weights(self, traj_gt: torch.Tensor,
                        masks: torch.Tensor) -> torch.Tensor:
        if not self.turn_loss_weights:
            return traj_gt.new_ones((traj_gt.shape[0], ))
        weights = traj_gt.new_ones((traj_gt.shape[0], ))
        valid = (1 - masks).bool()
        for idx in range(traj_gt.shape[0]):
            pts = traj_gt[idx, valid[idx], :2]
            bucket = self._classify_turn_bucket(pts)
            weights[idx] = float(self.turn_loss_weights.get(bucket, 1.0))
        if self.normalize_turn_weights and weights.numel() > 0:
            weights = weights / weights.mean().clamp(min=1e-6)
        return weights

    def _classify_turn_bucket(self, pts: torch.Tensor) -> str:
        if pts.shape[0] < 2:
            return 'static_slow'
        path = torch.norm(pts[1:] - pts[:-1], dim=1).sum()
        net = torch.norm(pts[-1])
        if float(path.detach()) < self.static_path_thr or \
                float(net.detach()) < self.static_path_thr:
            return 'static_slow'
        heading = self._heading_change_deg(pts)
        lat_ratio = self._lateral_ratio(pts)
        if heading <= self.straight_deg and \
                lat_ratio <= self.straight_lateral_ratio:
            return 'straight'
        if heading <= self.mild_deg and lat_ratio <= self.mild_lateral_ratio:
            return 'mild_turn'
        return 'sharp_turn'

    @staticmethod
    def _heading_change_deg(pts: torch.Tensor,
                            segment_min_disp: float = 0.05) -> float:
        if pts.shape[0] < 3:
            return 0.0
        deltas = pts[1:] - pts[:-1]
        norms = torch.norm(deltas, dim=1)
        valid = torch.nonzero(norms >= segment_min_disp).flatten()
        if valid.numel() < 2:
            return 0.0
        v0 = deltas[valid[0]]
        v1 = deltas[valid[-1]]
        a0 = torch.atan2(v0[1], v0[0])
        a1 = torch.atan2(v1[1], v1[0])
        diff = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
        return float(torch.abs(diff).detach() * 180.0 / math.pi)

    @staticmethod
    def _lateral_ratio(pts: torch.Tensor) -> float:
        if pts.shape[0] < 2:
            return 0.0
        end = pts[-1]
        net = torch.norm(end)
        if float(net.detach()) < 1e-6:
            return 0.0
        direction = end / net.clamp(min=1e-6)
        normal = torch.stack((-direction[1], direction[0]))
        lateral = torch.max(torch.abs(torch.matmul(pts, normal)))
        return float((lateral / net.clamp(min=1e-6)).detach())


def mode_ade_fde(traj: torch.Tensor,
                 traj_gt: torch.Tensor,
                 masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return per-mode ADE and FDE for every sample."""
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    valid = 1 - masks_rpt
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(torch.sum(torch.pow(dist, exponent=2), dim=3),
                     exponent=0.5)
    ade = torch.sum(dist * valid, dim=2) / torch.clip(
        torch.sum(valid, dim=2), min=1)

    lengths = torch.sum(1 - masks, dim=1).long()
    last_inds = torch.clamp(lengths, min=1) - 1
    gather_inds = last_inds[:, None, None, None].repeat(1, num_modes, 1, 2)
    traj_last = torch.gather(traj[..., :2], dim=2,
                             index=gather_inds).squeeze(2)
    gt_last = torch.gather(traj_gt_rpt, dim=2,
                           index=gather_inds).squeeze(2)
    fde = torch.pow(torch.sum(torch.pow(gt_last - traj_last, exponent=2),
                              dim=2), exponent=0.5)
    invalid = lengths <= 0
    if invalid.any():
        ade[invalid] = 0
        fde[invalid] = 0
    return ade, fde

def min_ade(traj: torch.Tensor, traj_gt: torch.Tensor,
            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes average displacement error for the best trajectory is a set,
    with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape
    [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape
    [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error, shape
    [batch_size]
    """
    num_modes = traj.shape[1]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    err = traj_gt_rpt - traj[:, :, :, 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=3)
    err = torch.pow(err, exponent=0.5)
    err = torch.sum(err * (1 - masks_rpt), dim=2) / \
        torch.clip(torch.sum((1 - masks_rpt), dim=2), min=1)
    err, inds = torch.min(err, dim=1)

    return err, inds

def traj_nll(
        pred_dist: torch.Tensor,
        traj_gt: torch.Tensor,
        masks: torch.Tensor):
    """
    Computes negative log likelihood of ground truth trajectory under a
    predictive distribution with a single mode,
    with a bivariate Gaussian distribution predicted at each time in the
    prediction horizon

    :param pred_dist: parameters of a bivariate Gaussian distribution,
    shape [batch_size, sequence_length, 5]
    :param traj_gt: ground truth trajectory,
    shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth,
    shape [batch_size, sequence_length]
    :return:
    """
    mu_x = pred_dist[:, :, 0]
    mu_y = pred_dist[:, :, 1]
    x = traj_gt[:, :, 0]
    y = traj_gt[:, :, 1]

    sig_x = pred_dist[:, :, 2]
    sig_y = pred_dist[:, :, 3]
    rho = pred_dist[:, :, 4]
    ohr = torch.pow(1 - torch.pow(rho, 2), -0.5)

    nll = 0.5 * torch.pow(ohr, 2) * \
        (torch.pow(sig_x, 2) * torch.pow(x - mu_x, 2) + torch.pow(sig_y, 2) *
         torch.pow(y - mu_y, 2) - 2 * rho * torch.pow(sig_x, 1) *
         torch.pow(sig_y, 1) * (x - mu_x) * (y - mu_y)) - \
        torch.log(sig_x * sig_y * ohr) + 1.8379

    nll[nll.isnan()] = 0
    nll[nll.isinf()] = 0

    nll = torch.sum(nll * (1 - masks), dim=1) / (torch.sum((1 - masks), dim=1) + 1e-5)
    # Note: Normalizing with torch.sum((1 - masks), dim=1) makes values
    # somewhat comparable for trajectories of
    # different lengths

    return nll

def min_fde(traj: torch.Tensor, traj_gt: torch.Tensor,
            masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes final displacement error for the best trajectory is a set,
    with respect to ground truth
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory, shape
    [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth, shape
    [batch_size, sequence_length]
    :return errs, inds: errors and indices for modes with min error,
    shape [batch_size]
    """
    num_modes = traj.shape[1]
    lengths = torch.sum(1 - masks, dim=1).long()
    valid_mask = lengths > 0
    traj = traj[valid_mask]
    traj_gt = traj_gt[valid_mask]
    masks = masks[valid_mask]
    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    lengths = torch.sum(1 - masks, dim=1).long()
    inds = lengths.unsqueeze(1).unsqueeze(
        2).unsqueeze(3).repeat(1, num_modes, 1, 2) - 1

    traj_last = torch.gather(traj[..., :2], dim=2, index=inds).squeeze(2)
    traj_gt_last = torch.gather(traj_gt_rpt, dim=2, index=inds).squeeze(2)

    err = traj_gt_last - traj_last[..., 0:2]
    err = torch.pow(err, exponent=2)
    err = torch.sum(err, dim=2)
    err = torch.pow(err, exponent=0.5)
    err, inds = torch.min(err, dim=1)

    return err, inds


def miss_rate(
        traj: torch.Tensor,
        traj_gt: torch.Tensor,
        masks: torch.Tensor,
        dist_thresh: float = 2) -> torch.Tensor:
    """
    Computes miss rate for mini batch of trajectories,
    with respect to ground truth and given distance threshold
    :param traj: predictions, shape [batch_size, num_modes, sequence_length, 2]
    :param traj_gt: ground truth trajectory,
    shape [batch_size, sequence_length, 2]
    :param masks: masks for varying length ground truth,
    shape [batch_size, sequence_length]
    :param dist_thresh: distance threshold for computing miss rate.
    :return errs, inds: errors and indices for modes with min error,
    shape [batch_size]
    """
    num_modes = traj.shape[1]

    traj_gt_rpt = traj_gt.unsqueeze(1).repeat(1, num_modes, 1, 1)
    masks_rpt = masks.unsqueeze(1).repeat(1, num_modes, 1)
    dist = traj_gt_rpt - traj[:, :, :, 0:2]
    dist = torch.pow(dist, exponent=2)
    dist = torch.sum(dist, dim=3)
    dist = torch.pow(dist, exponent=0.5)
    dist[masks_rpt.bool()] = -math.inf
    dist, _ = torch.max(dist, dim=2)
    dist, _ = torch.min(dist, dim=1)
    m_r = torch.sum(torch.as_tensor(dist > dist_thresh)) / len(dist)

    return m_r
