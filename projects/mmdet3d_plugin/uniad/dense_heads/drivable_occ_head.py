# Copyright (c) OpenMMLab. All rights reserved.
"""Dense static-occupancy head for the KL drivable task.

Drivable space here is a single, static, current-frame dense mask (where the
ground / free space is). The original UniAD ``PansegformerHead`` is built for
HD-map *vector* panoptic segmentation (countable lane/crossing instances + one
stuff class); feeding it only the single drivable stuff class leaves its entire
things detection branch running dead (``things_ratio=0``, zero gradient).

``DrivableOccHead`` replaces that machinery with a small dense conv head in the
spirit of ``OccHead`` (it reuses ``OccHead``'s ``SimpleConv2d`` BEV stack), but
with NO queries, NO transformer, and NO future-time dimension -- purely a
current-frame BEV -> single-channel drivable logit. It matches the exact
``forward_train`` / ``forward_test`` signatures the detector already calls, so
``UniADTrackLidar`` needs no changes -- only the ``seg_head`` config block swaps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.builder import HEADS, build_loss

from .occ_head_plugin import SimpleConv2d
from .seg_head_plugin import IOU


@HEADS.register_module()
class DrivableOccHead(BaseModule):
    """Dense current-frame drivable-occupancy head.

    Args:
        bev_h, bev_w (int): BEV grid size; the head outputs a mask at this
            resolution (== ``canvas_size``).
        canvas_size (tuple[int]): (H, W) the eval/IOU is computed at. Defaults
            to ``(bev_h, bev_w)``.
        in_channels (int): BEV embedding channels.
        proj_channels (int): hidden conv width.
        num_conv (int): number of conv layers in the projection stack.
        loss_dice / loss_mask (dict): loss configs (Dice + sigmoid-BCE).
        eval_drivable_only (bool): zero out lane/divider/crossing/contour IoU
            keys (drivable-only task). Kept for eval-key parity with panseg.
        pos_weight (float): positive-class weight for the BCE term.
    """

    def __init__(self,
                 bev_h,
                 bev_w,
                 in_channels=256,
                 canvas_size=None,
                 proj_channels=256,
                 num_conv=4,
                 loss_dice=None,
                 loss_mask=None,
                 eval_drivable_only=True,
                 pos_weight=1.0,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = tuple(canvas_size) if canvas_size is not None \
            else (bev_h, bev_w)
        self.eval_drivable_only = eval_drivable_only
        self.register_buffer(
            'pos_weight', torch.tensor(float(pos_weight)), persistent=False)

        # BEV -> single-channel logit, same spatial resolution (no upsample:
        # BEV is already at canvas_size). Reuses OccHead's SimpleConv2d.
        self.decoder = SimpleConv2d(
            in_channels=in_channels,
            out_channels=1,
            conv_channels=proj_channels,
            num_conv=num_conv,
        )

        self.loss_dice = build_loss(loss_dice) if loss_dice else None
        self.loss_mask = build_loss(loss_mask) if loss_mask else None

    def _to_bchw(self, bev_feat):
        """Accept BEV as [HW, B, C] / [B, HW, C] / [B, C, H, W] -> [B,C,H,W].

        The detector's ``_bev_for_seg_head`` hands us [HW, B, C]; we also
        tolerate the other layouts for robustness.
        """
        if bev_feat.dim() == 4:
            return bev_feat
        if bev_feat.dim() == 3:
            hw = self.bev_h * self.bev_w
            if bev_feat.shape[0] == hw:        # [HW, B, C]
                b = bev_feat.shape[1]
                c = bev_feat.shape[2]
                return bev_feat.permute(1, 2, 0).reshape(
                    b, c, self.bev_h, self.bev_w)
            if bev_feat.shape[1] == hw:        # [B, HW, C]
                b = bev_feat.shape[0]
                c = bev_feat.shape[2]
                return bev_feat.permute(0, 2, 1).reshape(
                    b, c, self.bev_h, self.bev_w)
        raise ValueError(
            'DrivableOccHead expects BEV [HW,B,C], [B,HW,C] or [B,C,H,W], '
            f'got {tuple(bev_feat.shape)}.')

    @auto_fp16(apply_to=('bev_feat', ))
    def forward(self, bev_feat):
        """BEV -> drivable logit. Returns dict with 'logit' [B,1,H,W]."""
        x = self._to_bchw(bev_feat)
        logit = self.decoder(x)                # [B, 1, H, W]
        if logit.shape[-2:] != self.canvas_size:
            logit = F.interpolate(
                logit, size=self.canvas_size, mode='bilinear',
                align_corners=False)
        return {'logit': logit}

    @staticmethod
    def _drivable_gt_train(gt_lane_masks, device):
        """Per-batch drivable GT for training.

        ``gt_lane_masks`` is a list (len bs) of [K, H, W] masks; drivable is
        the last (stuff) channel. Returns [B, 1, H, W] float in {0,1}.
        """
        masks = []
        for m in gt_lane_masks:
            masks.append(m[-1].to(device).float())
        return torch.stack(masks, 0).unsqueeze(1)

    def forward_train(self,
                      bev_feat=None,
                      img_metas=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None):
        """Returns (losses_dict, pred_dict). Labels/bboxes are ignored --
        drivable is a classless dense mask."""
        pred = self(bev_feat)
        logit = pred['logit']                                  # [B,1,H,W]
        target = self._drivable_gt_train(gt_lane_masks, logit.device)

        losses = dict()
        if self.loss_dice is not None:
            # Project-local DiceLoss (projects/.../losses/dice_loss.py) takes
            # probabilities (no internal sigmoid) shaped [N, H, W].
            prob = logit.sigmoid().squeeze(1)          # [B, H, W]
            losses['loss_drivable_dice'] = self.loss_dice(
                prob, target.squeeze(1))
        bce = F.binary_cross_entropy_with_logits(
            logit, target, pos_weight=self.pos_weight.to(logit))
        losses['loss_drivable_bce'] = bce
        return losses, pred

    def forward_test(self,
                     bev_feat=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     img_metas=None,
                     rescale=False):
        """Returns [dict(pts_bbox={'drivable': mask}, ret_iou={...})].

        Replicates the minimal output the eval path consumes; drivable GT is
        the last channel ``gt_lane_masks[0][0, -1]`` (matching PansegformerHead).
        """
        n = len(img_metas) if img_metas is not None else 1
        bbox_list = [dict() for _ in range(n)]

        pred = self(bev_feat)
        logit = pred['logit']                          # [B,1,H,W]
        drivable_pred = (logit[0, 0].sigmoid() > 0.5).long()

        with torch.no_grad():
            drivable_gt = gt_lane_masks[0][0, -1].to(drivable_pred.device).long()
            drivable_iou, drivable_intersection, drivable_union = IOU(
                drivable_pred.view(1, -1), drivable_gt.view(1, -1))
            zero = drivable_intersection.new_zeros(())
            ret_iou = {
                'drivable_intersection': drivable_intersection,
                'drivable_union': drivable_union,
                'lanes_intersection': zero, 'lanes_union': zero,
                'divider_intersection': zero, 'divider_union': zero,
                'crossing_intersection': zero, 'crossing_union': zero,
                'contour_intersection': zero, 'contour_union': zero,
                'drivable_iou': drivable_iou,
                'lanes_iou': zero, 'divider_iou': zero,
                'crossing_iou': zero, 'contour_iou': zero,
            }

        pts_bbox = {'drivable': drivable_pred}
        for result_dict in bbox_list:
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
        return bbox_list
