# ---------------------------------------------------------------------------
# LidarDrivableHead: a drivable-only segmentation head for the KL LiDAR project.
#
# It is a faithful *subset* of PansegformerHead: it keeps the exact compute that
# produces the drivable IoU baseline -- the deformable BEV encoder, the stuff
# SegMaskHead and the stuff query -- and physically removes the entire things
# (object-detection) machinery that ran dead (things_ratio==0) under the
# drivable-only task: the transformer decoder, query_embedding, cls/reg
# branches, things_mask_head, the Hungarian assigners and the focal/bbox/iou
# losses + get_bboxes decode. The single-class stuff classification head was
# also pruned (a no-op with one stuff class).
#
# This does NOT touch PansegformerHead, which the 5 panoptic configs still use.
# ---------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
from torch.nn.init import normal_

from mmdet.models.builder import HEADS, build_loss
from mmdet.models.utils.builder import TRANSFORMER
from mmdet.core import reduce_mean

from .seg_head_plugin import IOU  # noqa: F401 (registers seg_head_plugin)


@TRANSFORMER.register_module()
class SegDeformableEncoder(BaseModule):
    """Encoder half of SegDeformableTransformer, decoder physically removed.

    Replicates the BEV-feature preprocessing + encoder call from
    SegDeformableTransformer.forward (the part before the decoder) verbatim, so
    the produced ``memory`` is bit-for-bit identical to what PansegformerHead
    fed into its stuff_mask_head. No query_embedding, no decoder.
    """

    def __init__(self, encoder=None, num_feature_levels=1, **kwargs):
        super(SegDeformableEncoder, self).__init__(**kwargs)
        self.encoder = build_transformer_layer_sequence(encoder)
        self.embed_dims = self.encoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H - 0.5, H, dtype=torch.float32,
                               device=device),
                torch.linspace(0.5, W - 0.5, W, dtype=torch.float32,
                               device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, mlvl_feats, mlvl_masks, mlvl_pos_embeds, **kwargs):
        """Returns memory [H*W, bs, C] (encoder output), identical to the
        encoder stage of SegDeformableTransformer.forward."""
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, _, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1, )), (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(m) for m in mlvl_masks], 1)
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)  # (H*W, bs, C)
        lvl_pos_embed_flatten = lvl_pos_embed_flatten.permute(1, 0, 2)
        memory = self.encoder(
            query=feat_flatten,
            key=None,
            value=None,
            query_pos=lvl_pos_embed_flatten,
            query_key_padding_mask=mask_flatten,
            spatial_shapes=spatial_shapes,
            reference_points=reference_points,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            **kwargs)
        return memory, mask_flatten


@HEADS.register_module()
class LidarDrivableHead(BaseModule):
    """Drivable-only seg head: BEV deformable encoder + stuff SegMaskHead.

    Detector contract matches PansegformerHead exactly (forward_train /
    forward_test signatures), so UniADTrackLidar needs no change -- only the
    config's seg_head block is swapped.
    """

    def __init__(self,
                 bev_h,
                 bev_w,
                 canvas_size,
                 pc_range,
                 in_channels=256,
                 embed_dims=256,
                 num_stuff_classes=1,
                 stuff_label_offset=3,
                 transformer=None,
                 positional_encoding=None,
                 stuff_transformer_head=None,
                 loss_mask=dict(type='DiceLoss', loss_weight=2.0),
                 eval_drivable_only=True,
                 init_cfg=None,
                 **kwargs):
        super(LidarDrivableHead, self).__init__(init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stuff_classes = num_stuff_classes
        self.stuff_label_offset = stuff_label_offset
        self.num_dec_stuff = stuff_transformer_head['num_decoder_layers']
        self.eval_drivable_only = eval_drivable_only
        self.fp16_enabled = False

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.encoder = TRANSFORMER.build(transformer)
        self.stuff_mask_head = TRANSFORMER.build(stuff_transformer_head)
        self.loss_mask = build_loss(loss_mask)
        self._init_layers()

    def _init_layers(self):
        # stuff query: split into (query, query_pos) of embed_dims each.
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)

    def init_weights(self):
        self.encoder.init_weights()

    def forward(self, bev_feat):
        """bev_feat: [H*W, bs, C] (as fed by UniADTrackLidar._bev_for_seg_head).
        Returns dict with args_tuple = (memory, memory_mask, hw_lvl)."""
        _, bs, _ = bev_feat.shape
        mlvl_feats = [
            torch.reshape(bev_feat,
                          (bs, self.bev_h, self.bev_w, -1)).permute(0, 3, 1, 2)
        ]
        img_masks = mlvl_feats[0].new_zeros((bs, self.bev_h, self.bev_w))
        hw_lvl = [feat_lvl.shape[-2:] for feat_lvl in mlvl_feats]
        mlvl_masks = []
        mlvl_positional_encodings = []
        for feat in mlvl_feats:
            mlvl_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(
                    torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        memory, memory_mask = self.encoder(mlvl_feats, mlvl_masks,
                                           mlvl_positional_encodings)
        memory = memory.permute(1, 0, 2)  # [bs, H*W, C]
        args_tuple = (memory, memory_mask, hw_lvl)
        return {'args_tuple': args_tuple}

    def forward_train(self,
                      bev_feat=None,
                      img_metas=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None):
        """Drivable is classless+boxless, so gt_lane_bboxes/labels are ignored;
        the drivable mask is taken from gt_lane_masks. Returns (losses, pred)."""
        pred_seg_dict = self(bev_feat)
        losses = self.loss(pred_seg_dict['args_tuple'], gt_lane_labels,
                           gt_lane_masks)
        return losses, pred_seg_dict

    def loss(self, args_tuple, gt_labels_list, gt_masks_list):
        """Stuff-only loss, mirroring the verified stuff mask path of
        PansegformerHead (drivable == single stuff class).

        gt_labels_list[i] holds the drivable label (== stuff_label_offset); the
        stuff GT mask is gt_masks_list[i]. Emits loss keys identical to the
        PansegformerHead drivable-only path so the detector's
        loss_weighted_and_prefixed(..., 'map') is unchanged."""
        memory, memory_mask, hw_lvl = args_tuple
        BS = memory.shape[0]
        stuff_query, stuff_query_pos = torch.split(self.stuff_query.weight,
                                                   self.embed_dims, dim=1)
        stuff_query = stuff_query.unsqueeze(0).expand(BS, -1, -1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(BS, -1, -1)

        mask_stuff, mask_inter_stuff, _ = self.stuff_mask_head(
            memory, memory_mask, None, stuff_query, None, stuff_query_pos,
            hw_lvl=hw_lvl)
        mask_stuff = mask_stuff.squeeze(-1)
        mask_inter_stuff = torch.stack(mask_inter_stuff, 0).squeeze(-1)

        mask_preds_stuff = []
        mask_preds_inter_stuff = [[] for _ in range(self.num_dec_stuff)]
        for i in range(BS):
            mask_preds_stuff.append(mask_stuff[i].reshape(-1, *hw_lvl[0]))
            for j in range(self.num_dec_stuff):
                mask_preds_inter_stuff[j].append(
                    mask_inter_stuff[j][i].reshape(-1, *hw_lvl[0]))
        mask_preds_stuff = torch.cat(mask_preds_stuff, 0)
        mask_preds_inter_stuff = [
            torch.cat(each, 0) for each in mask_preds_inter_stuff
        ]

        loss_dict = self._stuff_losses(mask_preds_stuff, mask_preds_inter_stuff,
                                       gt_labels_list, gt_masks_list, BS)
        return loss_dict

    def _stuff_losses(self, mask_preds_stuff, mask_preds_inter_stuff,
                      gt_stuff_labels_list, gt_stuff_masks_list, BS):
        """Build stuff GT + compute mask losses. Mirrors the mask path of the
        verified PansegformerHead stuff branch (the cls_stuff head was pruned:
        with a single stuff class its FocalLoss is a no-op, logged 0.0000).
        Returns a loss dict with the mask keys of the drivable-only path."""
        device = mask_preds_stuff.device
        mask_stuff_gt, mask_weight_stuff = [], []
        num_total_pos_stuff = 0
        for i in range(BS):
            num_total_pos_stuff += len(gt_stuff_labels_list[i])
            select_stuff_index = gt_stuff_labels_list[i] - \
                self.stuff_label_offset
            mask_weight_i_stuff = torch.zeros([self.num_stuff_classes])
            mask_weight_i_stuff[select_stuff_index] = 1
            stuff_masks = torch.zeros(
                (self.num_stuff_classes, *gt_stuff_masks_list[i].shape[-2:]),
                device=gt_stuff_masks_list[i].device).to(torch.bool)
            stuff_masks[select_stuff_index] = gt_stuff_masks_list[i].to(
                torch.bool)
            mask_stuff_gt.append(stuff_masks)
            mask_weight_stuff.append(mask_weight_i_stuff)

        mask_weight_stuff = torch.cat(mask_weight_stuff, 0).to(device)
        mask_stuff_gt = torch.cat(mask_stuff_gt, 0).to(torch.float)
        num_total_pos_stuff = mask_preds_stuff.new_tensor([num_total_pos_stuff])
        num_total_pos_stuff = torch.clamp(reduce_mean(num_total_pos_stuff),
                                          min=1).item()

        if mask_preds_stuff.shape[0] == 0:
            loss_mask_stuff = (0 * mask_preds_stuff).sum()
        else:
            mask_preds = F.interpolate(mask_preds_stuff.unsqueeze(0),
                                       scale_factor=2.0,
                                       mode='bilinear').squeeze(0)
            mask_targets_stuff = F.interpolate(mask_stuff_gt.unsqueeze(0),
                                               size=mask_preds.shape[-2:],
                                               mode='bilinear').squeeze(0)
            loss_mask_stuff = self.loss_mask(mask_preds, mask_targets_stuff,
                                             mask_weight_stuff,
                                             avg_factor=num_total_pos_stuff)

        loss_mask_stuff_list = []
        for j in range(len(mask_preds_inter_stuff)):
            mp = mask_preds_inter_stuff[j]
            if mp.shape[0] == 0:
                loss_mask_stuff_list.append((0 * mp).sum())
            else:
                mp = F.interpolate(mp.unsqueeze(0), scale_factor=2.0,
                                   mode='bilinear').squeeze(0)
                loss_mask_stuff_list.append(self.loss_mask(
                    mp, mask_targets_stuff, mask_weight_stuff,
                    avg_factor=num_total_pos_stuff))

        loss_dict = {'loss_mask_stuff': loss_mask_stuff}
        for i in range(len(loss_mask_stuff_list)):
            loss_dict[f'd{i}.loss_mask_stuff_f'] = loss_mask_stuff_list[i]
        return loss_dict

    def get_drivable_bboxes(self, args_tuple, img_metas, rescale=False):
        """Decode the stuff/drivable mask. Verbatim from PansegformerHead."""
        memory, memory_mask, hw_lvl = args_tuple
        results = []
        for img_id in range(len(img_metas)):
            i = img_id
            ori_shape = (self.canvas_size[0], self.canvas_size[1], 3)
            stuff_query = self.stuff_query.weight[None, :, :self.embed_dims]
            stuff_query_pos = self.stuff_query.weight[None, :, self.embed_dims:]
            mask_stuff, _, _ = \
                self.stuff_mask_head(memory[i:i + 1], memory_mask[i:i + 1],
                                     None, stuff_query, None, stuff_query_pos,
                                     hw_lvl=hw_lvl)
            attn_map = mask_stuff.squeeze(-1)
            mask_pred = attn_map.reshape(-1, *hw_lvl[0])
            mask_pred = F.interpolate(mask_pred.unsqueeze(0),
                                      size=ori_shape[:2],
                                      mode='bilinear').squeeze(0)
            drivable = mask_pred[-1] > 0.5
            results.append({
                'drivable': drivable,
                'score_list': mask_pred,
            })
        return results

    def forward_test(self,
                     pts_feats=None,
                     gt_lane_labels=None,
                     gt_lane_masks=None,
                     img_metas=None,
                     rescale=False):
        bbox_list = [dict() for _ in range(len(img_metas))]
        pred_seg_dict = self(pts_feats)
        results = self.get_drivable_bboxes(
            pred_seg_dict['args_tuple'], img_metas, rescale=rescale)

        with torch.no_grad():
            drivable_pred = results[0]['drivable']
            drivable_gt = gt_lane_masks[0][0, -1]
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
                'drivable_iou': drivable_iou, 'lanes_iou': zero,
                'divider_iou': zero, 'crossing_iou': zero, 'contour_iou': zero,
            }
        for result_dict, pts_bbox in zip(bbox_list, results):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['ret_iou'] = ret_iou
            result_dict['args_tuple'] = pred_seg_dict['args_tuple']
        return bbox_list
