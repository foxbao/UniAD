import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (MultiScaleDeformableAttention,
                                         build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import BaseModule
from mmdet.models import HEADS
from mmdet.models.utils.builder import TRANSFORMER
from torch.nn.init import normal_


@TRANSFORMER.register_module()
class SegDeformableEncoder(BaseModule):
    """Encoder-only SegDeformableTransformer subset for drivable masks."""

    def __init__(self, encoder=None, num_feature_levels=1, **kwargs):
        super().__init__(**kwargs)
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
        for lvl, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5,
                               height - 0.5,
                               height,
                               dtype=torch.float32,
                               device=device),
                torch.linspace(0.5,
                               width - 0.5,
                               width,
                               dtype=torch.float32,
                               device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (
                valid_ratios[:, None, lvl, 0] * width)
            reference_points_list.append(torch.stack((ref_x, ref_y), -1))
        reference_points = torch.cat(reference_points_list, 1)
        return reference_points[:, :, None] * valid_ratios[:, None]

    @staticmethod
    def get_valid_ratio(mask):
        _, height, width = mask.shape
        valid_h = torch.sum(~mask[:, :, 0], 1)
        valid_w = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_h.float() / height
        valid_ratio_w = valid_w.float() / width
        return torch.stack([valid_ratio_w, valid_ratio_h], -1)

    def forward(self, mlvl_feats, mlvl_masks, mlvl_pos_embeds, **kwargs):
        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(
                zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            _, _, height, width = feat.shape
            spatial_shapes.append((height, width))
            feat = feat.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
            lvl_pos_embed_flatten.append(lvl_pos_embed)

        feat_flatten = torch.cat(feat_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat_flatten.device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1, )),
             (spatial_shapes[:, 0] * spatial_shapes[:, 1]).cumsum(0)[:-1]))
        valid_ratios = torch.stack(
            [self.get_valid_ratio(mask) for mask in mlvl_masks], 1)
        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=feat_flatten.device)

        feat_flatten = feat_flatten.permute(1, 0, 2)
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


@TRANSFORMER.register_module()
class SegDeformableEncoderTRTP(SegDeformableEncoder):
    """TRT-plugin configured alias for the drivable deformable encoder."""


@HEADS.register_module()
class LidarDrivableHead(BaseModule):
    """Drivable-only BEV segmentation head for LiDAR deployment."""

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
                 eval_drivable_only=True,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.canvas_size = canvas_size
        self.pc_range = pc_range
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_stuff_classes = num_stuff_classes
        self.stuff_label_offset = stuff_label_offset
        self.eval_drivable_only = eval_drivable_only
        self.fp16_enabled = False

        self.positional_encoding = build_positional_encoding(
            positional_encoding)
        self.encoder = TRANSFORMER.build(transformer)
        self.stuff_mask_head = TRANSFORMER.build(stuff_transformer_head)
        self._init_layers()

    def _init_layers(self):
        self.stuff_query = nn.Embedding(self.num_stuff_classes,
                                        self.embed_dims * 2)

    def init_weights(self):
        self.encoder.init_weights()

    def forward(self, bev_feat):
        _, batch_size, _ = bev_feat.shape
        mlvl_feats = [
            torch.reshape(bev_feat,
                          (batch_size, self.bev_h, self.bev_w,
                           -1)).permute(0, 3, 1, 2)
        ]
        img_masks = mlvl_feats[0].new_zeros(
            (batch_size, self.bev_h, self.bev_w))
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
        memory = memory.permute(1, 0, 2)
        return memory, memory_mask, hw_lvl

    def forward_test_trt(self, bev_feat):
        memory, memory_mask, hw_lvl = self(bev_feat)
        batch_size = memory.shape[0]
        stuff_query, stuff_query_pos = torch.split(
            self.stuff_query.weight, self.embed_dims, dim=1)
        stuff_query = stuff_query.unsqueeze(0).expand(batch_size, -1, -1)
        stuff_query_pos = stuff_query_pos.unsqueeze(0).expand(
            batch_size, -1, -1)

        mask_stuff, _, _ = self.stuff_mask_head(
            memory, memory_mask, None, stuff_query, None, stuff_query_pos)
        attn_map = mask_stuff[..., 0]
        mask_pred = attn_map.reshape(batch_size, self.num_stuff_classes,
                                     hw_lvl[0][0], hw_lvl[0][1])
        mask_pred = F.interpolate(
            mask_pred, size=self.canvas_size, mode='bilinear')
        return mask_pred[0]


@HEADS.register_module()
class LidarDrivableHeadTRTP(LidarDrivableHead):
    """TensorRT-plugin configured alias for LiDAR drivable deployment."""
