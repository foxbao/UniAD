import copy
import os

import numpy as np
import torch
import torch.nn as nn
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_head, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from third_party.uniad_mmdet3d.models.detectors.mvx_two_stage import (
    MVXTwoStageDetector)
from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox, normalize_bbox

from ..dense_heads.track_head_plugin import (Instances, MemoryBank,
                                             QueryInteractionModule,
                                             RuntimeTrackerBase)


@DETECTORS.register_module()
class UniADTrackLidar(MVXTwoStageDetector):
    """LiDAR-only UniAD stage-1 tracker.

    This mirrors camera UniAD's detector/tracker boundary while using the
    LiDAR voxel backbone and BEVFormer encoder as the BEV front-end.
    """

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 point_cloud_range=None,
                 num_query=600,
                 embed_dims=256,
                 use_prev_bev=True,
                 video_test_mode=True,
                 return_query_feats=True,
                 loss_cfg=None,
                 qim_args=dict(
                     qim_type='QIMBase',
                     merger_dropout=0,
                     update_query_pos=True,
                     fp_ratio=0.3,
                     random_drop=0.1),
                 mem_args=dict(
                     memory_bank_type='MemoryBank',
                     memory_bank_score_thresh=0.0,
                     memory_bank_len=4),
                 score_thresh=0.2,
                 filter_score_thresh=0.1,
                 miss_tolerance=5,
                 gt_iou_threshold=0.0,
                 freeze_lidar_backbone=False,
                 freeze_bev_encoder=False,
                 seg_head=None,
                 task_loss_weight=None,
                 queue_length=4,
                 with_sdc=False,
                 **kwargs):
        super().__init__(
            pts_voxel_layer=pts_voxel_layer,
            pts_voxel_encoder=pts_voxel_encoder,
            pts_middle_encoder=pts_middle_encoder,
            pts_fusion_layer=pts_fusion_layer,
            img_backbone=img_backbone,
            pts_backbone=pts_backbone,
            img_neck=img_neck,
            pts_neck=pts_neck,
            pts_bbox_head=pts_bbox_head,
            img_roi_head=img_roi_head,
            img_rpn_head=img_rpn_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)

        if kwargs:
            raise TypeError(f'Unexpected UniADTrackLidar kwargs: {kwargs}')
        self.point_cloud_range = point_cloud_range
        if self.point_cloud_range is None and hasattr(self.pts_bbox_head,
                                                      'bbox_coder'):
            self.point_cloud_range = self.pts_bbox_head.bbox_coder.pc_range
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.use_prev_bev = use_prev_bev
        self.video_test_mode = video_test_mode
        self.return_query_feats = return_query_feats
        self.fp16_enabled = False
        self.gt_iou_threshold = gt_iou_threshold
        self.freeze_lidar_backbone = freeze_lidar_backbone
        self.freeze_bev_encoder = freeze_bev_encoder
        self.queue_length = queue_length
        self.with_sdc = with_sdc
        self.query_embedding = nn.Embedding(
            self.num_query + int(with_sdc), self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.sdc_query_index = self.num_query if with_sdc else None
        if freeze_lidar_backbone:
            self._freeze_modules(self._lidar_backbone_modules())
        if freeze_bev_encoder:
            self._freeze_modules(self._bev_encoder_modules())

        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance)
        self.query_interact = QueryInteractionModule(
            qim_args,
            dim_in=self.embed_dims,
            hidden_dim=self.embed_dims,
            dim_out=self.embed_dims)
        self.memory_bank = MemoryBank(
            mem_args,
            dim_in=self.embed_dims,
            hidden_dim=self.embed_dims,
            dim_out=self.embed_dims)
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length)
        if loss_cfg is None:
            raise ValueError('UniADTrackLidar requires loss_cfg.')
        loss_cfg = copy.deepcopy(loss_cfg)
        if self.with_sdc:
            loss_cfg['sdc_query_index'] = self.sdc_query_index
        self.criterion = build_loss(loss_cfg)
        self.seg_head = build_head(seg_head) if seg_head is not None else None
        self.task_loss_weight = dict(track=1.0, map=1.0)
        if task_loss_weight is not None:
            self.task_loss_weight.update(task_loss_weight)
        self.test_track_instances = None
        self.scene_token = None
        self.timestamp = None
        self.l2g_t = None
        self.l2g_r_mat = None
        self.test_frame_token = None
        self._test_prev_bev = None
        self._test_scene_token = None

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Keep tracker query names aligned with UniAD while loading detector ckpts.

        LiDAR BEVFormer detector checkpoints used the top-level
        ``query_embedding`` / ``reference_points`` keys for detector queries.
        In UniADTrackLidar those names intentionally mean tracker queries, as
        in camera UniAD.  When a source checkpoint has no tracker modules, drop
        those old detector-query keys and let the tracker query initialize
        randomly.  Legacy LiDAR tracker checkpoints from the previous naming
        scheme are still accepted by remapping ``track_*`` keys.
        """
        own_state = self.state_dict()
        marker_keys = [
            prefix + 'query_interact.self_attn.in_proj_weight',
            prefix + 'memory_bank.save_proj.weight',
            prefix + 'criterion.code_weights',
        ]
        source_has_tracker = any(key in state_dict for key in marker_keys)

        legacy_pairs = {
            prefix + 'track_query_embedding.weight':
            prefix + 'query_embedding.weight',
            prefix + 'track_reference_points.weight':
            prefix + 'reference_points.weight',
            prefix + 'track_reference_points.bias':
            prefix + 'reference_points.bias',
        }
        for old_key, new_key in legacy_pairs.items():
            if old_key in state_dict:
                state_dict[new_key] = state_dict[old_key]
                state_dict.pop(old_key)

        tracker_keys = [
            prefix + 'query_embedding.weight',
            prefix + 'reference_points.weight',
            prefix + 'reference_points.bias',
        ]
        for key in tracker_keys:
            own_key = key[len(prefix):] if key.startswith(prefix) else key
            if key not in state_dict:
                continue
            if not source_has_tracker:
                state_dict.pop(key)
                continue
            if own_key in own_state and state_dict[key].shape != own_state[
                    own_key].shape:
                state_dict.pop(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def _lidar_backbone_modules(self):
        return [
            getattr(self, 'pts_voxel_encoder', None),
            getattr(self, 'pts_middle_encoder', None),
            getattr(self, 'pts_backbone', None),
        ]

    def _bev_encoder_modules(self):
        return [
            self.pts_bbox_head.lidar_input_proj,
            self.pts_bbox_head.bev_embedding,
            self.pts_bbox_head.positional_encoding,
            self.pts_bbox_head.transformer.encoder,
        ]

    @staticmethod
    def _freeze_modules(modules):
        for module in modules:
            if module is None:
                continue
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    @staticmethod
    def _unwrap_single_bev(pts_feats):
        if not isinstance(pts_feats, (list, tuple)) or len(pts_feats) != 1:
            raise ValueError('UniADTrackLidar expects one BEV feature level, '
                             f'got {type(pts_feats)}.')
        return pts_feats[0]

    @staticmethod
    def _wrap_single_bev(bev):
        return [bev]

    def encode_bev(self, lidar_bev, prev_bev=None, queue_meta=None):
        if not self.use_prev_bev:
            prev_bev = None
        return self.pts_bbox_head.get_bev_features(
            lidar_bev, prev_bev=prev_bev, queue_meta=queue_meta)

    def valid_prev_bev(self, prev_bev, queue_meta):
        if prev_bev is None or not self.use_prev_bev:
            return None
        if queue_meta is None:
            return prev_bev
        if any(meta is None or not meta.get('prev_bev_exists', False)
               for meta in queue_meta):
            return None
        return prev_bev

    @staticmethod
    def _normalize_history_points(history_points, batch_size):
        if history_points is None or len(history_points) == 0:
            return [[] for _ in range(batch_size)]
        if batch_size == 1 and isinstance(history_points[0], torch.Tensor):
            return [list(history_points)]
        if len(history_points) == batch_size and isinstance(
                history_points[0], (list, tuple)):
            return [list(sample_history) for sample_history in history_points]
        if isinstance(history_points[0], (list, tuple)):
            per_sample = [[] for _ in range(batch_size)]
            for step_points in history_points:
                if len(step_points) != batch_size:
                    raise ValueError('history_points collate shape mismatch.')
                for batch_idx, points in enumerate(step_points):
                    per_sample[batch_idx].append(points)
            return per_sample
        raise TypeError(f'Unsupported history_points structure: '
                        f'{type(history_points)}')

    @staticmethod
    def current_queue_meta(img_metas):
        current = []
        for meta in img_metas:
            queue_metas = meta.get('queue_metas')
            if queue_metas is None:
                current.append(None)
                continue
            last_idx = max(queue_metas.keys())
            current.append(queue_metas[last_idx])
        return current

    def obtain_history_bev(self, history_points, img_metas):
        if not self.use_prev_bev or history_points is None:
            return None
        batch_size = len(img_metas)
        history_by_sample = self._normalize_history_points(
            history_points, batch_size)
        if not history_by_sample or len(history_by_sample[0]) == 0:
            return None
        num_history = len(history_by_sample[0])
        if any(len(sample_history) != num_history
               for sample_history in history_by_sample):
            raise ValueError('All samples must share the same history length.')

        queue_metas = [meta['queue_metas'] for meta in img_metas]
        prev_bev = None
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                for step in range(num_history):
                    step_points = [
                        sample_history[step]
                        for sample_history in history_by_sample
                    ]
                    step_lidar_bev = self.extract_lidar_bev_from_points(
                        step_points, img_metas)
                    step_meta = None
                    if prev_bev is not None:
                        step_meta = [
                            sample_queue_metas[step]
                            for sample_queue_metas in queue_metas
                        ]
                        prev_bev = self.valid_prev_bev(prev_bev, step_meta)
                        if prev_bev is None:
                            step_meta = None
                    prev_bev = self.encode_bev(
                        step_lidar_bev, prev_bev, queue_meta=step_meta)
        finally:
            if was_training:
                self.train()
        return prev_bev

    def _predict_prev_bev(self, history_points, img_metas, current_meta):
        if not self.use_prev_bev:
            return None
        if self.video_test_mode and len(img_metas) == 1:
            current_scene = (current_meta[0].get('scene_token')
                             if current_meta and current_meta[0] else None)
            if current_scene != self._test_scene_token:
                self._test_prev_bev = None
                self._test_scene_token = current_scene
            cached = self.valid_prev_bev(self._test_prev_bev, current_meta)
            if cached is not None:
                return cached
        prev_bev = self.obtain_history_bev(history_points, img_metas)
        return self.valid_prev_bev(prev_bev, current_meta)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_lidar_backbone:
            self._freeze_modules(self._lidar_backbone_modules())
        if self.freeze_bev_encoder:
            self._freeze_modules(self._bev_encoder_modules())
        return self

    @property
    def with_seg_head(self):
        return getattr(self, 'seg_head', None) is not None

    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight.get(prefix, 1.0)
        return {
            f'{prefix}.{key}': value * loss_factor
            for key, value in loss_dict.items()
        }

    @staticmethod
    def _first_batch_queue(value):
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(
                value[0], (list, tuple)):
            return list(value[0])
        if isinstance(value, (list, tuple)):
            return list(value)
        raise TypeError(f'Expected a queue list, got {type(value)}.')

    @staticmethod
    def _first_batch_metas(img_metas):
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1:
            img_metas = img_metas[0]
        if not isinstance(img_metas, dict):
            raise TypeError(f'Expected queued img_metas dict, got '
                            f'{type(img_metas)}.')
        return img_metas

    @classmethod
    def _current_img_metas(cls, img_metas):
        img_metas = cls._first_batch_metas(img_metas)
        if not img_metas:
            return []
        return [copy.deepcopy(img_metas[max(img_metas.keys())])]

    @staticmethod
    def _as_queue_metas(meta):
        if isinstance(meta, dict) and meta and all(
                isinstance(key, int) for key in meta.keys()):
            return meta
        if isinstance(meta, dict):
            queue_metas = meta.get('queue_metas')
            if isinstance(queue_metas, dict):
                return queue_metas
        return None

    @staticmethod
    def _normalize_test_history_points(points, history_points):
        if history_points is not None:
            return history_points
        points_queue = points
        if isinstance(points_queue, (list, tuple)) and len(points_queue) == 1:
            points_queue = points_queue[0]
        if isinstance(points_queue, (list, tuple)) and len(points_queue) > 1:
            return list(points_queue[:-1])
        return history_points

    @classmethod
    def _normalize_test_img_metas(cls, img_metas):
        while (isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and
               isinstance(img_metas[0], (list, tuple))):
            img_metas = img_metas[0]
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and \
                isinstance(img_metas[0], dict):
            queue_metas = cls._as_queue_metas(img_metas[0])
            if queue_metas is not None:
                current_idx = max(queue_metas.keys())
                current_meta = copy.deepcopy(queue_metas[current_idx])
                current_meta['queue_metas'] = queue_metas
                return [current_meta], True
            return img_metas, False
        if isinstance(img_metas, dict):
            queue_metas = cls._as_queue_metas(img_metas)
            if queue_metas is not None:
                current_idx = max(queue_metas.keys())
                current_meta = copy.deepcopy(queue_metas[current_idx])
                current_meta['queue_metas'] = queue_metas
                return [current_meta], True
            return [img_metas], False
        return img_metas, False

    @staticmethod
    def _seg_batch_list(value, item_ndim):
        if value is None:
            return None
        while (isinstance(value, (list, tuple)) and len(value) == 1 and
               isinstance(value[0], (list, tuple))):
            value = value[0]
        if isinstance(value, (list, tuple)):
            if len(value) == 1 and isinstance(value[0], torch.Tensor):
                return UniADTrackLidar._seg_batch_list(value[0], item_ndim)
            return list(value)
        if isinstance(value, torch.Tensor):
            if value.dim() == item_ndim:
                return [value]
            if value.dim() == item_ndim + 1:
                return [value[i] for i in range(value.size(0))]
        return value

    @staticmethod
    def _seg_test_list(value, item_ndim):
        if value is None:
            return None
        while (isinstance(value, (list, tuple)) and len(value) == 1 and
               isinstance(value[0], (list, tuple))):
            value = value[0]
        if isinstance(value, torch.Tensor):
            if value.dim() == item_ndim:
                return [value.unsqueeze(0)]
            if value.dim() == item_ndim + 1:
                return [value]
        if isinstance(value, (list, tuple)):
            if len(value) == 1 and isinstance(value[0], torch.Tensor):
                if value[0].dim() == item_ndim:
                    return [value[0].unsqueeze(0)]
                if value[0].dim() == item_ndim + 1:
                    return list(value)
            return list(value)
        return value

    @staticmethod
    def _bev_for_seg_head(bev_embed):
        if bev_embed.dim() == 4:
            return bev_embed.flatten(2).permute(2, 0, 1).contiguous()
        if bev_embed.dim() == 3:
            if bev_embed.shape[0] < bev_embed.shape[1]:
                return bev_embed.permute(1, 0, 2).contiguous()
            return bev_embed
        raise ValueError('seg_head expects BEV shape [HW, B, C], '
                         '[B, HW, C], or [B, C, H, W], got '
                         f'{tuple(bev_embed.shape)}.')

    @staticmethod
    def _detach_track_instances(track_instances):
        """Detach and clone all tensors in track_instances to break view refs."""
        for field in track_instances._fields:
            val = getattr(track_instances, field)
            if isinstance(val, torch.Tensor):
                setattr(track_instances, field, val.detach().clone())
        return track_instances

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(
            query[..., :dim // 2])
        track_instances.query = query
        track_instances.output_embedding = torch.zeros(
            (num_queries, dim // 2), device=device)
        track_instances.obj_idxes = torch.full(
            (num_queries, ), -1, dtype=torch.long, device=device)
        track_instances.matched_gt_idxes = torch.full(
            (num_queries, ), -1, dtype=torch.long, device=device)
        track_instances.disappear_time = torch.zeros(
            (num_queries, ), dtype=torch.long, device=device)
        track_instances.iou = torch.zeros(
            (num_queries, ), dtype=torch.float, device=device)
        track_instances.scores = torch.zeros(
            (num_queries, ), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (num_queries, ), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (num_queries, 10), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (num_queries, self.pts_bbox_head.num_classes),
            dtype=torch.float,
            device=device)
        track_instances.mem_bank = torch.zeros(
            (num_queries, self.mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device)
        track_instances.mem_padding_mask = torch.ones(
            (num_queries, self.mem_bank_len), dtype=torch.bool, device=device)
        track_instances.save_period = torch.zeros(
            (num_queries, ), dtype=torch.float32, device=device)
        return track_instances.to(device)

    def _copy_tracks_for_loss(self, tgt_instances):
        device = self.query_embedding.weight.device
        track_instances = Instances((1, 1))
        track_instances.obj_idxes = copy.deepcopy(tgt_instances.obj_idxes)
        track_instances.matched_gt_idxes = copy.deepcopy(
            tgt_instances.matched_gt_idxes)
        track_instances.disappear_time = copy.deepcopy(
            tgt_instances.disappear_time)
        track_instances.scores = torch.zeros(
            (len(track_instances), ), dtype=torch.float, device=device)
        track_instances.track_scores = torch.zeros(
            (len(track_instances), ), dtype=torch.float, device=device)
        track_instances.pred_boxes = torch.zeros(
            (len(track_instances), 10), dtype=torch.float, device=device)
        track_instances.iou = torch.zeros(
            (len(track_instances), ), dtype=torch.float, device=device)
        track_instances.pred_logits = torch.zeros(
            (len(track_instances), self.pts_bbox_head.num_classes),
            dtype=torch.float,
            device=device)
        track_instances.save_period = copy.deepcopy(tgt_instances.save_period)
        return track_instances.to(device)

    @staticmethod
    def _sanitize_logits(logits):
        return torch.nan_to_num(
            logits, nan=0.0, posinf=50.0, neginf=-50.0).clamp(
                min=-50.0, max=50.0)

    @staticmethod
    def _sanitize_boxes(boxes):
        boxes = torch.nan_to_num(boxes, nan=0.0, posinf=0.0, neginf=0.0)
        if boxes.size(-1) < 6:
            return boxes

        parts = [
            boxes[..., :2],
            boxes[..., 2:4].clamp(min=-5.0, max=5.0),
            boxes[..., 4:5],
            boxes[..., 5:6].clamp(min=-5.0, max=5.0),
        ]
        if boxes.size(-1) >= 10:
            parts.extend([
                boxes[..., 6:8],
                boxes[..., 8:10].clamp(min=-50.0, max=50.0),
                boxes[..., 10:],
            ])
        else:
            parts.append(boxes[..., 6:])
        return torch.cat(parts, dim=-1)

    def _sanitize_track_outputs(self, output_classes, output_coords,
                                output_past_trajs):
        output_classes = self._sanitize_logits(output_classes)
        output_coords = self._sanitize_boxes(output_coords)
        output_past_trajs = torch.nan_to_num(
            output_past_trajs, nan=0.0, posinf=0.0, neginf=0.0)
        return output_classes, output_coords, output_past_trajs

    @staticmethod
    def _check_finite_tensor(name, value):
        if not torch.isfinite(value).all():
            bad_count = (~torch.isfinite(value.detach())).sum().item()
            raise FloatingPointError(
                f'Non-finite UniADTrackLidar tensor in {name}: '
                f'shape={tuple(value.shape)}, bad_count={bad_count}')
        return value

    def _track_instances2results(self, track_instances, img_metas,
                                 with_mask=True):
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and \
                isinstance(img_metas[0], dict) and 0 in img_metas[0]:
            img_metas = [img_metas[0][max(img_metas[0].keys())]]
        box_type_3d = img_metas[0]['box_type_3d']
        if len(track_instances) == 0:
            empty_boxes = track_instances.pred_boxes.new_zeros((0, 9))
            boxes_3d = box_type_3d(
                empty_boxes.detach().cpu(),
                box_dim=empty_boxes.size(-1),
                origin=(0.5, 0.5, 0.5))
            empty_scores = track_instances.pred_boxes.new_zeros((0, ))
            empty_labels = torch.zeros(
                (0, ),
                dtype=torch.long,
                device=track_instances.pred_boxes.device)
            empty_indices = torch.zeros_like(empty_labels)
            return dict(
                boxes_3d=boxes_3d,
                scores_3d=empty_scores.detach().cpu(),
                labels_3d=empty_labels.detach().cpu(),
                track_scores=empty_scores.detach().cpu(),
                track_ids=empty_labels.detach().cpu(),
                bbox_index=empty_indices.detach().cpu(),
                mask=empty_labels.detach().cpu().bool(),
                track_bbox_results=[[
                    boxes_3d,
                    empty_scores.detach().cpu(),
                    empty_labels.detach().cpu(),
                    empty_indices.detach().cpu(),
                    empty_labels.detach().cpu().bool()
                ]])

        cls_scores = track_instances.pred_logits.sigmoid()
        scores, labels = cls_scores.max(dim=-1)
        max_num = min(self.pts_bbox_head.bbox_coder.max_num, scores.size(0))
        scores, bbox_index = scores.topk(max_num)
        labels = labels[bbox_index]
        track_scores = track_instances.scores[bbox_index]
        track_ids = track_instances.obj_idxes[bbox_index]
        bboxes = denormalize_bbox(track_instances.pred_boxes[bbox_index],
                                  self.point_cloud_range)
        post_center_range = self.pts_bbox_head.bbox_coder.post_center_range
        if post_center_range is not None:
            post_center_range = bboxes.new_tensor(post_center_range)
            mask = (bboxes[..., :3] >= post_center_range[:3]).all(1)
            mask &= (bboxes[..., :3] <= post_center_range[3:]).all(1)
        else:
            mask = torch.ones_like(scores, dtype=torch.bool)
        out_mask = mask if with_mask else torch.ones_like(mask)
        if with_mask:
            bboxes = bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            track_scores = track_scores[mask]
            track_ids = track_ids[mask]

        boxes_3d = box_type_3d(
            bboxes.detach().cpu(),
            box_dim=bboxes.size(-1),
            origin=(0.5, 0.5, 0.5))
        return dict(
            boxes_3d=boxes_3d,
            scores_3d=scores.detach().cpu(),
            labels_3d=labels.detach().cpu(),
            track_scores=track_scores.detach().cpu(),
            track_ids=track_ids.detach().cpu(),
            bbox_index=bbox_index.detach().cpu(),
            mask=out_mask.detach().cpu(),
            track_bbox_results=[[
                boxes_3d,
                scores.detach().cpu(),
                labels.detach().cpu(),
                bbox_index.detach().cpu(),
                out_mask.detach().cpu()
            ]])

    def select_active_track_query(self, track_instances, active_index,
                                  img_metas, with_mask=True):
        result_dict = self._track_instances2results(
            track_instances[active_index], img_metas, with_mask=with_mask)
        bbox_index = result_dict['bbox_index'].to(
            track_instances.output_embedding.device)
        mask = result_dict['mask'].to(track_instances.output_embedding.device)
        bbox_index = bbox_index[mask]
        result_dict['track_query_embeddings'] = (
            track_instances.output_embedding[active_index][bbox_index])
        result_dict['track_query_matched_idxes'] = (
            track_instances.matched_gt_idxes[active_index][bbox_index])
        return result_dict

    def select_sdc_track_query(self, sdc_instance, img_metas):
        """Pull the SDC slot's outputs for downstream stage-2 consumers.

        Mirrors UniAD camera ``select_sdc_track_query``.  ``sdc_instance``
        must be a single-row Instances; we keep ``with_mask=False`` so the
        output is preserved regardless of bbox post-center-range filtering.
        """
        out = {}
        result_dict = self._track_instances2results(
            sdc_instance, img_metas, with_mask=False)
        out['sdc_boxes_3d'] = result_dict['boxes_3d']
        out['sdc_scores_3d'] = result_dict['scores_3d']
        out['sdc_track_scores'] = result_dict['track_scores']
        out['sdc_track_bbox_results'] = result_dict['track_bbox_results']
        out['sdc_embedding'] = sdc_instance.output_embedding[0]
        return out

    @staticmethod
    def _meta_pose(meta, device):
        ego2global = torch.as_tensor(
            meta.get('ego2global', torch.eye(4)),
            dtype=torch.float32,
            device=device)
        timestamp = torch.as_tensor(
            float(meta.get('timestamp', 0.0)),
            dtype=torch.float32,
            device=device)
        return ego2global[:3, :3], ego2global[:3, 3], timestamp

    def _is_new_test_clip(self, meta, scene_token, timestamp):
        if self.test_track_instances is None:
            return True
        if scene_token != self.scene_token:
            return True

        prev_token = meta.get('prev', None)
        if prev_token == '':
            return True
        if prev_token is not None:
            if self.test_frame_token is not None and prev_token != self.test_frame_token:
                return True

        if self.timestamp is not None:
            time_gap = torch.abs(timestamp - self.timestamp)
            if bool((time_gap > 1.0).detach().cpu().item()):
                return True
        return False

    def velo_update(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta):
        time_delta = torch.nan_to_num(
            time_delta.type(torch.float), nan=0.0, posinf=0.0,
            neginf=0.0).clamp(min=0.0, max=2.0)
        num_query = ref_pts.size(0)
        ref_pts = torch.nan_to_num(
            ref_pts, nan=0.0, posinf=0.0, neginf=0.0)
        velocity = torch.nan_to_num(
            velocity, nan=0.0, posinf=0.0, neginf=0.0).clamp(
                min=-50.0, max=50.0)
        velo_pad = torch.cat((velocity, velocity.new_zeros((num_query, 1))),
                             dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.point_cloud_range
        reference_points[..., 0:1] = (
            reference_points[..., 0:1] *
            (pc_range[3] - pc_range[0]) + pc_range[0])
        reference_points[..., 1:2] = (
            reference_points[..., 1:2] *
            (pc_range[4] - pc_range[1]) + pc_range[1])
        reference_points[..., 2:3] = (
            reference_points[..., 2:3] *
            (pc_range[5] - pc_range[2]) + pc_range[2])
        reference_points = reference_points + velo_pad * time_delta
        ref_pts = reference_points @ l2g_r1 + l2g_t1 - l2g_t2
        ref_pts = ref_pts @ l2g_r2.T.type(torch.float)
        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2])
        ref_pts = torch.nan_to_num(
            ref_pts, nan=0.5, posinf=1.0, neginf=0.0).clamp(
                min=1e-4, max=1.0 - 1e-4)
        return inverse_sigmoid(ref_pts)

    def _forward_single_frame_inference(self,
                                        points,
                                        img_metas,
                                        track_instances,
                                        prev_bev=None,
                                        l2g_r1=None,
                                        l2g_t1=None,
                                        l2g_r2=None,
                                        l2g_t2=None,
                                        time_delta=None):
        active_inst = track_instances[track_instances.obj_idxes >= 0]
        other_inst = track_instances[track_instances.obj_idxes < 0]
        if l2g_r2 is not None and len(active_inst) > 0 and l2g_r1 is not None:
            ref_pts = self.velo_update(
                active_inst.ref_pts,
                active_inst.pred_boxes[:, -2:],
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta=time_delta)
            dim = active_inst.query.shape[-1]
            active_inst.ref_pts = self.reference_points(
                active_inst.query[..., :dim // 2])
            active_inst.ref_pts[..., :2] = ref_pts[..., :2]
        track_instances = Instances.cat([other_inst, active_inst])

        bev_embed, bev_pos = self.get_bevs(
            points, img_metas, prev_bev=prev_bev)
        det_output = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed),
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts)
        output_classes = det_output['all_cls_scores']
        output_coords = det_output['all_bbox_preds']
        output_past_trajs = det_output['all_past_traj_preds']
        last_ref_pts = det_output['last_ref_points']
        query_feats = det_output['query_feats']
        output_classes, output_coords, output_past_trajs = \
            self._sanitize_track_outputs(output_classes, output_coords,
                                         output_past_trajs)
        last_ref_pts = torch.nan_to_num(
            last_ref_pts, nan=0.0, posinf=0.0, neginf=0.0)

        out = dict(
            pred_logits=output_classes,
            pred_boxes=output_coords,
            ref_pts=last_ref_pts,
            bev_embed=bev_embed,
            bev_pos=bev_pos,
            query_embeddings=query_feats,
            all_past_traj_preds=output_past_trajs)

        track_scores = output_classes[-1, 0].sigmoid().max(dim=-1).values
        track_instances.scores = track_scores
        track_instances.track_scores = track_scores
        track_instances.pred_logits = output_classes[-1, 0]
        track_instances.pred_boxes = output_coords[-1, 0]
        track_instances.pred_past_trajs = output_past_trajs[-1, 0]
        track_instances.output_embedding = query_feats[-1][0]
        track_instances.ref_pts = last_ref_pts[0]
        if self.with_sdc and self.sdc_query_index is not None:
            # Reserve the last query as SDC; obj_id=-2 keeps it out of the
            # detection track set used for AMOTA evaluation.
            track_instances.obj_idxes[self.sdc_query_index] = -2
        self.track_base.update(track_instances, None)

        active_index = (
            (track_instances.obj_idxes >= 0)
            & (track_instances.scores >= self.track_base.filter_score_thresh))
        out.update(
            self.select_active_track_query(track_instances, active_index,
                                           img_metas))
        if self.with_sdc and self.sdc_query_index is not None:
            out.update(
                self.select_sdc_track_query(
                    track_instances[track_instances.obj_idxes == -2],
                    img_metas))
        out['track_instances_fordet'] = track_instances

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        out_track_instances = self.query_interact(
            dict(
                init_track_instances=self._generate_empty_tracks(),
                track_instances=track_instances))
        out['track_instances'] = out_track_instances
        out['track_obj_idxes'] = track_instances.obj_idxes
        return out

    def _bev_pos(self, batch_size, device, dtype):
        return self.pts_bbox_head.positional_encoding(
            batch_size, device, dtype)

    def extract_lidar_bev_from_points(self, points, img_metas):
        from torch import Tensor
        if isinstance(points, Tensor):
            points = [points]
        voxels, num_points, coors = self.voxelize(points)
        try:
            voxel_features = self.pts_voxel_encoder(
                voxels, num_points, coors, None, img_metas)
        except TypeError:
            voxel_features = self.pts_voxel_encoder(
                voxels, num_points, coors)
        middle_dtype = next(self.pts_middle_encoder.parameters()).dtype
        voxel_features = voxel_features.to(dtype=middle_dtype)
        batch_size = int(coors[-1, 0].item()) + 1
        if self.freeze_lidar_backbone:
            with torch.no_grad():
                x = self.pts_middle_encoder(voxel_features, coors, batch_size)
                x = self.pts_backbone(x)
            x = tuple(t.detach() for t in x) if isinstance(x, tuple) else x.detach()
        else:
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
        pts_feats = self.pts_neck(x) if self.with_pts_neck else x
        return self._unwrap_single_bev(pts_feats)

    def get_bevs(self, points, img_metas, prev_bev=None):
        lidar_bev = self.extract_lidar_bev_from_points(points, img_metas)
        prev_bev = self.valid_prev_bev(prev_bev, img_metas)
        bev_embed = self.encode_bev(
            lidar_bev, prev_bev=prev_bev, queue_meta=img_metas)
        bev_pos = self._bev_pos(
            bev_embed.size(0), bev_embed.device, bev_embed.dtype)
        return bev_embed, bev_pos

    @auto_fp16(apply_to=('points', ))
    def _forward_single_frame_train(self,
                                    points,
                                    img_metas,
                                    track_instances,
                                    prev_bev=None,
                                    l2g_r1=None,
                                    l2g_t1=None,
                                    l2g_r2=None,
                                    l2g_t2=None,
                                    time_delta=None):
        bev_embed, bev_pos = self.get_bevs(
            points, img_metas, prev_bev=prev_bev)
        det_output = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed),
            object_query_embeds=track_instances.query,
            ref_points=track_instances.ref_pts)

        output_classes = det_output['all_cls_scores']
        output_coords = det_output['all_bbox_preds']
        output_past_trajs = det_output['all_past_traj_preds']
        last_ref_pts = det_output['last_ref_points']
        query_feats = det_output['query_feats']
        self._check_finite_tensor('all_cls_scores', output_classes)
        self._check_finite_tensor('all_bbox_preds', output_coords)
        if getattr(self.criterion, 'loss_past_traj_weight', 0.0) > 0:
            self._check_finite_tensor('all_past_traj_preds',
                                      output_past_trajs)
        output_classes, output_coords, output_past_trajs = \
            self._sanitize_track_outputs(output_classes, output_coords,
                                         output_past_trajs)
        last_ref_pts = torch.nan_to_num(
            last_ref_pts, nan=0.0, posinf=0.0, neginf=0.0)

        out = dict(
            pred_logits=output_classes[-1],
            pred_boxes=output_coords[-1],
            pred_past_trajs=output_past_trajs[-1],
            ref_pts=last_ref_pts,
            bev_embed=bev_embed,
            bev_pos=bev_pos)
        with torch.no_grad():
            track_scores = output_classes[-1, 0].sigmoid().max(dim=-1).values

        nb_dec = output_classes.size(0)
        track_instances_list = [
            self._copy_tracks_for_loss(track_instances)
            for _ in range(nb_dec - 1)
        ]
        track_instances.output_embedding = query_feats[-1][0]
        velo = output_coords[-1, 0, :, -2:]
        if l2g_r2 is not None:
            ref_pts = self.velo_update(
                last_ref_pts[0], velo, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                time_delta=time_delta)
        else:
            ref_pts = last_ref_pts[0]

        dim = track_instances.query.shape[-1]
        track_instances.ref_pts = self.reference_points(
            track_instances.query[..., :dim // 2])
        track_instances.ref_pts[..., :2] = ref_pts[..., :2]
        track_instances_list.append(track_instances)

        for dec_id in range(nb_dec):
            track_instances = track_instances_list[dec_id]
            track_instances.scores = track_scores
            track_instances.pred_logits = output_classes[dec_id, 0]
            track_instances.pred_boxes = output_coords[dec_id, 0]
            track_instances.pred_past_trajs = output_past_trajs[dec_id, 0]
            out['track_instances'] = track_instances
            track_instances, _ = self.criterion.match_for_single_frame(
                out, dec_id, if_step=(dec_id == nb_dec - 1))

        active_index = (
            (track_instances.obj_idxes >= 0)
            & (track_instances.iou >= self.gt_iou_threshold)
            & (track_instances.matched_gt_idxes >= 0))
        out['active_track_instances'] = track_instances[active_index]
        out['active_track_query_embeddings'] = (
            track_instances.output_embedding[active_index])
        out['active_track_query_matched_idxes'] = (
            track_instances.matched_gt_idxes[active_index])
        out.update(
            self.select_active_track_query(track_instances, active_index,
                                           img_metas))
        if self.with_sdc and self.sdc_query_index is not None:
            # The criterion sets obj_idxes[sdc_query_index] = -2 during
            # match_for_single_frame; index by position instead of mask to
            # ensure exactly one row even before that side-effect runs.
            sdc_slice = track_instances[
                self.sdc_query_index:self.sdc_query_index + 1]
            out.update(self.select_sdc_track_query(sdc_slice, img_metas))

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        out_track_instances = self.query_interact(
            dict(
                init_track_instances=self._generate_empty_tracks(),
                track_instances=track_instances))
        out['track_instances'] = out_track_instances
        return out

    @auto_fp16(apply_to=('points', ))
    def forward_track_train(self,
                            points=None,
                            img_metas=None,
                            gt_bboxes_3d=None,
                            gt_labels_3d=None,
                            gt_inds=None,
                            gt_past_traj=None,
                            gt_past_traj_mask=None,
                            gt_sdc_bbox=None,
                            gt_sdc_label=None,
                            l2g_t=None,
                            l2g_r_mat=None,
                            timestamp=None,
                            **kwargs):
        points_queue = self._first_batch_queue(points)
        img_metas = self._first_batch_metas(img_metas)
        if os.environ.get('UNIAD_DEBUG_TRAIN_SAMPLE', '0') == '1':
            try:
                from mmcv.runner import get_dist_info
                rank, _ = get_dist_info()
            except Exception:
                rank = 0
            current_meta = img_metas[-1] if img_metas else {}
            print(
                f'[UNIAD_DEBUG_TRAIN_SAMPLE] rank={rank} '
                f'scene={current_meta.get("scene_token", "")} '
                f'token={current_meta.get("token", "")} '
                f'timestamp={current_meta.get("timestamp", "")}',
                flush=True)
        num_frames = len(points_queue)
        device = points_queue[-1].device
        if self.with_sdc and (gt_sdc_bbox is None or gt_sdc_label is None):
            raise ValueError('with_sdc=True requires gt_sdc_bbox and '
                             'gt_sdc_label in the training batch.')

        gt_instances_list = []
        for frame_idx in range(num_frames):
            gt_instances = Instances((1, 1))
            boxes = gt_bboxes_3d[0][frame_idx].tensor.to(device)
            gt_instances.boxes = normalize_bbox(boxes, self.point_cloud_range)
            gt_instances.labels = gt_labels_3d[0][frame_idx].to(device)
            gt_instances.obj_ids = gt_inds[0][frame_idx].to(device)
            gt_instances.past_traj = gt_past_traj[0][frame_idx].to(device)
            gt_instances.past_traj_mask = gt_past_traj_mask[0][frame_idx].to(
                device)
            if self.with_sdc and gt_sdc_bbox is not None:
                num_obj = boxes.shape[0]
                sd_box = gt_sdc_bbox[0][frame_idx].tensor.to(device)
                sd_box = normalize_bbox(sd_box, self.point_cloud_range)[:1]
                sd_label = gt_sdc_label[0][frame_idx].to(device)[:1]
                if num_obj > 0:
                    # Instances enforces all fields share length.  The loss
                    # only reads sdc_boxes[:1] / sdc_labels[0:1], so rows past
                    # the first are never used; expanding satisfies the length
                    # check without copying memory.
                    gt_instances.sdc_boxes = (
                        sd_box.expand(num_obj, -1).contiguous())
                    gt_instances.sdc_labels = (
                        sd_label.expand(num_obj).contiguous())
                else:
                    # Keep one SDC target even on frames with no regular
                    # objects.  Bypass the length check because Instances
                    # length is defined by regular-object fields.
                    gt_instances.get_fields()['sdc_boxes'] = sd_box.contiguous()
                    gt_instances.get_fields()['sdc_labels'] = (
                        sd_label.contiguous())
            gt_instances_list.append(gt_instances)
        self.criterion.initialize_for_single_clip(gt_instances_list)

        track_instances = self._generate_empty_tracks()
        prev_bev = None
        frame_res = None
        for frame_idx in range(num_frames):
            frame_metas = [copy.deepcopy(img_metas[frame_idx])]
            if frame_idx == num_frames - 1:
                l2g_r2 = None
                l2g_t2 = None
                time_delta = None
            else:
                l2g_r2 = l2g_r_mat[0][frame_idx + 1].to(device)
                l2g_t2 = l2g_t[0][frame_idx + 1].to(device)
                time_delta = (timestamp[0][frame_idx + 1] -
                              timestamp[0][frame_idx]).to(device)
            frame_res = self._forward_single_frame_train(
                points_queue[frame_idx],
                frame_metas,
                track_instances,
                prev_bev=prev_bev,
                l2g_r1=l2g_r_mat[0][frame_idx].to(device),
                l2g_t1=l2g_t[0][frame_idx].to(device),
                l2g_r2=l2g_r2,
                l2g_t2=l2g_t2,
                time_delta=time_delta)
            track_instances = frame_res['track_instances']
            prev_bev = frame_res['bev_embed'].detach()

        get_keys = [
            'bev_embed', 'bev_pos', 'track_query_embeddings',
            'track_query_matched_idxes', 'track_bbox_results'
        ]
        if self.with_sdc:
            get_keys += [
                'sdc_boxes_3d', 'sdc_scores_3d', 'sdc_track_scores',
                'sdc_track_bbox_results', 'sdc_embedding'
            ]
        outs_track = {k: frame_res[k] for k in get_keys if k in frame_res}
        return self.criterion.losses_dict, outs_track

    @auto_fp16(apply_to=('points', ))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_bbox=None,
                      gt_sdc_label=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      **kwargs):
        losses, outs_track = self.forward_track_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            gt_sdc_bbox=gt_sdc_bbox,
            gt_sdc_label=gt_sdc_label,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            timestamp=timestamp,
            **kwargs)
        losses = self.loss_weighted_and_prefixed(losses, prefix='track')

        if self.with_seg_head and gt_lane_labels is not None:
            current_metas = self._current_img_metas(img_metas)
            gt_lane_labels = self._seg_batch_list(gt_lane_labels, 1)
            gt_lane_bboxes = self._seg_batch_list(gt_lane_bboxes, 2)
            gt_lane_masks = self._seg_batch_list(gt_lane_masks, 3)
            bev_embed = self._bev_for_seg_head(outs_track['bev_embed'])
            losses_seg, _ = self.seg_head.forward_train(
                bev_embed,
                current_metas,
                gt_lane_labels,
                gt_lane_bboxes,
                gt_lane_masks)
            losses.update(
                self.loss_weighted_and_prefixed(losses_seg, prefix='map'))

        sanitized_losses = {}
        for key, value in sorted(losses.items()):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f'Non-finite training loss in {key}: '
                    f'{value.detach().cpu()}')
            sanitized_losses[key] = value
        return sanitized_losses

    def simple_test(self, points, img_metas, img=None, history_points=None,
                    **kwargs):
        history_points = self._normalize_test_history_points(
            points, history_points)
        points_queue = points
        if isinstance(points, (list, tuple)) and len(points) == 1 and isinstance(
                points[0], (list, tuple)):
            points_queue = points[0]
        if isinstance(points_queue, (list, tuple)):
            points = points_queue[-1]
        img_metas, has_queue_meta = self._normalize_test_img_metas(img_metas)

        has_queue_meta = has_queue_meta or (
            img_metas and isinstance(img_metas[0], dict) and
            'queue_metas' in img_metas[0])
        if has_queue_meta:
            queue_current_meta = self.current_queue_meta(img_metas)
            current_meta = []
            for base_meta, queue_meta in zip(img_metas, queue_current_meta):
                merged_meta = copy.deepcopy(base_meta)
                if queue_meta is not None:
                    merged_meta.update(queue_meta)
                current_meta.append(merged_meta)
            prev_bev = self._predict_prev_bev(
                history_points, img_metas, current_meta)
        else:
            current_meta = img_metas
            prev_bev = self._test_prev_bev

        if len(current_meta) != 1:
            raise NotImplementedError(
                'UniADTrackLidar test currently supports batch size 1.')
        meta = current_meta[0]
        device = points.device if isinstance(points, torch.Tensor) else \
            next(self.parameters()).device
        l2g_r2, l2g_t2, timestamp = self._meta_pose(meta, device)
        scene_token = meta.get('scene_token', '')
        frame_token = meta.get('token', None)
        is_new_clip = self._is_new_test_clip(meta, scene_token, timestamp)
        if is_new_clip:
            self.track_base.clear()
            track_instances = self._generate_empty_tracks()
            l2g_r1, l2g_t1, time_delta = None, None, None
            self._test_prev_bev = None
            self._test_scene_token = scene_token
            if has_queue_meta:
                prev_bev = self.obtain_history_bev(history_points, img_metas)
                prev_bev = self.valid_prev_bev(prev_bev, current_meta)
            else:
                prev_bev = None
            torch.cuda.empty_cache()
        else:
            track_instances = self.test_track_instances
            l2g_r1 = self.l2g_r_mat
            l2g_t1 = self.l2g_t
            time_delta = timestamp - self.timestamp

        frame_res = self._forward_single_frame_inference(
            points,
            current_meta,
            track_instances,
            prev_bev=prev_bev,
            l2g_r1=l2g_r1,
            l2g_t1=l2g_t1,
            l2g_r2=l2g_r2,
            l2g_t2=l2g_t2,
            time_delta=time_delta)

        self.test_track_instances = self._detach_track_instances(
            frame_res['track_instances'])
        self._test_prev_bev = frame_res['bev_embed'].detach().clone()
        self.scene_token = scene_token
        self.test_frame_token = frame_token
        self.timestamp = timestamp
        self.l2g_r_mat = l2g_r2
        self.l2g_t = l2g_t2

        get_keys = [
            'track_query_embeddings',
            'track_query_matched_idxes', 'track_bbox_results', 'boxes_3d',
            'scores_3d', 'labels_3d', 'track_scores', 'track_ids'
        ]
        if getattr(self, 'with_motion_head', False) or self.with_seg_head:
            get_keys = ['bev_embed', 'bev_pos'] + get_keys
        if self.with_sdc and (
                getattr(self, 'with_motion_head', False)
                or getattr(self, 'with_planning_head', False)):
            # Stage-1 evaluation does not consume SDC outputs; only export
            # them when a downstream stage-2 head is attached.
            get_keys += [
                'sdc_boxes_3d', 'sdc_scores_3d', 'sdc_track_scores',
                'sdc_track_bbox_results', 'sdc_embedding'
            ]
        result = {k: frame_res[k] for k in get_keys if k in frame_res}
        if (self.with_seg_head and kwargs.get('gt_lane_labels') is not None and
                kwargs.get('gt_lane_masks') is not None):
            gt_lane_labels = self._seg_test_list(
                kwargs.get('gt_lane_labels'), 1)
            gt_lane_masks = self._seg_test_list(
                kwargs.get('gt_lane_masks'), 3)
            bev_embed = self._bev_for_seg_head(frame_res['bev_embed'])
            result_seg = self.seg_head.forward_test(
                bev_embed,
                gt_lane_labels,
                gt_lane_masks,
                current_meta,
                rescale=False)
            if result_seg:
                result['map'] = result_seg[0].get('pts_bbox', {})
                if 'ret_iou' in result_seg[0]:
                    result['ret_iou'] = result_seg[0]['ret_iou']
        det_results = self.pts_bbox_head.predict_by_feat(
            dict(
                all_cls_scores=frame_res['pred_logits'],
                all_bbox_preds=frame_res['pred_boxes'],
                last_ref_points=frame_res['ref_pts']),
            current_meta)
        result.update(
            boxes_3d_det=det_results[0]['boxes_3d'],
            scores_3d_det=det_results[0]['scores_3d'],
            labels_3d_det=det_results[0]['labels_3d'])
        return [dict(pts_bbox=result)]

    def simple_test_track_queue(self, points, img_metas):
        """Run TrackFormer sequentially over one complete test queue.

        Normal E2E inference uses history frames only to build temporal BEV
        and predicts tracks at the current frame. This diagnostic path keeps
        tracker state across every queued frame so OccWorld can audit a fully
        predicted five-frame instance history.
        """
        points_queue = self._first_batch_queue(points)
        normalized_metas, has_queue_meta = self._normalize_test_img_metas(
            img_metas)
        if len(normalized_metas) != 1 or not has_queue_meta:
            raise ValueError(
                'Track-queue export requires batch size one with queue_metas')
        queue_metas = normalized_metas[0]['queue_metas']
        ordered_keys = sorted(queue_metas)
        if len(points_queue) != len(ordered_keys):
            raise ValueError(
                'Track-queue points/meta length mismatch: '
                f'{len(points_queue)} vs {len(ordered_keys)}')

        self.track_base.clear()
        track_instances = self._generate_empty_tracks()
        prev_bev = None
        previous_r = None
        previous_t = None
        previous_timestamp = None
        queue_results = []
        export_keys = (
            'boxes_3d', 'scores_3d', 'labels_3d',
            'track_scores', 'track_ids')
        for points_frame, meta_key in zip(points_queue, ordered_keys):
            frame_meta = copy.deepcopy(queue_metas[meta_key])
            device = (
                points_frame.device if isinstance(points_frame, torch.Tensor)
                else next(self.parameters()).device)
            current_r, current_t, current_timestamp = self._meta_pose(
                frame_meta, device)
            time_delta = (
                None if previous_timestamp is None
                else current_timestamp - previous_timestamp)
            frame_res = self._forward_single_frame_inference(
                points_frame,
                [frame_meta],
                track_instances,
                prev_bev=prev_bev,
                l2g_r1=previous_r,
                l2g_t1=previous_t,
                l2g_r2=current_r,
                l2g_t2=current_t,
                time_delta=time_delta)
            track_instances = self._detach_track_instances(
                frame_res['track_instances'])
            prev_bev = frame_res['bev_embed'].detach().clone()
            result = {
                key: frame_res[key]
                for key in export_keys if key in frame_res
            }
            result.update(
                sample_idx=int(frame_meta.get('sample_idx', -1)),
                scene_token=str(frame_meta.get('scene_token', '')),
                token=str(frame_meta.get('token', '')),
                timestamp=float(frame_meta.get('timestamp', 0.0)),
                ego2global=np.asarray(
                    frame_meta.get('ego2global', np.eye(4)),
                    dtype=np.float64))
            queue_results.append(result)
            previous_r = current_r
            previous_t = current_t
            previous_timestamp = current_timestamp

        self.track_base.clear()
        return [dict(pts_bbox=dict(track_queue_results=queue_results))]
