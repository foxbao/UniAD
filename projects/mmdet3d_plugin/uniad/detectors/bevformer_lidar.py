from typing import Optional, Sequence

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS
from third_party.uniad_mmdet3d.models.detectors.mvx_two_stage import (
    MVXTwoStageDetector)
from torch import Tensor


@DETECTORS.register_module()
class BEVFormerLidar(MVXTwoStageDetector):
    """LiDAR-only BEVFormer detector adapted to the old MMCV runner stack."""

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
                 point_cloud_range: Optional[Sequence[float]] = None,
                 num_query: int = 600,
                 embed_dims: int = 256,
                 use_prev_bev: bool = True,
                 video_test_mode: bool = True,
                 return_query_feats: bool = False):
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
        self.point_cloud_range = point_cloud_range
        self.num_query = num_query
        self.embed_dims = embed_dims
        self.use_prev_bev = use_prev_bev
        self.video_test_mode = video_test_mode
        self.return_query_feats = return_query_feats
        self.fp16_enabled = False

        self._test_prev_bev = None
        self._test_scene_token = None

    @staticmethod
    def _unwrap_single_bev(pts_feats) -> Tensor:
        if not isinstance(pts_feats, (list, tuple)) or len(pts_feats) != 1:
            raise ValueError('BEVFormerLidar expects one BEV feature '
                             f'level, got {type(pts_feats)}.')
        return pts_feats[0]

    @staticmethod
    def _wrap_single_bev(bev: Tensor):
        return [bev]

    def extract_lidar_bev_from_points(self, points, img_metas) -> Tensor:
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
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        pts_feats = self.pts_neck(x) if self.with_pts_neck else x
        return self._unwrap_single_bev(pts_feats)

    def encode_bev(self, lidar_bev: Tensor, prev_bev: Optional[Tensor] = None,
                   queue_meta: Optional[Sequence[dict]] = None) -> Tensor:
        if not self.use_prev_bev:
            prev_bev = None
        return self.pts_bbox_head.get_bev_features(
            lidar_bev, prev_bev=prev_bev, queue_meta=queue_meta)

    def valid_prev_bev(self, prev_bev: Optional[Tensor],
                       queue_meta: Optional[Sequence[dict]]) -> Optional[Tensor]:
        if prev_bev is None or not self.use_prev_bev:
            return None
        if queue_meta is None:
            return prev_bev
        if any(meta is None or not meta.get('prev_bev_exists', False)
               for meta in queue_meta):
            return None
        return prev_bev

    @staticmethod
    def _normalize_history_points(history_points, batch_size: int):
        if history_points is None or len(history_points) == 0:
            return [[] for _ in range(batch_size)]
        if batch_size == 1 and isinstance(history_points[0], Tensor):
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

    def _extract_current_bev_embed(self, points, img_metas, history_points):
        lidar_bev = self.extract_lidar_bev_from_points(points, img_metas)
        prev_bev = self.obtain_history_bev(history_points, img_metas)
        current_meta = self.current_queue_meta(img_metas)
        prev_bev = self.valid_prev_bev(prev_bev, current_meta)
        bev_embed = self.encode_bev(
            lidar_bev, prev_bev, queue_meta=current_meta)
        return bev_embed, current_meta

    @auto_fp16(apply_to=('points', ))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      history_points=None,
                      **kwargs):
        bev_embed, _ = self._extract_current_bev_embed(
            points, img_metas, history_points)
        preds = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed))
        gt_bboxes_3d = [boxes.to(bev_embed.device) for boxes in gt_bboxes_3d]
        gt_labels_3d = [
            torch.as_tensor(labels, device=bev_embed.device, dtype=torch.long)
            for labels in gt_labels_3d
        ]
        return self.pts_bbox_head.loss_by_feat(
            preds, gt_bboxes_3d, gt_labels_3d)

    def simple_test(self, points, img_metas, img=None, history_points=None,
                    **kwargs):
        if isinstance(img_metas, dict):
            img_metas = [img_metas]
        lidar_bev = self.extract_lidar_bev_from_points(points, img_metas)
        current_meta = self.current_queue_meta(img_metas)
        prev_bev = self._predict_prev_bev(history_points, img_metas,
                                          current_meta)
        bev_embed = self.encode_bev(
            lidar_bev, prev_bev, queue_meta=current_meta)
        preds = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed))
        results = self.pts_bbox_head.predict_by_feat(
            preds, img_metas, return_query_feats=self.return_query_feats)
        if self.video_test_mode and len(img_metas) == 1:
            self._test_prev_bev = bev_embed.detach()
        return [dict(pts_bbox=result) for result in results]

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
