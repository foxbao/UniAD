import copy

import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from ..dense_heads.track_head_plugin import (Instances, MemoryBank,
                                             QueryInteractionModule,
                                             RuntimeTrackerBase)
from .bevformer_lidar import BEVFormerLidar


@DETECTORS.register_module()
class UniADTrackLidar(BEVFormerLidar):
    """LiDAR-only UniAD stage-1 tracker.

    This keeps the LiDAR BEVFormer detector/front-end compatible with
    ``base_bevformer_lidar.py`` checkpoints, then adds UniAD's query
    interaction, memory bank, and clip matcher for track training.
    """

    def __init__(self,
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
                 queue_length=4,
                 **kwargs):
        kwargs['return_query_feats'] = True
        super().__init__(**kwargs)

        self.gt_iou_threshold = gt_iou_threshold
        self.freeze_bev_encoder = freeze_bev_encoder
        self.queue_length = queue_length
        if freeze_lidar_backbone:
            for module in [
                    self.pts_voxel_encoder, self.pts_middle_encoder,
                    self.pts_backbone, self.pts_neck
            ]:
                if module is None:
                    continue
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False

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
        self.criterion = build_loss(loss_cfg)
        self.test_track_instances = None
        self.scene_token = None
        self.timestamp = None
        self.l2g_t = None
        self.l2g_r_mat = None

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

    def _generate_empty_tracks(self):
        track_instances = Instances((1, 1))
        num_queries, dim = self.query_embedding.weight.shape
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        track_instances.ref_pts = self.reference_points(query[..., :dim // 2])
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

    def velo_update(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                    time_delta):
        time_delta = time_delta.type(torch.float)
        num_query = ref_pts.size(0)
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
        ref_pts = ref_pts @ torch.linalg.inv(l2g_r2).type(torch.float)
        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2])
        return inverse_sigmoid(ref_pts)

    def _bev_pos(self, batch_size, device, dtype):
        return self.pts_bbox_head.positional_encoding(
            batch_size, device, dtype)

    def get_bevs(self, points, img_metas, prev_bev=None):
        lidar_bev = self.extract_lidar_bev_from_points(points, img_metas)
        prev_bev = self.valid_prev_bev(prev_bev, img_metas)
        if self.freeze_bev_encoder:
            with torch.no_grad():
                bev_embed = self.encode_bev(
                    lidar_bev, prev_bev=prev_bev, queue_meta=img_metas)
        else:
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

        if self.memory_bank is not None:
            track_instances = self.memory_bank(track_instances)
        out_track_instances = self.query_interact(
            dict(
                init_track_instances=self._generate_empty_tracks(),
                track_instances=track_instances))
        out['track_instances'] = out_track_instances
        return out

    @auto_fp16(apply_to=('points', ))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      **kwargs):
        points_queue = self._first_batch_queue(points)
        img_metas = self._first_batch_metas(img_metas)
        num_frames = len(points_queue)
        device = points_queue[-1].device

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

        return self.criterion.losses_dict

    def simple_test(self, points, img_metas, img=None, history_points=None,
                    **kwargs):
        points_queue = points
        if isinstance(points, (list, tuple)) and len(points) == 1 and isinstance(
                points[0], (list, tuple)):
            points_queue = points[0]
        if isinstance(points_queue, (list, tuple)):
            points = points_queue[-1]
        if isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and \
                isinstance(img_metas[0], dict) and 0 in img_metas[0]:
            img_metas = [img_metas[0][max(img_metas[0].keys())]]
        elif isinstance(img_metas, dict):
            img_metas = [img_metas]

        if img_metas and isinstance(img_metas[0], dict) and \
                'queue_metas' in img_metas[0]:
            current_meta = self.current_queue_meta(img_metas)
            prev_bev = self._predict_prev_bev(
                history_points, img_metas, current_meta)
        else:
            current_meta = img_metas
            prev_bev = None

        bev_embed, _ = self.get_bevs(points, current_meta, prev_bev=prev_bev)
        query_embeds, ref_points = self._detector_query_inputs(
            bev_embed.size(0), bev_embed.device, bev_embed.dtype)
        preds = self.pts_bbox_head.get_detections(
            self._wrap_single_bev(bev_embed),
            object_query_embeds=query_embeds,
            ref_points=ref_points)
        results = self.pts_bbox_head.predict_by_feat(preds, img_metas)
        return [dict(pts_bbox=result) for result in results]
