import torch
import torch.nn as nn
from mmdet.core.bbox import build_bbox_coder
from mmdet.models import DETECTORS, build_head, build_loss
from mmdet.models.utils.transformer import inverse_sigmoid
from third_party.uniad_mmdet3d.models.detectors.mvx_two_stage import (
    MVXTwoStageDetector)

from projects.mmdet3d_plugin.core.bbox.util import denormalize_bbox_trt
from projects.mmdet3d_plugin.uniad.functions import inverse

from ..dense_heads.track_head_plugin import (MemoryBankTRTP,
                                             QueryInteractionModuleTRTP,
                                             RuntimeTrackerBase)
from .uniad_track import track_base_update_trt_script_v5


def index_bool2long_trt(bool_index):
    long_index = bool_index.long()
    long_index = (
        torch.arange(1, long_index.shape[-1] + 1, device=long_index.device) *
        long_index)
    long_index = long_index[long_index.nonzero(as_tuple=True)[0]]
    return long_index - torch.ones_like(long_index)


@DETECTORS.register_module()
class UniADTrackLidarTRT(MVXTwoStageDetector):
    """LiDAR-only UniAD stage-1 TensorRT export boundary.

    The sparse voxel encoder and LiDAR backbone+neck are exported as separate
    engines.  This detector starts from dense ``lidar_bev`` and owns only the
    BEV encoder, tracking decoder, and TensorRT-friendly track state update.
    """

    def __init__(
        self,
        pts_bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
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
        bbox_coder=dict(
            type='DETRTrack3DCoder',
            post_center_range=[-64.0, -48.0, -10.0, 64.0, 48.0, 10.0],
            pc_range=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0],
            max_num=300,
            num_classes=13,
            score_threshold=0.0,
            with_nms=False,
            iou_thres=0.3),
        point_cloud_range=None,
        embed_dims=256,
        num_query=600,
        num_classes=13,
        score_thresh=0.4,
        filter_score_thresh=0.35,
        miss_tolerance=5,
        gt_iou_threshold=0.0,
        queue_length=5,
        video_test_mode=True,
        with_sdc=True,
        **kwargs,
    ):
        super().__init__(
            pts_bbox_head=pts_bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        if kwargs:
            raise TypeError(f'Unexpected UniADTrackLidarTRT kwargs: {kwargs}')

        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.num_classes = num_classes
        self.point_cloud_range = point_cloud_range or bbox_coder.get(
            'pc_range')
        self.pc_range = self.point_cloud_range
        self.queue_length = queue_length
        self.with_sdc = with_sdc
        self.sdc_query_index = self.num_query if with_sdc else None
        self.video_test_mode = video_test_mode

        self.query_embedding = nn.Embedding(
            self.num_query + int(with_sdc), self.embed_dims * 2)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.track_base = RuntimeTrackerBase(
            score_thresh=score_thresh,
            filter_score_thresh=filter_score_thresh,
            miss_tolerance=miss_tolerance)
        self.query_interact = QueryInteractionModuleTRTP(
            qim_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims)
        self.memory_bank = MemoryBankTRTP(
            mem_args,
            dim_in=embed_dims,
            hidden_dim=embed_dims,
            dim_out=embed_dims)
        self.mem_bank_len = (
            0 if self.memory_bank is None else self.memory_bank.max_his_length)
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.criterion = build_loss(loss_cfg) if loss_cfg is not None else None
        self.gt_iou_threshold = gt_iou_threshold
        self.bev_h = self.pts_bbox_head.bev_h
        self.bev_w = self.pts_bbox_head.bev_w
        self.inverse = inverse

    def get_bevs_trt(self, lidar_bev, prev_bev, shift, use_prev_bev):
        bev_embed = self.pts_bbox_head.get_bev_features_trt(
            lidar_bev,
            prev_bev=prev_bev,
            shift=shift,
            use_prev_bev=use_prev_bev)
        bev_pos = self.pts_bbox_head.positional_encoding(
            lidar_bev.size(0), lidar_bev.device, lidar_bev.dtype)
        return bev_embed, bev_pos

    def velo_update_trt(self, ref_pts, velocity, l2g_r1, l2g_t1, l2g_r2,
                        l2g_t2, time_delta):
        time_delta = time_delta.float()
        num_query = ref_pts.size(0)
        velo_pad = torch.cat(
            (velocity, velocity.new_zeros((num_query, 1))), dim=-1)

        reference_points = ref_pts.sigmoid().clone()
        pc_range = self.pc_range
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
        # FP16 numerical hygiene (same class as MapLaneEncoderTRT): the naive
        # form `reference_points @ l2g_r1 + l2g_t1 - l2g_t2` first adds the ego
        # global translation l2g_t1 (~thousands of metres) and only then
        # subtracts l2g_t2 (same magnitude) -> catastrophic cancellation in FP16
        # (ULP ~2 m at 3000). Compute the inter-frame translation delta
        # (l2g_t1 - l2g_t2, a small metric offset) in fp32 first, so no
        # ~3000 m intermediate ever enters the graph. Math-identical:
        #   (a @ R + t1) - t2 == a @ R + (t1 - t2)
        trans_delta = (l2g_t1.float() - l2g_t2.float()).to(dtype=ref_pts.dtype)
        ref_pts = reference_points @ l2g_r1 + trans_delta
        g2l_r = self.inverse(l2g_r2[None, ...])[0].float()
        ref_pts = ref_pts @ g2l_r

        ref_pts[..., 0:1] = (ref_pts[..., 0:1] - pc_range[0]) / (
            pc_range[3] - pc_range[0])
        ref_pts[..., 1:2] = (ref_pts[..., 1:2] - pc_range[1]) / (
            pc_range[4] - pc_range[1])
        ref_pts[..., 2:3] = (ref_pts[..., 2:3] - pc_range[2]) / (
            pc_range[5] - pc_range[2])
        return inverse_sigmoid(ref_pts)

    def _forward_single_frame_inference_preprocess_trt(
        self,
        track_instances,
        l2g_r1,
        l2g_t1,
        l2g_r2,
        l2g_t2,
        time_delta,
        use_prev_bev,
    ):
        active_index = index_bool2long_trt(track_instances[3] >= 0)
        other_index = index_bool2long_trt(track_instances[3] < 0)
        active_inst = [item[active_index] for item in track_instances]
        other_inst = [item[other_index] for item in track_instances]

        condition = use_prev_bev.reshape(-1)[0].float()
        active_inst_query = active_inst[0]
        active_inst_velo = active_inst[-5][:, -2:]
        active_inst_ref_pts = active_inst[1]
        updated_ref_pts = self.velo_update_trt(
            active_inst_ref_pts,
            active_inst_velo,
            l2g_r1,
            l2g_t1,
            l2g_r2,
            l2g_t2,
            time_delta=time_delta)[0]
        active_inst_ref_pts = (
            condition * updated_ref_pts +
            (1 - condition) * active_inst_ref_pts)

        query_generated_ref_pts = self.reference_points(
            active_inst_query[..., :self.embed_dims])
        active_inst[1] = (
            query_generated_ref_pts * condition +
            active_inst[1] * (1 - condition))
        active_inst[1][..., :2] = (
            active_inst_ref_pts[..., :2] * condition +
            active_inst[1][..., :2] * (1 - condition))

        for i in range(len(track_instances)):
            track_instances[i] = torch.cat((other_inst[i], active_inst[i]),
                                           dim=0)
        return track_instances, track_instances[0], track_instances[1]

    def _forward_single_frame_inference_trt(
        self,
        lidar_bev,
        track_instances,
        prev_bev,
        shift,
        l2g_r1,
        l2g_t1,
        l2g_r2,
        l2g_t2,
        time_delta,
        use_prev_bev,
    ):
        track_instances, query, ref_pts = (
            self._forward_single_frame_inference_preprocess_trt(
                track_instances,
                l2g_r1,
                l2g_t1,
                l2g_r2,
                l2g_t2,
                time_delta,
                use_prev_bev))
        bev_embed, bev_pos = self.get_bevs_trt(
            lidar_bev, prev_bev, shift, use_prev_bev)
        output_classes, output_coords, all_past_traj_preds, last_ref_pts, \
            query_feats = self.pts_bbox_head.get_detections_trt(
                bev_embed,
                object_query_embeds=query,
                ref_points=ref_pts)
        return (track_instances, bev_embed, bev_pos, output_classes,
                output_coords, all_past_traj_preds, last_ref_pts, query_feats)

    def track2mop_trt(self, track_instances, output_classes, output_coords,
                      last_ref_pts, query_feats, max_obj_id):
        track_instances[7] = output_classes[-1, 0, :].sigmoid().max(
            dim=-1).values
        track_instances[-4] = output_classes[-1, 0]
        track_instances[-5] = output_coords[-1, 0]
        track_instances[2] = query_feats[-1][0]
        track_instances[1] = last_ref_pts[0]
        if self.sdc_query_index is not None:
            track_instances[3][self.sdc_query_index] = -2
        track_instances[3], track_instances[5], max_obj_id = \
            track_base_update_trt_script_v5(
                track_instances[3], track_instances[5], track_instances[7],
                track_instances[-5].shape[0], self.track_base.score_thresh,
                self.track_base.filter_score_thresh, max_obj_id,
                self.track_base.miss_tolerance)

        active_index = (
            (track_instances[3] >= 0) &
            (track_instances[7] >= self.track_base.filter_score_thresh))
        active_index = index_bool2long_trt(active_index)
        (track_query_embeddings, track_query_matched_idxes,
         bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
         labels, track_scores, bbox_index, obj_idxes, mask,
         track_bbox_results1, track_bbox_results2) = \
            self.select_active_track_query_trt(track_instances, active_index)

        if self.memory_bank is not None:
            track_instances[2], track_instances[-2], track_instances[-1], \
                track_instances[-3] = self.memory_bank.forward_trt(
                    track_instances[-2],
                    track_instances[2],
                    track_instances[-3],
                    track_instances[7],
                    track_instances[-1])
        query_interact_track_instances = self.query_interact.forward_trt(
            self._generate_empty_tracks_trt(), track_instances)

        return (track_query_embeddings, track_query_matched_idxes,
                bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
                labels, track_scores, bbox_index, obj_idxes, mask,
                track_bbox_results1, track_bbox_results2, max_obj_id,
                track_instances, query_interact_track_instances)

    def select_active_track_query_trt(self,
                                      track_instances,
                                      active_index,
                                      with_mask=True):
        active_track_instances = [item[active_index] for item in track_instances]
        (bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
         labels, track_scores, bbox_index, obj_idxes, mask,
         track_bbox_results1, track_bbox_results2) = \
            self._track_instances2results_trt(
                active_track_instances, with_mask=with_mask)
        mask = index_bool2long_trt(mask.bool())
        track_query_embeddings = track_instances[2][active_index][bbox_index][
            mask]
        track_query_matched_idxes = track_instances[4][active_index][
            bbox_index][mask]
        return (track_query_embeddings, track_query_matched_idxes,
                bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
                labels, track_scores, bbox_index, obj_idxes, mask,
                track_bbox_results1, track_bbox_results2)

    def _track_instances2results_trt(self, track_instances, with_mask=True):
        bbox_list = [
            track_instances[-4],
            track_instances[-5],
            track_instances[7],
            track_instances[3],
        ]
        (bboxes_dict_bboxes, scores, labels, track_scores, obj_idxes,
         bbox_index, bboxes_dict_mask) = self.bbox_coder_decode_trt(
             bbox_list, with_mask=with_mask)
        bottom_center = bboxes_dict_bboxes[:, :3]
        bboxes_gravity_center = torch.zeros_like(bottom_center)
        xy_mask = torch.zeros_like(bottom_center)
        xy_mask[:, 0:2] = torch.ones_like(xy_mask[:, 0:2])
        bboxes_gravity_center = (
            bboxes_gravity_center * (1 - xy_mask) + bottom_center * xy_mask)
        z_mask = torch.zeros_like(bottom_center)
        z_mask[:, 2] = torch.ones_like(z_mask[:, 2])
        bboxes_gravity_center = (
            bboxes_gravity_center * (1 - z_mask) + bottom_center * z_mask +
            bboxes_dict_bboxes[:, 3:6] * z_mask * 0.5)
        bboxes_yaw = bboxes_dict_bboxes[:, 6]
        return (bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
                labels, track_scores, bbox_index, obj_idxes,
                bboxes_dict_mask, scores, labels)

    def bbox_coder_decode_trt(self, bbox_list, with_mask=True):
        return self.decode_single_trt(
            bbox_list[0], bbox_list[1], bbox_list[2], bbox_list[3],
            with_mask)

    def decode_single_trt(self,
                          cls_scores,
                          bbox_preds,
                          track_scores,
                          obj_idxes,
                          with_mask=True):
        max_num = torch.minimum(
            torch.tensor(cls_scores.size(0)),
            torch.tensor(self.bbox_coder.max_num)).int()
        cls_scores = cls_scores.sigmoid()
        _, indexs = cls_scores.max(dim=-1)
        labels = indexs % self.bbox_coder.num_classes
        _, bbox_index = track_scores.topk(max_num)

        labels = labels[bbox_index]
        bbox_preds = bbox_preds[bbox_index]
        track_scores = track_scores[bbox_index]
        obj_idxes = obj_idxes[bbox_index]
        final_box_preds = denormalize_bbox_trt(
            bbox_preds, self.bbox_coder.pc_range)
        final_scores = track_scores
        final_preds = labels

        if self.bbox_coder.score_threshold is not None:
            thresh_mask = final_scores > self.bbox_coder.score_threshold

        self.bbox_coder.post_center_range = torch.tensor(
            self.bbox_coder.post_center_range, device=final_scores.device)
        mask = (final_box_preds[..., :3] >=
                self.bbox_coder.post_center_range[:3]).all(1)
        mask = mask & ((final_box_preds[..., :3] <=
                        self.bbox_coder.post_center_range[3:]).all(1))
        if self.bbox_coder.score_threshold:
            mask = mask & thresh_mask
        if not with_mask:
            mask = torch.ones_like(mask) > 0

        keep = index_bool2long_trt(mask)
        boxes3d = final_box_preds[keep]
        scores = final_scores[keep]
        labels = final_preds[keep]
        track_scores = track_scores[keep]
        obj_idxes = obj_idxes[keep]
        return (boxes3d, scores, labels, track_scores, obj_idxes, bbox_index,
                mask.int())

    def _generate_empty_tracks_trt(self):
        num_queries, dim = self.query_embedding.weight.shape
        device = self.query_embedding.weight.device
        query = self.query_embedding.weight
        query.requires_grad = False
        ref_pts = self.reference_points(query[..., :dim // 2]).to(device)

        pred_boxes_init = torch.zeros(
            (ref_pts.shape[0], 10), dtype=torch.float, device=device)
        output_embedding = torch.zeros(
            (num_queries, dim >> 1), device=device)
        obj_idxes = torch.full(
            (ref_pts.shape[0], ), -1, dtype=torch.int, device=device)
        matched_gt_idxes = torch.full(
            (ref_pts.shape[0], ), -1, dtype=torch.int, device=device)
        disappear_time = torch.zeros(
            (ref_pts.shape[0], ), dtype=torch.int, device=device)
        iou = torch.zeros((ref_pts.shape[0], ), dtype=torch.float,
                          device=device)
        scores = torch.zeros(
            (ref_pts.shape[0], ), dtype=torch.float, device=device)
        track_scores = torch.zeros(
            (ref_pts.shape[0], ), dtype=torch.float, device=device)
        pred_logits = torch.zeros(
            (ref_pts.shape[0], self.num_classes),
            dtype=torch.float,
            device=device)
        mem_bank = torch.zeros(
            (ref_pts.shape[0], self.mem_bank_len, dim // 2),
            dtype=torch.float32,
            device=device)
        mem_padding_mask = torch.ones(
            (ref_pts.shape[0], self.mem_bank_len),
            dtype=torch.int,
            device=device)
        save_period = torch.zeros(
            (ref_pts.shape[0], ), dtype=torch.float32, device=device)
        return [
            query, ref_pts, output_embedding, obj_idxes, matched_gt_idxes,
            disappear_time, iou, scores, track_scores, pred_boxes_init,
            pred_logits, mem_bank, mem_padding_mask, save_period
        ]

    def simple_test_track_preprocess_trt(self, prev_track_instances,
                                         prev_timestamp, prev_l2g_r_mat,
                                         prev_l2g_t, l2g_t, l2g_r_mat,
                                         scene_token_changed, timestamp):
        empty_track_instances = self._generate_empty_tracks_trt()

        def shape_len_1(prev, empty, changed):
            padded = torch.cat([
                empty,
                torch.ones(prev.shape[0] - empty.shape[0]).to(empty) *
                (-10**4)
            ],
                               dim=0)
            item = padded * changed + prev * (1 - changed)
            return item[index_bool2long_trt(item != (-10**4))]

        def shape_len_2(prev, empty, changed):
            padded = torch.cat([
                empty,
                torch.ones(prev.shape[0] - empty.shape[0],
                           empty.shape[1]).to(empty) * (-10**4)
            ],
                               dim=0)
            item = padded * changed + prev * (1 - changed)
            return item[index_bool2long_trt((item != (-10**4))[:, 0])]

        def shape_len_3(prev, empty, changed):
            padded = torch.cat([
                empty,
                torch.ones(prev.shape[0] - empty.shape[0], empty.shape[1],
                           empty.shape[2]).to(empty) * (-10**4)
            ],
                               dim=0)
            item = padded * changed + prev * (1 - changed)
            return item[index_bool2long_trt((item != (-10**4))[:, 0, 0])]

        changed = scene_token_changed.reshape(-1)[0].float()
        track_instances = []
        for i in range(0, 3):
            track_instances.append(
                shape_len_2(prev_track_instances[i], empty_track_instances[i],
                            changed))
        for i in range(3, 9):
            track_instances.append(
                shape_len_1(prev_track_instances[i], empty_track_instances[i],
                            changed))
        for i in range(9, 11):
            track_instances.append(
                shape_len_2(prev_track_instances[i], empty_track_instances[i],
                            changed))
        track_instances.append(
            shape_len_3(prev_track_instances[11], empty_track_instances[11],
                        changed))
        track_instances.append(
            shape_len_2(prev_track_instances[12], empty_track_instances[12],
                        changed))
        track_instances.append(
            shape_len_1(prev_track_instances[13], empty_track_instances[13],
                        changed))

        time_delta = (
            torch.zeros_like(timestamp)[0] * changed +
            (timestamp[0] - prev_timestamp[0]) * (1 - changed))
        identity = torch.eye(
            3, device=l2g_r_mat.device,
            dtype=l2g_r_mat.dtype).unsqueeze(0)
        l2g_r1 = identity * changed + prev_l2g_r_mat * (1 - changed)
        l2g_t1 = torch.zeros_like(l2g_t) * changed + prev_l2g_t * (
            1 - changed)
        l2g_r2 = identity * changed + l2g_r_mat * (1 - changed)
        l2g_t2 = torch.zeros_like(l2g_t) * changed + l2g_t * (1 - changed)
        return (time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2,
                [item.detach() for item in track_instances], timestamp,
                l2g_t, l2g_r_mat)

    def simple_test_track_trt(self, prev_track_intances, prev_timestamp,
                              prev_l2g_r_mat, prev_l2g_t, scene_token_changed,
                              timestamp, l2g_r_mat, l2g_t, prev_bev,
                              lidar_bev, shift, use_prev_bev, max_obj_id):
        (time_delta, l2g_r1, l2g_t1, l2g_r2, l2g_t2, track_instances,
         prev_timestamp, prev_l2g_t, prev_l2g_r_mat) = \
            self.simple_test_track_preprocess_trt(
                prev_track_intances, prev_timestamp, prev_l2g_r_mat,
                prev_l2g_t, l2g_t, l2g_r_mat, scene_token_changed,
                timestamp)

        (track_instances, bev_embed, bev_pos, output_classes, output_coords,
         all_past_traj_preds, last_ref_pts, query_feats) = \
            self._forward_single_frame_inference_trt(
                lidar_bev, track_instances, prev_bev, shift, l2g_r1, l2g_t1,
                l2g_r2, l2g_t2, time_delta, use_prev_bev)
        output_classes = output_classes.float().detach()
        output_coords = output_coords.float().detach()
        last_ref_pts = last_ref_pts.float().detach()
        query_feats = query_feats.float().detach()

        (track_query_embeddings, track_query_matched_idxes,
         bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
         labels, track_scores, bbox_index, obj_idxes, mask,
         track_bbox_results1, track_bbox_results2, max_obj_id_out,
         track_instances_fordet, track_instances) = self.track2mop_trt(
             track_instances, output_classes, output_coords, last_ref_pts,
             query_feats, max_obj_id)

        return (track_instances, prev_timestamp, prev_l2g_t, prev_l2g_r_mat,
                bev_embed, bev_pos, output_classes, output_coords,
                all_past_traj_preds, last_ref_pts, query_feats,
                track_query_embeddings, track_query_matched_idxes,
                bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
                labels, track_scores, bbox_index, obj_idxes, mask,
                track_bbox_results1, track_bbox_results2, max_obj_id_out,
                track_instances_fordet)

    def forward_track_lidar_trt(
        self,
        prev_track_intances0,
        prev_track_intances1,
        prev_track_intances2,
        prev_track_intances3,
        prev_track_intances4,
        prev_track_intances5,
        prev_track_intances6,
        prev_track_intances7,
        prev_track_intances8,
        prev_track_intances9,
        prev_track_intances10,
        prev_track_intances11,
        prev_track_intances12,
        prev_track_intances13,
        prev_timestamp,
        prev_l2g_r_mat,
        prev_l2g_t,
        prev_bev,
        lidar_bev,
        shift,
        timestamp,
        l2g_r_mat,
        l2g_t,
        use_prev_bev,
        max_obj_id,
    ):
        scene_token_changed = 1 - use_prev_bev
        prev_track_intances = [
            prev_track_intances0, prev_track_intances1,
            prev_track_intances2, prev_track_intances3,
            prev_track_intances4, prev_track_intances5,
            prev_track_intances6, prev_track_intances7,
            prev_track_intances8, prev_track_intances9,
            prev_track_intances10, prev_track_intances11,
            prev_track_intances12, prev_track_intances13
        ]
        (prev_track_instances_out, prev_timestamp_out, prev_l2g_t_out,
         prev_l2g_r_mat_out, bev_embed, bev_pos, output_classes,
         output_coords, all_past_traj_preds, last_ref_pts, query_feats,
         track_query_embeddings, track_query_matched_idxes,
         bboxes_dict_bboxes, bboxes_gravity_center, bboxes_yaw, scores,
         labels, track_scores, bbox_index, obj_idxes, mask,
         track_bbox_results1, track_bbox_results2, max_obj_id_out,
         track_instances_fordet) = self.simple_test_track_trt(
             prev_track_intances, prev_timestamp, prev_l2g_r_mat, prev_l2g_t,
             scene_token_changed, timestamp, l2g_r_mat, l2g_t, prev_bev,
             lidar_bev, shift, use_prev_bev, max_obj_id)

        (prev_track_intances0_out, prev_track_intances1_out,
         prev_track_intances2_out, prev_track_intances3_out,
         prev_track_intances4_out, prev_track_intances5_out,
         prev_track_intances6_out, prev_track_intances7_out,
         prev_track_intances8_out, prev_track_intances9_out,
         prev_track_intances10_out, prev_track_intances11_out,
         prev_track_intances12_out,
         prev_track_intances13_out) = prev_track_instances_out

        return (prev_track_intances0_out, prev_track_intances1_out,
                prev_track_intances3_out, prev_track_intances4_out,
                prev_track_intances5_out, prev_track_intances6_out,
                prev_track_intances8_out, prev_track_intances9_out,
                prev_track_intances11_out, prev_track_intances12_out,
                prev_track_intances13_out, prev_timestamp_out,
                prev_l2g_t_out, prev_l2g_r_mat_out, bev_embed,
                bboxes_dict_bboxes, scores, labels.int(), bbox_index.int(),
                obj_idxes.int(), max_obj_id_out.int())

    def forward(self, *args, **kwargs):
        return self.forward_track_lidar_trt(*args, **kwargs)


@DETECTORS.register_module()
class UniADTrackDrivableLidarTRT(UniADTrackLidarTRT):
    """Track TRT boundary with an additional drivable-mask head.

    The sparse encoder and LiDAR BEV encoder/track decoder stay identical to
    ``UniADTrackLidarTRT``.  This subclass only converts the exported
    ``bev_embed`` back to the segmentation-head convention and appends the
    drivable score map as a dense-engine output.
    """

    def __init__(self, seg_head=None, **kwargs):
        super().__init__(**kwargs)
        if seg_head is None:
            raise ValueError('UniADTrackDrivableLidarTRT requires seg_head.')
        self.seg_head = build_head(seg_head)

    @staticmethod
    def _bev_for_seg_head(bev_embed):
        if bev_embed.dim() == 3:
            return bev_embed
        if bev_embed.dim() != 4:
            raise ValueError('seg_head expects BEV shape [B, C, H, W] or '
                             f'[HW, B, C], got {tuple(bev_embed.shape)}.')
        batch_size, channels, bev_h, bev_w = bev_embed.shape
        return bev_embed.permute(2, 3, 0, 1).reshape(
            bev_h * bev_w, batch_size, channels).contiguous()

    def forward_track_drivable_lidar_trt(
        self,
        prev_track_intances0,
        prev_track_intances1,
        prev_track_intances2,
        prev_track_intances3,
        prev_track_intances4,
        prev_track_intances5,
        prev_track_intances6,
        prev_track_intances7,
        prev_track_intances8,
        prev_track_intances9,
        prev_track_intances10,
        prev_track_intances11,
        prev_track_intances12,
        prev_track_intances13,
        prev_timestamp,
        prev_l2g_r_mat,
        prev_l2g_t,
        prev_bev,
        lidar_bev,
        shift,
        timestamp,
        l2g_r_mat,
        l2g_t,
        use_prev_bev,
        max_obj_id,
    ):
        track_outputs = super().forward_track_lidar_trt(
            prev_track_intances0, prev_track_intances1,
            prev_track_intances2, prev_track_intances3,
            prev_track_intances4, prev_track_intances5,
            prev_track_intances6, prev_track_intances7,
            prev_track_intances8, prev_track_intances9,
            prev_track_intances10, prev_track_intances11,
            prev_track_intances12, prev_track_intances13, prev_timestamp,
            prev_l2g_r_mat, prev_l2g_t, prev_bev, lidar_bev, shift,
            timestamp, l2g_r_mat, l2g_t, use_prev_bev, max_obj_id)
        bev_embed = track_outputs[14]
        drivable_score = self.seg_head.forward_test_trt(
            self._bev_for_seg_head(bev_embed))
        return track_outputs + (drivable_score, )

    def forward(self, *args, **kwargs):
        return self.forward_track_drivable_lidar_trt(*args, **kwargs)
