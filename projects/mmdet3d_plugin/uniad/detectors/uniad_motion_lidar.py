import torch
from mmdet.models import DETECTORS, build_head

from .uniad_track_lidar import UniADTrackLidarTRT


@DETECTORS.register_module()
class UniADMotionLidarTRT(UniADTrackLidarTRT):
    """LiDAR-only TensorRT export boundary for tracking plus MotionHead."""

    def __init__(self,
                 motion_head=None,
                 planning_head=None,
                 task_loss_weight=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.motion_head = build_head(motion_head) if motion_head else None
        self.planning_head = build_head(planning_head) if planning_head else None
        self.task_loss_weight = task_loss_weight or dict(
            track=1.0, motion=1.0)

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_planning_head(self):
        return hasattr(self, 'planning_head') and self.planning_head is not None

    @staticmethod
    def _bev_for_motion_head_trt(bev_embed):
        return bev_embed.flatten(2).permute(2, 0, 1).contiguous()

    def _empty_lane_query_trt(self, bev_embed):
        return (bev_embed.new_zeros((1, 0, self.motion_head.embed_dims)),
                bev_embed.new_zeros((1, 0, self.motion_head.embed_dims)))

    def select_sdc_track_query_trt(self, track_instances):
        if not self.with_sdc or self.sdc_query_index is None:
            raise RuntimeError('LiDAR motion TRT export expects with_sdc=True.')

        sdc_index = torch.tensor(
            [self.sdc_query_index],
            dtype=torch.long,
            device=track_instances[0].device)
        sdc_instance = [item[sdc_index] for item in track_instances]
        (_, sdc_gravity_center, sdc_yaw, _, _, _, _, _, _, sdc_scores,
         sdc_labels) = self._track_instances2results_trt(
             sdc_instance, with_mask=False)
        sdc_embedding = track_instances[2][sdc_index][0]
        return (sdc_embedding, sdc_gravity_center, sdc_yaw, sdc_scores,
                sdc_labels)

    def forward_e2e_lidar_trt(
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
        if not self.with_motion_head:
            raise RuntimeError('UniADMotionLidarTRT requires motion_head.')

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

        (sdc_embedding, sdc_gravity_center, sdc_yaw, sdc_track_scores,
         sdc_labels) = self.select_sdc_track_query_trt(track_instances_fordet)

        bev_motion = self._bev_for_motion_head_trt(bev_embed)
        lane_query, lane_query_pos = self._empty_lane_query_trt(bev_motion)
        outputs_traj_scores, outputs_trajs, valid_traj_masks = \
            self.motion_head.forward_test_trt(
                bev_motion.float().detach(),
                track_query_embeddings.float().detach(),
                track_bbox_results1.float().detach(),
                track_bbox_results2.long().detach(),
                bboxes_gravity_center.float().detach(),
                bboxes_yaw.float().detach(),
                sdc_embedding.float().detach(),
                sdc_gravity_center.float().detach(),
                sdc_yaw.float().detach(),
                sdc_track_scores.float().detach(),
                sdc_labels.long().detach(),
                lane_query.float().detach(),
                lane_query_pos.float().detach())

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
                obj_idxes.int(), max_obj_id_out.int(),
                outputs_traj_scores[0, 0], outputs_trajs[0, 0],
                outputs_traj_scores[1, 0], outputs_trajs[1, 0],
                outputs_traj_scores[-1, 0], outputs_trajs[-1, 0],
                valid_traj_masks[0])

    def forward_e2e_lidar_plan_trt(
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
        command,
    ):
        if not self.with_motion_head:
            raise RuntimeError('UniADMotionLidarTRT requires motion_head.')
        if not self.with_planning_head:
            raise RuntimeError('LiDAR plan TRT export requires planning_head.')

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

        (sdc_embedding, sdc_gravity_center, sdc_yaw, sdc_track_scores,
         sdc_labels) = self.select_sdc_track_query_trt(track_instances_fordet)

        bev_motion = self._bev_for_motion_head_trt(bev_embed)
        lane_query, lane_query_pos = self._empty_lane_query_trt(bev_motion)
        (outputs_traj_scores, outputs_trajs, valid_traj_masks,
         inter_states, out_track_query, track_query_pos, sdc_traj_query,
         sdc_track_query, sdc_track_query_pos, motion_track_scores) = \
            self.motion_head.forward_test_trt_with_queries(
                bev_motion.float().detach(),
                track_query_embeddings.float().detach(),
                track_bbox_results1.float().detach(),
                track_bbox_results2.long().detach(),
                bboxes_gravity_center.float().detach(),
                bboxes_yaw.float().detach(),
                sdc_embedding.float().detach(),
                sdc_gravity_center.float().detach(),
                sdc_yaw.float().detach(),
                sdc_track_scores.float().detach(),
                sdc_labels.long().detach(),
                lane_query.float().detach(),
                lane_query_pos.float().detach())

        seg_out = bev_motion.new_zeros(
            (1, 1, 1, self.planning_head.bev_h, self.planning_head.bev_w))
        bev_pos_plan = bev_pos.permute(0, 2, 1).reshape(
            bev_pos.shape[0], bev_pos.shape[2], self.planning_head.bev_h,
            self.planning_head.bev_w)
        sdc_traj = self.planning_head.forward_test_trt(
            bev_motion.float().detach(),
            sdc_traj_query.float().detach(),
            sdc_track_query.float().detach(),
            bev_pos_plan.float().detach(),
            seg_out,
            [command.detach().long()])

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
                obj_idxes.int(), max_obj_id_out.int(),
                outputs_traj_scores[0, 0], outputs_trajs[0, 0],
                outputs_traj_scores[1, 0], outputs_trajs[1, 0],
                outputs_traj_scores[-1, 0], outputs_trajs[-1, 0],
                valid_traj_masks[0], sdc_traj)

    def forward(self, *args, **kwargs):
        return self.forward_e2e_lidar_trt(*args, **kwargs)
