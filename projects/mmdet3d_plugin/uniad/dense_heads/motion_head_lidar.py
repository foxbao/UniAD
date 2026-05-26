import torch
from mmdet.models import HEADS

from .motion_head import MotionHead


@HEADS.register_module()
class MotionHeadLidar(MotionHead):
    """LiDAR-only MotionHead variant.

    It keeps UniAD's trajectory decoder/loss, but trains only on active
    object track queries.  SDC and map lane queries are intentionally omitted
    for the first LiDAR motion stage.
    """

    def _load_anchors(self, anchor_info_path):
        super()._load_anchors(anchor_info_path)
        if self.kmeans_anchors.size(0) != self.num_anchor_group:
            self.kmeans_anchors = self.kmeans_anchors[:self.num_anchor_group]
        if self.kmeans_anchors.size(2) != self.predict_steps:
            self.kmeans_anchors = self.kmeans_anchors[:, :, :self.predict_steps]

    def forward_train(self,
                      bev_embed,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      outs_track=None,
                      outs_seg=None):
        outs_track = outs_track or {}
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        all_matched_idxes = [outs_track['track_query_matched_idxes']]
        track_boxes = outs_track['track_bbox_results']

        if track_query.size(2) == 0:
            zero = bev_embed.sum() * 0
            losses = dict(
                loss_traj=zero,
                l_class=zero.detach(),
                l_reg=zero.detach(),
                min_ade=zero.detach(),
                min_fde=zero.detach(),
                mr=zero.detach())
            for layer_id in range(self.motionformer.num_layers - 1):
                losses[f'd{layer_id}.loss_traj'] = zero
                losses[f'd{layer_id}.l_class'] = zero.detach()
                losses[f'd{layer_id}.l_reg'] = zero.detach()
                losses[f'd{layer_id}.min_ade'] = zero.detach()
                losses[f'd{layer_id}.min_fde'] = zero.detach()
                losses[f'd{layer_id}.mr'] = zero.detach()
            outs_motion = dict(
                all_matched_idxes=all_matched_idxes,
                track_query=track_query.new_zeros((1, 0, self.embed_dims)),
                track_query_pos=track_query.new_zeros((1, 0, self.embed_dims)),
                traj_query=track_query.new_zeros(
                    (self.motionformer.num_layers, 1, 0, self.num_anchor,
                     self.embed_dims)))
            return dict(
                losses=losses,
                outs_motion=outs_motion,
                track_boxes=track_boxes)

        lane_query = track_query.new_zeros((1, 0, self.embed_dims))
        lane_query_pos = track_query.new_zeros((1, 0, self.embed_dims))

        outs_motion = self(
            bev_embed,
            track_query,
            lane_query,
            lane_query_pos,
            track_boxes)
        loss_inputs = [
            gt_bboxes_3d, gt_fut_traj, gt_fut_traj_mask, outs_motion,
            all_matched_idxes, track_boxes
        ]
        losses = self.loss(*loss_inputs)

        def filter_vehicle_query(outs_motion, all_matched_idxes,
                                 gt_labels_3d, vehicle_id_list):
            if all_matched_idxes[0].numel() == 0:
                outs_motion['all_matched_idxes'] = all_matched_idxes
                return outs_motion, all_matched_idxes
            query_label = gt_labels_3d[0][-1][all_matched_idxes[0]]
            vehicle_mask = torch.zeros_like(query_label, dtype=torch.bool)
            for veh_id in vehicle_id_list:
                vehicle_mask |= query_label == veh_id
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :,
                                                                  vehicle_mask]
            outs_motion['track_query'] = outs_motion['track_query'][:,
                                                                    vehicle_mask]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:,
                                                                            vehicle_mask]
            all_matched_idxes[0] = all_matched_idxes[0][vehicle_mask]
            return outs_motion, all_matched_idxes

        outs_motion, all_matched_idxes = filter_vehicle_query(
            outs_motion, all_matched_idxes, gt_labels_3d,
            self.vehicle_id_list)
        outs_motion['all_matched_idxes'] = all_matched_idxes
        return dict(
            losses=losses,
            outs_motion=outs_motion,
            track_boxes=track_boxes)

    def forward_test(self, bev_embed, outs_track=None, outs_seg=None):
        """LiDAR-only motion prediction for online tracking results."""
        outs_track = outs_track or {}
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        track_boxes = outs_track['track_bbox_results']

        if track_query.size(2) == 0:
            empty_traj = track_query.new_zeros(
                (0, self.num_anchor, self.predict_steps, 5))
            empty_scores = track_query.new_zeros((0, self.num_anchor))
            outs_motion = dict(
                track_query=track_query.new_zeros((1, 0, self.embed_dims)),
                track_query_pos=track_query.new_zeros((1, 0, self.embed_dims)),
                traj_query=track_query.new_zeros(
                    (self.motionformer.num_layers, 1, 0, self.num_anchor,
                     self.embed_dims)))
            return [dict(traj=empty_traj.cpu(),
                         traj_scores=empty_scores.cpu())], outs_motion

        lane_query = track_query.new_zeros((1, 0, self.embed_dims))
        lane_query_pos = track_query.new_zeros((1, 0, self.embed_dims))
        outs_motion = self(
            bev_embed,
            track_query,
            lane_query,
            lane_query_pos,
            track_boxes)
        traj_results = self.get_trajs(outs_motion, track_boxes)
        return traj_results, outs_motion
