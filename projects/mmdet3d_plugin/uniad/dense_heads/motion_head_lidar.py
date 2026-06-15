import torch
from mmdet.models import HEADS

from .motion_head import MotionHead
from .motion_head_plugin.map_lane_encoder import MapLaneEncoder


@HEADS.register_module()
class MotionHeadLidar(MotionHead):
    """LiDAR-only MotionHead variant.

    It keeps UniAD's trajectory decoder/loss, but trains only on active
    object track queries.  SDC and map lane queries are intentionally omitted
    for the first LiDAR motion stage.
    """

    def __init__(self, *args, map_lane_encoder=None, **kwargs):
        super().__init__(*args, **kwargs)
        if map_lane_encoder is not None:
            self.map_lane_encoder = MapLaneEncoder(
                pc_range=self.pc_range,
                embed_dims=self.embed_dims,
                **map_lane_encoder)
        else:
            self.map_lane_encoder = None

    def _build_lane_query(self, track_query, outs_track):
        if self.map_lane_encoder is None:
            return (track_query.new_zeros((1, 0, self.embed_dims)),
                    track_query.new_zeros((1, 0, self.embed_dims)))
        ego2global = outs_track.get('ego2global')
        if ego2global is None:
            return (track_query.new_zeros((1, 0, self.embed_dims)),
                    track_query.new_zeros((1, 0, self.embed_dims)))
        return self.map_lane_encoder(
            ego2global, device=track_query.device, dtype=track_query.dtype)

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

        # Append SDC query/GT to the tail so MotionFormer attends to it
        # alongside the active object queries; it is split back out below.
        with_sdc = (
            'sdc_embedding' in outs_track
            and gt_sdc_fut_traj is not None
            and gt_sdc_fut_traj_mask is not None)
        if with_sdc:
            sdc_match_index = torch.zeros(
                (1,),
                dtype=all_matched_idxes[0].dtype,
                device=all_matched_idxes[0].device)
            sdc_match_index[0] = gt_fut_traj[0].shape[0]
            all_matched_idxes = [
                torch.cat([all_matched_idxes[0], sdc_match_index], dim=0)
            ]
            gt_fut_traj[0] = torch.cat(
                [gt_fut_traj[0], gt_sdc_fut_traj[0]], dim=0)
            gt_fut_traj_mask[0] = torch.cat(
                [gt_fut_traj_mask[0], gt_sdc_fut_traj_mask[0]], dim=0)
            track_query = torch.cat(
                [track_query, outs_track['sdc_embedding'][None, None, None, :]],
                dim=2)
            sdc_track_boxes = outs_track['sdc_track_bbox_results']
            track_boxes[0][0].tensor = torch.cat(
                [track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor],
                dim=0)
            track_boxes[0][1] = torch.cat(
                [track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
            track_boxes[0][2] = torch.cat(
                [track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
            track_boxes[0][3] = torch.cat(
                [track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

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

        lane_query, lane_query_pos = self._build_lane_query(
            track_query, outs_track)

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

        # Split SDC slot back out so downstream planning can consume it,
        # leaving traj/track_query as object-only for the vehicle filter.
        if with_sdc:
            all_matched_idxes[0] = all_matched_idxes[0][:-1]
            outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
            outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
            outs_motion['sdc_track_query_pos'] = (
                outs_motion['track_query_pos'][:, -1])
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
            outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
            outs_motion['track_query_pos'] = (
                outs_motion['track_query_pos'][:, :-1])

        def filter_vehicle_query(outs_motion, all_matched_idxes,
                                 gt_labels_3d, vehicle_id_list):
            if all_matched_idxes[0].numel() == 0:
                outs_motion['all_matched_idxes'] = all_matched_idxes
                return outs_motion, all_matched_idxes
            matched_gt = all_matched_idxes[0]
            gt_labels = gt_labels_3d[0][-1].to(matched_gt.device)
            valid_match = (
                (matched_gt >= 0)
                & (matched_gt < gt_labels.numel()))
            vehicle_mask = torch.zeros_like(matched_gt, dtype=torch.bool)
            if valid_match.any():
                query_label = gt_labels[matched_gt[valid_match]]
                valid_vehicle_mask = torch.zeros_like(
                    query_label, dtype=torch.bool)
                for veh_id in vehicle_id_list:
                    valid_vehicle_mask |= query_label == veh_id
                vehicle_mask[valid_match] = valid_vehicle_mask
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

        with_sdc = (
            'sdc_embedding' in outs_track
            and outs_track.get('sdc_embedding') is not None)
        if with_sdc:
            track_query = torch.cat(
                [track_query, outs_track['sdc_embedding'][None, None, None, :]],
                dim=2)
            sdc_track_boxes = outs_track['sdc_track_bbox_results']
            track_boxes[0][0].tensor = torch.cat(
                [track_boxes[0][0].tensor, sdc_track_boxes[0][0].tensor],
                dim=0)
            track_boxes[0][1] = torch.cat(
                [track_boxes[0][1], sdc_track_boxes[0][1]], dim=0)
            track_boxes[0][2] = torch.cat(
                [track_boxes[0][2], sdc_track_boxes[0][2]], dim=0)
            track_boxes[0][3] = torch.cat(
                [track_boxes[0][3], sdc_track_boxes[0][3]], dim=0)

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

        lane_query, lane_query_pos = self._build_lane_query(
            track_query, outs_track)
        outs_motion = self(
            bev_embed,
            track_query,
            lane_query,
            lane_query_pos,
            track_boxes)
        traj_results = self.get_trajs(outs_motion, track_boxes)
        _, scores, labels, _, _ = track_boxes[0]
        outs_motion['track_scores'] = scores[None, :]

        # Split SDC out before vehicle filter (camera's [-1]=0 trick relies
        # on its vehicle_id_list including 0; ours does not, so we remove
        # the SDC slot explicitly and filter the remainder).
        if with_sdc:
            outs_motion['sdc_traj_query'] = outs_motion['traj_query'][:, :, -1]
            outs_motion['sdc_track_query'] = outs_motion['track_query'][:, -1]
            outs_motion['sdc_track_query_pos'] = (
                outs_motion['track_query_pos'][:, -1])
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :, :-1]
            outs_motion['track_query'] = outs_motion['track_query'][:, :-1]
            outs_motion['track_query_pos'] = (
                outs_motion['track_query_pos'][:, :-1])
            outs_motion['track_scores'] = outs_motion['track_scores'][:, :-1]
            labels = labels[:-1]

        if labels.numel() > 0:
            vehicle_mask = torch.zeros_like(labels, dtype=torch.bool)
            for veh_id in self.vehicle_id_list:
                vehicle_mask |= labels == veh_id
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :,
                                                                  vehicle_mask]
            outs_motion['track_query'] = outs_motion['track_query'][:,
                                                                    vehicle_mask]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:,
                                                                            vehicle_mask]
            outs_motion['track_scores'] = outs_motion['track_scores'][:,
                                                                      vehicle_mask]

        return traj_results, outs_motion
