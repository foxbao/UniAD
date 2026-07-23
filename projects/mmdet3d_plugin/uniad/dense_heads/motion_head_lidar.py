import torch
from mmdet.models import HEADS

from .motion_head import MotionHead


@HEADS.register_module()
class MotionHeadLidar(MotionHead):
    """LiDAR-only MotionHead variant.

    It keeps UniAD's trajectory decoder/loss, trains object track queries, and
    appends the SDC query when SDC supervision is available.

    The HD-map lane prior is built and owned by the detector
    (UniADMotionLidar); this head is a pure consumer that reads lane_query
    from the outs_map dict, mirroring how camera UniAD's MotionHead consumes
    the seg head's outs_seg.

    map_local_k: if set, each map-enabled agent attends only its K-nearest
    valid lanes (MTR-style local map collection) instead of the global lane
    set. None (default) keeps the global behavior.
    map_agent_scope:
        all: every agent may attend to the map.
        sdc_only: only the appended SDC query may attend to the map.
        none: all agents use the no-map branch.
    """

    def __init__(self, *args, map_local_k=None, map_agent_scope='all',
                 **kwargs):
        super().__init__(*args, **kwargs)
        if map_agent_scope not in ('all', 'sdc_only', 'none'):
            raise ValueError(
                'map_agent_scope must be one of all/sdc_only/none, got '
                f'{map_agent_scope}')
        self.map_local_k = map_local_k
        self.map_agent_scope = map_agent_scope

    def _map_agent_mask(self, track_query, with_sdc):
        """Return a bool (B, A) mask for agents allowed to attend map lanes."""
        if self.map_agent_scope == 'all':
            return None
        batch_size, num_agents = track_query.shape[0], track_query.shape[2]
        mask = torch.zeros(
            (batch_size, num_agents),
            dtype=torch.bool,
            device=track_query.device)
        if self.map_agent_scope == 'sdc_only' and with_sdc and num_agents > 0:
            mask[:, -1] = True
        return mask

    def _lane_inputs(self, track_query, outs_map):
        """Read lane_query from the detector-built outs_map dict.

        Falls back to empty lane tensors (the shapes MotionFormer already
        tolerates) when no HD-map lane prior is available. Returns
        (lane_query, lane_query_pos, lane_valid, lane_centroids).
        """
        outs_map = outs_map or {}
        lane_query = outs_map.get('lane_query')
        if lane_query is None:
            return (track_query.new_zeros((1, 0, self.embed_dims)),
                    track_query.new_zeros((1, 0, self.embed_dims)), None, None)
        return (lane_query, outs_map['lane_query_pos'],
                outs_map['lane_valid'], outs_map.get('lane_centroids'))

    def _load_anchors(self, anchor_info_path):
        super()._load_anchors(anchor_info_path)
        if self.kmeans_anchors.size(0) != self.num_anchor_group:
            self.kmeans_anchors = self.kmeans_anchors[:self.num_anchor_group]
        if self.kmeans_anchors.size(2) != self.predict_steps:
            self.kmeans_anchors = self.kmeans_anchors[:, :, :self.predict_steps]

    def _attach_planning_actor_context(self, outs_motion, track_boxes,
                                       actor_mask, with_sdc):
        """Expose detached online actor futures for collision-aware planning."""
        all_scores = outs_motion.get('all_traj_scores')
        all_preds = outs_motion.get('all_traj_preds')
        if all_scores is None or all_preds is None:
            return

        scores = all_scores[-1]
        preds = all_preds[-1]
        batch_size = scores.size(0)
        if isinstance(actor_mask, (list, tuple)):
            actor_masks = list(actor_mask)
        elif batch_size == 1:
            actor_masks = [actor_mask]
        else:
            actor_masks = [actor_mask[index] for index in range(batch_size)]

        future_list = []
        size_list = []
        yaw_list = []
        score_list = []
        for batch_index in range(batch_size):
            boxes, track_scores = track_boxes[batch_index][:2]
            num_objects = scores.size(1) - int(with_sdc)
            num_objects = min(num_objects, len(boxes.tensor))
            mask = actor_masks[batch_index].to(
                device=scores.device, dtype=torch.bool).reshape(-1)
            num_objects = min(num_objects, mask.numel())
            mask = mask[:num_objects]

            if num_objects == 0 or not mask.any():
                future_list.append(preds.new_zeros(
                    (0, preds.size(-2), 2)))
                size_list.append(preds.new_zeros((0, 2)))
                yaw_list.append(preds.new_zeros((0,)))
                score_list.append(preds.new_zeros((0,)))
                continue

            object_scores = scores[batch_index, :num_objects]
            object_preds = preds[batch_index, :num_objects]
            mode_index = object_scores.argmax(dim=-1)
            gather_index = mode_index[:, None, None, None].expand(
                -1, 1, object_preds.size(-2), object_preds.size(-1))
            selected_preds = object_preds.gather(
                1, gather_index).squeeze(1)[..., :2]

            centers = boxes.gravity_center[:num_objects, :2].to(
                device=preds.device, dtype=preds.dtype)
            box_tensor = boxes.tensor[:num_objects].to(
                device=preds.device, dtype=preds.dtype)
            # mmdet3d LiDAR boxes store length/width in tensor columns 3/4.
            sizes = box_tensor[:, 3:5].abs().clamp_min(0.1)
            yaws = boxes.yaw[:num_objects].to(
                device=preds.device, dtype=preds.dtype)
            detection_scores = track_scores[:num_objects].to(
                device=preds.device, dtype=preds.dtype).clamp(0.0, 1.0)
            mode_scores = object_scores.max(dim=-1).values.exp().clamp(
                0.0, 1.0)

            future_list.append(
                (centers[:, None] + selected_preds)[mask])
            size_list.append(sizes[mask])
            yaw_list.append(yaws[mask])
            score_list.append((detection_scores * mode_scores)[mask])

        max_actors = max((value.size(0) for value in future_list), default=0)
        planning_steps = preds.size(-2)
        actor_future = preds.new_zeros(
            (batch_size, max_actors, planning_steps, 2))
        actor_sizes = preds.new_zeros((batch_size, max_actors, 2))
        actor_yaws = preds.new_zeros((batch_size, max_actors))
        actor_scores = preds.new_zeros((batch_size, max_actors))
        actor_valid = torch.zeros(
            (batch_size, max_actors), device=preds.device, dtype=torch.bool)
        for batch_index, future in enumerate(future_list):
            count = future.size(0)
            if count == 0:
                continue
            actor_future[batch_index, :count] = future
            actor_sizes[batch_index, :count] = size_list[batch_index]
            actor_yaws[batch_index, :count] = yaw_list[batch_index]
            actor_scores[batch_index, :count] = score_list[batch_index]
            actor_valid[batch_index, :count] = True

        outs_motion['planning_actor_future'] = actor_future.detach()
        outs_motion['planning_actor_sizes'] = actor_sizes.detach()
        outs_motion['planning_actor_yaws'] = actor_yaws.detach()
        outs_motion['planning_actor_scores'] = actor_scores.detach()
        outs_motion['planning_actor_valid'] = actor_valid

    def forward_train(self,
                      bev_embed,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      outs_track=None,
                      outs_seg=None,
                      outs_map=None):
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

        lane_query, lane_query_pos, lane_valid, lane_centroids = \
            self._lane_inputs(track_query, outs_map)
        outs_motion = self(
            bev_embed,
            track_query,
            lane_query,
            lane_query_pos,
            track_boxes,
            lane_key_padding_mask=(
                None if lane_valid is None else ~lane_valid),
            lane_centroids=lane_centroids,
            map_agent_mask=self._map_agent_mask(track_query, with_sdc))
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
            self._attach_planning_actor_context(
                outs_motion, track_boxes, vehicle_mask, with_sdc)
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

    def forward_test(self, bev_embed, outs_track=None, outs_seg=None,
                     outs_map=None):
        """LiDAR-only motion prediction for online tracking results."""
        outs_track = outs_track or {}
        track_query = outs_track['track_query_embeddings'][None, None, ...]
        # Appending the SDC slot below must not mutate the tracker result:
        # ``boxes_3d`` and ``track_bbox_results`` share the same box object.
        # Keep the exported TrackFormer boxes and scores aligned for other
        # consumers such as online OccWorld instance anchoring.
        track_boxes = [[
            value.clone() if hasattr(value, 'clone') else value
            for value in row
        ] for row in outs_track['track_bbox_results']]

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

        lane_query, lane_query_pos, lane_valid, lane_centroids = \
            self._lane_inputs(track_query, outs_map)
        outs_motion = self(
            bev_embed,
            track_query,
            lane_query,
            lane_query_pos,
            track_boxes,
            lane_key_padding_mask=(
                None if lane_valid is None else ~lane_valid),
            lane_centroids=lane_centroids,
            map_agent_mask=self._map_agent_mask(track_query, with_sdc))
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

        vehicle_mask = torch.zeros_like(labels, dtype=torch.bool)
        if labels.numel() > 0:
            for veh_id in self.vehicle_id_list:
                vehicle_mask |= labels == veh_id
        self._attach_planning_actor_context(
            outs_motion, track_boxes, vehicle_mask, with_sdc)
        if labels.numel() > 0:
            outs_motion['traj_query'] = outs_motion['traj_query'][:, :,
                                                                  vehicle_mask]
            outs_motion['track_query'] = outs_motion['track_query'][:,
                                                                    vehicle_mask]
            outs_motion['track_query_pos'] = outs_motion['track_query_pos'][:,
                                                                            vehicle_mask]
            outs_motion['track_scores'] = outs_motion['track_scores'][:,
                                                                      vehicle_mask]

        return traj_results, outs_motion
