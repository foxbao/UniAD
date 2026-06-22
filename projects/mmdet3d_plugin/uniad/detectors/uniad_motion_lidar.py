import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_head

from ..dense_heads.motion_head_plugin.map_lane_encoder import MapLaneEncoder
from .uniad_track_lidar import UniADTrackLidar


@DETECTORS.register_module()
class UniADMotionLidar(UniADTrackLidar):
    """LiDAR-only stage-2 model for tracking plus motion prediction."""

    def __init__(self,
                 motion_head=None,
                 occ_head=None,
                 planning_head=None,
                 llm_head=None,
                 task_loss_weight=None,
                 map_lane_encoder=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.motion_head = build_head(motion_head) if motion_head else None
        self.occ_head = build_head(occ_head) if occ_head else None
        self.planning_head = (
            build_head(planning_head) if planning_head else None)
        self.llm_head = build_head(llm_head) if llm_head else None
        self.task_loss_weight = task_loss_weight or dict(
            track=1.0, motion=1.0, occ=1.0, planning=1.0)

        # HD-map lane prior. It is a detector-level module (an external survey
        # input, not a perception output), encoding surveyed lanes into
        # lane_query that MotionFormer's MapInteraction consumes via outs_map.
        # Dims are sourced from the motion head so they always match.
        self.map_lane_encoder = None
        if map_lane_encoder is not None:
            assert self.motion_head is not None, (
                'map_lane_encoder requires a motion_head to consume lane_query')
            self.map_lane_encoder = MapLaneEncoder(
                pc_range=self.motion_head.pc_range,
                embed_dims=self.motion_head.embed_dims,
                **map_lane_encoder)

    def _build_outs_map(self, bev_embed, ego2global):
        """Encode HD-map lanes for the current frame into an outs_map dict.

        Returns lane_query=None when no encoder is configured or no
        ego2global is available, so the motion head's consumer path stays
        uniform (it falls back to empty lane tensors).
        """
        if self.map_lane_encoder is None or ego2global is None:
            return dict(lane_query=None, lane_query_pos=None, lane_valid=None,
                        lane_centroids=None)
        lane_query, lane_query_pos, lane_valid, lane_centroids = \
            self.map_lane_encoder(
                ego2global, device=bev_embed.device, dtype=bev_embed.dtype)
        return dict(lane_query=lane_query, lane_query_pos=lane_query_pos,
                    lane_valid=lane_valid, lane_centroids=lane_centroids)

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @property
    def with_occ_head(self):
        return hasattr(self, 'occ_head') and self.occ_head is not None

    @property
    def with_planning_head(self):
        return (hasattr(self, 'planning_head')
                and self.planning_head is not None)

    @property
    def with_llm_head(self):
        return hasattr(self, 'llm_head') and self.llm_head is not None

    def _current_caption(self, img_metas):
        """Extract the current-frame VLM caption from img_metas.

        img_metas for this stack is an integer-keyed metas_map
        ({0:..., N-1:{... 'gt_caption':...}}); the caption sits on the last
        frame. Mirrors _current_frame_ego2global's unwrapping of the nested
        list/dict layouts. Returns None when absent.
        """
        if img_metas is None:
            return None
        while (isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and
               isinstance(img_metas[0], (list, tuple))):
            img_metas = img_metas[0]
        if isinstance(img_metas, (list, tuple)):
            meta = img_metas[-1] if img_metas else None
        else:
            meta = img_metas
        if not isinstance(meta, dict):
            return None
        if 'gt_caption' in meta:
            return meta['gt_caption']
        # integer-keyed metas_map: caption lives on the last frame
        if meta and all(isinstance(k, int) for k in meta.keys()):
            last = meta[max(meta.keys())]
            if isinstance(last, dict):
                return last.get('gt_caption')
        return None

    @staticmethod
    def _bev_for_motion_head(bev_embed):
        if bev_embed.dim() == 4:
            return bev_embed.flatten(2).permute(2, 0, 1).contiguous()
        if bev_embed.dim() == 3:
            return bev_embed
        raise ValueError('MotionHead expects BEV shape [HW, B, C] or '
                         f'[B, C, H, W], got {tuple(bev_embed.shape)}.')

    @staticmethod
    def _bev_pos_for_planning_head(bev_pos, bev_h, bev_w):
        """Reshape [B, HW, C] -> [B, C, H, W] for PlanningHead.

        The lidar tracker's LearnedBEVPositionalEncoding already flattens
        spatial dims, while PlanningHead's forward calls
        ``rearrange(bev_pos, 'b c h w -> (h w) b c')`` which assumes the
        4D camera convention.  Restore the 4D layout to keep the head
        unchanged.
        """
        if bev_pos.dim() == 4:
            return bev_pos
        if bev_pos.dim() == 3:
            b, hw, c = bev_pos.shape
            assert hw == bev_h * bev_w, (
                f'bev_pos length {hw} != bev_h*bev_w '
                f'{bev_h}*{bev_w}={bev_h * bev_w}')
            return bev_pos.permute(0, 2, 1).reshape(b, c, bev_h, bev_w)
        raise ValueError('bev_pos must be [B, HW, C] or [B, C, H, W], '
                         f'got {tuple(bev_pos.shape)}.')

    def loss_weighted_and_prefixed(self, loss_dict, prefix=''):
        loss_factor = self.task_loss_weight.get(prefix, 1.0)
        return {
            f'{prefix}.{key}': value * loss_factor
            for key, value in loss_dict.items()
        }

    def _current_frame_ego2global(self, img_metas):
        """Extract ego2global from img_metas for the current frame."""
        if img_metas is None:
            return None
        while (isinstance(img_metas, (list, tuple)) and len(img_metas) == 1 and
               isinstance(img_metas[0], (list, tuple))):
            img_metas = img_metas[0]
        if isinstance(img_metas, (list, tuple)):
            meta = img_metas[-1] if img_metas else None
        else:
            meta = img_metas
        if meta is None:
            return None
        if isinstance(meta, dict) and 'ego2global' in meta:
            return meta['ego2global']
        if isinstance(meta, dict) and 'queue_metas' in meta:
            qm = meta['queue_metas']
            if isinstance(qm, dict) and qm:
                last = qm[max(qm.keys())]
                if isinstance(last, dict):
                    return last.get('ego2global')
            if isinstance(qm, (list, tuple)) and qm:
                last = qm[-1]
                if isinstance(last, dict):
                    return last.get('ego2global')
        if isinstance(meta, dict) and meta and all(
                isinstance(key, int) for key in meta.keys()):
            last = meta[max(meta.keys())]
            if isinstance(last, dict):
                return last.get('ego2global')
        return None

    def _fill_empty_occ_query(self, outs_motion, bev_embed):
        if outs_motion['track_query'].shape[1] != 0:
            return outs_motion
        embed_dims = self.motion_head.embed_dims
        num_layers = self.motion_head.motionformer.num_layers
        num_anchor = self.motion_head.num_anchor
        outs_motion['track_query'] = torch.zeros(
            (1, 1, embed_dims), device=bev_embed.device, dtype=bev_embed.dtype)
        outs_motion['track_query_pos'] = torch.zeros_like(
            outs_motion['track_query'])
        outs_motion['traj_query'] = torch.zeros(
            (num_layers, 1, 1, num_anchor, embed_dims),
            device=bev_embed.device,
            dtype=bev_embed.dtype)
        outs_motion['all_matched_idxes'] = [
            torch.full((1,), -1, device=bev_embed.device, dtype=torch.long)
        ]
        return outs_motion

    @staticmethod
    def _wrap_occ_eval_tensor(tensor, target_dim):
        while isinstance(tensor, torch.Tensor) and tensor.dim() < target_dim:
            tensor = tensor.unsqueeze(0)
        return tensor

    @auto_fp16(apply_to=('points', ))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_lane_labels=None,
                      gt_lane_bboxes=None,
                      gt_lane_masks=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      gt_sdc_fut_traj=None,
                      gt_sdc_fut_traj_mask=None,
                      sdc_planning=None,
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
                      gt_segmentation=None,
                      gt_instance=None,
                      gt_occ_img_is_valid=None,
                      **kwargs):
        losses = dict()
        losses_track, outs_track = self.forward_track_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_inds=gt_inds,
            gt_past_traj=gt_past_traj,
            gt_past_traj_mask=gt_past_traj_mask,
            l2g_t=l2g_t,
            l2g_r_mat=l2g_r_mat,
            timestamp=timestamp,
            **kwargs)
        losses.update(
            self.loss_weighted_and_prefixed(losses_track, prefix='track'))

        outs_seg = dict()
        if self.with_seg_head and gt_lane_labels is not None:
            current_metas = self._current_img_metas(img_metas)
            gt_lane_labels = self._seg_batch_list(gt_lane_labels, 1)
            gt_lane_bboxes = self._seg_batch_list(gt_lane_bboxes, 2)
            gt_lane_masks = self._seg_batch_list(gt_lane_masks, 3)
            bev_seg = self._bev_for_seg_head(outs_track['bev_embed'])
            losses_seg, outs_seg = self.seg_head.forward_train(
                bev_seg,
                current_metas,
                gt_lane_labels,
                gt_lane_bboxes,
                gt_lane_masks)
            losses.update(
                self.loss_weighted_and_prefixed(losses_seg, prefix='map'))

        bev_embed = None
        outs_motion = dict()
        if self.with_motion_head:
            bev_embed = self._bev_for_motion_head(outs_track['bev_embed'])
            # Pass ego2global so motion head can encode HD map lanes
            if img_metas is not None:
                e2g = self._current_frame_ego2global(img_metas)
                if e2g is not None:
                    outs_track['ego2global'] = torch.as_tensor(
                        e2g, device=bev_embed.device, dtype=torch.float32)
            ret_dict_motion = self.motion_head.forward_train(
                bev_embed,
                gt_bboxes_3d,
                gt_labels_3d,
                gt_fut_traj=gt_fut_traj,
                gt_fut_traj_mask=gt_fut_traj_mask,
                gt_sdc_fut_traj=gt_sdc_fut_traj,
                gt_sdc_fut_traj_mask=gt_sdc_fut_traj_mask,
                outs_track=outs_track,
                outs_seg=outs_seg,
                outs_map=self._build_outs_map(
                    bev_embed, outs_track.get('ego2global')))
            outs_motion = ret_dict_motion['outs_motion']
            outs_motion['bev_pos'] = outs_track.get('bev_pos')
            losses.update(
                self.loss_weighted_and_prefixed(
                    ret_dict_motion['losses'], prefix='motion'))

        if self.with_occ_head:
            if not self.with_motion_head:
                raise RuntimeError('OccHead requires MotionHead outputs.')
            if bev_embed is None:
                bev_embed = self._bev_for_motion_head(outs_track['bev_embed'])
            outs_motion = self._fill_empty_occ_query(outs_motion, bev_embed)
            losses_occ = self.occ_head.forward_train(
                bev_embed,
                outs_motion,
                gt_inds_list=gt_inds,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid)
            losses.update(
                self.loss_weighted_and_prefixed(losses_occ, prefix='occ'))

        if self.with_planning_head:
            if not self.with_motion_head:
                raise RuntimeError(
                    'PlanningHead requires MotionHead outputs.')
            outs_motion['bev_pos'] = self._bev_pos_for_planning_head(
                outs_motion['bev_pos'],
                self.planning_head.bev_h,
                self.planning_head.bev_w)
            outs_planning = self.planning_head.forward_train(
                bev_embed,
                outs_motion,
                sdc_planning=sdc_planning,
                sdc_planning_mask=sdc_planning_mask,
                command=command,
                gt_future_boxes=gt_future_boxes)
            losses.update(
                self.loss_weighted_and_prefixed(
                    outs_planning['losses'], prefix='planning'))

        if self.with_llm_head:
            if not self.with_motion_head:
                raise RuntimeError('LLMBridgeHead requires MotionHead outputs.')
            losses_llm = self.llm_head.forward_train(
                outs_motion, outs_track=outs_track,
                gt_caption=self._current_caption(img_metas))
            losses.update(
                self.loss_weighted_and_prefixed(losses_llm, prefix='llm'))

        sanitized_losses = {}
        for key, value in sorted(losses.items()):
            if not torch.isfinite(value).all():
                raise FloatingPointError(
                    f'Non-finite UniADMotionLidar loss in {key}: '
                    f'{value.detach().cpu()}')
            sanitized_losses[key] = value
        return sanitized_losses

    def simple_test(self, points, img_metas, img=None, history_points=None,
                    **kwargs):
        results = super().simple_test(
            points,
            img_metas,
            img=img,
            history_points=history_points,
            **kwargs)

        if not self.with_motion_head:
            return results

        result = results[0]['pts_bbox']
        bev_embed = self._bev_for_motion_head(result['bev_embed'])
        # Pass ego2global for HD map encoding
        e2g = self._current_frame_ego2global(img_metas)
        if e2g is not None:
            result['ego2global'] = torch.as_tensor(
                e2g, device=bev_embed.device, dtype=torch.float32)
        result_motion, outs_motion = self.motion_head.forward_test(
            bev_embed, outs_track=result,
            outs_map=self._build_outs_map(bev_embed, result.get('ego2global')))
        result.update(result_motion[0])

        if self.with_occ_head and kwargs.get('gt_segmentation') is not None:
            outs_motion['bev_pos'] = result.get('bev_pos')
            occ_no_query = outs_motion['track_query'].shape[1] == 0
            gt_segmentation = self._wrap_occ_eval_tensor(
                kwargs.get('gt_segmentation'), 5)
            gt_instance = self._wrap_occ_eval_tensor(
                kwargs.get('gt_instance'), 5)
            gt_occ_img_is_valid = self._wrap_occ_eval_tensor(
                kwargs.get('gt_occ_img_is_valid'), 3)
            outs_occ = self.occ_head.forward_test(
                bev_embed,
                outs_motion,
                no_query=occ_no_query,
                gt_segmentation=gt_segmentation,
                gt_instance=gt_instance,
                gt_img_is_valid=gt_occ_img_is_valid)
            for key in ('pred_ins_logits', 'pred_ins_sigmoid'):
                outs_occ.pop(key, None)
            results[0]['occ'] = outs_occ
        else:
            outs_occ = dict()

        if self.with_planning_head:
            sdc_planning = kwargs.get('sdc_planning')
            sdc_planning_mask = kwargs.get('sdc_planning_mask')
            command = kwargs.get('command')
            if command is None:
                # Without a command, planning would crash on
                # navi_embed[command]; skip rather than guess a value.
                return results
            outs_motion.setdefault('bev_pos', result.get('bev_pos'))
            outs_motion['bev_pos'] = self._bev_pos_for_planning_head(
                outs_motion['bev_pos'],
                self.planning_head.bev_h,
                self.planning_head.bev_w)
            # PlanningHead.forward_test reads outs_occflow['seg_out']
            # unconditionally; supply at least an empty dict so the head
            # can short-circuit when use_col_optim=False (the col-optim
            # branch is the only consumer of seg_out).
            occ_for_plan = outs_occ if 'seg_out' in outs_occ else {
                'seg_out': bev_embed.new_zeros((1, 1, 1, 1, 1)).long()}
            result_planning = self.planning_head.forward_test(
                bev_embed, outs_motion, occ_for_plan, command)
            results[0]['planning'] = dict(
                planning_gt=dict(
                    segmentation=kwargs.get('gt_segmentation'),
                    sdc_planning=sdc_planning,
                    sdc_planning_mask=sdc_planning_mask,
                    command=command,
                ),
                result_planning=result_planning,
            )

        # LLM caption from LiDAR queries (consumes track_bbox_results /
        # track_query_embeddings, so run before the cleanup below strips them).
        # NB: if planning is enabled and command is None, the early return above
        # skips this; planning is off in the LLM smoke-test config.
        if self.with_llm_head:
            results[0]['llm_caption'] = self.llm_head.forward_test(
                outs_motion, outs_track=result)

        for key in ('bev_embed', 'bev_pos', 'track_query_embeddings',
                    'track_query_matched_idxes', 'track_bbox_results'):
            result.pop(key, None)
        return results
