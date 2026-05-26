import torch
from mmcv.runner import auto_fp16
from mmdet.models import DETECTORS, build_head

from .uniad_track_lidar import UniADTrackLidar


@DETECTORS.register_module()
class UniADMotionLidar(UniADTrackLidar):
    """LiDAR-only stage-2 model for tracking plus motion prediction."""

    def __init__(self,
                 motion_head=None,
                 task_loss_weight=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.motion_head = build_head(motion_head) if motion_head else None
        self.task_loss_weight = task_loss_weight or dict(track=1.0, motion=1.0)

    @property
    def with_motion_head(self):
        return hasattr(self, 'motion_head') and self.motion_head is not None

    @staticmethod
    def _bev_for_motion_head(bev_embed):
        if bev_embed.dim() == 4:
            return bev_embed.flatten(2).permute(2, 0, 1).contiguous()
        if bev_embed.dim() == 3:
            return bev_embed
        raise ValueError('MotionHead expects BEV shape [HW, B, C] or '
                         f'[B, C, H, W], got {tuple(bev_embed.shape)}.')

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
            if isinstance(qm, (list, tuple)) and qm:
                last = qm[-1]
                if isinstance(last, dict):
                    return last.get('ego2global')
        return None

    @auto_fp16(apply_to=('points', ))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_inds=None,
                      gt_fut_traj=None,
                      gt_fut_traj_mask=None,
                      gt_past_traj=None,
                      gt_past_traj_mask=None,
                      l2g_t=None,
                      l2g_r_mat=None,
                      timestamp=None,
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
                outs_track=outs_track)
            losses_motion = ret_dict_motion['losses']
            losses.update(
                self.loss_weighted_and_prefixed(
                    losses_motion, prefix='motion'))

        for key, value in losses.items():
            losses[key] = torch.nan_to_num(value)
        return losses

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
        result_motion, _ = self.motion_head.forward_test(
            bev_embed, outs_track=result)
        result.update(result_motion[0])

        for key in ('bev_embed', 'bev_pos', 'track_query_embeddings',
                    'track_query_matched_idxes', 'track_bbox_results'):
            result.pop(key, None)
        return results
