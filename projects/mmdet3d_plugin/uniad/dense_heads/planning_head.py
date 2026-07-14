#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
from .planning_head_plugin import CollisionNonlinearOptimizer
from .map_multimodal_planner import MapMultimodalPlanner
import numpy as np
import copy

@HEADS.register_module(force=True)
class PlanningHeadSingleMode(nn.Module):
    def __init__(self,
                 bev_h=200,
                 bev_w=200,
                 embed_dims=256,
                 planning_steps=6,
                 loss_planning=None,
                 loss_collision=None,
                 planning_eval=False,
                 use_col_optim=False,
                 col_optim_args=dict(
                    occ_filter_range=5.0,
                    sigma=1.0, 
                    alpha_collision=5.0,
                 ),
                 with_adapter=False,
                 use_map_lane=False,
                 map_local_k=None,
                 map_attn_layers=1,
                 map_gate_init=-2.0,
                 map_delta_init='zeros',
                 map_fusion_mode='gated_residual',
                 map_fusion_position='pre_bev',
                 map_force_scale=1.0,
                 lane_anchor_mode='none',
                 lane_anchor_scale=1.0,
                 lane_anchor_offset_scale=0.0,
                 lane_anchor_sample_mode='whole_lane',
                 lane_anchor_reference='absolute',
                 lane_anchor_select_mode='first_point',
                 lane_anchor_candidate_k=8,
                 lane_anchor_direction_mode='forward',
                 lane_anchor_init_alpha=0.3,
                 lane_anchor_static_gate_loss_weight=0.0,
                 lane_anchor_static_disp_thresh=0.5,
                 lane_anchor_utility_gate_loss_weight=0.0,
                 lane_anchor_utility_target_mode='least_squares',
                 lane_anchor_utility_grid_size=21,
                 lane_anchor_utility_min_improvement=0.0,
                 lane_anchor_selector_loss_weight=0.0,
                 lane_anchor_selector_teacher_force=True,
                 lane_anchor_selector_sample_teacher=True,
                 lane_anchor_selector_temperature=1.0,
                 ablate_lane_anchor=False,
                 ablate_ego_status='none',
                 planning_motion_loss_weights=None,
                 use_goal=False,
                 multimodal_planner=None,
                ):
        """
        Single Mode Planning Head for Autonomous Driving.

        Args:
            embed_dims (int): Embedding dimensions. Default: 256.
            planning_steps (int): Number of steps for motion planning. Default: 6.
            loss_planning (dict): Configuration for planning loss. Default: None.
            loss_collision (dict): Configuration for collision loss. Default: None.
            planning_eval (bool): Whether to use planning for evaluation. Default: False.
            use_col_optim (bool): Whether to use collision optimization. Default: False.
            col_optim_args (dict): Collision optimization arguments. Default: dict(occ_filter_range=5.0, sigma=1.0, alpha_collision=5.0).
        """
        super(PlanningHeadSingleMode, self).__init__()

        # Nuscenes
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.navi_embed = nn.Embedding(3, embed_dims)
        self.reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, planning_steps * 2),
        )
        self.loss_planning = build_loss(loss_planning)
        self.planning_steps = planning_steps
        self.planning_eval = planning_eval
        self.planning_motion_loss_weights = (
            planning_motion_loss_weights or None)
        self.map_multimodal_planner = None
        if multimodal_planner is not None:
            self.map_multimodal_planner = MapMultimodalPlanner(
                embed_dims=embed_dims,
                planning_steps=planning_steps,
                **multimodal_planner)
        
        #### planning head
        # Goal-conditioned planning (feat/plan-goal). When use_goal is False the
        # head is byte-identical to upstream: fuser_dim stays 3, no goal_encoder
        # is created, and forward concatenates only the original three queries.
        # This keeps base_e2e_lidar_plan / _mapfuse / _mapfuse_v2 unchanged.
        # ablate_ego_status: EVAL-only diagnostic to disambiguate "map fusion is
        # too weak" from "ego-status dominates the plan" (CVPR2024 "Is Ego Status
        # All You Need?"). The sdc_track_query carries the ego's detected
        # position+velocity (the strongest ego-status pathway in this planner;
        # confirmed by the runaway-frame analysis). Zeroing/perturbing it at
        # inference severs that pathway; comparing the map-on/off delta with vs
        # without ego-status tells us whether the kinematic prior was capping the
        # map's contribution. 'none' = unchanged (default). 'zero' = zero the
        # sdc_track_query. 'noise' = replace with unit-scaled Gaussian noise
        # (keeps "an ego exists" but destroys the state). No effect on training.
        # Modes act on the ego-status pathways feeding plan_query:
        #   none  : unchanged (default)
        #   zero  : zero sdc_track_query (detected ego box: position+velocity)
        #   noise : replace sdc_track_query with Gaussian noise
        #   traj  : zero sdc_traj_query (motion-predicted ego future trajectory —
        #           the kinematic-trend pathway; suspected true ego-status route,
        #           since runaway static frames copied the ego's motion trend)
        #   both  : zero BOTH sdc_track_query and sdc_traj_query
        assert ablate_ego_status in ('none', 'zero', 'noise', 'traj', 'both'), (
            'ablate_ego_status must be none/zero/noise/traj/both, got '
            f'{ablate_ego_status}')
        self.ablate_ego_status = ablate_ego_status
        self.use_goal = use_goal
        if use_goal:
            self.goal_encoder = nn.Sequential(
                nn.Linear(2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims))
        fuser_dim = 4 if use_goal else 3
        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)

        self.use_map_lane = use_map_lane
        self.map_local_k = map_local_k
        assert map_fusion_mode in ('gated_residual', 'force_residual'), (
            'map_fusion_mode must be gated_residual/force_residual, got '
            f'{map_fusion_mode}')
        self.map_fusion_mode = map_fusion_mode
        assert map_fusion_position in ('pre_bev', 'post_bev'), (
            'map_fusion_position must be pre_bev/post_bev, got '
            f'{map_fusion_position}')
        self.map_fusion_position = map_fusion_position
        self.map_force_scale = float(map_force_scale)
        assert lane_anchor_mode in (
                'none', 'replace', 'residual', 'blend', 'learned_blend'), (
            'lane_anchor_mode must be none/replace/residual/blend/'
            'learned_blend, got '
            f'{lane_anchor_mode}')
        self.lane_anchor_mode = lane_anchor_mode
        self.lane_anchor_scale = float(lane_anchor_scale)
        self.lane_anchor_offset_scale = float(lane_anchor_offset_scale)
        self.lane_anchor_init_alpha = float(lane_anchor_init_alpha)
        assert 0.0 < self.lane_anchor_init_alpha < 1.0, (
            'lane_anchor_init_alpha must be in (0, 1), got '
            f'{lane_anchor_init_alpha}')
        self.lane_anchor_static_gate_loss_weight = float(
            lane_anchor_static_gate_loss_weight)
        self.lane_anchor_static_disp_thresh = float(
            lane_anchor_static_disp_thresh)
        self.lane_anchor_utility_gate_loss_weight = float(
            lane_anchor_utility_gate_loss_weight)
        assert self.lane_anchor_utility_gate_loss_weight >= 0.0, (
            'lane_anchor_utility_gate_loss_weight must be >= 0, got '
            f'{lane_anchor_utility_gate_loss_weight}')
        assert lane_anchor_utility_target_mode in (
            'least_squares', 'eval_grid'), (
            'lane_anchor_utility_target_mode must be least_squares/eval_grid, '
            f'got {lane_anchor_utility_target_mode}')
        self.lane_anchor_utility_target_mode = \
            lane_anchor_utility_target_mode
        self.lane_anchor_utility_grid_size = int(
            lane_anchor_utility_grid_size)
        assert self.lane_anchor_utility_grid_size >= 2, (
            'lane_anchor_utility_grid_size must be >= 2, got '
            f'{lane_anchor_utility_grid_size}')
        self.lane_anchor_utility_min_improvement = float(
            lane_anchor_utility_min_improvement)
        assert self.lane_anchor_utility_min_improvement >= 0.0, (
            'lane_anchor_utility_min_improvement must be >= 0, got '
            f'{lane_anchor_utility_min_improvement}')
        assert lane_anchor_sample_mode in ('whole_lane', 'local_forward'), (
            'lane_anchor_sample_mode must be whole_lane/local_forward, got '
            f'{lane_anchor_sample_mode}')
        self.lane_anchor_sample_mode = lane_anchor_sample_mode
        assert lane_anchor_reference in ('absolute', 'relative_start'), (
            'lane_anchor_reference must be absolute/relative_start, got '
            f'{lane_anchor_reference}')
        self.lane_anchor_reference = lane_anchor_reference
        assert lane_anchor_select_mode in (
                'first_point', 'closest_point', 'best_endpoint',
                'learned_selector', 'soft_selector',
                'straight_through_selector'), (
            'lane_anchor_select_mode must be first_point/closest_point/'
            'best_endpoint/learned_selector/soft_selector/'
            'straight_through_selector, got '
            f'{lane_anchor_select_mode}')
        self.lane_anchor_select_mode = lane_anchor_select_mode
        self.lane_anchor_candidate_k = int(lane_anchor_candidate_k)
        assert self.lane_anchor_candidate_k > 0, (
            'lane_anchor_candidate_k must be > 0, got '
            f'{lane_anchor_candidate_k}')
        assert lane_anchor_direction_mode in ('forward', 'bidirectional'), (
            'lane_anchor_direction_mode must be forward/bidirectional, got '
            f'{lane_anchor_direction_mode}')
        self.lane_anchor_direction_mode = lane_anchor_direction_mode
        self.lane_anchor_selector_loss_weight = float(
            lane_anchor_selector_loss_weight)
        self.lane_anchor_selector_teacher_force = bool(
            lane_anchor_selector_teacher_force)
        self.lane_anchor_selector_sample_teacher = bool(
            lane_anchor_selector_sample_teacher)
        self.lane_anchor_selector_temperature = float(
            lane_anchor_selector_temperature)
        self.ablate_lane_anchor = bool(ablate_lane_anchor)
        assert self.lane_anchor_selector_temperature > 0.0, (
            'lane_anchor_selector_temperature must be > 0, got '
            f'{lane_anchor_selector_temperature}')
        if use_map_lane:
            module_prefix = (
                'post_map' if map_fusion_position == 'post_bev' else 'map')
            map_attn_layer = nn.TransformerDecoderLayer(
                embed_dims, 8, dim_feedforward=embed_dims*2,
                dropout=0.1, batch_first=False)
            setattr(
                self,
                f'{module_prefix}_attn_module',
                nn.TransformerDecoder(map_attn_layer, map_attn_layers))
            setattr(
                self,
                f'{module_prefix}_delta_proj',
                nn.Linear(embed_dims, embed_dims))
            setattr(
                self,
                f'{module_prefix}_gate',
                nn.Sequential(
                    nn.Linear(embed_dims * 2, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims, embed_dims)))
            map_delta_proj = getattr(
                self, f'{module_prefix}_delta_proj')
            map_gate = getattr(self, f'{module_prefix}_gate')
            # map_delta_init controls how the map->plan residual projection
            # starts. 'zeros' (default, preserves the historical behaviour of
            # base_e2e_lidar_plan_mapfuse) makes the map delta identically 0 at
            # step 0; combined with a strongly negative map_gate_init this
            # starves the map branch of gradient and it converges to a no-op.
            # 'small' seeds a tiny random projection so the branch receives a
            # non-degenerate gradient from the first step, giving the map a fair
            # chance to be learned. The gate bias is a separate lever
            # (map_gate_init): relax it toward 0 to open the fusion channel.
            assert map_delta_init in ('zeros', 'small'), (
                f'map_delta_init must be zeros/small, got {map_delta_init}')
            if map_delta_init == 'zeros':
                nn.init.zeros_(map_delta_proj.weight)
            else:
                nn.init.xavier_uniform_(map_delta_proj.weight, gain=0.1)
            nn.init.zeros_(map_delta_proj.bias)
            nn.init.constant_(map_gate[-1].bias, map_gate_init)
        
        self.mlp_fuser = nn.Sequential(
                nn.Linear(embed_dims*fuser_dim, embed_dims),
                nn.LayerNorm(embed_dims),
                nn.ReLU(inplace=True),
            )
        
        self.pos_embed = nn.Embedding(1, embed_dims)
        self.loss_collision = []
        for cfg in loss_collision:
            self.loss_collision.append(build_loss(cfg))
        self.loss_collision = nn.ModuleList(self.loss_collision)
        
        self.use_col_optim = use_col_optim
        self.occ_filter_range = col_optim_args['occ_filter_range']
        self.sigma = col_optim_args['sigma']
        self.alpha_collision = col_optim_args['alpha_collision']

        # TODO: reimplement it with down-scaled feature_map
        self.with_adapter = with_adapter
        if with_adapter:
            bev_adapter_block = nn.Sequential(
                nn.Conv2d(embed_dims, embed_dims // 2, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=1),
            )
            N_Blocks = 3
            bev_adapter = [copy.deepcopy(bev_adapter_block) for _ in range(N_Blocks)]
            self.bev_adapter = nn.Sequential(*bev_adapter)

        if lane_anchor_mode == 'learned_blend':
            anchor_ctl_dim = embed_dims + planning_steps * 2 * 3
            self.lane_anchor_gate_head = nn.Sequential(
                nn.Linear(anchor_ctl_dim, embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 2, 1))
            self.lane_anchor_residual_head = nn.Sequential(
                nn.Linear(anchor_ctl_dim, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, planning_steps * 2))
            init_logit = math.log(
                self.lane_anchor_init_alpha
                / (1.0 - self.lane_anchor_init_alpha))
            nn.init.zeros_(self.lane_anchor_gate_head[-1].weight)
            nn.init.constant_(self.lane_anchor_gate_head[-1].bias, init_logit)
            nn.init.zeros_(self.lane_anchor_residual_head[-1].weight)
            nn.init.zeros_(self.lane_anchor_residual_head[-1].bias)

        if lane_anchor_select_mode in (
                'learned_selector', 'soft_selector',
                'straight_through_selector'):
            selector_ctl_dim = embed_dims + planning_steps * 2 * 3
            self.lane_anchor_selector_head = nn.Sequential(
                nn.Linear(selector_ctl_dim, embed_dims // 2),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims // 2, 1))
            nn.init.zeros_(self.lane_anchor_selector_head[-1].weight)
            nn.init.zeros_(self.lane_anchor_selector_head[-1].bias)
           
    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      outs_map=None,
                      sdc_goal=None,
                      ):
        """
        Perform forward planning training with the given inputs.
        Args:
            bev_embed (torch.Tensor): The input bird's eye view feature map.
            outs_motion (dict): A dictionary containing the motion outputs.
            outs_occflow (dict): A dictionary containing the occupancy flow outputs.
            sdc_planning (torch.Tensor, optional): The self-driving car's planned trajectory.
            sdc_planning_mask (torch.Tensor, optional): The mask for the self-driving car's planning.
            command (torch.Tensor, optional): The driving command issued to the self-driving car.
            gt_future_boxes (torch.Tensor, optional): The ground truth future bounding boxes.
            img_metas (list[dict], optional): A list of metadata information about the input images.

        Returns:
            ret_dict (dict): A dictionary containing the losses and planning outputs.
        """
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']

        occ_mask = None
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query,
                             sdc_track_query, command, outs_map=outs_map,
                             outs_motion=outs_motion,
                             sdc_goal=sdc_goal,
                             sdc_planning=sdc_planning,
                             sdc_planning_mask=sdc_planning_mask)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={},
                     command=None, outs_map=None, sdc_goal=None,
                     sdc_planning=None, sdc_planning_mask=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']

        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query,
                             sdc_track_query, command, outs_map=outs_map,
                             outs_motion=outs_motion,
                             sdc_goal=sdc_goal,
                             sdc_planning=sdc_planning,
                             sdc_planning_mask=sdc_planning_mask)
        return outs_planning

    def _lane_memory(self, outs_map, batch_size, device, dtype):
        if not self.use_map_lane or outs_map is None:
            return None, None

        lane_query = outs_map.get('lane_query')
        if lane_query is None or lane_query.size(1) == 0:
            return None, None

        lane_query = lane_query.to(device=device, dtype=dtype)
        lane_query_pos = outs_map.get('lane_query_pos')
        if lane_query_pos is not None:
            lane_query_pos = lane_query_pos.to(device=device, dtype=dtype)
            lane_mem = lane_query + lane_query_pos
        else:
            lane_mem = lane_query

        if lane_mem.size(0) == 1 and batch_size != 1:
            lane_mem = lane_mem.expand(batch_size, -1, -1)
        if lane_mem.size(0) != batch_size:
            raise ValueError(
                'lane_query batch does not match plan_query batch: '
                f'{lane_mem.size(0)} vs {batch_size}')

        lane_valid = outs_map.get('lane_valid')
        if lane_valid is None:
            lane_mask = torch.zeros(
                lane_mem.shape[:2], device=device, dtype=torch.bool)
        else:
            lane_valid = lane_valid.to(device=device).bool()
            if lane_valid.size(0) == 1 and batch_size != 1:
                lane_valid = lane_valid.expand(batch_size, -1)
            lane_mask = ~lane_valid

        if (self.map_local_k is not None and self.map_local_k > 0
                and outs_map.get('lane_centroids') is not None):
            lane_centroids = outs_map['lane_centroids'].to(
                device=device, dtype=dtype)
            if lane_centroids.size(0) == 1 and batch_size != 1:
                lane_centroids = lane_centroids.expand(batch_size, -1, -1)
            lane_dist = torch.linalg.norm(lane_centroids, dim=-1)
            lane_dist = lane_dist.masked_fill(lane_mask, float('inf'))
            k = min(self.map_local_k, lane_dist.size(1))
            keep_idx = lane_dist.topk(k, dim=-1, largest=False).indices
            keep = torch.zeros_like(lane_mask, dtype=torch.bool)
            keep.scatter_(1, keep_idx, True)
            keep = keep & torch.isfinite(lane_dist)
            lane_mask = lane_mask | ~keep

        if (~lane_mask).sum() == 0:
            return None, None

        all_masked = lane_mask.all(dim=1)
        if all_masked.any():
            lane_mask = lane_mask.clone()
            lane_mem = lane_mem.clone()
            lane_mask[all_masked] = False
            lane_mem[all_masked] = 0

        lane_mem = rearrange(lane_mem, 'b m c -> m b c')
        return lane_mem, lane_mask

    def _apply_map_lane_attention(self, plan_query, outs_map):
        lane_mem, lane_mask = self._lane_memory(
            outs_map, plan_query.size(1), plan_query.device,
            plan_query.dtype)
        if lane_mem is None:
            return plan_query, None
        module_prefix = (
            'post_map'
            if self.map_fusion_position == 'post_bev'
            else 'map')
        map_attn_module = getattr(
            self, f'{module_prefix}_attn_module')
        map_delta_proj = getattr(
            self, f'{module_prefix}_delta_proj')
        map_gate_module = getattr(self, f'{module_prefix}_gate')
        map_context = map_attn_module(
            plan_query, lane_mem, memory_key_padding_mask=lane_mask)
        if self.map_fusion_mode == 'force_residual':
            map_delta = map_context - plan_query
            applied_delta = self.map_force_scale * map_delta
            gate_mean = plan_query.new_tensor(self.map_force_scale)
        else:
            map_delta = map_delta_proj(map_context - plan_query)
            map_gate = torch.sigmoid(
                map_gate_module(
                    torch.cat([plan_query, map_context], dim=-1)))
            applied_delta = map_gate * map_delta
            gate_mean = map_gate.mean()

        delta_norm = map_delta.norm(dim=-1).mean()
        applied_norm = applied_delta.norm(dim=-1).mean()
        query_norm = plan_query.norm(dim=-1).mean().clamp_min(1e-6)
        stats = dict(
            map_fusion_gate_mean=gate_mean,
            map_fusion_delta_norm=delta_norm,
            map_fusion_applied_norm=applied_norm,
            map_fusion_relative_norm=applied_norm / query_norm,
        )
        return plan_query + applied_delta, stats

    def _lane_anchor_trajectory(self, outs_map, device, dtype, ref_traj=None):
        if (self.lane_anchor_mode == 'none' or outs_map is None
                or outs_map.get('lane_points') is None):
            return None

        lane_points = outs_map['lane_points'].to(device=device, dtype=dtype)
        lane_valid = outs_map.get('lane_valid')
        if lane_valid is None:
            lane_valid = torch.ones(
                lane_points.shape[:2], device=device, dtype=torch.bool)
        else:
            lane_valid = lane_valid.to(device=device).bool()

        if (self.lane_anchor_sample_mode == 'local_forward'
                and self.lane_anchor_select_mode == 'best_endpoint'
                and ref_traj is not None):
            return self._best_endpoint_lane_anchor(
                lane_points, lane_valid, dtype, ref_traj)

        if self.lane_anchor_select_mode == 'closest_point':
            lane_dist = torch.linalg.norm(
                lane_points[..., :2], dim=-1).min(dim=-1).values
        else:
            lane_dist = torch.linalg.norm(lane_points[..., 0, :2], dim=-1)
        lane_dist = lane_dist.masked_fill(~lane_valid, float('inf'))
        best = lane_dist.argmin(dim=1)
        has_lane = torch.isfinite(lane_dist.gather(1, best[:, None])).squeeze(1)
        if not has_lane.any():
            return None

        batch_idx = torch.arange(lane_points.size(0), device=device)
        anchor_points = lane_points[batch_idx, best]
        if self.lane_anchor_sample_mode == 'local_forward':
            return self._local_forward_lane_anchor(
                anchor_points, has_lane, dtype, ref_traj)

        if anchor_points.size(1) == self.planning_steps:
            anchor = anchor_points
        else:
            src_idx = torch.linspace(
                0, anchor_points.size(1) - 1, self.planning_steps,
                device=device)
            left = src_idx.floor().long()
            right = src_idx.ceil().long()
            alpha = (src_idx - left.to(dtype))[:, None]
            anchor = (anchor_points[:, left] * (1 - alpha)
                      + anchor_points[:, right] * alpha)
        anchor = anchor * has_lane[:, None, None].to(dtype)
        return anchor * self.lane_anchor_scale

    def _anchor_ref_score(self, anchor, has_lane, ref_traj, ref_mask=None):
        ref_xy = ref_traj[..., :2].to(anchor)
        step_error = torch.linalg.norm(anchor - ref_xy, dim=-1)
        if ref_mask is None:
            endpoint = step_error[:, -1]
            traj = step_error.mean(dim=1)
            has_ref = torch.ones_like(has_lane)
        else:
            ref_mask = ref_mask.to(device=anchor.device, dtype=torch.bool)
            has_ref = ref_mask.any(dim=1)
            last = ref_mask.long().sum(dim=1).clamp_min(1) - 1
            endpoint = step_error.gather(1, last[:, None]).squeeze(1)
            traj = (step_error * ref_mask.to(step_error)).sum(dim=1) / \
                ref_mask.sum(dim=1).clamp_min(1).to(step_error)
        score = endpoint + 0.25 * traj
        return score.masked_fill(~(has_lane & has_ref), float('inf'))

    def _lane_anchor_candidates(self, lane_points, lane_valid, dtype,
                                sample_ref_traj, score_ref_traj=None,
                                score_ref_mask=None):
        """Build K local lane anchors and score them against a separate ref."""
        if score_ref_traj is None:
            score_ref_traj = sample_ref_traj
        lane_dist = torch.linalg.norm(
            lane_points[..., :2], dim=-1).min(dim=-1).values
        lane_dist = lane_dist.masked_fill(~lane_valid, float('inf'))
        if not torch.isfinite(lane_dist).any():
            return None

        batch_size, num_lanes = lane_dist.shape
        k = min(self.lane_anchor_candidate_k, num_lanes)
        cand_idx = lane_dist.topk(k, dim=1, largest=False).indices
        batch_idx = torch.arange(batch_size, device=lane_points.device)

        cand_anchors = []
        cand_scores = []
        cand_valid = []
        for rank in range(k):
            idx = cand_idx[:, rank]
            has_lane = torch.isfinite(lane_dist[batch_idx, idx])
            points = lane_points[batch_idx, idx]
            anchor = self._local_forward_lane_anchor(
                points, has_lane, dtype, sample_ref_traj)

            if self.lane_anchor_direction_mode == 'bidirectional':
                rev_points = torch.flip(points, dims=[1])
                rev_anchor = self._local_forward_lane_anchor(
                    rev_points, has_lane, dtype, sample_ref_traj)
                forward_sample_score = self._anchor_ref_score(
                    anchor, has_lane, sample_ref_traj)
                reverse_sample_score = self._anchor_ref_score(
                    rev_anchor, has_lane, sample_ref_traj)
                use_rev = reverse_sample_score < forward_sample_score
                anchor = torch.where(use_rev[:, None, None], rev_anchor, anchor)

            score = self._anchor_ref_score(
                anchor, has_lane, score_ref_traj, score_ref_mask)

            cand_anchors.append(anchor)
            cand_scores.append(score)
            cand_valid.append(has_lane)

        return (torch.stack(cand_anchors, dim=1),
                torch.stack(cand_scores, dim=1),
                torch.stack(cand_valid, dim=1))

    def _best_endpoint_lane_anchor(self, lane_points, lane_valid, dtype,
                                   ref_traj):
        """Choose the local lane anchor most compatible with base planner."""
        cands = self._lane_anchor_candidates(
            lane_points, lane_valid, dtype, ref_traj)
        if cands is None:
            return None
        anchors, scores, candidate_valid = cands
        scores = scores.masked_fill(~candidate_valid, float('inf'))
        batch_size = anchors.size(0)
        batch_idx = torch.arange(batch_size, device=lane_points.device)
        best = scores.argmin(dim=1)
        has_best = torch.isfinite(scores.gather(1, best[:, None])).squeeze(1)
        if not has_best.any():
            return None
        best_anchor = anchors[batch_idx, best]
        return best_anchor * has_best[:, None, None].to(dtype)

    def _selector_logits(self, plan_query, base_traj, anchors):
        batch_size, num_cands = anchors.shape[:2]
        query_feat = rearrange(plan_query, 'p b c -> b (p c)')
        query_feat = query_feat[:, None].expand(-1, num_cands, -1)
        base_xy = base_traj[..., :2][:, None].expand(-1, num_cands, -1, -1)
        anchor_xy = anchors[..., :2]
        geom_feat = torch.cat([
            base_xy.reshape(batch_size, num_cands, -1),
            anchor_xy.reshape(batch_size, num_cands, -1),
            (anchor_xy - base_xy).reshape(batch_size, num_cands, -1),
        ], dim=-1)
        feat = torch.cat([query_feat, geom_feat.to(query_feat)], dim=-1)
        logits = self.lane_anchor_selector_head(feat).squeeze(-1)
        return logits

    def _planning_reference(self, sdc_planning, sdc_planning_mask, base_traj):
        if sdc_planning is None:
            return None, None
        gt_ref = sdc_planning[0, :, :self.planning_steps, :2]
        if gt_ref.size(0) != base_traj.size(0):
            return None, None
        gt_ref = gt_ref.to(base_traj).detach()
        gt_mask = None
        if sdc_planning_mask is not None:
            gt_mask = sdc_planning_mask[0, :, :self.planning_steps]
            if gt_mask.dim() == 3:
                gt_mask = gt_mask.any(dim=-1)
            gt_mask = gt_mask.to(device=base_traj.device, dtype=torch.bool)
        return gt_ref, gt_mask

    def _select_learned_anchor(self, anchors, pred_idx, selector_probs):
        if self.lane_anchor_select_mode == 'soft_selector':
            weights = selector_probs
        else:
            hard_weights = F.one_hot(
                pred_idx, num_classes=anchors.size(1)).to(selector_probs)
            if (self.training and self.lane_anchor_select_mode ==
                    'straight_through_selector'):
                weights = (
                    hard_weights - selector_probs.detach() + selector_probs)
            else:
                weights = hard_weights
        return torch.sum(weights[..., None, None] * anchors, dim=1)

    def _learned_selector_lane_anchor(self, outs_map, plan_query, base_traj,
                                      sdc_planning, sdc_planning_mask):
        if (self.lane_anchor_mode == 'none' or outs_map is None
                or outs_map.get('lane_points') is None):
            return None, None

        lane_points = outs_map['lane_points'].to(
            device=base_traj.device, dtype=base_traj.dtype)
        lane_valid = outs_map.get('lane_valid')
        if lane_valid is None:
            lane_valid = torch.ones(
                lane_points.shape[:2], device=base_traj.device,
                dtype=torch.bool)
        else:
            lane_valid = lane_valid.to(device=base_traj.device).bool()

        base_ref = base_traj.detach()
        gt_ref, gt_mask = self._planning_reference(
            sdc_planning, sdc_planning_mask, base_traj)
        sample_ref = base_ref
        if (self.training and self.lane_anchor_selector_sample_teacher
                and gt_ref is not None):
            sample_ref = gt_ref
        score_ref = gt_ref if gt_ref is not None else sample_ref
        cands = self._lane_anchor_candidates(
            lane_points, lane_valid, base_traj.dtype, sample_ref,
            score_ref_traj=score_ref, score_ref_mask=gt_mask)
        if cands is None:
            return None, None
        anchors, oracle_scores, candidate_valid = cands
        if not candidate_valid.any():
            return None, None

        selector_logits = self._selector_logits(
            plan_query, base_traj, anchors)
        selector_logits = selector_logits.masked_fill(~candidate_valid, -1e4)
        finite_oracle = torch.isfinite(oracle_scores) & candidate_valid
        target_scores = oracle_scores.masked_fill(~finite_oracle, 1e4)
        oracle_target = target_scores.argmin(dim=1)
        pred_idx = selector_logits.argmax(dim=1)
        selector_probs = torch.softmax(
            selector_logits / self.lane_anchor_selector_temperature, dim=1)
        if self.lane_anchor_select_mode in (
                'soft_selector', 'straight_through_selector'):
            lane_anchor = self._select_learned_anchor(
                anchors, pred_idx, selector_probs)
        elif self.training and self.lane_anchor_selector_teacher_force:
            select_idx = oracle_target
            batch_idx = torch.arange(
                base_traj.size(0), device=base_traj.device)
            lane_anchor = anchors[batch_idx, select_idx]
        else:
            select_idx = pred_idx
            batch_idx = torch.arange(
                base_traj.size(0), device=base_traj.device)
            lane_anchor = anchors[batch_idx, select_idx]
        has_candidate = candidate_valid.any(dim=1)
        if gt_ref is None:
            has_target_ref = torch.zeros_like(has_candidate)
        elif gt_mask is None:
            has_target_ref = torch.ones_like(has_candidate)
        else:
            has_target_ref = gt_mask.any(dim=1)
        target_valid = has_candidate & has_target_ref & finite_oracle.any(dim=1)
        selected_l2 = self._anchor_ref_score(
            lane_anchor, has_candidate, score_ref, gt_mask)
        entropy = -(selector_probs * selector_probs.clamp_min(1e-8).log()) \
            .sum(dim=1)
        batch_idx = torch.arange(
            base_traj.size(0), device=base_traj.device)
        oracle_anchor = anchors[batch_idx, oracle_target]
        pred_anchor = anchors[batch_idx, pred_idx]
        stats = dict(
            lane_anchor_selector_logits=selector_logits,
            lane_anchor_selector_probs=selector_probs,
            lane_anchor_selector_target=oracle_target,
            lane_anchor_selector_pred=pred_idx,
            lane_anchor_selector_valid=target_valid,
            lane_anchor_selector_num_valid=candidate_valid.sum(dim=1),
            lane_anchor_selector_oracle_l2=oracle_scores.gather(
                1, oracle_target[:, None]).squeeze(1),
            lane_anchor_selector_pred_l2=oracle_scores.gather(
                1, pred_idx[:, None]).squeeze(1),
            lane_anchor_selector_selected_l2=selected_l2,
            lane_anchor_selector_entropy=entropy,
            lane_anchor_selector_max_prob=selector_probs.max(dim=1).values,
            lane_anchor_selector_oracle_anchor=oracle_anchor,
            lane_anchor_selector_pred_anchor=pred_anchor,
            lane_anchor_selector_selected_anchor=lane_anchor,
        )
        return lane_anchor, stats

    def _local_forward_lane_anchor(self, anchor_points, has_lane, dtype,
                                   ref_traj=None):
        """Sample a short-horizon lane anchor from the point nearest ego."""
        origin = anchor_points.new_zeros(anchor_points.size(0), 1, 2)
        point_dist = torch.linalg.norm(anchor_points[..., :2], dim=-1)
        start_idx = point_dist.argmin(dim=1)

        deltas = anchor_points[:, 1:] - anchor_points[:, :-1]
        seg_len = torch.linalg.norm(deltas, dim=-1).clamp_min(1e-6)
        cum_s = torch.cat([
            torch.zeros(anchor_points.size(0), 1, device=anchor_points.device,
                        dtype=dtype),
            torch.cumsum(seg_len, dim=1)
        ], dim=1)

        if ref_traj is None:
            target_s = torch.arange(
                1, self.planning_steps + 1, device=anchor_points.device,
                dtype=dtype)[None] * 1.0
        else:
            ref_pts = torch.cat([origin, ref_traj[..., :2]], dim=1)
            ref_step = torch.linalg.norm(
                ref_pts[:, 1:] - ref_pts[:, :-1], dim=-1)
            target_s = torch.cumsum(ref_step, dim=1)

        start_s = cum_s.gather(1, start_idx[:, None])
        sample_s = start_s + target_s
        sample_s = torch.minimum(sample_s, cum_s[:, -1:])

        right = torch.searchsorted(cum_s.contiguous(), sample_s.contiguous())
        right = right.clamp(min=1, max=anchor_points.size(1) - 1)
        left = right - 1
        left_s = cum_s.gather(1, left)
        right_s = cum_s.gather(1, right)
        denom = (right_s - left_s).clamp_min(1e-6)
        alpha = ((sample_s - left_s) / denom).unsqueeze(-1)

        left_pts = anchor_points.gather(
            1, left[..., None].expand(-1, -1, 2))
        right_pts = anchor_points.gather(
            1, right[..., None].expand(-1, -1, 2))
        anchor = left_pts * (1 - alpha) + right_pts * alpha
        if self.lane_anchor_reference == 'relative_start':
            start_pts = anchor_points.gather(
                1, start_idx[:, None, None].expand(-1, 1, 2))
            anchor = anchor - start_pts
        anchor = anchor * has_lane[:, None, None].to(dtype)
        return anchor * self.lane_anchor_scale

    def _lane_anchor_control_feature(self, plan_query, base_traj, lane_anchor):
        query_feat = rearrange(plan_query, 'p b c -> b (p c)')
        base_xy = base_traj[..., :2]
        anchor_xy = lane_anchor[..., :2]
        geom_feat = torch.cat([
            base_xy.reshape(base_xy.size(0), -1),
            anchor_xy.reshape(anchor_xy.size(0), -1),
            (anchor_xy - base_xy).reshape(base_xy.size(0), -1),
        ], dim=-1)
        return torch.cat([query_feat, geom_feat.to(query_feat)], dim=-1)

    def _apply_learned_lane_anchor(self, plan_query, base_traj, lane_anchor):
        control_feat = self._lane_anchor_control_feature(
            plan_query, base_traj, lane_anchor)
        gate = torch.sigmoid(self.lane_anchor_gate_head(control_feat))
        residual = self.lane_anchor_residual_head(control_feat).view(
            -1, self.planning_steps, 2)
        anchor_target = lane_anchor + residual
        final_traj = base_traj + gate[:, None] * (anchor_target - base_traj)
        return final_traj, gate.squeeze(-1), residual

    def forward(self, 
                bev_embed, 
                occ_mask, 
                bev_pos, 
                sdc_traj_query,
                sdc_track_query,
                command,
                outs_map=None,
                outs_motion=None,
                sdc_goal=None,
                sdc_planning=None,
                sdc_planning_mask=None):
        """
        Forward pass for PlanningHeadSingleMode.

        Args:
            bev_embed (torch.Tensor): Bird's eye view feature embedding.
            occ_mask (torch.Tensor): Instance mask for occupancy.
            bev_pos (torch.Tensor): BEV position.
            sdc_traj_query (torch.Tensor): SDC trajectory query.
            sdc_track_query (torch.Tensor): SDC track query.
            command (int): Driving command.

        Returns:
            dict: A dictionary containing SDC trajectory and all SDC trajectories.
        """
        sdc_track_query = sdc_track_query.detach()
        # EVAL-only ego-status ablation (no-op when 'none', i.e. all training
        # configs). Two ego pathways feed plan_query: sdc_track_query (detected
        # ego box: position+velocity) and sdc_traj_query (motion-predicted ego
        # future trajectory: the kinematic trend). Sever them per mode to test
        # whether ego-status dominance caps the map contribution.
        if self.ablate_ego_status in ('zero', 'both'):
            sdc_track_query = torch.zeros_like(sdc_track_query)
        elif self.ablate_ego_status == 'noise':
            sdc_track_query = torch.randn_like(sdc_track_query)
        sdc_traj_query = sdc_traj_query[-1]
        if self.ablate_ego_status in ('traj', 'both'):
            sdc_traj_query = torch.zeros_like(sdc_traj_query)
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1,P,-1)
        
        
        navi_embed = self.navi_embed.weight[command]
        navi_embed = navi_embed[None].expand(-1,P,-1)
        if self.use_goal:
            # sdc_goal arrives as (B,1,2) after collation; take (x,y) and encode
            # into a per-mode conditioning token. The goal is only a directional
            # prior (no endpoint loss) — see get_sdc_goal / loss().
            goal_xy = sdc_goal.reshape(1, -1)[:, :2].to(sdc_traj_query)
            goal_embed = self.goal_encoder(goal_xy)[:, None, :].expand(-1, P, -1)
            plan_query = torch.cat(
                [sdc_traj_query, sdc_track_query, navi_embed, goal_embed], dim=-1)
        else:
            plan_query = torch.cat([sdc_traj_query, sdc_track_query, navi_embed], dim=-1)

        plan_query = self.mlp_fuser(plan_query).max(1, keepdim=True)[0]   # expand, then fuse  # [1, 6, 768] -> [1, 1, 256]
        plan_query = rearrange(plan_query, 'b p c -> p b c')
        
        bev_pos = rearrange(bev_pos, 'b c h w -> (h w) b c')
        bev_feat = bev_embed +  bev_pos
        
        ##### Plugin adapter #####
        if self.with_adapter:
            bev_feat = rearrange(bev_feat, '(h w) b c -> b c h w', h=self.bev_h, w=self.bev_w)
            bev_feat = bev_feat + self.bev_adapter(bev_feat)  # residual connection
            bev_feat = rearrange(bev_feat, 'b c h w -> (h w) b c')
        ##########################
      
        pos_embed = self.pos_embed.weight
        plan_query = plan_query + pos_embed[None]  # [1, 1, 256]
        map_fusion_stats = None
        if self.map_fusion_position == 'pre_bev':
            plan_query, map_fusion_stats = self._apply_map_lane_attention(
                plan_query, outs_map)
        
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        if self.map_fusion_position == 'post_bev':
            plan_query, map_fusion_stats = self._apply_map_lane_attention(
                plan_query, outs_map)
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        sdc_traj_base = sdc_traj_all
        lane_anchor_gate = None
        lane_anchor_residual = None
        lane_selector_stats = None
        lane_anchor = None
        if not self.ablate_lane_anchor:
            if self.lane_anchor_select_mode in (
                    'learned_selector', 'soft_selector',
                    'straight_through_selector'):
                lane_anchor, lane_selector_stats = \
                    self._learned_selector_lane_anchor(
                        outs_map, plan_query, sdc_traj_all, sdc_planning,
                        sdc_planning_mask)
            else:
                lane_anchor = self._lane_anchor_trajectory(
                    outs_map, sdc_traj_all.device, sdc_traj_all.dtype,
                    ref_traj=sdc_traj_all.detach())
        if lane_anchor is not None:
            if self.lane_anchor_mode == 'replace':
                sdc_traj_all = lane_anchor + (
                    self.lane_anchor_offset_scale * sdc_traj_all)
            elif self.lane_anchor_mode == 'residual':
                sdc_traj_all = sdc_traj_all + lane_anchor
            elif self.lane_anchor_mode == 'blend':
                alpha = self.lane_anchor_offset_scale
                sdc_traj_all = sdc_traj_all + alpha * (
                    lane_anchor - sdc_traj_all)
            elif self.lane_anchor_mode == 'learned_blend':
                sdc_traj_all, lane_anchor_gate, lane_anchor_residual = \
                    self._apply_learned_lane_anchor(
                        plan_query, sdc_traj_all, lane_anchor)
        multimodal_outputs = None
        if self.map_multimodal_planner is not None:
            multimodal_outputs = self.map_multimodal_planner(
                plan_query, sdc_traj_all, outs_map=outs_map,
                outs_motion=outs_motion)
            sdc_traj_all = multimodal_outputs['multimodal_selected_traj']
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)

        ret = dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
            sdc_traj_base=sdc_traj_base,
        )
        if lane_anchor is not None:
            ret['lane_anchor'] = lane_anchor
        if lane_anchor_gate is not None:
            ret['lane_anchor_gate'] = lane_anchor_gate
        if lane_anchor_residual is not None:
            ret['lane_anchor_residual'] = lane_anchor_residual
        if lane_selector_stats is not None:
            ret.update(lane_selector_stats)
        if map_fusion_stats is not None:
            ret.update(map_fusion_stats)
        if multimodal_outputs is not None:
            if self.training:
                ret.update(multimodal_outputs)
            else:
                ret['multimodal_selected_index'] = \
                    multimodal_outputs['multimodal_selected_index']
                ret['multimodal_fallback_index'] = \
                    multimodal_outputs['multimodal_fallback_index']
                ret['multimodal_candidate_count'] = \
                    multimodal_outputs['multimodal_candidate_valid'].sum(
                        dim=-1)
                for key in (
                        'multimodal_map_selected_index',
                        'multimodal_utility_probability',
                        'multimodal_utility_score',
                        'multimodal_selected_predicted_cost',
                        'multimodal_fallback_predicted_cost',
                        'multimodal_selected_map_traj',
                        'multimodal_fallback_traj',
                        'multimodal_audit_indices',
                        'multimodal_audit_valid',
                        'multimodal_audit_raw_candidates',
                        'multimodal_audit_refined_candidates',
                        'multimodal_audit_logits',
                        'multimodal_audit_selection_probabilities',
                        'multimodal_audit_predicted_horizon_costs',
                        'multimodal_set_indices',
                        'multimodal_set_variant',
                        'multimodal_set_valid',
                        'multimodal_set_candidates',
                        'multimodal_set_raw_candidates',
                        'multimodal_set_refined_candidates',
                        'multimodal_set_safety_features',
                        'multimodal_set_base_horizon_costs',
                        'multimodal_set_cost_delta',
                        'multimodal_set_collision_logits',
                        'multimodal_set_predicted_horizon_costs',
                        'multimodal_set_selection_cost',
                        'multimodal_set_selection_probabilities',
                        'multimodal_set_selected_position',
                        'multimodal_set_guarded'):
                    if multimodal_outputs.get(key) is not None:
                        ret[key] = multimodal_outputs[key]
        return ret

    def collision_optimization(self, sdc_traj_all, occ_mask):
        """
        Optimize SDC trajectory with occupancy instance mask.

        Args:
            sdc_traj_all (torch.Tensor): SDC trajectory tensor.
            occ_mask (torch.Tensor): Occupancy flow instance mask. 
        Returns:
            torch.Tensor: Optimized SDC trajectory tensor.
        """
        pos_xy_t = []
        valid_occupancy_num = 0
        
        if occ_mask.shape[2] == 1:
            occ_mask = occ_mask.squeeze(2)
        occ_horizon = occ_mask.shape[1]
        assert occ_horizon == 5

        for t in range(self.planning_steps):
            cur_t = min(t+1, occ_horizon-1)
            pos_xy = torch.nonzero(occ_mask[0][cur_t], as_tuple=False)
            pos_xy = pos_xy[:, [1, 0]]
            pos_xy[:, 0] = (pos_xy[:, 0] - self.bev_h//2) * 0.5 + 0.25
            pos_xy[:, 1] = (pos_xy[:, 1] - self.bev_w//2) * 0.5 + 0.25

            # filter the occupancy in range
            keep_index = torch.sum((sdc_traj_all[0, t, :2][None, :] - pos_xy[:, :2])**2, axis=-1) < self.occ_filter_range**2
            pos_xy_t.append(pos_xy[keep_index].cpu().detach().numpy())
            valid_occupancy_num += torch.sum(keep_index>0)
        if valid_occupancy_num == 0:
            return sdc_traj_all
        
        col_optimizer = CollisionNonlinearOptimizer(self.planning_steps, 0.5, self.sigma, self.alpha_collision, pos_xy_t)
        col_optimizer.set_reference_trajectory(sdc_traj_all[0].cpu().detach().numpy())
        sol = col_optimizer.solve()
        sdc_traj_optim = np.stack([sol.value(col_optimizer.position_x), sol.value(col_optimizer.position_y)], axis=-1)
        return torch.tensor(sdc_traj_optim[None], device=sdc_traj_all.device, dtype=sdc_traj_all.dtype)
    
    @staticmethod
    def _wrap_pi(angle):
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _heading_change_deg(points, min_segment_disp=0.05):
        if points.size(0) < 2:
            return 0.0
        origin = points.new_zeros((1, points.size(1)))
        pts = torch.cat([origin, points], dim=0)
        deltas = pts[1:, :2] - pts[:-1, :2]
        norms = torch.linalg.norm(deltas, dim=-1)
        valid = torch.nonzero(norms >= min_segment_disp, as_tuple=False)
        if valid.numel() < 2:
            return 0.0
        first = int(valid[0, 0])
        last = int(valid[-1, 0])
        v0 = deltas[first]
        v1 = deltas[last]
        a0 = math.atan2(float(v0[1].detach().cpu()),
                        float(v0[0].detach().cpu()))
        a1 = math.atan2(float(v1[1].detach().cpu()),
                        float(v1[0].detach().cpu()))
        return abs(math.degrees(PlanningHeadSingleMode._wrap_pi(a1 - a0)))

    @staticmethod
    def _yaw_change_deg(traj, valid):
        if traj.size(-1) < 3:
            return 0.0
        idx = torch.nonzero(valid[:traj.size(0)], as_tuple=False)
        if idx.numel() < 2:
            return 0.0
        first = int(idx[0, 0])
        last = int(idx[-1, 0])
        yaw0 = float(traj[first, 2].detach().cpu())
        yaw1 = float(traj[last, 2].detach().cpu())
        if not np.isfinite(yaw0) or not np.isfinite(yaw1):
            return 0.0
        return abs(math.degrees(PlanningHeadSingleMode._wrap_pi(yaw1 - yaw0)))

    @staticmethod
    def _lateral_ratio(points):
        if points.size(0) < 2:
            return 0.0
        end = points[-1, :2]
        net = torch.linalg.norm(end)
        if float(net.detach().cpu()) < 1e-6:
            return 0.0
        direction = end / net.clamp_min(1e-6)
        normal = torch.stack([-direction[1], direction[0]])
        lateral = torch.max(torch.abs(points[:, :2] @ normal))
        return float((lateral / net.clamp_min(1e-6)).detach().cpu())

    @staticmethod
    def _planning_motion_bucket(sdc_planning, sdc_planning_mask):
        if sdc_planning.dim() == 4:
            traj = sdc_planning[0, 0]
        elif sdc_planning.dim() == 3:
            traj = sdc_planning[0] if sdc_planning.size(0) == 1 \
                else sdc_planning
        elif sdc_planning.dim() == 2:
            traj = sdc_planning
        else:
            return None

        if sdc_planning_mask.dim() == 4:
            valid = sdc_planning_mask[0, 0, :, 0] > 0
        elif sdc_planning_mask.dim() == 3:
            mask = sdc_planning_mask[0] if sdc_planning_mask.size(0) == 1 \
                else sdc_planning_mask
            valid = mask[:, 0] > 0 if mask.size(-1) == 3 else mask > 0
        elif sdc_planning_mask.dim() == 2:
            valid = (sdc_planning_mask[:, 0] > 0
                     if sdc_planning_mask.size(-1) == 3
                     else sdc_planning_mask.reshape(-1) > 0)
        else:
            valid = sdc_planning_mask.reshape(-1) > 0
        valid = valid[:traj.size(0)]
        idx = torch.nonzero(valid, as_tuple=False)
        if idx.numel() == 0:
            return None
        last = int(idx[-1, 0])
        points = traj[:last + 1][valid[:last + 1]]
        if points.numel() == 0:
            return None
        final_disp = torch.linalg.norm(points[-1, :2])
        final_disp = float(final_disp.detach().cpu())
        if final_disp < 0.5:
            return 'static'
        if final_disp < 2.0:
            return 'slow'
        turn_angle = max(
            PlanningHeadSingleMode._heading_change_deg(points),
            PlanningHeadSingleMode._yaw_change_deg(
                traj[:last + 1], valid[:last + 1]))
        if (turn_angle >= 15.0
                or PlanningHeadSingleMode._lateral_ratio(points) >= 0.15):
            return 'turning'
        return 'moving_straight'

    def _planning_motion_loss_weight(self, sdc_planning, sdc_planning_mask):
        if not self.planning_motion_loss_weights:
            return None, None
        bucket = self._planning_motion_bucket(
            sdc_planning, sdc_planning_mask)
        if bucket is None:
            return 'unknown', sdc_planning.new_tensor(1.0)
        weight = float(self.planning_motion_loss_weights.get(bucket, 1.0))
        return bucket, sdc_planning.new_tensor(weight)

    @staticmethod
    def _optimal_blend_gate_target(base_traj, map_traj, gt_traj, valid_mask,
                                   eps=1e-6):
        """Return the least-squares alpha on the base-to-map trajectory line."""
        delta = (map_traj[..., :2] - base_traj[..., :2]).detach()
        gt_delta = (gt_traj[..., :2] - base_traj[..., :2]).detach()
        valid = valid_mask.to(device=delta.device, dtype=delta.dtype)
        weighted_delta = delta * valid[..., None]
        numerator = (weighted_delta * gt_delta).sum(dim=(-2, -1))
        denominator = (weighted_delta * delta).sum(dim=(-2, -1))
        has_target = valid.bool().any(dim=-1) & (denominator > eps)
        target = (numerator / denominator.clamp_min(eps)).clamp(0.0, 1.0)
        target = torch.where(has_target, target, torch.zeros_like(target))
        return target.detach(), has_target

    @staticmethod
    def _eval_grid_gate_target(base_traj, map_traj, gt_traj, valid_mask,
                               grid_size=21, min_improvement=0.0, eps=1e-6):
        """Search alpha using the evaluator's 1s/2s/3s Euclidean L2."""
        base = base_traj[..., :2].detach()
        delta = (map_traj[..., :2] - base).detach()
        gt = gt_traj[..., :2].detach()
        horizon_idx = torch.tensor(
            [1, 3, 5], device=base.device, dtype=torch.long)
        horizon_idx = horizon_idx[horizon_idx < base.size(-2)]
        if horizon_idx.numel() == 0:
            empty = base.new_zeros(base.size(0))
            return empty, torch.zeros_like(empty, dtype=torch.bool), empty

        base = base.index_select(-2, horizon_idx)
        delta = delta.index_select(-2, horizon_idx)
        gt = gt.index_select(-2, horizon_idx)
        valid = valid_mask.to(device=base.device, dtype=torch.bool) \
            .index_select(-1, horizon_idx)
        has_target = valid.any(dim=-1) & \
            (delta.square().sum(dim=(-2, -1)) > eps)

        alphas = torch.linspace(
            0.0, 1.0, grid_size, device=base.device, dtype=base.dtype)
        candidates = base[:, None] + \
            alphas[None, :, None, None] * delta[:, None]
        errors = torch.linalg.norm(
            candidates - gt[:, None], dim=-1)
        valid_f = valid.to(errors.dtype)
        costs = (errors * valid_f[:, None]).sum(dim=-1) / \
            valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        best_cost, best_idx = costs.min(dim=1)
        base_cost = costs[:, 0]
        improvement = (base_cost - best_cost).clamp_min(0.0)
        target = alphas[best_idx]
        target = torch.where(
            has_target & (improvement >= min_improvement),
            target, torch.zeros_like(target))
        return target.detach(), has_target, improvement.detach()

    def _lane_anchor_utility_gate_loss(self, sdc_planning,
                                       sdc_planning_mask, outs_planning):
        gate = outs_planning['lane_anchor_gate'].reshape(-1)
        zero = gate.sum() * 0.0
        stats = dict(
            loss_lane_anchor_utility_gate=zero,
            lane_anchor_utility_valid_rate=zero.detach(),
            lane_anchor_utility_target_mean=zero.detach(),
            lane_anchor_utility_gate_mae=zero.detach(),
            lane_anchor_utility_use_rate=zero.detach(),
            lane_anchor_utility_improvement_mean=zero.detach(),
        )
        required = ('sdc_traj_base', 'lane_anchor', 'lane_anchor_residual')
        if any(key not in outs_planning for key in required):
            return stats

        base_traj = outs_planning['sdc_traj_base'][..., :2]
        map_traj = (
            outs_planning['lane_anchor'][..., :2]
            + outs_planning['lane_anchor_residual'][..., :2])
        gt_traj = sdc_planning[0, :, :self.planning_steps, :2]
        valid_mask = torch.any(
            sdc_planning_mask[0, :, :self.planning_steps], dim=-1)
        batch_size = min(
            gate.numel(), base_traj.size(0), map_traj.size(0),
            gt_traj.size(0), valid_mask.size(0))
        if batch_size == 0:
            return stats

        if self.lane_anchor_utility_target_mode == 'eval_grid':
            target, has_target, improvement = self._eval_grid_gate_target(
                base_traj[:batch_size], map_traj[:batch_size],
                gt_traj[:batch_size], valid_mask[:batch_size],
                grid_size=self.lane_anchor_utility_grid_size,
                min_improvement=self.lane_anchor_utility_min_improvement)
        else:
            target, has_target = self._optimal_blend_gate_target(
                base_traj[:batch_size], map_traj[:batch_size],
                gt_traj[:batch_size], valid_mask[:batch_size])
            improvement = torch.zeros_like(target)
        stats['lane_anchor_utility_valid_rate'] = \
            has_target.to(gate).mean().detach()
        if has_target.any():
            pred = gate[:batch_size][has_target]
            target = target[has_target].to(pred)
            improvement = improvement[has_target].to(pred)
            stats['loss_lane_anchor_utility_gate'] = (
                F.mse_loss(pred, target)
                * self.lane_anchor_utility_gate_loss_weight)
            stats['lane_anchor_utility_target_mean'] = target.mean().detach()
            stats['lane_anchor_utility_gate_mae'] = \
                (pred - target).abs().mean().detach()
            stats['lane_anchor_utility_use_rate'] = \
                (target > 0).to(pred).mean().detach()
            stats['lane_anchor_utility_improvement_mean'] = \
                improvement.mean().detach()
        return stats

    def loss(self, sdc_planning, sdc_planning_mask, outs_planning, future_gt_bbox=None):
        sdc_traj_all = outs_planning['sdc_traj_all'] # b, p, t, 5
        loss_dict = dict()
        planning_bucket, planning_weight = self._planning_motion_loss_weight(
            sdc_planning, sdc_planning_mask)
        for i in range(len(self.loss_collision)):
            loss_collision = self.loss_collision[i](sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :3], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1), future_gt_bbox[0][1:self.planning_steps+1])
            if planning_weight is not None:
                loss_collision = loss_collision * planning_weight
            loss_dict[f'loss_collision_{i}'] = loss_collision          
        loss_ade = self.loss_planning(sdc_traj_all, sdc_planning[0, :, :self.planning_steps, :2], torch.any(sdc_planning_mask[0, :, :self.planning_steps], dim=-1))
        if planning_weight is not None:
            loss_ade = loss_ade * planning_weight
        loss_dict.update(dict(loss_ade=loss_ade))
        if (self.lane_anchor_static_gate_loss_weight > 0
                and 'lane_anchor_gate' in outs_planning):
            gt_final = sdc_planning[0, :, self.planning_steps - 1, :2]
            gt_disp = torch.linalg.norm(gt_final, dim=-1)
            static_mask = gt_disp < self.lane_anchor_static_disp_thresh
            gate = outs_planning['lane_anchor_gate'].reshape(-1)
            static_weight = static_mask.to(gate).mean()
            loss_static_gate = gate.pow(2).mean() * static_weight
            loss_static_gate = loss_static_gate * \
                self.lane_anchor_static_gate_loss_weight
            loss_dict['loss_lane_anchor_static_gate'] = loss_static_gate
        if (self.lane_anchor_utility_gate_loss_weight > 0
                and 'lane_anchor_gate' in outs_planning):
            loss_dict.update(self._lane_anchor_utility_gate_loss(
                sdc_planning, sdc_planning_mask, outs_planning))
        if (self.training and self.lane_anchor_select_mode in
                ('learned_selector', 'soft_selector',
                 'straight_through_selector')):
            selector_zero = sum(
                parameter.sum() * 0.0
                for parameter in self.lane_anchor_selector_head.parameters())
            loss_dict['loss_lane_anchor_selector'] = selector_zero
            for key in (
                    'lane_anchor_selector_valid_rate',
                    'lane_anchor_selector_acc',
                    'lane_anchor_selector_oracle_l2',
                    'lane_anchor_selector_pred_l2',
                    'lane_anchor_selector_selected_l2',
                    'lane_anchor_selector_entropy',
                    'lane_anchor_selector_max_prob'):
                loss_dict[key] = selector_zero.detach()

            if (self.lane_anchor_selector_loss_weight > 0
                    and 'lane_anchor_selector_logits' in outs_planning):
                logits = outs_planning['lane_anchor_selector_logits']
                target = outs_planning['lane_anchor_selector_target']
                valid = outs_planning.get('lane_anchor_selector_valid')
                if valid is None:
                    valid = torch.ones_like(target, dtype=torch.bool)
                loss_dict['lane_anchor_selector_valid_rate'] = \
                    valid.to(logits).mean().detach()
                for key in (
                        'lane_anchor_selector_entropy',
                        'lane_anchor_selector_max_prob'):
                    if key in outs_planning:
                        loss_dict[key] = outs_planning[key].mean().detach()
                if valid.any():
                    loss_selector = F.cross_entropy(
                        logits[valid], target[valid])
                    loss_dict['loss_lane_anchor_selector'] = (
                        loss_selector * self.lane_anchor_selector_loss_weight)
                    pred = outs_planning['lane_anchor_selector_pred']
                    acc = (pred[valid] == target[valid]).to(logits).mean()
                    loss_dict['lane_anchor_selector_acc'] = acc.detach()
                    for key in (
                            'lane_anchor_selector_oracle_l2',
                            'lane_anchor_selector_pred_l2',
                            'lane_anchor_selector_selected_l2'):
                        if key in outs_planning:
                            loss_dict[key] = outs_planning[key][valid] \
                                .mean().detach()
        if (self.training and self.map_multimodal_planner is not None
                and 'multimodal_logits' in outs_planning):
            gt_traj = sdc_planning[0, :, :self.planning_steps, :3]
            gt_valid = torch.any(
                sdc_planning_mask[0, :, :self.planning_steps], dim=-1)
            planner_future_boxes = None
            if future_gt_bbox is not None:
                planner_future_boxes = future_gt_bbox[0][
                    1:self.planning_steps + 1]
            loss_dict.update(self.map_multimodal_planner.loss(
                outs_planning, gt_traj, gt_valid,
                future_gt_bbox=planner_future_boxes))
        if planning_bucket is not None:
            loss_dict['motion_bucket_weight'] = planning_weight.detach()
        for key in (
                'map_fusion_gate_mean',
                'map_fusion_delta_norm',
                'map_fusion_applied_norm',
                'map_fusion_relative_norm'):
            if key in outs_planning:
                loss_dict[key] = outs_planning[key].detach()
        return loss_dict
