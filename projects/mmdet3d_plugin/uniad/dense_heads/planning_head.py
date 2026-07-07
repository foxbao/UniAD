#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

import torch
import torch.nn as nn
import math
from mmdet.models.builder import HEADS, build_loss
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.functional import bivariate_gaussian_activation
from .planning_head_plugin import CollisionNonlinearOptimizer
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
                 planning_motion_loss_weights=None,
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
        
        #### planning head
        fuser_dim = 3
        attn_module_layer = nn.TransformerDecoderLayer(embed_dims, 8, dim_feedforward=embed_dims*2, dropout=0.1, batch_first=False)
        self.attn_module = nn.TransformerDecoder(attn_module_layer, 3)

        self.use_map_lane = use_map_lane
        self.map_local_k = map_local_k
        if use_map_lane:
            map_attn_layer = nn.TransformerDecoderLayer(
                embed_dims, 8, dim_feedforward=embed_dims*2,
                dropout=0.1, batch_first=False)
            self.map_attn_module = nn.TransformerDecoder(
                map_attn_layer, map_attn_layers)
            self.map_delta_proj = nn.Linear(embed_dims, embed_dims)
            self.map_gate = nn.Sequential(
                nn.Linear(embed_dims * 2, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, embed_dims))
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
                nn.init.zeros_(self.map_delta_proj.weight)
            else:
                nn.init.xavier_uniform_(self.map_delta_proj.weight, gain=0.1)
            nn.init.zeros_(self.map_delta_proj.bias)
            nn.init.constant_(self.map_gate[-1].bias, map_gate_init)
        
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
           
    def forward_train(self,
                      bev_embed, 
                      outs_motion={}, 
                      sdc_planning=None, 
                      sdc_planning_mask=None,
                      command=None,
                      gt_future_boxes=None,
                      outs_map=None,
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
                             sdc_track_query, command, outs_map=outs_map)
        loss_inputs = [sdc_planning, sdc_planning_mask, outs_planning, gt_future_boxes]
        losses = self.loss(*loss_inputs)
        ret_dict = dict(losses=losses, outs_motion=outs_planning)
        return ret_dict

    def forward_test(self, bev_embed, outs_motion={}, outs_occflow={},
                     command=None, outs_map=None):
        sdc_traj_query = outs_motion['sdc_traj_query']
        sdc_track_query = outs_motion['sdc_track_query']
        bev_pos = outs_motion['bev_pos']
        occ_mask = outs_occflow['seg_out']
        
        outs_planning = self(bev_embed, occ_mask, bev_pos, sdc_traj_query,
                             sdc_track_query, command, outs_map=outs_map)
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
            return plan_query
        map_context = self.map_attn_module(
            plan_query, lane_mem, memory_key_padding_mask=lane_mask)
        map_delta = self.map_delta_proj(map_context - plan_query)
        map_gate = torch.sigmoid(
            self.map_gate(torch.cat([plan_query, map_context], dim=-1)))
        return plan_query + map_gate * map_delta

    def forward(self, 
                bev_embed, 
                occ_mask, 
                bev_pos, 
                sdc_traj_query, 
                sdc_track_query, 
                command,
                outs_map=None):
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
        sdc_traj_query = sdc_traj_query[-1]
        P = sdc_traj_query.shape[1]
        sdc_track_query = sdc_track_query[:, None].expand(-1,P,-1)
        
        
        navi_embed = self.navi_embed.weight[command]
        navi_embed = navi_embed[None].expand(-1,P,-1)
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
        plan_query = self._apply_map_lane_attention(plan_query, outs_map)
        
        # plan_query: [1, 1, 256]
        # bev_feat: [40000, 1, 256]
        plan_query = self.attn_module(plan_query, bev_feat)   # [1, 1, 256]
        
        sdc_traj_all = self.reg_branch(plan_query).view((-1, self.planning_steps, 2))
        sdc_traj_all[...,:2] = torch.cumsum(sdc_traj_all[...,:2], dim=1)
        sdc_traj_all[0] = bivariate_gaussian_activation(sdc_traj_all[0])
        if self.use_col_optim and not self.training:
            # post process, only used when testing
            assert occ_mask is not None
            sdc_traj_all = self.collision_optimization(sdc_traj_all, occ_mask)
        
        return dict(
            sdc_traj=sdc_traj_all,
            sdc_traj_all=sdc_traj_all,
        )

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
        if planning_bucket is not None:
            loss_dict['motion_bucket_weight'] = planning_weight.detach()
        return loss_dict
