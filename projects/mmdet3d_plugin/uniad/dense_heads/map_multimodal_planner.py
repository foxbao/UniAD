import torch
import torch.nn as nn
import torch.nn.functional as F


class MapMultimodalPlanner(nn.Module):
    """Score map-derived trajectory candidates and refine the selected mode."""

    def __init__(self,
                 embed_dims=256,
                 planning_steps=6,
                 num_heads=8,
                 dropout=0.1,
                 coordinate_scale=20.0,
                 residual_scale=1.5,
                 fallback_logit_bias=2.0,
                 score_temperature=0.25,
                 score_loss_weight=1.0,
                 residual_loss_weight=1.0,
                 eval_horizon_indices=(1, 3, 5),
                 oracle_recall_tolerance=0.01,
                 ablate_map=False):
        super().__init__()
        self.planning_steps = int(planning_steps)
        self.coordinate_scale = float(coordinate_scale)
        self.residual_scale = float(residual_scale)
        self.fallback_logit_bias = float(fallback_logit_bias)
        self.score_temperature = float(score_temperature)
        self.score_loss_weight = float(score_loss_weight)
        self.residual_loss_weight = float(residual_loss_weight)
        self.eval_horizon_indices = tuple(int(x) for x in eval_horizon_indices)
        self.oracle_recall_tolerance = float(oracle_recall_tolerance)
        self.ablate_map = bool(ablate_map)
        assert self.coordinate_scale > 0.0
        assert self.residual_scale >= 0.0
        assert self.score_temperature > 0.0

        trajectory_dims = self.planning_steps * 2
        self.candidate_encoder = nn.Sequential(
            nn.Linear(trajectory_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims))
        self.source_embed = nn.Embedding(2, embed_dims)
        self.map_attention = nn.MultiheadAttention(
            embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(embed_dims)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dims * 2, embed_dims))
        self.ffn_norm = nn.LayerNorm(embed_dims)
        self.score_head = nn.Linear(embed_dims, 1)
        self.residual_head = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, trajectory_dims))

        nn.init.zeros_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)

    @staticmethod
    def _match_batch(tensor, batch_size):
        if tensor is None or tensor.size(0) == batch_size:
            return tensor
        if tensor.size(0) == 1:
            return tensor.expand(batch_size, *tensor.shape[1:])
        raise ValueError(
            f'Cannot match batch {tensor.size(0)} to {batch_size}.')

    def _candidate_inputs(self, fallback, outs_map):
        batch_size = fallback.size(0)
        candidates = None if (self.ablate_map or outs_map is None) \
            else outs_map.get('planning_candidates')
        candidate_valid = None if (self.ablate_map or outs_map is None) \
            else outs_map.get('planning_candidate_valid')
        if candidates is None:
            candidates = fallback.new_zeros(
                (batch_size, 0, self.planning_steps, 2))
            candidate_valid = torch.zeros(
                (batch_size, 0), device=fallback.device, dtype=torch.bool)
        else:
            candidates = self._match_batch(
                candidates.to(device=fallback.device, dtype=fallback.dtype),
                batch_size)
            if candidate_valid is None:
                candidate_valid = torch.ones(
                    candidates.shape[:2], device=fallback.device,
                    dtype=torch.bool)
            else:
                candidate_valid = self._match_batch(
                    candidate_valid.to(device=fallback.device,
                                       dtype=torch.bool), batch_size)
        raw_candidates = torch.cat([candidates, fallback[:, None]], dim=1)
        fallback_valid = torch.ones(
            (batch_size, 1), device=fallback.device, dtype=torch.bool)
        valid = torch.cat([candidate_valid, fallback_valid], dim=1)
        source = torch.zeros_like(valid, dtype=torch.long)
        source[:, -1] = 1
        return raw_candidates, valid, source

    def _lane_memory(self, outs_map, batch_size, device, dtype):
        if (self.ablate_map or outs_map is None
                or outs_map.get('lane_query') is None):
            return None, None
        memory = outs_map['lane_query'].to(device=device, dtype=dtype)
        position = outs_map.get('lane_query_pos')
        if position is not None:
            memory = memory + position.to(device=device, dtype=dtype)
        memory = self._match_batch(memory, batch_size)
        valid = outs_map.get('lane_valid')
        if valid is None:
            valid = torch.ones(
                memory.shape[:2], device=device, dtype=torch.bool)
        else:
            valid = self._match_batch(
                valid.to(device=device, dtype=torch.bool), batch_size)

        # MultiheadAttention cannot consume an all-masked memory row. A zero
        # sentinel keeps that row finite and contributes no surveyed-map signal.
        empty = ~valid.any(dim=1)
        if empty.any():
            memory = memory.clone()
            valid = valid.clone()
            memory[empty, 0] = 0.0
            valid[empty, 0] = True
        return memory, valid

    def forward(self, plan_query, fallback, outs_map=None):
        raw_candidates, valid, source = self._candidate_inputs(
            fallback, outs_map)
        batch_size, num_candidates = raw_candidates.shape[:2]
        context = plan_query.squeeze(0)
        if context.dim() == 1:
            context = context[None]
        context = self._match_batch(context, batch_size)

        feature = self.candidate_encoder(
            (raw_candidates / self.coordinate_scale).reshape(
                batch_size, num_candidates, -1))
        feature = feature + context[:, None] + self.source_embed(source)

        lane_memory, lane_valid = self._lane_memory(
            outs_map, batch_size, feature.device, feature.dtype)
        if lane_memory is not None and lane_memory.size(1) > 0:
            attended, _ = self.map_attention(
                feature, lane_memory, lane_memory,
                key_padding_mask=~lane_valid, need_weights=False)
            feature = self.attention_norm(feature + attended)
        feature = self.ffn_norm(feature + self.ffn(feature))

        logits = self.score_head(feature).squeeze(-1)
        fallback_bias = torch.zeros_like(logits)
        fallback_bias[:, -1] = self.fallback_logit_bias
        logits = logits + fallback_bias
        logits = logits.masked_fill(~valid, -1e4)
        residual = torch.tanh(self.residual_head(feature)).reshape(
            batch_size, num_candidates, self.planning_steps, 2)
        residual = residual * self.residual_scale
        refined_candidates = raw_candidates + residual

        probabilities = torch.softmax(logits, dim=-1)
        selected_index = logits.argmax(dim=-1)
        hard_weights = F.one_hot(
            selected_index, num_classes=num_candidates).to(probabilities)
        if self.training:
            selection_weights = (
                hard_weights + probabilities - probabilities.detach())
        else:
            selection_weights = hard_weights
        selected = torch.sum(
            refined_candidates * selection_weights[..., None, None], dim=1)
        return dict(
            multimodal_raw_candidates=raw_candidates,
            multimodal_refined_candidates=refined_candidates,
            multimodal_candidate_valid=valid,
            multimodal_logits=logits,
            multimodal_selected_index=selected_index,
            multimodal_selection_probabilities=probabilities,
            multimodal_residual=residual,
            multimodal_fallback_index=raw_candidates.new_full(
                (batch_size,), num_candidates - 1, dtype=torch.long),
            multimodal_selected_traj=selected,
        )

    def _metric_cost(self, candidates, gt, valid):
        batch_size, _, steps = candidates.shape[:3]
        horizon = torch.zeros(
            (steps,), device=candidates.device, dtype=torch.bool)
        for index in self.eval_horizon_indices:
            if index < steps:
                horizon[index] = True
        metric_valid = valid[:, :steps] & horizon[None]
        no_metric_horizon = ~metric_valid.any(dim=-1)
        if no_metric_horizon.any():
            metric_valid = metric_valid.clone()
            metric_valid[no_metric_horizon] = valid[
                no_metric_horizon, :steps]
        error = torch.linalg.norm(
            candidates - gt[:, None, :steps], dim=-1)
        weight = metric_valid.to(error.dtype)
        cost = (error * weight[:, None]).sum(dim=-1) / \
            weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        sample_valid = metric_valid.any(dim=-1)
        assert cost.size(0) == batch_size
        return cost, sample_valid

    def loss(self, outputs, gt, valid):
        logits = outputs['multimodal_logits']
        zero = sum(parameter.sum() * 0.0 for parameter in self.parameters())
        stats = dict(
            loss_multimodal_score=zero,
            loss_multimodal_residual=zero,
            multimodal_valid_rate=zero.detach(),
            multimodal_candidate_count=zero.detach(),
            multimodal_top1_oracle_recall=zero.detach(),
            multimodal_top5_oracle_recall=zero.detach(),
            multimodal_oracle_l2=zero.detach(),
            multimodal_selected_raw_l2=zero.detach(),
            multimodal_selected_refined_l2=zero.detach(),
            multimodal_regret=zero.detach(),
            multimodal_fallback_rate=zero.detach(),
            multimodal_residual_norm=zero.detach(),
        )
        batch_size = min(logits.size(0), gt.size(0), valid.size(0))
        if batch_size == 0:
            return stats
        raw = outputs['multimodal_raw_candidates'][:batch_size]
        logits = logits[:batch_size]
        candidate_valid = outputs['multimodal_candidate_valid'][:batch_size]
        gt = gt[:batch_size, :self.planning_steps, :2].to(raw)
        valid = valid[:batch_size, :self.planning_steps].to(torch.bool)
        costs, sample_valid = self._metric_cost(raw, gt, valid)
        stats['multimodal_valid_rate'] = sample_valid.to(raw).mean().detach()
        stats['multimodal_candidate_count'] = \
            candidate_valid.to(raw).sum(dim=-1).mean().detach()
        if not sample_valid.any():
            return stats

        masked_costs = costs.masked_fill(~candidate_valid, 1e4)
        target_probabilities = torch.softmax(
            -masked_costs / self.score_temperature, dim=-1).detach()
        log_probabilities = F.log_softmax(logits, dim=-1)
        score_loss = -(target_probabilities * log_probabilities).sum(dim=-1)
        stats['loss_multimodal_score'] = (
            score_loss[sample_valid].mean() * self.score_loss_weight)

        oracle_index = masked_costs.argmin(dim=-1)
        selected_index = outputs['multimodal_selected_index'][:batch_size]
        batch_index = torch.arange(batch_size, device=raw.device)
        oracle_raw = raw[batch_index, oracle_index]
        residual = outputs['multimodal_residual'][:batch_size]
        oracle_residual = residual[batch_index, oracle_index]
        residual_target = (gt - oracle_raw).clamp(
            min=-self.residual_scale, max=self.residual_scale).detach()
        residual_error = F.smooth_l1_loss(
            oracle_residual, residual_target, reduction='none').sum(dim=-1)
        valid_weight = valid.to(residual_error.dtype)
        residual_loss = (residual_error * valid_weight).sum(dim=-1) / \
            valid_weight.sum(dim=-1).clamp_min(1.0)
        stats['loss_multimodal_residual'] = (
            residual_loss[sample_valid].mean()
            * self.residual_loss_weight)

        oracle_cost = masked_costs[batch_index, oracle_index]
        selected_cost = masked_costs[batch_index, selected_index]
        refined = outputs['multimodal_refined_candidates'][:batch_size]
        refined_costs, _ = self._metric_cost(refined, gt, valid)
        selected_refined_cost = refined_costs[batch_index, selected_index]
        top_k = min(5, logits.size(-1))
        top_indices = logits.topk(top_k, dim=-1).indices
        top_costs = masked_costs.gather(dim=-1, index=top_indices)
        near_oracle = oracle_cost + self.oracle_recall_tolerance
        stats['multimodal_top1_oracle_recall'] = (
            selected_cost[sample_valid] <= near_oracle[sample_valid]
        ).to(raw).mean().detach()
        stats['multimodal_top5_oracle_recall'] = (
            top_costs[sample_valid] <= near_oracle[sample_valid, None]
        ).any(dim=-1).to(raw).mean().detach()
        stats['multimodal_oracle_l2'] = \
            oracle_cost[sample_valid].mean().detach()
        stats['multimodal_selected_raw_l2'] = \
            selected_cost[sample_valid].mean().detach()
        stats['multimodal_selected_refined_l2'] = \
            selected_refined_cost[sample_valid].mean().detach()
        stats['multimodal_regret'] = (
            selected_cost[sample_valid] - oracle_cost[sample_valid]
        ).mean().detach()
        fallback_index = outputs['multimodal_fallback_index'][:batch_size]
        stats['multimodal_fallback_rate'] = (
            selected_index[sample_valid] == fallback_index[sample_valid]
        ).to(raw).mean().detach()
        stats['multimodal_residual_norm'] = torch.linalg.norm(
            oracle_residual[sample_valid], dim=-1).mean().detach()
        return stats
