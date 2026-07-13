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
                 fallback_logit_bias=0.1,
                 score_temperature=0.25,
                 score_target_mode='multi_positive',
                 positive_cost_margin=0.05,
                 score_loss_weight=1.0,
                 residual_loss_weight=1.0,
                 eval_horizon_indices=(1, 3, 5),
                 oracle_recall_tolerance=0.01,
                 use_utility_gate=False,
                 utility_gate_init=-2.0,
                 utility_gate_loss_weight=1.0,
                 utility_min_improvement=0.01,
                 utility_threshold=0.5,
                 utility_target_mode='binary',
                 utility_regression_clip=2.0,
                 use_candidate_cost=False,
                 candidate_cost_loss_weight=1.0,
                 candidate_cost_ranking_weight=0.2,
                 candidate_cost_temperature=0.25,
                 candidate_cost_target_clip=10.0,
                 candidate_cost_background_weight=0.05,
                 candidate_cost_oracle_weight=2.0,
                 candidate_cost_fallback_weight=2.0,
                 candidate_cost_hard_weight=1.0,
                 candidate_cost_hard_count=16,
                 candidate_cost_near_margin=0.1,
                 candidate_cost_fallback_tiebreak=1e-4,
                 ablate_map=False):
        super().__init__()
        self.planning_steps = int(planning_steps)
        self.coordinate_scale = float(coordinate_scale)
        self.residual_scale = float(residual_scale)
        self.fallback_logit_bias = float(fallback_logit_bias)
        self.score_temperature = float(score_temperature)
        if score_target_mode not in ('softmax', 'multi_positive'):
            raise ValueError(
                'score_target_mode must be softmax/multi_positive, got '
                f'{score_target_mode}')
        self.score_target_mode = score_target_mode
        self.positive_cost_margin = float(positive_cost_margin)
        self.score_loss_weight = float(score_loss_weight)
        self.residual_loss_weight = float(residual_loss_weight)
        self.eval_horizon_indices = tuple(int(x) for x in eval_horizon_indices)
        self.oracle_recall_tolerance = float(oracle_recall_tolerance)
        self.use_utility_gate = bool(use_utility_gate)
        self.utility_gate_loss_weight = float(utility_gate_loss_weight)
        self.utility_min_improvement = float(utility_min_improvement)
        self.utility_threshold = float(utility_threshold)
        if utility_target_mode not in ('binary', 'improvement_regression'):
            raise ValueError(
                'utility_target_mode must be binary/improvement_regression, '
                f'got {utility_target_mode}')
        self.utility_target_mode = utility_target_mode
        self.utility_regression_clip = float(utility_regression_clip)
        self.use_candidate_cost = bool(use_candidate_cost)
        self.candidate_cost_loss_weight = float(candidate_cost_loss_weight)
        self.candidate_cost_ranking_weight = float(
            candidate_cost_ranking_weight)
        self.candidate_cost_temperature = float(candidate_cost_temperature)
        self.candidate_cost_target_clip = float(candidate_cost_target_clip)
        self.candidate_cost_background_weight = float(
            candidate_cost_background_weight)
        self.candidate_cost_oracle_weight = float(
            candidate_cost_oracle_weight)
        self.candidate_cost_fallback_weight = float(
            candidate_cost_fallback_weight)
        self.candidate_cost_hard_weight = float(candidate_cost_hard_weight)
        self.candidate_cost_hard_count = int(candidate_cost_hard_count)
        self.candidate_cost_near_margin = float(candidate_cost_near_margin)
        self.candidate_cost_fallback_tiebreak = float(
            candidate_cost_fallback_tiebreak)
        self.ablate_map = bool(ablate_map)
        assert self.coordinate_scale > 0.0
        assert self.residual_scale >= 0.0
        assert self.score_temperature > 0.0
        assert self.positive_cost_margin >= 0.0
        assert self.candidate_cost_temperature > 0.0
        assert self.candidate_cost_target_clip > 0.0
        assert self.candidate_cost_background_weight >= 0.0
        assert self.candidate_cost_oracle_weight >= 0.0
        assert self.candidate_cost_fallback_weight >= 0.0
        assert self.candidate_cost_hard_weight >= 0.0
        assert self.candidate_cost_hard_count >= 0
        assert self.candidate_cost_near_margin >= 0.0
        assert self.candidate_cost_fallback_tiebreak >= 0.0

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
        self.utility_gate_head = None
        self.utility_regression_head = None
        self.candidate_cost_head = None
        if self.use_utility_gate:
            self.utility_gate_head = nn.Sequential(
                nn.Linear(embed_dims * 3, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, 1))
            nn.init.zeros_(self.utility_gate_head[-1].weight)
            nn.init.constant_(
                self.utility_gate_head[-1].bias, float(utility_gate_init))
            if self.utility_target_mode == 'improvement_regression':
                utility_dims = embed_dims * 3 + trajectory_dims * 3
                self.utility_regression_head = nn.Sequential(
                    nn.Linear(utility_dims, embed_dims),
                    nn.ReLU(inplace=True),
                    nn.Linear(embed_dims, 1))
                nn.init.zeros_(self.utility_regression_head[-1].weight)
                nn.init.constant_(
                    self.utility_regression_head[-1].bias, -0.1)
        if self.use_candidate_cost:
            cost_input_dims = embed_dims + trajectory_dims
            self.candidate_cost_head = nn.Sequential(
                nn.Linear(cost_input_dims, embed_dims),
                nn.ReLU(inplace=True),
                nn.Linear(embed_dims, len(self.eval_horizon_indices)))
            nn.init.zeros_(self.candidate_cost_head[-1].weight)
            nn.init.zeros_(self.candidate_cost_head[-1].bias)

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
        if not self.use_utility_gate and not self.use_candidate_cost:
            fallback_bias = torch.zeros_like(logits)
            fallback_bias[:, -1] = self.fallback_logit_bias
            logits = logits + fallback_bias
        logits = logits.masked_fill(~valid, -1e4)
        residual = torch.tanh(self.residual_head(feature)).reshape(
            batch_size, num_candidates, self.planning_steps, 2)
        residual = residual * self.residual_scale
        if self.use_utility_gate or self.use_candidate_cost:
            # The safety fallback must remain byte-for-byte C2.3. Residual
            # capacity is reserved for explicit map candidates only.
            residual = residual * (source == 0).to(residual)[..., None, None]
        refined_candidates = raw_candidates + residual

        utility_logit = None
        utility_score = None
        utility_probability = None
        map_selected_index = None
        selected_map = None
        predicted_horizon_costs = None
        selected_predicted_cost = None
        fallback_predicted_cost = None
        fallback_index = num_candidates - 1
        fallback_traj = refined_candidates[:, -1]
        if self.use_candidate_cost:
            cost_input = torch.cat([
                feature,
                (refined_candidates / self.coordinate_scale).reshape(
                    batch_size, num_candidates, -1),
            ], dim=-1)
            predicted_horizon_costs = F.softplus(
                self.candidate_cost_head(cost_input))
            predicted_mean_cost = predicted_horizon_costs.mean(dim=-1)
            selection_cost = predicted_mean_cost.masked_fill(~valid, 1e4)
            selection_cost = selection_cost.clone()
            selection_cost[:, -1] -= self.candidate_cost_fallback_tiebreak
            probabilities = torch.softmax(
                -selection_cost / self.candidate_cost_temperature, dim=-1)
            selected_index = selection_cost.argmin(dim=-1)
            hard_weights = F.one_hot(
                selected_index, num_classes=num_candidates).to(probabilities)
            selected = torch.sum(
                refined_candidates * hard_weights[..., None, None],
                dim=1)
            batch_index = torch.arange(batch_size, device=feature.device)
            selected_predicted_cost = predicted_mean_cost[
                batch_index, selected_index]
            fallback_predicted_cost = predicted_mean_cost[:, -1]
        elif self.use_utility_gate and num_candidates > 1:
            map_logits = logits[:, :-1]
            map_probabilities = torch.softmax(map_logits, dim=-1)
            map_selected_index = map_logits.argmax(dim=-1)
            map_hard_weights = F.one_hot(
                map_selected_index,
                num_classes=num_candidates - 1).to(map_probabilities)
            if self.training:
                map_weights = (
                    map_hard_weights + map_probabilities
                    - map_probabilities.detach())
            else:
                map_weights = map_hard_weights
            selected_map = torch.sum(
                refined_candidates[:, :-1]
                * map_weights[..., None, None], dim=1)
            selected_map_feature = torch.sum(
                feature[:, :-1] * map_weights[..., None], dim=1)
            utility_input = torch.cat(
                [context, selected_map_feature, feature[:, -1]], dim=-1)
            if self.utility_target_mode == 'improvement_regression':
                trajectory_input = torch.cat([
                    selected_map, fallback_traj,
                    selected_map - fallback_traj], dim=-1).reshape(
                        batch_size, -1) / self.coordinate_scale
                utility_score = self.utility_regression_head(torch.cat(
                    [utility_input, trajectory_input], dim=-1)).squeeze(-1)
                utility_logit = utility_score
                utility_probability = torch.sigmoid(utility_score)
                map_hard = (
                    utility_score >= self.utility_min_improvement
                ).to(utility_score)
                gate_soft = torch.sigmoid(
                    utility_score - self.utility_min_improvement)
            else:
                utility_logit = self.utility_gate_head(
                    utility_input).squeeze(-1)
                utility_probability = torch.sigmoid(utility_logit)
                map_hard = (
                    utility_probability >= self.utility_threshold
                ).to(utility_probability)
                gate_soft = utility_probability
            if self.training:
                map_weight = (
                    map_hard + gate_soft - gate_soft.detach())
            else:
                map_weight = map_hard
            selected = fallback_traj + map_weight[:, None, None] * (
                selected_map - fallback_traj)
            selected_index = torch.where(
                map_hard.to(torch.bool), map_selected_index,
                map_selected_index.new_full(
                    map_selected_index.shape, fallback_index))
            probabilities = torch.cat([
                map_probabilities * utility_probability[:, None],
                (1.0 - utility_probability)[:, None]], dim=-1)
        else:
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
                refined_candidates * selection_weights[..., None, None],
                dim=1)
        return dict(
            multimodal_raw_candidates=raw_candidates,
            multimodal_refined_candidates=refined_candidates,
            multimodal_candidate_valid=valid,
            multimodal_logits=logits,
            multimodal_selected_index=selected_index,
            multimodal_selection_probabilities=probabilities,
            multimodal_residual=residual,
            multimodal_fallback_index=raw_candidates.new_full(
                (batch_size,), fallback_index, dtype=torch.long),
            multimodal_map_selected_index=map_selected_index,
            multimodal_utility_logit=utility_logit,
            multimodal_utility_score=utility_score,
            multimodal_utility_probability=utility_probability,
            multimodal_predicted_horizon_costs=predicted_horizon_costs,
            multimodal_selected_predicted_cost=selected_predicted_cost,
            multimodal_fallback_predicted_cost=fallback_predicted_cost,
            multimodal_selected_map_traj=selected_map,
            multimodal_fallback_traj=fallback_traj,
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

    def _horizon_costs(self, candidates, gt, valid):
        """Return per-candidate L2 targets at the configured horizons."""
        horizon_indices = torch.tensor(
            self.eval_horizon_indices, device=candidates.device,
            dtype=torch.long)
        horizon_valid = horizon_indices < candidates.size(2)
        safe_indices = horizon_indices.clamp(max=candidates.size(2) - 1)
        candidate_points = candidates.index_select(2, safe_indices)
        gt_points = gt.index_select(1, safe_indices)
        costs = torch.linalg.norm(
            candidate_points - gt_points[:, None], dim=-1)
        target_valid = valid.index_select(1, safe_indices) & \
            horizon_valid[None]
        return costs, target_valid

    def loss(self, outputs, gt, valid):
        logits = outputs['multimodal_logits']
        zero = sum(parameter.sum() * 0.0 for parameter in self.parameters())
        stats = dict(
            loss_multimodal_score=zero,
            loss_multimodal_residual=zero,
            loss_multimodal_utility_gate=zero,
            loss_multimodal_candidate_cost=zero,
            loss_multimodal_candidate_cost_ranking=zero,
            multimodal_valid_rate=zero.detach(),
            multimodal_candidate_count=zero.detach(),
            multimodal_top1_oracle_recall=zero.detach(),
            multimodal_top5_oracle_recall=zero.detach(),
            multimodal_positive_count=zero.detach(),
            multimodal_positive_probability=zero.detach(),
            multimodal_fallback_target_rate=zero.detach(),
            multimodal_oracle_l2=zero.detach(),
            multimodal_selected_raw_l2=zero.detach(),
            multimodal_selected_refined_l2=zero.detach(),
            multimodal_regret=zero.detach(),
            multimodal_fallback_rate=zero.detach(),
            multimodal_residual_norm=zero.detach(),
            multimodal_utility_target_rate=zero.detach(),
            multimodal_utility_probability=zero.detach(),
            multimodal_utility_accuracy=zero.detach(),
            multimodal_utility_target_mean=zero.detach(),
            multimodal_utility_mae=zero.detach(),
            multimodal_map_oracle_l2=zero.detach(),
            multimodal_pred_map_l2=zero.detach(),
            multimodal_fallback_l2=zero.detach(),
            multimodal_candidate_cost_mae=zero.detach(),
            multimodal_candidate_cost_selected_pred=zero.detach(),
            multimodal_candidate_cost_selected_actual=zero.detach(),
            multimodal_candidate_cost_fallback_pred=zero.detach(),
            multimodal_candidate_cost_fallback_actual=zero.detach(),
            multimodal_candidate_cost_oracle_l2=zero.detach(),
            multimodal_candidate_cost_regret=zero.detach(),
            multimodal_candidate_cost_near_oracle=zero.detach(),
            multimodal_candidate_cost_fallback_accuracy=zero.detach(),
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
        separate_map_ranking = (
            self.use_utility_gate or self.use_candidate_cost)
        if separate_map_ranking and logits.size(1) > 1:
            score_logits = logits[:, :-1]
            score_costs = masked_costs[:, :-1]
            score_valid = candidate_valid[:, :-1]
        else:
            score_logits = logits
            score_costs = masked_costs
            score_valid = candidate_valid
        score_sample_valid = sample_valid & score_valid.any(dim=-1)
        score_costs = score_costs.masked_fill(~score_valid, 1e4)
        score_oracle_index = score_costs.argmin(dim=-1)
        oracle_index = masked_costs.argmin(dim=-1)
        batch_index = torch.arange(batch_size, device=raw.device)
        oracle_cost = masked_costs[batch_index, oracle_index]
        score_oracle_cost = score_costs[batch_index, score_oracle_index]
        log_probabilities = F.log_softmax(score_logits, dim=-1)
        if self.score_target_mode == 'multi_positive':
            positive = score_valid & (
                score_costs <= score_oracle_cost[:, None]
                + self.positive_cost_margin)
            positive_log_probability = torch.logsumexp(
                log_probabilities.masked_fill(~positive, -1e4), dim=-1)
            score_loss = -positive_log_probability
            probabilities = torch.softmax(score_logits, dim=-1)
            stats['multimodal_positive_count'] = positive.to(raw).sum(
                dim=-1)[score_sample_valid].mean().detach()
            stats['multimodal_positive_probability'] = (
                probabilities * positive.to(probabilities)
            ).sum(dim=-1)[score_sample_valid].mean().detach()
            if not separate_map_ranking:
                stats['multimodal_fallback_target_rate'] = positive[
                    score_sample_valid, -1].to(raw).mean().detach()
        else:
            target_probabilities = torch.softmax(
                -score_costs / self.score_temperature, dim=-1).detach()
            score_loss = -(
                target_probabilities * log_probabilities).sum(dim=-1)
        stats['loss_multimodal_score'] = (
            score_loss[score_sample_valid].mean() * self.score_loss_weight)

        if self.use_utility_gate and not self.use_candidate_cost:
            fallback_cost = masked_costs[:, -1]
            map_selected_index = outputs[
                'multimodal_map_selected_index'][:batch_size]
            predicted_map_cost = score_costs[
                batch_index, map_selected_index]
            utility_target = score_sample_valid & (
                predicted_map_cost + self.utility_min_improvement
                < fallback_cost)
            utility_logit = outputs['multimodal_utility_logit'][:batch_size]
            utility_improvement = (fallback_cost - predicted_map_cost).clamp(
                min=-self.utility_regression_clip,
                max=self.utility_regression_clip).detach()
            if self.utility_target_mode == 'improvement_regression':
                stats['loss_multimodal_utility_gate'] = (
                    F.smooth_l1_loss(
                        utility_logit[sample_valid],
                        utility_improvement[sample_valid])
                    * self.utility_gate_loss_weight)
                utility_decision = (
                    utility_logit >= self.utility_min_improvement)
                stats['multimodal_utility_target_mean'] = utility_improvement[
                    sample_valid].mean().detach()
                stats['multimodal_utility_mae'] = (
                    utility_logit[sample_valid]
                    - utility_improvement[sample_valid]
                ).abs().mean().detach()
            else:
                stats['loss_multimodal_utility_gate'] = (
                    F.binary_cross_entropy_with_logits(
                        utility_logit[sample_valid],
                        utility_target[sample_valid].to(utility_logit))
                    * self.utility_gate_loss_weight)
                utility_decision = (torch.sigmoid(utility_logit) >= 0.5)
            utility_probability = torch.sigmoid(utility_logit)
            stats['multimodal_utility_target_rate'] = utility_target[
                sample_valid].to(raw).mean().detach()
            stats['multimodal_utility_probability'] = utility_probability[
                sample_valid].mean().detach()
            stats['multimodal_utility_accuracy'] = (
                utility_decision[sample_valid]
                == utility_target[sample_valid]
            ).to(raw).mean().detach()
            stats['multimodal_map_oracle_l2'] = score_oracle_cost[
                sample_valid].mean().detach()
            stats['multimodal_pred_map_l2'] = predicted_map_cost[
                sample_valid].mean().detach()
            stats['multimodal_fallback_l2'] = fallback_cost[
                sample_valid].mean().detach()
            stats['multimodal_fallback_target_rate'] = (
                ~utility_target[sample_valid]).to(raw).mean().detach()

        if self.use_candidate_cost:
            refined = outputs['multimodal_refined_candidates'][:batch_size]
            predicted_horizon_costs = outputs[
                'multimodal_predicted_horizon_costs'][:batch_size]
            target_horizon_costs, horizon_valid = self._horizon_costs(
                refined, gt, valid)
            target_horizon_costs = target_horizon_costs.detach().clamp(
                max=self.candidate_cost_target_clip)
            cost_valid = candidate_valid[..., None] & horizon_valid[:, None]
            horizon_weight = horizon_valid.to(raw)
            target_mean_cost = (
                target_horizon_costs * horizon_weight[:, None]
            ).sum(dim=-1) / horizon_weight.sum(
                dim=-1, keepdim=True).clamp_min(1.0)
            target_mean_cost = target_mean_cost.masked_fill(
                ~candidate_valid, 1e4)
            predicted_mean_cost = predicted_horizon_costs.mean(dim=-1)
            predicted_selection_cost = predicted_mean_cost.masked_fill(
                ~candidate_valid, 1e4)

            target_oracle_cost, target_oracle_index = \
                target_mean_cost.min(dim=-1)
            near_oracle = candidate_valid & (
                target_mean_cost <= target_oracle_cost[:, None]
                + self.candidate_cost_near_margin)
            cost_weight = candidate_valid.to(raw) * \
                self.candidate_cost_background_weight
            cost_weight = cost_weight + near_oracle.to(raw) * \
                self.candidate_cost_oracle_weight
            cost_weight[:, -1] += self.candidate_cost_fallback_weight
            hard_count = min(
                self.candidate_cost_hard_count,
                predicted_selection_cost.size(1))
            if hard_count > 0:
                hard_indices = predicted_selection_cost.topk(
                    hard_count, dim=-1, largest=False).indices
                hard_weight = torch.zeros_like(cost_weight)
                hard_weight.scatter_(1, hard_indices, 1.0)
                hard_weight = hard_weight * candidate_valid.to(raw)
                cost_weight = cost_weight + hard_weight * \
                    self.candidate_cost_hard_weight
            cost_weight = cost_weight[..., None] * cost_valid.to(raw)

            cost_error = F.smooth_l1_loss(
                predicted_horizon_costs, target_horizon_costs,
                reduction='none')
            cost_loss = (cost_error * cost_weight).sum(dim=(1, 2)) / \
                cost_weight.sum(dim=(1, 2)).clamp_min(1.0)
            cost_sample_valid = sample_valid & horizon_valid.any(dim=-1)
            if cost_sample_valid.any():
                stats['loss_multimodal_candidate_cost'] = (
                    cost_loss[cost_sample_valid].mean()
                    * self.candidate_cost_loss_weight)

                target_distribution = torch.softmax(
                    -target_mean_cost / self.candidate_cost_temperature,
                    dim=-1).detach()
                predicted_log_distribution = F.log_softmax(
                    -predicted_selection_cost
                    / self.candidate_cost_temperature, dim=-1)
                ranking_loss = -(
                    target_distribution
                    * predicted_log_distribution).sum(dim=-1)
                stats['loss_multimodal_candidate_cost_ranking'] = (
                    ranking_loss[cost_sample_valid].mean()
                    * self.candidate_cost_ranking_weight)

            selected_cost_index = outputs[
                'multimodal_selected_index'][:batch_size]
            selected_predicted_cost = predicted_mean_cost[
                batch_index, selected_cost_index]
            selected_actual_cost = target_mean_cost[
                batch_index, selected_cost_index]
            fallback_predicted_cost = predicted_mean_cost[:, -1]
            fallback_actual_cost = target_mean_cost[:, -1]

            def cost_sample_mean(value):
                weight = cost_sample_valid.to(value)
                return (value * weight).sum() / weight.sum().clamp_min(1.0)

            plain_cost_weight = cost_valid.to(raw)
            stats['multimodal_candidate_cost_mae'] = (
                (predicted_horizon_costs - target_horizon_costs).abs()
                * plain_cost_weight
            ).sum() / plain_cost_weight.sum().clamp_min(1.0)
            stats['multimodal_candidate_cost_mae'] = \
                stats['multimodal_candidate_cost_mae'].detach()
            stats['multimodal_candidate_cost_selected_pred'] = \
                cost_sample_mean(selected_predicted_cost).detach()
            stats['multimodal_candidate_cost_selected_actual'] = \
                cost_sample_mean(selected_actual_cost).detach()
            stats['multimodal_candidate_cost_fallback_pred'] = \
                cost_sample_mean(fallback_predicted_cost).detach()
            stats['multimodal_candidate_cost_fallback_actual'] = \
                cost_sample_mean(fallback_actual_cost).detach()
            stats['multimodal_candidate_cost_oracle_l2'] = \
                cost_sample_mean(target_oracle_cost).detach()
            stats['multimodal_candidate_cost_regret'] = cost_sample_mean(
                selected_actual_cost - target_oracle_cost).detach()
            near_oracle_selected = (
                selected_actual_cost <= target_oracle_cost
                + self.oracle_recall_tolerance).to(raw)
            stats['multimodal_candidate_cost_near_oracle'] = \
                cost_sample_mean(near_oracle_selected).detach()
            selected_fallback = selected_cost_index == \
                outputs['multimodal_fallback_index'][:batch_size]
            target_fallback = (
                target_oracle_index == candidate_valid.size(1) - 1)
            stats['multimodal_fallback_target_rate'] = cost_sample_mean(
                target_fallback.to(raw)).detach()
            fallback_accuracy = (
                selected_fallback == target_fallback).to(raw)
            stats['multimodal_candidate_cost_fallback_accuracy'] = \
                cost_sample_mean(fallback_accuracy).detach()
            for horizon_offset, horizon_index in enumerate(
                    self.eval_horizon_indices):
                valid_at_horizon = cost_valid[..., horizon_offset]
                if valid_at_horizon.any():
                    horizon_mae = (
                        predicted_horizon_costs[..., horizon_offset]
                        - target_horizon_costs[..., horizon_offset]
                    ).abs()[valid_at_horizon].mean().detach()
                else:
                    horizon_mae = zero.detach()
                stats[
                    f'multimodal_candidate_cost_mae_h{horizon_index}'] = \
                    horizon_mae

        selected_index = outputs['multimodal_selected_index'][:batch_size]
        residual_oracle_index = (
            score_oracle_index if separate_map_ranking else oracle_index)
        oracle_raw = raw[batch_index, residual_oracle_index]
        residual = outputs['multimodal_residual'][:batch_size]
        oracle_residual = residual[batch_index, residual_oracle_index]
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

        selected_cost = masked_costs[batch_index, selected_index]
        refined = outputs['multimodal_refined_candidates'][:batch_size]
        refined_costs, _ = self._metric_cost(refined, gt, valid)
        selected_refined_cost = refined_costs[batch_index, selected_index]
        top_k = min(5, score_logits.size(-1))
        top_indices = score_logits.topk(top_k, dim=-1).indices
        top_costs = score_costs.gather(dim=-1, index=top_indices)
        if self.use_utility_gate and not self.use_candidate_cost:
            score_selected_index = outputs[
                'multimodal_map_selected_index'][:batch_size]
        elif self.use_candidate_cost:
            score_selected_index = score_logits.argmax(dim=-1)
        else:
            score_selected_index = selected_index
        score_selected_cost = score_costs[
            batch_index, score_selected_index]
        stats['multimodal_top1_oracle_recall'] = (
            score_selected_cost[score_sample_valid]
            <= (score_oracle_cost + self.oracle_recall_tolerance)[
                score_sample_valid]
        ).to(raw).mean().detach()
        stats['multimodal_top5_oracle_recall'] = (
            top_costs[score_sample_valid]
            <= (score_oracle_cost + self.oracle_recall_tolerance)[
                score_sample_valid, None]
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
