"""OccWorld extension of the existing query-based UniAD occupancy head."""

import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import HEADS

from .occ_head import OccHead


def bev_to_world_layout(values: torch.Tensor) -> torch.Tensor:
    """Convert official BEV rows to image-aligned OccWorld rows."""
    if values.ndim < 2:
        raise ValueError('BEV values must end with [H,W]')
    return torch.flip(values, dims=(-2,))


def flow_to_world_layout(flow: torch.Tensor,
                         ignore_index: float = 255.0) -> torch.Tensor:
    """Convert official [dy,dx] flow to image-aligned OccWorld rows."""
    if flow.ndim < 3 or flow.shape[-3] != 2:
        raise ValueError('Flow must end with [2,H,W]')
    aligned = bev_to_world_layout(flow).clone()
    dy = aligned[..., 0, :, :]
    valid_dy = dy != ignore_index
    aligned[..., 0, :, :] = torch.where(valid_dy, -dy, dy)
    return aligned


def compose_incremental_flow_2d(
        flow: torch.Tensor,
        ignore_index: float = 255.0) -> torch.Tensor:
    """Compose Eulerian step flow into current-source cumulative flow.

    The input flow at step ``t`` is defined on the source cells of frame
    ``t``. Each current-frame cell is followed through those fields so the
    returned step ``t`` displacement is defined on the original frame-0
    source cell and points directly to frame ``t + 1``.
    """
    if flow.ndim != 5 or flow.shape[2] != 2:
        raise ValueError('Incremental flow must have shape [B,T,2,H,W]')
    batch_size, step_count, _, height, width = flow.shape
    valid = torch.all(flow != ignore_index, dim=2)
    safe_flow = torch.where(
        valid[:, :, None], flow, torch.zeros_like(flow))
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=flow.dtype),
        torch.arange(width, device=flow.device, dtype=flow.dtype),
        indexing='ij')
    base_y = grid_y[None].expand(batch_size, -1, -1)
    base_x = grid_x[None].expand(batch_size, -1, -1)
    cumulative = torch.zeros_like(flow[:, 0])
    active = torch.ones(
        (batch_size, height, width), dtype=torch.bool,
        device=flow.device)
    outputs = []
    for step in range(step_count):
        sample_y = base_y + cumulative[:, 0]
        sample_x = base_x + cumulative[:, 1]
        if height > 1:
            normalized_y = sample_y * (2.0 / (height - 1)) - 1.0
        else:
            normalized_y = torch.zeros_like(sample_y)
        if width > 1:
            normalized_x = sample_x * (2.0 / (width - 1)) - 1.0
        else:
            normalized_x = torch.zeros_like(sample_x)
        sample_grid = torch.stack(
            [normalized_x, normalized_y], dim=-1)
        sampled_flow = F.grid_sample(
            safe_flow[:, step], sample_grid, mode='nearest',
            padding_mode='zeros', align_corners=True)
        sampled_valid = F.grid_sample(
            valid[:, step, None].to(flow.dtype), sample_grid,
            mode='nearest', padding_mode='zeros',
            align_corners=True)[:, 0] > 0.5
        active = active & sampled_valid
        cumulative = cumulative + sampled_flow
        outputs.append(torch.where(
            active[:, None], cumulative,
            torch.full_like(cumulative, ignore_index)))
    return torch.stack(outputs, dim=1)


def forward_splat_2d(values: torch.Tensor,
                     flow: torch.Tensor) -> torch.Tensor:
    """Bilinearly splat source values using forward flow [dy, dx]."""
    if values.ndim != 4 or flow.ndim != 4:
        raise ValueError(
            'Forward splat expects values [B,C,H,W] and flow [B,2,H,W]')
    batch_size, channels, height, width = values.shape
    if tuple(flow.shape) != (batch_size, 2, height, width):
        raise ValueError('Forward-splat value and flow shapes do not match')
    grid_y, grid_x = torch.meshgrid(
        torch.arange(height, device=values.device, dtype=flow.dtype),
        torch.arange(width, device=values.device, dtype=flow.dtype),
        indexing='ij')
    destination_y = grid_y[None] + flow[:, 0]
    destination_x = grid_x[None] + flow[:, 1]
    y0 = torch.floor(destination_y)
    x0 = torch.floor(destination_x)
    output = values.new_zeros((batch_size, channels, height * width))
    flat_values = values.reshape(batch_size, channels, height * width)
    for y_index, y_weight in (
            (y0, 1.0 - (destination_y - y0)),
            (y0 + 1.0, destination_y - y0)):
        for x_index, x_weight in (
                (x0, 1.0 - (destination_x - x0)),
                (x0 + 1.0, destination_x - x0)):
            valid = (
                (y_index >= 0) & (y_index < height) &
                (x_index >= 0) & (x_index < width))
            flat_index = (
                y_index.clamp(0, height - 1).long() * width +
                x_index.clamp(0, width - 1).long())
            weight = (y_weight * x_weight * valid.to(flow.dtype)).reshape(
                batch_size, 1, height * width)
            output.scatter_add_(
                2,
                flat_index.reshape(batch_size, 1, height * width).expand(
                    -1, channels, -1),
                flat_values * weight.to(values.dtype))
    return output.reshape(batch_size, channels, height, width).clamp(0, 1)


def apply_physical_flow_fusion(
        prediction: torch.Tensor,
        observation_class: torch.Tensor,
        observation_known: torch.Tensor,
        warped_instance_probability: torch.Tensor,
        threshold: float,
        instance_class: int = 2) -> torch.Tensor:
    """Apply persistence plus flow-based instance arrival/departure."""
    if prediction.ndim != 5 or warped_instance_probability.ndim != 5:
        raise ValueError('Physical flow fusion expects 3D temporal volumes')
    batch_size, horizon_count, z_count, height, width = prediction.shape
    expected_current = (batch_size, z_count, height, width)
    expected_future = (
        batch_size, horizon_count - 1, z_count, height, width)
    if (tuple(observation_class.shape) != expected_current or
            tuple(observation_known.shape) != expected_current or
            tuple(warped_instance_probability.shape) != expected_future):
        raise ValueError('Physical flow-fusion shapes do not match')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Physical flow-fusion threshold must be in [0, 1]')
    fused = prediction.clone()
    known_future = observation_known[:, None].expand_as(fused[:, 1:])
    persistence = observation_class[:, None].expand_as(fused[:, 1:])
    fused[:, 1:] = torch.where(known_future, persistence, fused[:, 1:])
    current_instance = (
        observation_known & (observation_class == instance_class))
    current_instance = current_instance[:, None].to(
        warped_instance_probability.dtype)
    arrival = (
        warped_instance_probability - current_instance >= threshold)
    departure = (
        current_instance - warped_instance_probability >= threshold)
    fused[:, 1:] = torch.where(
        arrival, torch.full_like(fused[:, 1:], instance_class),
        fused[:, 1:])
    fused[:, 1:] = torch.where(
        departure, torch.zeros_like(fused[:, 1:]), fused[:, 1:])
    return fused


def apply_local_flow_overlay(
        raw_prediction: torch.Tensor,
        observation_class: torch.Tensor,
        observation_known: torch.Tensor,
        warped_instance_probability: torch.Tensor,
        threshold: float,
        instance_class: int = 2) -> torch.Tensor:
    """Overlay flow events on raw semantics without global persistence."""
    if raw_prediction.ndim != 5 or warped_instance_probability.ndim != 5:
        raise ValueError('Local flow overlay expects 3D temporal volumes')
    batch_size, horizon_count, z_count, height, width = (
        raw_prediction.shape)
    expected_current = (batch_size, z_count, height, width)
    expected_future = (
        batch_size, horizon_count - 1, z_count, height, width)
    if (tuple(observation_class.shape) != expected_current or
            tuple(observation_known.shape) != expected_current or
            tuple(warped_instance_probability.shape) != expected_future):
        raise ValueError('Local flow-overlay shapes do not match')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Local flow-overlay threshold must be in [0, 1]')
    current_instance = (
        observation_known & (observation_class == instance_class))
    current_instance = current_instance[:, None].to(
        warped_instance_probability.dtype)
    arrival = (
        warped_instance_probability - current_instance >= threshold)
    departure = (
        current_instance - warped_instance_probability >= threshold)
    fused = raw_prediction.clone()
    fused[:, 1:] = torch.where(
        arrival, torch.full_like(fused[:, 1:], instance_class),
        fused[:, 1:])
    fused[:, 1:] = torch.where(
        departure, torch.zeros_like(fused[:, 1:]), fused[:, 1:])
    return fused


def query_conditioned_flow_event_masks(
        observation_class: torch.Tensor,
        observation_known: torch.Tensor,
        warped_instance_probability: torch.Tensor,
        query_future_probability: torch.Tensor,
        flow_threshold: float,
        query_threshold: float,
        instance_class: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keep flow events only where query OCC agrees with their direction."""
    if (observation_class.shape != observation_known.shape or
            observation_class.ndim != 4 or
            warped_instance_probability.ndim != 5 or
            query_future_probability.ndim != 4):
        raise ValueError('Query-conditioned flow inputs have invalid shapes')
    batch_size, z_count, height, width = observation_class.shape
    expected_future = (
        batch_size, warped_instance_probability.shape[1],
        z_count, height, width)
    expected_query = (
        batch_size, warped_instance_probability.shape[1], height, width)
    if (tuple(warped_instance_probability.shape) != expected_future or
            tuple(query_future_probability.shape) != expected_query):
        raise ValueError('Query-conditioned flow shapes do not match')
    if not 0.0 <= flow_threshold <= 1.0:
        raise ValueError('Flow threshold must be in [0, 1]')
    if not 0.0 <= query_threshold <= 1.0:
        raise ValueError('Query threshold must be in [0, 1]')
    current_instance = (
        observation_known & (observation_class == instance_class))
    current_instance = current_instance[:, None].to(
        warped_instance_probability.dtype)
    arrival = (
        warped_instance_probability - current_instance >= flow_threshold)
    departure = (
        current_instance - warped_instance_probability >= flow_threshold)
    query_future = query_future_probability[:, :, None]
    return (arrival & (query_future >= query_threshold),
            departure & (query_future < query_threshold))


def apply_query_conditioned_local_flow_overlay(
        raw_prediction: torch.Tensor,
        observation_class: torch.Tensor,
        observation_known: torch.Tensor,
        warped_instance_probability: torch.Tensor,
        query_future_probability: torch.Tensor,
        flow_threshold: float,
        query_threshold: float,
        instance_class: int = 2) -> torch.Tensor:
    """Overlay only flow events consistent with future query OCC support."""
    if raw_prediction.ndim != 5:
        raise ValueError('Query-conditioned overlay expects [B,T,Z,H,W]')
    if raw_prediction.shape[1] != warped_instance_probability.shape[1] + 1:
        raise ValueError('Overlay horizon count does not match flow horizons')
    arrival, departure = query_conditioned_flow_event_masks(
        observation_class, observation_known, warped_instance_probability,
        query_future_probability, flow_threshold, query_threshold,
        instance_class=instance_class)
    fused = raw_prediction.clone()
    fused[:, 1:] = torch.where(
        arrival, torch.full_like(fused[:, 1:], instance_class), fused[:, 1:])
    fused[:, 1:] = torch.where(
        departure, torch.zeros_like(fused[:, 1:]), fused[:, 1:])
    return fused


def apply_physical_confidence_fusion(
        raw_prediction: torch.Tensor,
        physical_prediction: torch.Tensor,
        confidence_logits: torch.Tensor,
        threshold: float) -> torch.Tensor:
    """Select physical or raw future semantics with causal confidence."""
    if raw_prediction.shape != physical_prediction.shape:
        raise ValueError('Raw and physical prediction shapes must match')
    if raw_prediction.ndim != 5:
        raise ValueError('Confidence fusion expects [B,T,Z,H,W]')
    if tuple(confidence_logits.shape) != (
            raw_prediction.shape[0], raw_prediction.shape[1] - 1,
            *raw_prediction.shape[2:]):
        raise ValueError('Confidence logits must match future predictions')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Physical confidence threshold must be in [0, 1]')
    fused = raw_prediction.clone()
    use_physical = confidence_logits.sigmoid() >= threshold
    fused[:, 1:] = torch.where(
        use_physical, physical_prediction[:, 1:], raw_prediction[:, 1:])
    return fused


def physical_confidence_supervision(
        raw_prediction: torch.Tensor,
        physical_prediction: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = 255) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build labels only where raw/physical disagree and one is correct."""
    if (raw_prediction.shape != physical_prediction.shape or
            raw_prediction.shape != target.shape or
            raw_prediction.ndim != 5):
        raise ValueError('Confidence supervision tensors must match [B,T,Z,H,W]')
    valid = target != ignore_index
    disagreement = raw_prediction != physical_prediction
    raw_correct = raw_prediction == target
    physical_correct = physical_prediction == target
    selection = valid & disagreement & (raw_correct | physical_correct)
    return physical_correct, selection


def physical_confidence_signal_tensor(
        future_logits: torch.Tensor,
        warped_instance_probability: torch.Tensor,
        future_flow: torch.Tensor,
        observation_class: torch.Tensor,
        observation_known: torch.Tensor,
        future_change_logits: Optional[torch.Tensor] = None,
        dynamic_change_prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Build causal candidate signals as [B,C,T*Z,H,W]."""
    if future_logits.ndim != 6:
        raise ValueError('Future logits must have shape [B,T,C,Z,H,W]')
    batch_size, future_count, class_count, z_count, height, width = (
        future_logits.shape)
    future_shape = (batch_size, future_count, z_count, height, width)
    if tuple(warped_instance_probability.shape) != future_shape:
        raise ValueError('Warped instance probability does not match logits')
    if tuple(future_flow.shape) != (
            batch_size, future_count, 2, height, width):
        raise ValueError('Future flow does not match logits')
    if (tuple(observation_class.shape) !=
            (batch_size, z_count, height, width) or
            tuple(observation_known.shape) !=
            (batch_size, z_count, height, width)):
        raise ValueError('Observation tensors do not match future logits')
    if (future_change_logits is not None and
            tuple(future_change_logits.shape) != future_shape):
        raise ValueError('Future change logits do not match future logits')
    if (dynamic_change_prior is not None and
            tuple(dynamic_change_prior.shape) !=
            (batch_size, future_count, height, width)):
        raise ValueError('Dynamic change prior does not match future logits')

    dtype = future_logits.dtype
    future_probability = future_logits.softmax(dim=2)
    raw_confidence, raw_class = future_probability.max(dim=2)
    if class_count > 1:
        top_two = future_probability.topk(2, dim=2).values
        raw_margin = top_two[:, :, 0] - top_two[:, :, 1]
    else:
        raw_margin = raw_confidence
    observation_known = observation_known.bool()
    observation_one_hot = F.one_hot(
        observation_class.long().clamp(min=0, max=class_count - 1),
        num_classes=class_count)
    observation_one_hot = observation_one_hot.permute(
        0, 4, 1, 2, 3).to(dtype)
    observation_one_hot = (
        observation_one_hot * observation_known[:, None].to(dtype))
    observation_one_hot = observation_one_hot[:, None].expand(
        -1, future_count, -1, -1, -1, -1)
    current_instance = (
        observation_known & (observation_class == class_count - 1))
    current_instance = current_instance[:, None].expand(
        -1, future_count, -1, -1, -1).to(dtype)
    arrival_strength = (
        warped_instance_probability - current_instance).clamp(min=0)
    departure_strength = (
        current_instance - warped_instance_probability).clamp(min=0)
    flow_norm = torch.linalg.vector_norm(future_flow, dim=2)
    flow_norm = flow_norm[:, :, None].expand(
        -1, -1, z_count, -1, -1)
    if future_change_logits is None:
        change_probability = torch.zeros_like(warped_instance_probability)
    else:
        change_probability = future_change_logits.sigmoid()
    if dynamic_change_prior is None:
        dynamic_prior = torch.zeros_like(warped_instance_probability)
    else:
        dynamic_prior = dynamic_change_prior[:, :, None].expand(
            -1, -1, z_count, -1, -1)
    known = observation_known[:, None].expand(
        -1, future_count, -1, -1, -1).to(dtype)
    raw_class_normalized = raw_class.to(dtype)
    if class_count > 1:
        raw_class_normalized = raw_class_normalized / (class_count - 1)
    signals = [
        warped_instance_probability,
        current_instance,
        arrival_strength,
        departure_strength,
        flow_norm,
        change_probability,
        dynamic_prior,
        known,
        raw_confidence,
        raw_margin,
        raw_class_normalized,
    ]
    signals.extend(
        observation_one_hot[:, :, class_index]
        for class_index in range(class_count))
    signals.extend(
        future_probability[:, :, class_index]
        for class_index in range(class_count))
    stacked = torch.stack(signals, dim=2)
    return stacked.permute(0, 2, 1, 3, 4, 5).reshape(
        batch_size, len(signals), future_count * z_count, height, width)


class DenseWorldDecoder(nn.Module):
    """Predict current world, persistent future residuals and visibility."""

    def __init__(self, in_channels: int, hidden_channels: int,
                 horizon_count: int, class_count: int, z_count: int,
                 valid_prior: float = 0.16,
                 observation_semantic_logit_scale: float = 2.0,
                 observation_future_semantic_logit_scales: Optional[
                     Sequence[float]] = None,
                 use_future_change_gate: bool = False,
                 future_change_prior: float = 0.01,
                 dynamic_change_logit_scale: float = 0.0,
                 use_query_occupancy_adapter: bool = False,
                 query_occupancy_adapter_scale: float = 1.0,
                 history_count: int = 0,
                 use_wide_history_context: bool = False,
                 use_flow_warp: bool = False,
                 flow_parameterization: str = 'incremental',
                 flow_gate_logit_scale: float = 0.0,
                 use_physical_confidence: bool = False,
                 use_physical_confidence_signals: bool = False,
                 physical_confidence_prior: float = 0.8,
                 observation_valid_logit_scale: float = 4.0):
        super().__init__()
        if min(in_channels, hidden_channels, horizon_count,
               class_count, z_count) < 1:
            raise ValueError('DenseWorldDecoder dimensions must be positive')
        if horizon_count < 2:
            raise ValueError('DenseWorldDecoder needs current and future')
        if not 0.0 < valid_prior < 1.0:
            raise ValueError('valid_prior must be between zero and one')
        if min(observation_semantic_logit_scale,
               observation_valid_logit_scale) <= 0:
            raise ValueError('Observation logit scales must be positive')
        self.horizon_count = int(horizon_count)
        self.future_count = self.horizon_count - 1
        self.class_count = int(class_count)
        self.z_count = int(z_count)
        self.observation_semantic_logit_scale = float(
            observation_semantic_logit_scale)
        if observation_future_semantic_logit_scales is None:
            future_scales = [
                self.observation_semantic_logit_scale
            ] * self.future_count
        else:
            if len(observation_future_semantic_logit_scales) != (
                    self.future_count):
                raise ValueError(
                    'Future observation scales must match future horizons')
            future_scales = [
                float(value)
                for value in observation_future_semantic_logit_scales
            ]
            if any(value <= 0 for value in future_scales):
                raise ValueError(
                    'Future observation scales must be positive')
        # MMCV 1.5 serializes all buffers, including PyTorch buffers marked
        # non-persistent. Keep this inference constant as a plain tuple so it
        # does not change the checkpoint schema.
        self.observation_future_semantic_logit_scales = tuple(future_scales)
        self.observation_valid_logit_scale = float(
            observation_valid_logit_scale)
        self.use_future_change_gate = bool(use_future_change_gate)
        if not 0.0 < future_change_prior < 1.0:
            raise ValueError('Future change prior must be between zero and one')
        self.future_change_prior = float(future_change_prior)
        if dynamic_change_logit_scale < 0:
            raise ValueError(
                'Dynamic change logit scale must be non-negative')
        self.dynamic_change_logit_scale = float(
            dynamic_change_logit_scale)
        self.use_query_occupancy_adapter = bool(
            use_query_occupancy_adapter)
        if query_occupancy_adapter_scale < 0:
            raise ValueError(
                'Query occupancy adapter scale must be non-negative')
        self.query_occupancy_adapter_scale = float(
            query_occupancy_adapter_scale)
        self.history_count = int(history_count)
        if self.history_count < 0:
            raise ValueError('History count must be non-negative')
        self.use_wide_history_context = bool(use_wide_history_context)
        if self.use_wide_history_context and self.history_count < 1:
            raise ValueError('Wide history context requires causal history')
        self.use_flow_warp = bool(use_flow_warp)
        if self.use_flow_warp and self.history_count < 1:
            raise ValueError('Flow warp requires causal history')
        if flow_parameterization not in ('incremental', 'cumulative_current'):
            raise ValueError(
                'Flow parameterization must be incremental or '
                'cumulative_current')
        self.flow_parameterization = flow_parameterization
        if flow_gate_logit_scale < 0:
            raise ValueError('Flow gate logit scale must be non-negative')
        self.flow_gate_logit_scale = float(flow_gate_logit_scale)
        self.use_physical_confidence = bool(use_physical_confidence)
        if self.use_physical_confidence and not self.use_flow_warp:
            raise ValueError('Physical confidence requires flow warp')
        self.use_physical_confidence_signals = bool(
            use_physical_confidence_signals)
        if (self.use_physical_confidence_signals and
                not self.use_physical_confidence):
            raise ValueError(
                'Physical confidence signals require confidence prediction')
        if not 0.0 < physical_confidence_prior < 1.0:
            raise ValueError(
                'Physical confidence prior must be between zero and one')
        self.physical_confidence_prior = float(physical_confidence_prior)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        if self.history_count > 0:
            history_channels = (
                self.history_count * (self.class_count + 2) * self.z_count)
            self.history_encoder = nn.Sequential(
                nn.Conv2d(
                    history_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    hidden_channels, hidden_channels, 3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(inplace=True),
            )
            if self.use_wide_history_context:
                self.history_context_encoder = nn.Sequential(
                    nn.Conv2d(
                        history_channels, hidden_channels, 3, padding=1),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_channels, hidden_channels, 3,
                        padding=2, dilation=2),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_channels, hidden_channels, 3,
                        padding=4, dilation=4),
                )
            else:
                self.history_context_encoder = None
        else:
            self.history_encoder = None
            self.history_context_encoder = None
        self.current_head = nn.Conv2d(
            hidden_channels,
            self.class_count * self.z_count,
            kernel_size=1)
        self.future_residual_head = nn.Conv2d(
            hidden_channels,
            self.future_count * self.class_count * self.z_count,
            kernel_size=1)
        self.valid_head = nn.Conv2d(
            hidden_channels,
            self.horizon_count * self.z_count,
            kernel_size=1)
        if self.use_query_occupancy_adapter:
            self.query_occupancy_adapter = nn.Conv2d(
                self.horizon_count,
                self.future_count * self.z_count,
                kernel_size=1,
                bias=False)
        else:
            self.query_occupancy_adapter = None
        if self.use_future_change_gate:
            self.future_change_head = nn.Conv2d(
                hidden_channels,
                self.future_count * self.z_count,
                kernel_size=1)
            self.future_changed_class_head = nn.Conv2d(
                hidden_channels,
                self.future_count * self.class_count * self.z_count,
                kernel_size=1)
        else:
            self.future_change_head = None
            self.future_changed_class_head = None
        if self.use_flow_warp:
            self.future_flow_head = nn.Conv2d(
                hidden_channels, self.future_count * 2, kernel_size=1)
        else:
            self.future_flow_head = None
        if self.use_physical_confidence:
            self.physical_confidence_head = nn.Conv2d(
                hidden_channels,
                self.future_count * self.z_count,
                kernel_size=1)
        else:
            self.physical_confidence_head = None
        if self.use_physical_confidence_signals:
            signal_count = 11 + 2 * self.class_count
            signal_hidden_channels = min(hidden_channels, 16)
            self.physical_confidence_signal_head = nn.Sequential(
                nn.Conv3d(
                    signal_count, signal_hidden_channels, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(signal_hidden_channels, 1, kernel_size=1))
        else:
            self.physical_confidence_signal_head = None
        for head in (self.current_head, self.future_residual_head,
                     self.valid_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        nn.init.constant_(
            self.valid_head.bias,
            math.log(valid_prior / (1.0 - valid_prior)))
        if self.use_future_change_gate:
            nn.init.zeros_(self.future_change_head.weight)
            nn.init.constant_(
                self.future_change_head.bias,
                math.log(
                    self.future_change_prior /
                    (1.0 - self.future_change_prior)))
            nn.init.zeros_(self.future_changed_class_head.weight)
            nn.init.zeros_(self.future_changed_class_head.bias)
        if self.use_flow_warp:
            nn.init.zeros_(self.future_flow_head.weight)
            nn.init.zeros_(self.future_flow_head.bias)
        if self.query_occupancy_adapter is not None:
            nn.init.zeros_(self.query_occupancy_adapter.weight)
        if self.history_context_encoder is not None:
            nn.init.zeros_(self.history_context_encoder[-1].weight)
            nn.init.zeros_(self.history_context_encoder[-1].bias)
        if self.use_physical_confidence:
            nn.init.zeros_(self.physical_confidence_head.weight)
            nn.init.constant_(
                self.physical_confidence_head.bias,
                math.log(
                    self.physical_confidence_prior /
                    (1.0 - self.physical_confidence_prior)))
        if self.use_physical_confidence_signals:
            nn.init.zeros_(self.physical_confidence_signal_head[-1].weight)
            nn.init.zeros_(self.physical_confidence_signal_head[-1].bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Consume checkpoints produced by the short decay pilot before the
        # MMCV non-persistent-buffer behavior was identified.
        legacy_key = (
            prefix + 'observation_future_semantic_logit_scales')
        if legacy_key in state_dict:
            saved = state_dict.pop(legacy_key)
            expected = saved.new_tensor(
                self.observation_future_semantic_logit_scales)
            if saved.shape != expected.shape or not torch.allclose(
                    saved, expected):
                error_msgs.append(
                    f'{legacy_key} does not match configured future scales')
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def _observation_anchor(self, current_logits, valid_logits,
                            observation_state, observation_valid):
        if observation_state is None and observation_valid is None:
            return current_logits, valid_logits, None
        if observation_state is None or observation_valid is None:
            raise ValueError(
                'Observation state and valid mask must be provided together')
        batch_size, _, _, z_count, height, width = current_logits.shape
        expected_shape = (batch_size, z_count, height, width)
        if (tuple(observation_state.shape) != expected_shape or
                tuple(observation_valid.shape) != expected_shape):
            raise ValueError(
                'Current observation must have shape [B,Z,H,W], got '
                f'{tuple(observation_state.shape)} and '
                f'{tuple(observation_valid.shape)}')
        observation_state = observation_state.long()
        if torch.any((observation_state < 0) |
                     (observation_state > self.class_count)):
            raise ValueError('Current observation has an invalid state')
        known = observation_valid.bool() & (observation_state != 0)
        class_ids = (observation_state - 1).clamp(min=0)
        anchor_logits = F.one_hot(
            class_ids, num_classes=self.class_count)
        anchor_logits = anchor_logits.permute(0, 4, 1, 2, 3).to(
            current_logits.dtype)
        anchor_logits = (
            anchor_logits * self.observation_semantic_logit_scale)
        current_logits = torch.where(
            known[:, None, None],
            anchor_logits[:, None],
            current_logits)
        valid_logits = valid_logits + (
            known[:, None].to(valid_logits.dtype) *
            self.observation_valid_logit_scale)
        return current_logits, valid_logits, known

    def forward(self, bev_feature: torch.Tensor,
                observation_state: Optional[torch.Tensor] = None,
                observation_valid: Optional[torch.Tensor] = None,
                dynamic_occupancy_probability: Optional[
                    torch.Tensor] = None,
                history_state: Optional[torch.Tensor] = None,
                history_valid: Optional[torch.Tensor] = None):
        if bev_feature.ndim != 4:
            raise ValueError('BEV feature must have shape [B,C,H,W]')
        features = self.encoder(bev_feature)
        batch_size, _, height, width = features.shape
        history_features = None
        history_context_features = None
        if self.history_count > 0:
            if history_state is None or history_valid is None:
                raise ValueError(
                    'History-enabled decoder requires state and valid mask')
            expected_history_shape = (
                batch_size, self.history_count, self.z_count,
                height, width)
            if (tuple(history_state.shape) != expected_history_shape or
                    tuple(history_valid.shape) != expected_history_shape):
                raise ValueError(
                    'History must have shape '
                    f'{expected_history_shape}, got '
                    f'{tuple(history_state.shape)} and '
                    f'{tuple(history_valid.shape)}')
            if torch.any((history_state < 0) |
                         (history_state > self.class_count)):
                raise ValueError('History contains an invalid world state')
            history_valid = history_valid.bool()
            history_state = torch.where(
                history_valid, history_state.long(),
                torch.zeros_like(history_state, dtype=torch.long))
            history_one_hot = F.one_hot(
                history_state, num_classes=self.class_count + 1)
            history_one_hot = history_one_hot.permute(
                0, 1, 5, 2, 3, 4).to(features.dtype)
            history_input = torch.cat([
                history_one_hot,
                history_valid[:, :, None].to(features.dtype),
            ], dim=2).reshape(batch_size, -1, height, width)
            history_features = self.history_encoder(history_input)
            if self.history_context_encoder is not None:
                history_context_features = self.history_context_encoder(
                    history_input)
                history_features = (
                    history_features + history_context_features)
            features = features + history_features
        elif history_state is not None or history_valid is not None:
            raise ValueError(
                'History input was provided to a history-disabled decoder')
        current_logits = self.current_head(features).reshape(
            batch_size, 1, self.class_count,
            self.z_count, height, width)
        valid_logits = self.valid_head(features).reshape(
            batch_size, self.horizon_count,
            self.z_count, height, width)
        current_logits, valid_logits, observation_known = (
            self._observation_anchor(
                current_logits, valid_logits,
                observation_state, observation_valid))
        future_residual = self.future_residual_head(features).reshape(
            batch_size, self.future_count, self.class_count,
            self.z_count, height, width)
        future_base = current_logits.expand(
            -1, self.future_count, -1, -1, -1, -1)
        observation_class = None
        if observation_known is not None:
            observation_class = (observation_state.long() - 1).clamp(min=0)
            observation_anchor = F.one_hot(
                observation_class, num_classes=self.class_count)
            observation_anchor = observation_anchor.permute(
                0, 4, 1, 2, 3).to(current_logits.dtype)
            future_scales = current_logits.new_tensor(
                self.observation_future_semantic_logit_scales).view(
                    1, self.future_count, 1, 1, 1, 1)
            future_anchor = observation_anchor[:, None] * future_scales
            future_base = torch.where(
                observation_known[:, None, None],
                future_anchor, future_base)
        future_logits = future_base + future_residual
        future_change_logits = None
        future_changed_class_logits = None
        dynamic_change_prior = None
        future_flow = None
        warped_instance_probability = None
        flow_change_prior = None
        physical_confidence_logits = None
        query_instance_residual = None
        if dynamic_occupancy_probability is not None:
            expected_dynamic_shape = (
                batch_size, self.horizon_count, height, width)
            if tuple(dynamic_occupancy_probability.shape) != (
                    expected_dynamic_shape):
                raise ValueError(
                    'Dynamic occupancy probability must have shape '
                    f'{expected_dynamic_shape}, got '
                    f'{tuple(dynamic_occupancy_probability.shape)}')
            if (torch.any(dynamic_occupancy_probability < 0) or
                    torch.any(dynamic_occupancy_probability > 1)):
                raise ValueError(
                    'Dynamic occupancy probability must be in [0, 1]')
            dynamic_change_prior = torch.abs(
                dynamic_occupancy_probability[:, 1:] -
                dynamic_occupancy_probability[:, :1])
        if self.use_flow_warp:
            if observation_known is None or observation_class is None:
                raise ValueError('Flow warp requires the observation anchor')
            future_flow = self.future_flow_head(features).reshape(
                batch_size, self.future_count, 2, height, width)
            current_instance = (
                observation_known &
                (observation_class == self.class_count - 1))
            warped_instance = current_instance.to(features.dtype)
            warped_steps = []
            for horizon in range(self.future_count):
                if self.flow_parameterization == 'incremental':
                    warped_instance = forward_splat_2d(
                        warped_instance, future_flow[:, horizon])
                    warped_steps.append(warped_instance)
                else:
                    warped_steps.append(forward_splat_2d(
                        current_instance.to(features.dtype),
                        future_flow[:, horizon]))
            warped_instance_probability = torch.stack(warped_steps, dim=1)
            flow_change_prior = torch.abs(
                warped_instance_probability -
                current_instance[:, None].to(features.dtype))
        if self.use_future_change_gate:
            if observation_known is None:
                raise ValueError(
                    'Future change gate requires the observation anchor')
            future_change_logits = self.future_change_head(features).reshape(
                batch_size, self.future_count, self.z_count, height, width)
            if self.dynamic_change_logit_scale > 0:
                if dynamic_change_prior is None:
                    raise ValueError(
                        'Dynamic gate prior requires dynamic occupancy')
                future_change_logits = future_change_logits + (
                    dynamic_change_prior[:, :, None] *
                    self.dynamic_change_logit_scale)
            if self.flow_gate_logit_scale > 0:
                if flow_change_prior is None:
                    raise ValueError(
                        'Flow gate prior requires flow warp')
                future_change_logits = future_change_logits + (
                    flow_change_prior * self.flow_gate_logit_scale)
            future_changed_class_logits = (
                self.future_changed_class_head(features).reshape(
                    batch_size, self.future_count, self.class_count,
                    self.z_count, height, width))
            change_probability = future_change_logits.sigmoid()[:, :, None]
            gated_known_logits = (
                (1.0 - change_probability) * future_base +
                change_probability * future_changed_class_logits)
            future_logits = torch.where(
                observation_known[:, None, None],
                gated_known_logits, future_logits)
        future_logits_without_query_adapter = future_logits
        if self.query_occupancy_adapter is not None:
            if dynamic_occupancy_probability is None:
                raise ValueError(
                    'Query occupancy adapter requires dynamic occupancy')
            query_instance_residual = self.query_occupancy_adapter(
                dynamic_occupancy_probability *
                self.query_occupancy_adapter_scale).reshape(
                    batch_size, self.future_count, self.z_count,
                    height, width)
            instance_selector = F.one_hot(
                torch.as_tensor(
                    self.class_count - 1,
                    device=future_logits.device),
                num_classes=self.class_count).to(future_logits)
            future_logits = future_logits + (
                query_instance_residual[:, :, None] *
                instance_selector.view(1, 1, self.class_count, 1, 1, 1))
        if self.use_physical_confidence:
            physical_confidence_logits = self.physical_confidence_head(
                features).reshape(
                    batch_size, self.future_count, self.z_count,
                    height, width)
            if self.use_physical_confidence_signals:
                confidence_signals = physical_confidence_signal_tensor(
                    future_logits.detach(),
                    warped_instance_probability.detach(),
                    future_flow.detach(),
                    observation_class.detach(),
                    observation_known.detach(),
                    None if future_change_logits is None
                    else future_change_logits.detach(),
                    None if dynamic_change_prior is None
                    else dynamic_change_prior.detach())
                signal_residual = self.physical_confidence_signal_head(
                    confidence_signals).reshape(
                        batch_size, self.future_count, self.z_count,
                        height, width)
                physical_confidence_logits = (
                    physical_confidence_logits + signal_residual)
        world_logits = torch.cat([current_logits, future_logits], dim=1)
        world_logits_without_query_adapter = None
        if self.query_occupancy_adapter is not None:
            world_logits_without_query_adapter = torch.cat([
                current_logits, future_logits_without_query_adapter,
            ], dim=1)
        return {
            'world_logits': world_logits,
            'world_logits_without_query_adapter': (
                world_logits_without_query_adapter),
            'current_logits': current_logits,
            'future_residual': future_residual,
            'valid_logits': valid_logits,
            'observation_known_mask': observation_known,
            'observation_class': observation_class,
            'future_change_logits': future_change_logits,
            'future_changed_class_logits': future_changed_class_logits,
            'dynamic_occupancy_probability': dynamic_occupancy_probability,
            'dynamic_change_prior': dynamic_change_prior,
            'query_instance_residual': query_instance_residual,
            'history_features': history_features,
            'history_context_features': history_context_features,
            'future_flow': future_flow,
            'warped_instance_probability': warped_instance_probability,
            'flow_change_prior': flow_change_prior,
            'physical_confidence_logits': physical_confidence_logits,
        }


def masked_world_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Cross entropy for [B,T,C,Z,H,W] logits with unknown voxels masked."""
    if logits.ndim != 6 or target.ndim != 5:
        raise ValueError(
            'Expected logits [B,T,C,Z,H,W] and target [B,T,Z,H,W]')
    if (logits.shape[:2] != target.shape[:2] or
            logits.shape[3:] != target.shape[2:]):
        raise ValueError('World logits and target shapes are incompatible')
    batch_size, horizon_count, class_count = logits.shape[:3]
    flat_logits = logits.reshape(
        batch_size * horizon_count, class_count, *logits.shape[3:])
    flat_target = target.reshape(
        batch_size * horizon_count, *target.shape[2:]).long()
    per_voxel = F.cross_entropy(
        flat_logits, flat_target, weight=class_weights,
        ignore_index=ignore_index, reduction='none')
    valid = flat_target != ignore_index
    if not torch.any(valid):
        return per_voxel.sum() * 0.0
    return per_voxel[valid].mean()


def selected_world_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        selection: torch.Tensor,
        ignore_index: int = 255,
        class_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply world CE only on a boolean subset of otherwise valid voxels."""
    if selection.shape != target.shape:
        raise ValueError('World CE selection must match target shape')
    selected_target = target.masked_fill(~selection.bool(), ignore_index)
    return masked_world_cross_entropy(
        logits, selected_target,
        ignore_index=ignore_index,
        class_weights=class_weights)


def stable_known_world_selection(
        observation_known: torch.Tensor,
        observation_class: torch.Tensor,
        future_target: torch.Tensor,
        ignore_index: int = 255) -> torch.Tensor:
    """Select observed voxels whose future semantic state stays unchanged."""
    if observation_known.shape != observation_class.shape:
        raise ValueError(
            'Observation known mask and class map must have matching shapes')
    if future_target.ndim != observation_class.ndim + 1:
        raise ValueError(
            'Future world target must add one time dimension')
    if (future_target.shape[0] != observation_class.shape[0] or
            future_target.shape[2:] != observation_class.shape[1:]):
        raise ValueError(
            'Observation and future world target shapes are incompatible')
    return (
        observation_known.bool()[:, None] &
        (future_target != ignore_index) &
        (future_target == observation_class[:, None]))


def selected_binary_cross_entropy_with_logits(
        logits: torch.Tensor,
        target: torch.Tensor,
        selection: torch.Tensor,
        positive_weight: float = 1.0) -> torch.Tensor:
    """Binary cross entropy reduced only over the selected voxels."""
    if logits.shape != target.shape or selection.shape != target.shape:
        raise ValueError('Selected BCE tensors must have matching shapes')
    if positive_weight <= 0:
        raise ValueError('Selected BCE positive weight must be positive')
    per_voxel = F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype),
        pos_weight=logits.new_tensor(positive_weight),
        reduction='none')
    selection = selection.bool()
    if not torch.any(selection):
        return per_voxel.sum() * 0.0
    return per_voxel[selection].mean()


def selected_smooth_l1_loss(prediction: torch.Tensor,
                            target: torch.Tensor,
                            selection: torch.Tensor) -> torch.Tensor:
    """Smooth-L1 loss over explicitly selected flow components."""
    if (prediction.shape != target.shape or
            prediction.shape != selection.shape):
        raise ValueError(
            'Flow prediction, target and selection shapes must match')
    per_element = F.smooth_l1_loss(
        prediction, target.to(prediction.dtype), reduction='none')
    selection = selection.bool()
    if not selection.any():
        return per_element.sum() * 0.0
    return per_element[selection].mean()


def world_visibility_binary_cross_entropy(
        logits: torch.Tensor,
        target: torch.Tensor,
        positive_weight: float = 1.0) -> torch.Tensor:
    """Supervise whether each future world voxel is known or unknown."""
    if logits.ndim != 5 or target.ndim != 5 or logits.shape != target.shape:
        raise ValueError(
            'Visibility logits and target must have shape [B,T,Z,H,W]')
    if positive_weight <= 0:
        raise ValueError('Visibility positive weight must be positive')
    return F.binary_cross_entropy_with_logits(
        logits, target.to(logits.dtype),
        pos_weight=torch.as_tensor(
            positive_weight,
            dtype=logits.dtype,
            device=logits.device))


@HEADS.register_module()
class OccWorldHead(OccHead):
    """Keep UniAD instance occupancy and add dense 3D world prediction."""

    def __init__(self,
                 world_z_count: int = 10,
                 world_class_count: int = 3,
                 world_hidden_channels: int = 64,
                 world_ignore_index: int = 255,
                 world_loss_weight: float = 1.0,
                 world_current_loss_weight: float = 1.0,
                 world_future_loss_weight: float = 1.0,
                 world_future_transition_loss_weight: float = 0.0,
                 world_future_stability_loss_weight: float = 0.0,
                 world_future_change_gate_loss_weight: float = 0.0,
                 world_future_changed_class_loss_weight: float = 0.0,
                 world_future_change_positive_weight: float = 10.0,
                 world_flow_loss_weight: float = 0.0,
                 world_physical_confidence_loss_weight: float = 0.0,
                 world_physical_confidence_positive_weight: float = 1.0,
                 world_visibility_loss_weight: float = 1.0,
                 world_valid_positive_weight: float = 5.0,
                 world_valid_prior: float = 0.16,
                 world_use_observation_anchor: bool = False,
                 world_observation_semantic_logit_scale: float = 2.0,
                 world_observation_future_semantic_logit_scales: Optional[
                     Sequence[float]] = None,
                 world_use_future_change_gate: bool = False,
                 world_future_change_prior: float = 0.01,
                 world_dynamic_change_logit_scale: float = 0.0,
                 world_use_query_occupancy_adapter: bool = False,
                 world_query_occupancy_adapter_scale: float = 1.0,
                 world_history_count: int = 0,
                 world_use_wide_history_context: bool = False,
                 world_use_flow_warp: bool = False,
                 world_flow_parameterization: str = 'incremental',
                 world_flow_gate_logit_scale: float = 0.0,
                 world_align_bev_to_world_layout: bool = False,
                 world_physical_flow_fusion_threshold: Optional[
                     float] = None,
                 world_local_flow_overlay_threshold: Optional[
                     float] = None,
                 world_use_physical_confidence: bool = False,
                 world_use_physical_confidence_signals: bool = False,
                 world_physical_confidence_prior: float = 0.8,
                 world_physical_confidence_flow_threshold: float = 0.5,
                 world_physical_confidence_threshold: Optional[
                     float] = None,
                 world_observation_valid_logit_scale: float = 4.0,
                 world_class_weights: Optional[Sequence[float]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        loss_weights = (
            world_loss_weight, world_current_loss_weight,
            world_future_loss_weight,
            world_future_transition_loss_weight,
            world_future_stability_loss_weight,
            world_future_change_gate_loss_weight,
            world_future_changed_class_loss_weight,
            world_flow_loss_weight,
            world_physical_confidence_loss_weight,
            world_visibility_loss_weight)
        if any(weight < 0 for weight in loss_weights):
            raise ValueError('World loss weights must be non-negative')
        if world_valid_positive_weight <= 0:
            raise ValueError('world_valid_positive_weight must be positive')
        if world_future_change_positive_weight <= 0:
            raise ValueError(
                'world_future_change_positive_weight must be positive')
        if world_physical_confidence_positive_weight <= 0:
            raise ValueError(
                'world_physical_confidence_positive_weight must be positive')
        self.world_z_count = int(world_z_count)
        self.world_class_count = int(world_class_count)
        self.world_ignore_index = int(world_ignore_index)
        self.world_loss_weight = float(world_loss_weight)
        self.world_current_loss_weight = float(world_current_loss_weight)
        self.world_future_loss_weight = float(world_future_loss_weight)
        self.world_future_transition_loss_weight = float(
            world_future_transition_loss_weight)
        self.world_future_stability_loss_weight = float(
            world_future_stability_loss_weight)
        self.world_future_change_gate_loss_weight = float(
            world_future_change_gate_loss_weight)
        self.world_future_changed_class_loss_weight = float(
            world_future_changed_class_loss_weight)
        self.world_future_change_positive_weight = float(
            world_future_change_positive_weight)
        self.world_flow_loss_weight = float(world_flow_loss_weight)
        self.world_physical_confidence_loss_weight = float(
            world_physical_confidence_loss_weight)
        self.world_physical_confidence_positive_weight = float(
            world_physical_confidence_positive_weight)
        if world_physical_flow_fusion_threshold is not None:
            if not 0.0 <= world_physical_flow_fusion_threshold <= 1.0:
                raise ValueError(
                    'Physical flow-fusion threshold must be in [0, 1]')
            if not world_use_flow_warp:
                raise ValueError(
                    'Physical flow fusion requires flow warp')
        self.world_physical_flow_fusion_threshold = (
            None if world_physical_flow_fusion_threshold is None
            else float(world_physical_flow_fusion_threshold))
        if world_local_flow_overlay_threshold is not None:
            if not 0.0 <= world_local_flow_overlay_threshold <= 1.0:
                raise ValueError(
                    'Local flow-overlay threshold must be in [0, 1]')
            if not world_use_flow_warp:
                raise ValueError('Local flow overlay requires flow warp')
            if self.world_physical_flow_fusion_threshold is not None:
                raise ValueError(
                    'Legacy physical fusion and local overlay are exclusive')
        self.world_local_flow_overlay_threshold = (
            None if world_local_flow_overlay_threshold is None
            else float(world_local_flow_overlay_threshold))
        self.world_use_physical_confidence = bool(
            world_use_physical_confidence)
        if not 0.0 <= world_physical_confidence_flow_threshold <= 1.0:
            raise ValueError(
                'Physical confidence flow threshold must be in [0, 1]')
        self.world_physical_confidence_flow_threshold = float(
            world_physical_confidence_flow_threshold)
        if world_physical_confidence_threshold is not None:
            if not 0.0 <= world_physical_confidence_threshold <= 1.0:
                raise ValueError(
                    'Physical confidence threshold must be in [0, 1]')
            if not self.world_use_physical_confidence:
                raise ValueError(
                    'Physical confidence fusion requires its prediction head')
            if (self.world_physical_flow_fusion_threshold is not None or
                    self.world_local_flow_overlay_threshold is not None):
                raise ValueError(
                    'Physical confidence and direct flow fusion are exclusive')
        self.world_physical_confidence_threshold = (
            None if world_physical_confidence_threshold is None
            else float(world_physical_confidence_threshold))
        self.world_visibility_loss_weight = float(
            world_visibility_loss_weight)
        self.world_valid_positive_weight = float(
            world_valid_positive_weight)
        self.world_use_observation_anchor = bool(
            world_use_observation_anchor)
        # Official UniAD BEV tensors index rows from negative to positive Y,
        # while generated OccWorld volumes use image-aligned rows. Keep this
        # opt-in so checkpoints trained before the coordinate audit reproduce.
        self.world_align_bev_to_world_layout = bool(
            world_align_bev_to_world_layout)
        self.world_decoder = DenseWorldDecoder(
            in_channels=self.bev_proj_dim,
            hidden_channels=int(world_hidden_channels),
            horizon_count=self.n_future + 1,
            class_count=self.world_class_count,
            z_count=self.world_z_count,
            valid_prior=world_valid_prior,
            observation_semantic_logit_scale=(
                world_observation_semantic_logit_scale),
            observation_future_semantic_logit_scales=(
                world_observation_future_semantic_logit_scales),
            use_future_change_gate=world_use_future_change_gate,
            future_change_prior=world_future_change_prior,
            dynamic_change_logit_scale=(
                world_dynamic_change_logit_scale),
            use_query_occupancy_adapter=(
                world_use_query_occupancy_adapter),
            query_occupancy_adapter_scale=(
                world_query_occupancy_adapter_scale),
            history_count=world_history_count,
            use_wide_history_context=world_use_wide_history_context,
            use_flow_warp=world_use_flow_warp,
            flow_parameterization=world_flow_parameterization,
            flow_gate_logit_scale=world_flow_gate_logit_scale,
            use_physical_confidence=world_use_physical_confidence,
            use_physical_confidence_signals=(
                world_use_physical_confidence_signals),
            physical_confidence_prior=world_physical_confidence_prior,
            observation_valid_logit_scale=(
                world_observation_valid_logit_scale))
        if world_class_weights is None:
            class_weights = torch.empty(0, dtype=torch.float32)
        else:
            if len(world_class_weights) != self.world_class_count:
                raise ValueError(
                    'world_class_weights length must equal world_class_count')
            class_weights = torch.as_tensor(
                world_class_weights, dtype=torch.float32)
        self.register_buffer(
            'world_class_weights', class_weights, persistent=True)

    def _project_world_bev(self, bev_feat: torch.Tensor) -> torch.Tensor:
        if bev_feat.ndim != 3:
            raise ValueError('OccWorld BEV must have shape [HW,B,C]')
        spatial_size, batch_size, channels = bev_feat.shape
        expected_size = self.bev_size[0] * self.bev_size[1]
        if spatial_size != expected_size:
            raise ValueError(
                f'BEV length {spatial_size} does not match {expected_size}')
        base_state = bev_feat.permute(1, 2, 0).reshape(
            batch_size, channels, self.bev_size[0], self.bev_size[1])
        if self.bevslicer:
            base_state = self.bev_sampler(base_state)
        world_bev = self.bev_light_proj(base_state)
        if self.world_align_bev_to_world_layout:
            world_bev = bev_to_world_layout(world_bev)
        return world_bev

    def forward_world(self, bev_feat: torch.Tensor,
                      current_world_state=None,
                      current_world_valid=None,
                      dynamic_occupancy_probability=None,
                      history_world_state=None,
                      history_world_valid=None):
        if self.world_use_observation_anchor:
            if current_world_state is None or current_world_valid is None:
                raise ValueError(
                    'Observation anchor requires current world input')
        else:
            current_world_state = None
            current_world_valid = None
        return self.world_decoder(
            self._project_world_bev(bev_feat),
            observation_state=current_world_state,
            observation_valid=current_world_valid,
            dynamic_occupancy_probability=(
                dynamic_occupancy_probability),
            history_state=history_world_state,
            history_valid=history_world_valid)

    def _dynamic_occupancy_probability(self, pred_ins_logits,
                                       outs_dict):
        probability = pred_ins_logits.sigmoid()
        if self.test_with_track_score and 'track_scores' in outs_dict:
            track_scores = outs_dict['track_scores'].to(probability)
            probability = probability * track_scores[
                :, :, None, None, None]
        dynamic_probability = probability.max(
            dim=1).values[:, :self.n_future + 1]
        if self.world_align_bev_to_world_layout:
            dynamic_probability = bev_to_world_layout(dynamic_probability)
        return dynamic_probability

    def loss_world(self, outputs, gt_world_occ: torch.Tensor,
                   gt_world_valid: torch.Tensor,
                   gt_flow: Optional[torch.Tensor] = None):
        class_weights = (
            None if self.world_class_weights.numel() == 0
            else self.world_class_weights.to(outputs['world_logits']))
        current_loss = masked_world_cross_entropy(
            outputs['world_logits'][:, :1], gt_world_occ[:, :1],
            ignore_index=self.world_ignore_index,
            class_weights=class_weights)
        future_loss = masked_world_cross_entropy(
            outputs['world_logits'][:, 1:], gt_world_occ[:, 1:],
            ignore_index=self.world_ignore_index,
            class_weights=class_weights)
        visibility_loss = world_visibility_binary_cross_entropy(
            outputs['valid_logits'], gt_world_valid,
            positive_weight=self.world_valid_positive_weight)
        losses = {
            'loss_world_current_ce': (
                current_loss * self.world_loss_weight *
                self.world_current_loss_weight),
            'loss_world_future_ce': (
                future_loss * self.world_loss_weight *
                self.world_future_loss_weight),
            'loss_world_visibility': (
                visibility_loss * self.world_loss_weight *
                self.world_visibility_loss_weight),
        }
        if self.world_future_transition_loss_weight > 0:
            observation_known = outputs['observation_known_mask']
            observation_class = outputs['observation_class']
            if observation_known is None or observation_class is None:
                raise ValueError(
                    'Transition loss requires the observation anchor')
            future_target = gt_world_occ[:, 1:]
            transition = (
                observation_known[:, None] &
                (future_target != self.world_ignore_index) &
                (future_target != observation_class[:, None]))
            transition_loss = selected_world_cross_entropy(
                outputs['world_logits'][:, 1:], future_target,
                transition,
                ignore_index=self.world_ignore_index,
                class_weights=class_weights)
            losses['loss_world_future_transition_ce'] = (
                transition_loss * self.world_loss_weight *
                self.world_future_transition_loss_weight)
        if self.world_future_stability_loss_weight > 0:
            observation_known = outputs['observation_known_mask']
            observation_class = outputs['observation_class']
            if observation_known is None or observation_class is None:
                raise ValueError(
                    'Stability loss requires the observation anchor')
            future_target = gt_world_occ[:, 1:]
            stable_known = stable_known_world_selection(
                observation_known, observation_class, future_target,
                ignore_index=self.world_ignore_index)
            stability_loss = selected_world_cross_entropy(
                outputs['world_logits'][:, 1:], future_target,
                stable_known,
                ignore_index=self.world_ignore_index,
                class_weights=class_weights)
            losses['loss_world_future_stability_ce'] = (
                stability_loss * self.world_loss_weight *
                self.world_future_stability_loss_weight)
        gate_loss_enabled = self.world_future_change_gate_loss_weight > 0
        changed_loss_enabled = (
            self.world_future_changed_class_loss_weight > 0)
        if gate_loss_enabled or changed_loss_enabled:
            change_logits = outputs['future_change_logits']
            changed_class_logits = outputs[
                'future_changed_class_logits']
            observation_known = outputs['observation_known_mask']
            observation_class = outputs['observation_class']
            if (change_logits is None or changed_class_logits is None or
                    observation_known is None or observation_class is None):
                raise ValueError(
                    'Change-gate losses require the future change gate')
            future_target = gt_world_occ[:, 1:]
            gate_selection = (
                observation_known[:, None] &
                (future_target != self.world_ignore_index))
            changed_target = (
                future_target != observation_class[:, None])
            if gate_loss_enabled:
                gate_loss = selected_binary_cross_entropy_with_logits(
                    change_logits, changed_target,
                    gate_selection,
                    positive_weight=(
                        self.world_future_change_positive_weight))
                losses['loss_world_future_change_gate'] = (
                    gate_loss * self.world_loss_weight *
                    self.world_future_change_gate_loss_weight)
            if changed_loss_enabled:
                changed_class_loss = selected_world_cross_entropy(
                    changed_class_logits, future_target,
                    gate_selection & changed_target,
                    ignore_index=self.world_ignore_index,
                    class_weights=class_weights)
                losses['loss_world_future_changed_class_ce'] = (
                    changed_class_loss * self.world_loss_weight *
                    self.world_future_changed_class_loss_weight)
        if self.world_flow_loss_weight > 0:
            future_flow = outputs['future_flow']
            if future_flow is None or gt_flow is None:
                raise ValueError(
                    'Flow loss requires predicted and ground-truth flow')
            if gt_flow.ndim != 5 or gt_flow.shape[2] != 2:
                raise ValueError('gt_flow must have shape [B,T,2,H,W]')
            flow_target = gt_flow[:, :future_flow.shape[1]].to(future_flow)
            if flow_target.shape != future_flow.shape:
                raise ValueError(
                    'Predicted and ground-truth flow shapes do not match')
            if self.world_align_bev_to_world_layout:
                flow_target = flow_to_world_layout(
                    flow_target, ignore_index=self.world_ignore_index)
            if self.world_decoder.flow_parameterization == (
                    'cumulative_current'):
                flow_target = compose_incremental_flow_2d(
                    flow_target, ignore_index=self.world_ignore_index)
            flow_selection = flow_target != self.world_ignore_index
            flow_loss = selected_smooth_l1_loss(
                future_flow, flow_target, flow_selection)
            losses['loss_world_flow'] = (
                flow_loss * self.world_loss_weight *
                self.world_flow_loss_weight)
        if self.world_physical_confidence_loss_weight > 0:
            confidence_logits = outputs['physical_confidence_logits']
            observation_class = outputs['observation_class']
            observation_known = outputs['observation_known_mask']
            warped_instance = outputs['warped_instance_probability']
            if (confidence_logits is None or observation_class is None or
                    observation_known is None or warped_instance is None):
                raise ValueError(
                    'Physical confidence loss requires its head and flow warp')
            raw_prediction = outputs['world_logits'].detach().argmax(dim=2)
            physical_prediction = apply_physical_flow_fusion(
                raw_prediction, observation_class, observation_known,
                warped_instance,
                threshold=self.world_physical_confidence_flow_threshold,
                instance_class=self.world_class_count - 1)
            confidence_target, confidence_selection = (
                physical_confidence_supervision(
                    raw_prediction, physical_prediction, gt_world_occ,
                    ignore_index=self.world_ignore_index))
            confidence_loss = selected_binary_cross_entropy_with_logits(
                confidence_logits, confidence_target[:, 1:],
                confidence_selection[:, 1:],
                positive_weight=(
                    self.world_physical_confidence_positive_weight))
            losses['loss_world_physical_confidence'] = (
                confidence_loss * self.world_loss_weight *
                self.world_physical_confidence_loss_weight)
        return losses

    def forward_train(self, bev_feat, outs_dict, gt_inds_list=None,
                      gt_segmentation=None, gt_instance=None,
                      gt_img_is_valid=None, gt_world_occ=None,
                      gt_world_valid=None, current_world_state=None,
                      current_world_valid=None,
                      history_world_state=None,
                      history_world_valid=None,
                      gt_flow=None):
        losses, occ_predictions = super().forward_train(
            bev_feat, outs_dict,
            gt_inds_list=gt_inds_list,
            gt_segmentation=gt_segmentation,
            gt_instance=gt_instance,
            gt_img_is_valid=gt_img_is_valid,
            history_world_state=history_world_state,
            history_world_valid=history_world_valid,
            return_occ_predictions=True)
        if gt_world_occ is not None and self.world_loss_weight > 0:
            if gt_world_valid is None:
                raise ValueError(
                    'gt_world_valid is required with gt_world_occ')
            world_outputs = self.forward_world(
                bev_feat, current_world_state, current_world_valid,
                dynamic_occupancy_probability=(
                    self._dynamic_occupancy_probability(
                        occ_predictions['pred_ins_logits'], outs_dict)),
                history_world_state=history_world_state,
                history_world_valid=history_world_valid)
            losses.update(self.loss_world(
                world_outputs, gt_world_occ, gt_world_valid,
                gt_flow=gt_flow))
        return losses

    def forward_test(self, bev_feat, outs_dict, no_query=False,
                     gt_segmentation=None, gt_instance=None,
                     gt_img_is_valid=None, current_world_state=None,
                     current_world_valid=None,
                     history_world_state=None,
                     history_world_valid=None):
        outputs = super().forward_test(
            bev_feat, outs_dict, no_query=no_query,
            gt_segmentation=gt_segmentation,
            gt_instance=gt_instance,
            gt_img_is_valid=gt_img_is_valid,
            history_world_state=history_world_state,
            history_world_valid=history_world_valid)
        pred_ins_sigmoid = outputs.get('pred_ins_sigmoid')
        if pred_ins_sigmoid is None:
            dynamic_occupancy_probability = bev_feat.new_zeros((
                bev_feat.shape[1], self.n_future + 1,
                self.bev_size[0], self.bev_size[1]))
        else:
            dynamic_occupancy_probability = pred_ins_sigmoid.max(
                dim=1).values[:, :self.n_future + 1]
        if self.world_align_bev_to_world_layout:
            dynamic_occupancy_probability = bev_to_world_layout(
                dynamic_occupancy_probability)
        world_outputs = self.forward_world(
            bev_feat, current_world_state, current_world_valid,
            dynamic_occupancy_probability=dynamic_occupancy_probability,
            history_world_state=history_world_state,
            history_world_valid=history_world_valid)
        outputs.update(world_outputs)
        raw_world_prediction = world_outputs['world_logits'].argmax(dim=2)
        world_prediction = self._fuse_world_prediction(
            raw_world_prediction, world_outputs)
        ablation_logits = world_outputs[
            'world_logits_without_query_adapter']
        if ablation_logits is not None:
            outputs['query_adapter_ablation_world_pred'] = (
                self._fuse_world_prediction(
                    ablation_logits.argmax(dim=2), world_outputs))
        outputs['world_pred'] = world_prediction
        outputs['world_valid_probability'] = world_outputs[
            'valid_logits'].sigmoid()
        for key in (
                'planning_actor_future', 'planning_actor_boxes_3d',
                'planning_actor_scores', 'planning_actor_valid'):
            if key in outs_dict:
                outputs[key] = outs_dict[key]
        return outputs

    def _fuse_world_prediction(self, raw_world_prediction, world_outputs):
        world_prediction = raw_world_prediction
        if self.world_physical_confidence_threshold is not None:
            physical_prediction = apply_physical_flow_fusion(
                raw_world_prediction,
                world_outputs['observation_class'],
                world_outputs['observation_known_mask'],
                world_outputs['warped_instance_probability'],
                threshold=self.world_physical_confidence_flow_threshold,
                instance_class=self.world_class_count - 1)
            world_prediction = apply_physical_confidence_fusion(
                raw_world_prediction, physical_prediction,
                world_outputs['physical_confidence_logits'],
                threshold=self.world_physical_confidence_threshold)
        elif self.world_local_flow_overlay_threshold is not None:
            world_prediction = apply_local_flow_overlay(
                raw_world_prediction,
                world_outputs['observation_class'],
                world_outputs['observation_known_mask'],
                world_outputs['warped_instance_probability'],
                threshold=self.world_local_flow_overlay_threshold,
                instance_class=self.world_class_count - 1)
        elif self.world_physical_flow_fusion_threshold is not None:
            world_prediction = apply_physical_flow_fusion(
                raw_world_prediction,
                world_outputs['observation_class'],
                world_outputs['observation_known_mask'],
                world_outputs['warped_instance_probability'],
                threshold=self.world_physical_flow_fusion_threshold,
                instance_class=self.world_class_count - 1)
        return world_prediction
