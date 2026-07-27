from types import SimpleNamespace

import torch
import torch.nn as nn

from tools.analysis_tools.convert_occworld_flow_checkpoint import (
    convert_state_dict,
)

from projects.mmdet3d_plugin.uniad.dense_heads.occworld_head import (
    DenseWorldDecoder,
    OccWorldHead,
    apply_physical_confidence_fusion,
    apply_physical_flow_fusion,
    apply_local_flow_overlay,
    apply_query_conditioned_local_flow_overlay,
    bev_to_world_layout,
    compose_incremental_flow_2d,
    flow_to_world_layout,
    forward_splat_2d,
    masked_world_cross_entropy,
    physical_confidence_supervision,
    physical_confidence_signal_tensor,
    selected_binary_cross_entropy_with_logits,
    selected_smooth_l1_loss,
    selected_world_cross_entropy,
    stable_known_world_selection,
    world_visibility_binary_cross_entropy,
)
from projects.mmdet3d_plugin.uniad.dense_heads.occ_head import OccHead


def test_occ_head_test_without_gt_returns_zero_shape_for_empty_queries():
    head = SimpleNamespace(n_future=4, bev_size=(3, 5))
    bev_feat = torch.zeros((15, 2, 8))

    output = OccHead.forward_test(
        head, bev_feat, outs_dict={}, no_query=True)

    assert output['seg_gt'] is None
    assert output['ins_seg_gt'] is None
    assert tuple(output['seg_out'].shape) == (2, 5, 1, 3, 5)
    assert tuple(output['ins_seg_out'].shape) == (2, 5, 3, 5)


def test_compose_incremental_flow_follows_moving_source():
    flow = torch.full((1, 2, 2, 3, 5), 255.0)
    flow[0, 0, :, 1, 1] = torch.tensor([0.0, 1.0])
    flow[0, 1, :, 1, 2] = torch.tensor([0.0, 1.5])

    cumulative = compose_incremental_flow_2d(flow)

    assert torch.equal(
        cumulative[0, 0, :, 1, 1], torch.tensor([0.0, 1.0]))
    assert torch.equal(
        cumulative[0, 1, :, 1, 1], torch.tensor([0.0, 2.5]))
    assert torch.all(cumulative[0, :, :, 0, 0] == 255.0)


def test_compose_incremental_flow_invalidates_broken_trajectory():
    flow = torch.full((1, 2, 2, 2, 3), 255.0)
    flow[0, 0, :, 0, 0] = torch.tensor([0.0, 1.0])

    cumulative = compose_incremental_flow_2d(flow)

    assert torch.equal(
        cumulative[0, 0, :, 0, 0], torch.tensor([0.0, 1.0]))
    assert torch.all(cumulative[0, 1, :, 0, 0] == 255.0)


def test_checkpoint_conversion_prefix_sums_flow_heads():
    weight = torch.arange(8.0).reshape(8, 1, 1, 1)
    bias = torch.arange(8.0)
    state_dict = {
        'occ_head.world_decoder.future_flow_head.weight': weight,
        'occ_head.world_decoder.future_flow_head.bias': bias,
        'untouched': torch.tensor(3.0),
    }

    converted, matched = convert_state_dict(state_dict, future_count=4)

    expected_weight = weight.reshape(4, 2, 1, 1, 1).cumsum(dim=0)
    expected_bias = bias.reshape(4, 2).cumsum(dim=0)
    torch.testing.assert_close(
        converted[matched[0]].reshape_as(expected_weight), expected_weight)
    torch.testing.assert_close(
        converted[matched[1]].reshape_as(expected_bias), expected_bias)
    assert converted['untouched'].item() == 3.0


def test_bev_to_world_layout_flips_only_height_axis():
    bev = torch.tensor([[[[1, 2, 3], [4, 5, 6]]]])

    aligned = bev_to_world_layout(bev)

    assert aligned.tolist() == [[[[4, 5, 6], [1, 2, 3]]]]


def test_flow_to_world_layout_flips_height_negates_dy_and_keeps_ignore():
    flow = torch.tensor([[[
        [1.0, 255.0],
        [2.0, 3.0],
    ], [
        [4.0, 255.0],
        [5.0, 6.0],
    ]]])

    aligned = flow_to_world_layout(flow)

    assert aligned[0, 0].tolist() == [[-2.0, -3.0], [-1.0, 255.0]]
    assert aligned[0, 1].tolist() == [[5.0, 6.0], [4.0, 255.0]]


def test_occworld_head_aligns_projected_bev_rows_when_enabled():
    head = OccWorldHead.__new__(OccWorldHead)
    nn.Module.__init__(head)
    head.bev_size = (2, 3)
    head.bevslicer = False
    head.bev_light_proj = nn.Identity()
    head.world_align_bev_to_world_layout = True
    bev = torch.arange(6.0).reshape(6, 1, 1)

    projected = head._project_world_bev(bev)

    assert projected[0, 0].tolist() == [[3.0, 4.0, 5.0],
                                       [0.0, 1.0, 2.0]]


def test_occworld_head_aligns_training_dynamic_probability_rows():
    head = OccWorldHead.__new__(OccWorldHead)
    nn.Module.__init__(head)
    head.n_future = 1
    head.test_with_track_score = False
    head.world_align_bev_to_world_layout = True
    logits = torch.tensor([[[
        [[-2.0, -1.0], [1.0, 2.0]],
        [[-4.0, -3.0], [3.0, 4.0]],
    ]]])

    probability = head._dynamic_occupancy_probability(logits, {})

    expected = logits.sigmoid().max(dim=1).values.flip(-2)
    torch.testing.assert_close(probability, expected)


def test_physical_flow_fusion_moves_instance_and_keeps_persistence():
    prediction = torch.tensor([[[[[2, 0, 1]]], [[[2, 0, 1]]]]])
    observation_class = torch.tensor([[[[2, 0, 1]]]])
    observation_known = torch.ones_like(
        observation_class, dtype=torch.bool)
    warped = torch.tensor([[[[[0.0, 1.0, 0.0]]]]])

    fused = apply_physical_flow_fusion(
        prediction, observation_class, observation_known,
        warped, threshold=0.7)

    assert fused[0, 1, 0, 0].tolist() == [0, 2, 1]


def test_local_flow_overlay_preserves_raw_outside_instance_events():
    prediction = torch.tensor([[
        [[[0, 0, 2, 2]]],
        [[[1, 0, 1, 2]]],
    ]])
    observation_class = torch.tensor([[[[0, 0, 2, 2]]]])
    observation_known = torch.ones_like(
        observation_class, dtype=torch.bool)
    warped = torch.tensor([[[[[0.0, 1.0, 0.0, 0.0]]]]])

    fused = apply_local_flow_overlay(
        prediction, observation_class, observation_known,
        warped, threshold=0.7)

    assert fused[0, 1, 0, 0].tolist() == [1, 2, 0, 0]


def test_query_conditioned_overlay_rejects_inconsistent_flow_events():
    prediction = torch.tensor([[
        [[[2, 0, 0]]],
        [[[1, 0, 1]]],
    ]])
    observation_class = torch.tensor([[[[2, 0, 0]]]])
    observation_known = torch.ones_like(
        observation_class, dtype=torch.bool)
    warped = torch.tensor([[[[[0.0, 1.0, 1.0]]]]])
    query_future = torch.tensor([[[[0.8, 0.9, 0.1]]]])

    fused = apply_query_conditioned_local_flow_overlay(
        prediction, observation_class, observation_known,
        warped, query_future,
        flow_threshold=0.7, query_threshold=0.5)

    # Query OCC blocks departure at 0 and arrival at 2, but supports arrival 1.
    assert fused[0, 1, 0, 0].tolist() == [1, 2, 1]


def test_physical_confidence_selects_only_supervised_disagreements():
    raw = torch.tensor([
        [[[[0, 1, 2]]], [[[0, 1, 2]]]],
    ])
    physical = raw.clone()
    physical[:, 1, 0, 0] = torch.tensor([1, 0, 2])
    target = raw.clone()
    target[:, 1, 0, 0] = torch.tensor([0, 0, 255])

    confidence_target, selection = physical_confidence_supervision(
        raw, physical, target)
    logits = torch.tensor([[[[[-10.0, 10.0, 10.0]]]]])
    fused = apply_physical_confidence_fusion(
        raw, physical, logits, threshold=0.5)

    assert selection[0, 1, 0, 0].tolist() == [True, True, False]
    assert confidence_target[0, 1, 0, 0].tolist() == [False, True, False]
    assert fused[0, 1, 0, 0].tolist() == [0, 0, 2]


def test_forward_splat_moves_value_with_dy_dx_flow():
    values = torch.zeros((1, 1, 3, 4))
    values[0, 0, 1, 1] = 1.0
    flow = torch.zeros((1, 2, 3, 4))
    flow[0, 1, 1, 1] = 1.0

    warped = forward_splat_2d(values, flow)

    assert warped[0, 0, 1, 1] == 0
    assert warped[0, 0, 1, 2] == 1
    assert warped.sum() == 1


def test_forward_splat_is_differentiable_for_fractional_flow():
    values = torch.zeros((1, 1, 3, 4))
    values[0, 0, 1, 1] = 1.0
    flow = torch.zeros((1, 2, 3, 4), requires_grad=True)
    with torch.no_grad():
        flow[0, 1, 1, 1] = 0.5

    warped = forward_splat_2d(values, flow)
    loss = warped[0, 0, 1, 2]
    loss.backward()

    torch.testing.assert_close(warped[0, 0, 1, 1], torch.tensor(0.5))
    torch.testing.assert_close(warped[0, 0, 1, 2], torch.tensor(0.5))
    assert flow.grad[0, 1, 1, 1] != 0


def test_dense_world_decoder_emits_one_3d_volume_per_horizon():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=2, class_count=3, z_count=2)

    outputs = decoder(torch.randn(1, 4, 5, 6))
    logits = outputs['world_logits']

    assert logits.shape == (1, 2, 3, 2, 5, 6)
    torch.testing.assert_close(logits, torch.zeros_like(logits))
    assert outputs['current_logits'].shape == (1, 1, 3, 2, 5, 6)
    assert outputs['future_residual'].shape == (1, 1, 3, 2, 5, 6)
    assert outputs['valid_logits'].shape == (1, 2, 2, 5, 6)
    torch.testing.assert_close(
        outputs['world_logits'][:, 1:],
        outputs['current_logits'].expand(-1, 1, -1, -1, -1, -1))
    torch.testing.assert_close(
        outputs['valid_logits'].sigmoid(),
        torch.full_like(outputs['valid_logits'], 0.16))


def test_causal_observation_anchor_preserves_known_state_at_initialization():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=2,
        valid_prior=0.16,
        observation_semantic_logit_scale=2.0,
        observation_valid_logit_scale=4.0)
    state = torch.zeros((1, 2, 3, 4), dtype=torch.long)
    valid = torch.zeros_like(state, dtype=torch.bool)
    state[0, 0, 0, :4] = torch.tensor([1, 2, 3, 2])
    valid[0, 0, 0, :3] = True

    outputs = decoder(
        torch.randn(1, 4, 3, 4),
        observation_state=state,
        observation_valid=valid)
    prediction = outputs['world_logits'].argmax(dim=2)
    known = valid & (state != 0)

    assert prediction[0, 0, 0, 0, :3].tolist() == [0, 1, 2]
    torch.testing.assert_close(
        prediction[:, 1:],
        prediction[:, :1].expand(-1, 2, -1, -1, -1))
    torch.testing.assert_close(
        outputs['current_logits'][0, 0, :, 1, 2, 3],
        torch.zeros(3))
    torch.testing.assert_close(outputs['observation_known_mask'], known)
    valid_probability = outputs['valid_logits'].sigmoid()
    expected_known = torch.sigmoid(
        torch.logit(torch.tensor(0.16)) + 4.0)
    torch.testing.assert_close(
        valid_probability[0, :, 0, 0, 0],
        expected_known.expand(3))
    torch.testing.assert_close(
        valid_probability[0, :, 1, 2, 3],
        torch.full((3,), 0.16))


def test_future_anchor_decay_preserves_argmax_with_smaller_margins():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=5, class_count=3, z_count=1,
        observation_semantic_logit_scale=2.0,
        observation_future_semantic_logit_scales=[1.5, 1.0, 0.75, 0.5])
    state = torch.full((1, 1, 2, 2), 3)
    valid = torch.ones_like(state, dtype=torch.bool)

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        observation_state=state,
        observation_valid=valid)

    assert outputs['world_logits'].argmax(dim=2)[
        0, :, 0, 0, 0].tolist() == [2, 2, 2, 2, 2]
    torch.testing.assert_close(
        outputs['world_logits'][0, :, 2, 0, 0, 0],
        torch.tensor([2.0, 1.5, 1.0, 0.75, 0.5]))
    torch.testing.assert_close(
        outputs['world_logits'][0, :, :2, 0, 0, 0],
        torch.zeros((5, 2)))
    assert not any(
        'observation_future_semantic_logit_scales' in key
        for key in decoder.state_dict())


def test_future_anchor_decay_validates_horizon_count_and_scale():
    try:
        DenseWorldDecoder(
            in_channels=4, hidden_channels=3,
            horizon_count=3, class_count=3, z_count=1,
            observation_future_semantic_logit_scales=[1.0])
    except ValueError as error:
        assert 'match future horizons' in str(error)
    else:
        raise AssertionError('Expected a future-scale length error')

    try:
        DenseWorldDecoder(
            in_channels=4, hidden_channels=3,
            horizon_count=3, class_count=3, z_count=1,
            observation_future_semantic_logit_scales=[1.0, 0.0])
    except ValueError as error:
        assert 'must be positive' in str(error)
    else:
        raise AssertionError('Expected a positive future-scale error')


def test_future_change_gate_starts_closed_and_preserves_persistence():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=1,
        use_future_change_gate=True,
        future_change_prior=0.01)
    state = torch.full((1, 1, 2, 2), 2)
    valid = torch.ones_like(state, dtype=torch.bool)

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        observation_state=state,
        observation_valid=valid)

    assert outputs['world_logits'].argmax(dim=2)[
        0, :, 0, 0, 0].tolist() == [1, 1, 1]
    torch.testing.assert_close(
        outputs['future_change_logits'].sigmoid(),
        torch.full_like(outputs['future_change_logits'], 0.01))
    torch.testing.assert_close(
        outputs['future_changed_class_logits'],
        torch.zeros_like(outputs['future_changed_class_logits']))


def test_official_dynamic_occupancy_biases_future_change_gate():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=1,
        use_future_change_gate=True,
        future_change_prior=0.01,
        dynamic_change_logit_scale=10.0)
    state = torch.full((1, 1, 2, 2), 2)
    valid = torch.ones_like(state, dtype=torch.bool)
    dynamic = torch.zeros((1, 3, 2, 2))
    dynamic[:, 1, 0, 0] = 1.0

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        observation_state=state,
        observation_valid=valid,
        dynamic_occupancy_probability=dynamic)
    change_probability = outputs['future_change_logits'].sigmoid()

    assert change_probability[0, 0, 0, 0, 0] > 0.99
    torch.testing.assert_close(
        change_probability[0, 0, 0, 1, 1], torch.tensor(0.01))
    assert outputs['dynamic_change_prior'].shape == (1, 2, 2, 2)


def test_history_encoder_accepts_ordered_causal_world_states():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=2,
        history_count=2)
    state = torch.tensor([[
        [[[1, 2], [3, 0]], [[0, 1], [2, 3]]],
        [[[2, 3], [1, 0]], [[1, 2], [3, 0]]],
    ]])
    valid = state != 0

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        history_state=state,
        history_valid=valid)

    assert outputs['history_features'].shape == (1, 3, 2, 2)
    assert outputs['world_logits'].shape == (1, 3, 3, 2, 2, 2)


def test_history_encoder_rejects_missing_history():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=2, class_count=3, z_count=1,
        history_count=2)

    try:
        decoder(torch.randn(1, 4, 2, 2))
    except ValueError as error:
        assert 'requires state and valid mask' in str(error)
    else:
        raise AssertionError('Expected missing history to fail')


def test_flow_warp_starts_as_exact_instance_persistence():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=1,
        history_count=2, use_future_change_gate=True,
        use_flow_warp=True, flow_gate_logit_scale=10.0)
    state = torch.tensor([[[[1, 3], [2, 1]]]])
    valid = torch.ones_like(state, dtype=torch.bool)
    history = state[:, None].expand(-1, 2, -1, -1, -1).clone()
    history_valid = torch.ones_like(history, dtype=torch.bool)

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        observation_state=state,
        observation_valid=valid,
        history_state=history,
        history_valid=history_valid)

    assert torch.count_nonzero(outputs['future_flow']) == 0
    assert torch.count_nonzero(outputs['flow_change_prior']) == 0
    expected_instance = (state == 3).float()
    torch.testing.assert_close(
        outputs['warped_instance_probability'],
        expected_instance[:, None].expand(-1, 2, -1, -1, -1))


def test_cumulative_flow_warps_each_horizon_from_current_source():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=1,
        history_count=1, use_flow_warp=True,
        flow_parameterization='cumulative_current')
    with torch.no_grad():
        decoder.future_flow_head.bias.reshape(2, 2)[:, 1] = torch.tensor(
            [1.0, 2.0])
    state = torch.tensor([[[[3, 1, 1, 1]]]])
    valid = torch.ones_like(state, dtype=torch.bool)

    outputs = decoder(
        torch.randn(1, 4, 1, 4),
        observation_state=state,
        observation_valid=valid,
        history_state=state[:, None],
        history_valid=valid[:, None])

    warped = outputs['warped_instance_probability']
    assert warped[0, 0, 0].tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert warped[0, 1, 0].tolist() == [[0.0, 0.0, 1.0, 0.0]]


def test_wide_history_context_is_zero_residual_at_initialization():
    base = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=2, class_count=3, z_count=1,
        history_count=2)
    wide = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=2, class_count=3, z_count=1,
        history_count=2, use_wide_history_context=True)
    missing, unexpected = wide.load_state_dict(
        base.state_dict(), strict=False)
    assert unexpected == []
    assert all('history_context_encoder' in key for key in missing)
    state = torch.tensor([[[[[1, 2], [3, 1]]],
                           [[[2, 3], [1, 1]]]]])
    valid = torch.ones_like(state, dtype=torch.bool)
    feature = torch.randn(1, 4, 2, 2)

    base_output = base(
        feature, history_state=state, history_valid=valid)
    wide_output = wide(
        feature, history_state=state, history_valid=valid)

    torch.testing.assert_close(
        wide_output['history_context_features'],
        torch.zeros_like(wide_output['history_context_features']))
    torch.testing.assert_close(
        wide_output['world_logits'], base_output['world_logits'])


def test_physical_confidence_head_starts_at_configured_prior():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=1,
        history_count=1, use_flow_warp=True,
        use_physical_confidence=True,
        physical_confidence_prior=0.8)
    state = torch.tensor([[[[1, 3], [2, 1]]]])
    valid = torch.ones_like(state, dtype=torch.bool)

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        observation_state=state,
        observation_valid=valid,
        history_state=state[:, None],
        history_valid=valid[:, None])
    confidence = outputs['physical_confidence_logits']

    assert confidence.shape == (1, 2, 1, 2, 2)
    torch.testing.assert_close(
        confidence.sigmoid(), torch.full_like(confidence, 0.8))
    confidence.sum().backward()
    assert decoder.physical_confidence_head.bias.grad is not None


def test_candidate_signal_confidence_is_zero_residual_at_initialization():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=8,
        horizon_count=3, class_count=3, z_count=2,
        history_count=1, use_future_change_gate=True,
        use_flow_warp=True, use_physical_confidence=True,
        use_physical_confidence_signals=True,
        physical_confidence_prior=0.8)
    state = torch.tensor([
        [[[1, 3], [2, 1]], [[3, 1], [1, 2]]]
    ])
    valid = torch.ones_like(state, dtype=torch.bool)

    outputs = decoder(
        torch.randn(1, 4, 2, 2),
        observation_state=state,
        observation_valid=valid,
        history_state=state[:, None],
        history_valid=valid[:, None])
    confidence = outputs['physical_confidence_logits']

    torch.testing.assert_close(
        confidence.sigmoid(), torch.full_like(confidence, 0.8))
    confidence.sum().backward()
    assert decoder.physical_confidence_signal_head[-1].weight.grad is not None
    assert torch.count_nonzero(
        decoder.physical_confidence_signal_head[-1].weight.grad) > 0


def test_physical_confidence_signal_tensor_has_candidate_level_shape():
    future_logits = torch.zeros((1, 2, 3, 2, 2, 3))
    future_logits[:, :, 2] = 1.0
    warped = torch.zeros((1, 2, 2, 2, 3))
    warped[:, 1] = 0.75
    flow = torch.zeros((1, 2, 2, 2, 3))
    flow[:, 1, 0] = 2.0
    observation_class = torch.tensor([
        [[[0, 2, 1], [0, 0, 1]], [[2, 0, 0], [1, 1, 0]]]
    ])
    observation_known = torch.ones_like(
        observation_class, dtype=torch.bool)
    change_logits = torch.zeros_like(warped)
    dynamic_prior = torch.zeros((1, 2, 2, 3))

    signals = physical_confidence_signal_tensor(
        future_logits, warped, flow, observation_class,
        observation_known, change_logits, dynamic_prior)

    assert signals.shape == (1, 17, 4, 2, 3)
    assert torch.all(signals[:, 4, :2] == 0)
    assert torch.all(signals[:, 4, 2:] == 2)


def test_selected_smooth_l1_ignores_unselected_flow():
    prediction = torch.zeros((1, 2, 2, 1, 2), requires_grad=True)
    target = torch.ones_like(prediction)
    selection = torch.zeros_like(prediction, dtype=torch.bool)
    selection[..., 0] = True

    loss = selected_smooth_l1_loss(prediction, target, selection)
    loss.backward()

    assert torch.count_nonzero(prediction.grad[..., 0]) > 0
    assert torch.count_nonzero(prediction.grad[..., 1]) == 0


def test_masked_world_cross_entropy_ignores_unknown_voxels():
    logits = torch.zeros((1, 1, 3, 1, 1, 2))
    target = torch.tensor([[[[[0, 255]]]]])
    baseline = masked_world_cross_entropy(logits, target)
    logits[..., 1] = 1000.0

    torch.testing.assert_close(
        masked_world_cross_entropy(logits, target), baseline)


def test_masked_world_cross_entropy_backpropagates_on_valid_voxels():
    logits = torch.zeros(
        (1, 2, 3, 2, 3, 4), requires_grad=True)
    target = torch.zeros((1, 2, 2, 3, 4), dtype=torch.long)
    target[:, :, :, 0, 0] = 255

    loss = masked_world_cross_entropy(logits, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[..., 0, 0]) == 0


def test_selected_world_cross_entropy_only_backpropagates_on_subset():
    logits = torch.zeros((1, 2, 3, 1, 1, 3), requires_grad=True)
    target = torch.tensor([[[[[0, 1, 2]]], [[[2, 1, 0]]]]])
    selection = torch.zeros_like(target, dtype=torch.bool)
    selection[..., 1] = True

    loss = selected_world_cross_entropy(logits, target, selection)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[..., 0]).item() == 0
    assert torch.count_nonzero(logits.grad[..., 1]).item() > 0
    assert torch.count_nonzero(logits.grad[..., 2]).item() == 0


def test_stable_known_world_selection_excludes_changes_and_unknowns():
    observation_known = torch.tensor([[[[True, True, True, False]]]])
    observation_class = torch.tensor([[[[0, 1, 2, 1]]]])
    future_target = torch.tensor([
        [[[[0, 2, 255, 1]]],
         [[[1, 1, 2, 1]]]],
    ])

    selection = stable_known_world_selection(
        observation_known, observation_class, future_target)

    expected = torch.tensor([
        [[[[True, False, False, False]]],
         [[[False, True, True, False]]]],
    ])
    assert torch.equal(selection, expected)


def test_occworld_stability_loss_only_backpropagates_on_stable_known_voxels():
    head = SimpleNamespace(
        world_class_weights=torch.tensor([]),
        world_ignore_index=255,
        world_valid_positive_weight=1.0,
        world_loss_weight=1.0,
        world_current_loss_weight=0.0,
        world_future_loss_weight=0.0,
        world_visibility_loss_weight=0.0,
        world_future_transition_loss_weight=0.0,
        world_future_stability_loss_weight=1.0,
        world_future_change_gate_loss_weight=0.0,
        world_future_changed_class_loss_weight=0.0,
        world_flow_loss_weight=0.0,
        world_physical_confidence_loss_weight=0.0)
    world_logits = torch.zeros(
        (1, 3, 3, 1, 1, 4), requires_grad=True)
    outputs = {
        'world_logits': world_logits,
        'valid_logits': torch.zeros((1, 3, 1, 1, 4)),
        'observation_known_mask': torch.tensor(
            [[[[True, True, True, False]]]]),
        'observation_class': torch.tensor([[[[0, 1, 2, 1]]]]),
    }
    target = torch.tensor([
        [[[[0, 1, 2, 255]]],
         [[[0, 2, 255, 1]]],
         [[[1, 1, 2, 1]]]],
    ])
    valid = target != 255

    losses = OccWorldHead.loss_world(head, outputs, target, valid)
    assert 'loss_world_future_stability_ce' in losses
    losses['loss_world_future_stability_ce'].backward()

    future_gradient = world_logits.grad[:, 1:]
    assert torch.count_nonzero(future_gradient[..., 0]).item() > 0
    assert torch.count_nonzero(future_gradient[:, 0, ..., 1]).item() == 0
    assert torch.count_nonzero(future_gradient[:, 1, ..., 1]).item() > 0
    assert torch.count_nonzero(future_gradient[:, 1, ..., 2]).item() > 0
    assert torch.count_nonzero(future_gradient[..., 3]).item() == 0


def test_selected_binary_cross_entropy_only_uses_selected_voxels():
    logits = torch.zeros((1, 2, 1, 1, 3), requires_grad=True)
    target = torch.tensor([[[[[0.0, 1.0, 0.0]]],
                            [[[1.0, 0.0, 1.0]]]]])
    selection = torch.zeros_like(target, dtype=torch.bool)
    selection[..., 1] = True

    loss = selected_binary_cross_entropy_with_logits(
        logits, target, selection, positive_weight=3.0)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.count_nonzero(logits.grad[..., 0]).item() == 0
    assert torch.count_nonzero(logits.grad[..., 1]).item() > 0
    assert torch.count_nonzero(logits.grad[..., 2]).item() == 0


def test_persistence_and_visibility_heads_all_receive_gradients():
    decoder = DenseWorldDecoder(
        in_channels=4, hidden_channels=3,
        horizon_count=3, class_count=3, z_count=2)
    outputs = decoder(torch.randn(1, 4, 5, 6))
    target = torch.zeros((1, 3, 2, 5, 6), dtype=torch.long)
    valid = torch.ones((1, 3, 2, 5, 6), dtype=torch.bool)
    loss = (
        masked_world_cross_entropy(outputs['world_logits'], target) +
        world_visibility_binary_cross_entropy(
            outputs['valid_logits'], valid, positive_weight=2.0))

    loss.backward()

    assert decoder.current_head.bias.grad is not None
    assert decoder.future_residual_head.bias.grad is not None
    assert decoder.valid_head.bias.grad is not None
