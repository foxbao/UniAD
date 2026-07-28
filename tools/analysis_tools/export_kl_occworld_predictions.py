#!/usr/bin/env python
"""Export dense OccWorld predictions for frozen manifest splits."""

import argparse
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel

import projects.mmdet3d_plugin  # noqa: F401
from third_party.uniad_mmdet3d.datasets.builder import (
    build_dataloader,
    build_dataset,
)
from third_party.uniad_mmdet3d.models.builder import build_model
from tools.data_converter.kl_occworld_track_adapter import (
    track_result_to_occworld_instances,
)


def _checkpoint_epoch(path: Path) -> int:
    match = re.fullmatch(r'epoch_(\d+)\.pth', path.name)
    if match is None:
        raise ValueError(f'Checkpoint has no epoch number: {path}')
    return int(match.group(1))


def _discover_checkpoints(args) -> list:
    if args.checkpoint:
        return [args.checkpoint]
    paths = sorted(
        args.checkpoint_dir.glob(args.checkpoint_glob),
        key=_checkpoint_epoch)
    if not paths:
        raise FileNotFoundError(
            f'No checkpoints below {args.checkpoint_dir}')
    return paths


def _reference_indices(dataset):
    return [
        int(dataset.data_infos[raw_index].get('sample_idx', raw_index))
        for raw_index in dataset.valid_data_indices
    ]


def _shard_dataset_by_scene(dataset, shard_count: int,
                            shard_index: int) -> dict:
    """Select a deterministic, scene-disjoint subset of dataset references."""
    if shard_count < 1:
        raise ValueError('shard_count must be positive')
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f'shard_index must be in [0, {shard_count}), got {shard_index}')
    raw_indices = list(dataset.valid_data_indices)
    groups = {}
    for raw_index in raw_indices:
        scene_token = str(dataset.data_infos[raw_index].get('scene_token', ''))
        groups.setdefault(scene_token, []).append(raw_index)
    shards = [[] for _ in range(shard_count)]
    loads = [0] * shard_count
    for scene_token, values in sorted(
            groups.items(), key=lambda item: (-len(item[1]), item[0])):
        target = min(range(shard_count), key=lambda i: (loads[i], i))
        shards[target].extend(values)
        loads[target] += len(values)
    selected = set(shards[shard_index])
    positions = [
        position for position, raw_index in enumerate(raw_indices)
        if raw_index in selected
    ]
    dataset.valid_data_indices = [raw_indices[position] for position in positions]
    if hasattr(dataset, 'flag'):
        dataset.flag = dataset.flag[np.asarray(positions, dtype=np.int64)]
    selected_scenes = sorted({
        str(dataset.data_infos[raw_index].get('scene_token', ''))
        for raw_index in dataset.valid_data_indices
    })
    return {
        'shard_count': int(shard_count),
        'shard_index': int(shard_index),
        'reference_count': len(dataset.valid_data_indices),
        'scene_count': len(selected_scenes),
        'scene_tokens': selected_scenes,
        'all_shard_reference_counts': loads,
    }


def _prediction_filename(label_path: Path) -> str:
    suffix = '__occworld_sequence.npz'
    if not label_path.name.endswith(suffix):
        raise ValueError(f'Unexpected OccWorld label name: {label_path}')
    return label_path.name[:-len(suffix)] + '__occworld_prediction.npz'


def _load_current_anchor(root: Path, reference_index: int):
    path = root / f'{reference_index:06d}' / 'occworld_current_anchor.npz'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing current-anchor override for {reference_index}: {path}')
    with np.load(path, allow_pickle=False) as anchor:
        if int(anchor['reference_index']) != reference_index:
            raise ValueError(f'Wrong reference index in {path}')
        state = np.asarray(anchor['current_world_state_3d'], dtype=np.int64)
        valid = np.asarray(anchor['current_world_valid_3d'], dtype=np.bool_)
    if state.shape != (10, 120, 160) or valid.shape != state.shape:
        raise ValueError(f'Unexpected current-anchor shape in {path}')
    return state, valid, path


def _override_current_anchor(batch: dict, state: np.ndarray,
                             valid: np.ndarray):
    for key, value in (
            ('current_world_state', state),
            ('current_world_valid', valid)):
        if key not in batch:
            raise KeyError(f'Batch lacks {key}')
        target = batch[key].data[0]
        if target.ndim != 4 or target.shape[0] != 1:
            raise ValueError(
                f'Unexpected {key} batch shape {tuple(target.shape)}')
        tensor = torch.from_numpy(value).to(dtype=target.dtype)
        target[0].copy_(tensor)


def _load_online_inputs(root: Path, reference_index: int):
    path = root / f'{reference_index:06d}' / 'occworld_online_input.npz'
    if not path.exists():
        raise FileNotFoundError(
            f'Missing online-input override for {reference_index}: {path}')
    with np.load(path, allow_pickle=False) as inputs:
        if int(inputs['reference_index']) != reference_index:
            raise ValueError(f'Wrong reference index in {path}')
        payload = {
            'current_world_state': np.asarray(
                inputs['current_world_state_3d'], dtype=np.int64),
            'current_world_valid': np.asarray(
                inputs['current_world_valid_3d'], dtype=np.bool_),
            'history_world_state': np.asarray(
                inputs['history_world_state_3d'], dtype=np.int64),
            'history_world_valid': np.asarray(
                inputs['history_world_valid_3d'], dtype=np.bool_),
        }
    current_shape = (10, 120, 160)
    history_shape = (5, *current_shape)
    if (payload['current_world_state'].shape != current_shape or
            payload['current_world_valid'].shape != current_shape):
        raise ValueError(f'Unexpected current online-input shape in {path}')
    if (payload['history_world_state'].shape != history_shape or
            payload['history_world_valid'].shape != history_shape):
        raise ValueError(f'Unexpected history online-input shape in {path}')
    for prefix in ('current', 'history'):
        state = payload[f'{prefix}_world_state']
        valid = payload[f'{prefix}_world_valid']
        if np.any((state < 0) | (state > 3)):
            raise ValueError(f'Invalid {prefix} world state in {path}')
        if np.any((~valid) & (state != 0)):
            raise ValueError(
                f'Invalid {prefix} voxels carry a world state in {path}')
    return payload, path


def _override_online_inputs(batch: dict, payload: dict):
    expected_ndim = {
        'current_world_state': 4,
        'current_world_valid': 4,
        'history_world_state': 5,
        'history_world_valid': 5,
    }
    for key, ndim in expected_ndim.items():
        if key not in batch:
            raise KeyError(f'Batch lacks {key}')
        if key not in payload:
            raise KeyError(f'Online input lacks {key}')
        target = batch[key].data[0]
        if target.ndim != ndim or target.shape[0] != 1:
            raise ValueError(
                f'Unexpected {key} batch shape {tuple(target.shape)}')
        value = np.asarray(payload[key])
        if tuple(value.shape) != tuple(target.shape[1:]):
            raise ValueError(
                f'{key} override shape {value.shape} does not match '
                f'{tuple(target.shape[1:])}')
        tensor = torch.from_numpy(value).to(dtype=target.dtype)
        target[0].copy_(tensor)


def _motion_actor_diagnostic_payload(occ: dict,
                                     step_seconds: float = 0.5) -> dict:
    """Convert detached MotionHead actor geometry into a stable NPZ schema."""
    if step_seconds <= 0:
        raise ValueError('Motion actor step seconds must be positive')
    required = (
        'planning_actor_future', 'planning_actor_boxes_3d',
        'planning_actor_scores', 'planning_actor_valid')
    missing = [key for key in required if key not in occ]
    if missing:
        raise RuntimeError(
            'Motion actor diagnostic is missing: ' + ', '.join(missing))

    values = {
        key: occ[key].detach().cpu()
        for key in required
    }
    future = values['planning_actor_future']
    boxes = values['planning_actor_boxes_3d']
    scores = values['planning_actor_scores']
    valid = values['planning_actor_valid']
    if future.ndim != 4 or future.shape[0] != 1 or future.shape[-1] != 2:
        raise ValueError(
            f'Unexpected planning actor future shape {tuple(future.shape)}')
    actor_count = future.shape[1]
    if boxes.shape != (1, actor_count, 7):
        raise ValueError(
            f'Unexpected planning actor box shape {tuple(boxes.shape)}')
    if scores.shape != (1, actor_count):
        raise ValueError(
            f'Unexpected planning actor score shape {tuple(scores.shape)}')
    if valid.shape != (1, actor_count):
        raise ValueError(
            f'Unexpected planning actor valid shape {tuple(valid.shape)}')

    planning_steps = future.shape[2]
    return {
        'motion_actor_future_xy': future[0].numpy().astype(
            np.float32, copy=False),
        'motion_actor_boxes_3d': boxes[0].numpy().astype(
            np.float32, copy=False),
        'motion_actor_scores': scores[0].numpy().astype(
            np.float32, copy=False),
        'motion_actor_valid': valid[0].numpy().astype(
            np.bool_, copy=False),
        'motion_actor_step_times_s': (
            np.arange(1, planning_steps + 1, dtype=np.float32) *
            np.float32(step_seconds)),
        'motion_actor_box_z_origin': np.asarray('bottom'),
    }


def _load_checkpoint(model, path: Path):
    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict


def _reset_test_state(model):
    model.scene_token = None
    model._test_scene_token = None
    model._test_prev_bev = None
    model.test_frame_token = None
    model.timestamp = None
    model.l2g_r_mat = None
    model.l2g_t = None
    model.test_track_instances = None
    model.track_base.clear()


def export_checkpoint(model, wrapped_model, loader, dataset,
                      checkpoint_path: Path, split: str,
                      output_root: Path,
                      save_track_boxes: bool = False,
                      save_raw_world_prediction: bool = False,
                      save_query_dynamic_diagnostic: bool = False,
                      save_query_residual_ablation: bool = False,
                      save_motion_actor_diagnostic: bool = False,
                      save_motion_actor_overlay_diagnostic: bool = False,
                      motion_actor_step_seconds: float = 0.5,
                      track_score_threshold: float = 0.1,
                      current_anchor_root: Path = None,
                      online_input_root: Path = None):
    if current_anchor_root is not None and online_input_root is not None:
        raise ValueError(
            'Current-anchor and complete online-input overrides are '
            'mutually exclusive')
    _load_checkpoint(model, checkpoint_path)
    model.eval()
    _reset_test_state(model)
    epoch = _checkpoint_epoch(checkpoint_path)
    epoch_root = output_root / split / f'epoch_{epoch:03d}'
    references = _reference_indices(dataset)
    if len(references) != len(dataset):
        raise ValueError('Dataset reference mapping is inconsistent')
    output_paths = []
    for position, batch in enumerate(loader):
        reference_index = references[position]
        anchor_path = None
        online_input_path = None
        if current_anchor_root is not None:
            state, valid, anchor_path = _load_current_anchor(
                current_anchor_root, reference_index)
            _override_current_anchor(batch, state, valid)
        if online_input_root is not None:
            online_inputs, online_input_path = _load_online_inputs(
                online_input_root, reference_index)
            _override_online_inputs(batch, online_inputs)
        with torch.no_grad():
            results = wrapped_model(
                return_loss=False, rescale=True, **batch)
        if len(results) != 1 or 'occ' not in results[0]:
            raise RuntimeError(
                f'No OccWorld output for reference {reference_index}')
        occ = results[0]['occ']
        if ('world_pred' not in occ or
                'world_valid_probability' not in occ):
            raise RuntimeError(
                f'Incomplete OccWorld output for {reference_index}')
        prediction = occ['world_pred'].detach().cpu()
        query_residual_ablation = None
        if save_query_residual_ablation:
            query_residual_ablation = occ.get(
                'query_adapter_ablation_world_pred')
            if query_residual_ablation is None:
                raise RuntimeError(
                    'Query-residual ablation requires an enabled adapter')
            query_residual_ablation = (
                query_residual_ablation.detach().cpu())
        motion_actor_diagnostic = None
        if save_motion_actor_diagnostic:
            motion_actor_diagnostic = _motion_actor_diagnostic_payload(
                occ, step_seconds=motion_actor_step_seconds)
        motion_actor_arrival = None
        if save_motion_actor_overlay_diagnostic:
            motion_actor_arrival = occ.get('motion_actor_arrival_mask')
            if motion_actor_arrival is None:
                raise RuntimeError(
                    'Motion actor overlay diagnostic requires an enabled '
                    'motion actor overlay')
            motion_actor_arrival = motion_actor_arrival.detach().cpu()
        raw_prediction = None
        if save_raw_world_prediction:
            if 'world_logits' not in occ:
                raise RuntimeError(
                    f'No raw world logits for reference {reference_index}')
            raw_prediction = occ['world_logits'].argmax(
                dim=2).detach().cpu()
        query_dynamic_probability = None
        observation_class = None
        observation_known = None
        if save_query_dynamic_diagnostic:
            query_dynamic_probability = occ.get(
                'dynamic_occupancy_probability')
            observation_class = occ.get('observation_class')
            observation_known = occ.get('observation_known_mask')
            if (query_dynamic_probability is None or
                    observation_class is None or observation_known is None):
                raise RuntimeError(
                    'Query dynamic diagnostic requires world query, '
                    'observation class and observation known outputs')
            query_dynamic_probability = (
                query_dynamic_probability.detach().cpu())
            observation_class = observation_class.detach().cpu()
            observation_known = observation_known.detach().cpu()
        valid_probability = occ[
            'world_valid_probability'].detach().cpu()
        future_change_logits = occ.get('future_change_logits')
        future_change_probability = None
        if future_change_logits is not None:
            future_change_probability = (
                future_change_logits.sigmoid().detach().cpu())
        future_flow = occ.get('future_flow')
        if future_flow is not None:
            future_flow = future_flow.detach().cpu()
        flow_change_prior = occ.get('flow_change_prior')
        if flow_change_prior is not None:
            flow_change_prior = flow_change_prior.detach().cpu()
        changed_class_logits = occ.get('future_changed_class_logits')
        changed_class_prediction = None
        if changed_class_logits is not None:
            changed_class_prediction = changed_class_logits.argmax(
                dim=2).detach().cpu()
        warped_instance_probability = occ.get(
            'warped_instance_probability')
        if warped_instance_probability is not None:
            warped_instance_probability = (
                warped_instance_probability.detach().cpu())
        physical_confidence_logits = occ.get(
            'physical_confidence_logits')
        physical_confidence_probability = None
        if physical_confidence_logits is not None:
            physical_confidence_probability = (
                physical_confidence_logits.sigmoid().detach().cpu())
        if prediction.ndim != 5 or prediction.shape[0] != 1:
            raise ValueError(
                f'Unexpected world prediction shape {prediction.shape}')
        prediction = prediction[0].numpy().astype(np.uint8, copy=False)
        if query_residual_ablation is not None:
            if query_residual_ablation.shape != prediction[None].shape:
                raise ValueError(
                    'Unexpected query-residual ablation shape '
                    f'{query_residual_ablation.shape}')
            query_residual_ablation = (
                query_residual_ablation[0].numpy().astype(
                    np.uint8, copy=False))
        if raw_prediction is not None:
            if raw_prediction.shape != prediction[None].shape:
                raise ValueError(
                    f'Unexpected raw prediction shape {raw_prediction.shape}')
            raw_prediction = raw_prediction[0].numpy().astype(
                np.uint8, copy=False)
        if query_dynamic_probability is not None:
            if (query_dynamic_probability.ndim != 4 or
                    query_dynamic_probability.shape != (
                        1, prediction.shape[0], prediction.shape[-2],
                        prediction.shape[-1])):
                raise ValueError(
                    'Unexpected query dynamic probability shape '
                    f'{query_dynamic_probability.shape}')
            expected_observation = (1, *prediction.shape[1:])
            if (observation_class.shape != expected_observation or
                    observation_known.shape != expected_observation):
                raise ValueError('Unexpected observation diagnostic shape')
            query_dynamic_probability = (
                query_dynamic_probability[0].numpy().astype(
                    np.float32, copy=False))
            observation_class = observation_class[0].numpy().astype(
                np.uint8, copy=False)
            observation_known = observation_known[0].numpy().astype(
                np.bool_, copy=False)
        valid_probability = valid_probability[0].numpy().astype(
            np.float32, copy=False)
        if future_change_probability is not None:
            if (future_change_probability.ndim != 5 or
                    future_change_probability.shape[0] != 1):
                raise ValueError(
                    'Unexpected future change probability shape '
                    f'{future_change_probability.shape}')
            future_change_probability = (
                future_change_probability[0].numpy().astype(
                    np.float16, copy=False))
        if future_flow is not None:
            if future_flow.ndim != 5 or future_flow.shape[0] != 1:
                raise ValueError(
                    f'Unexpected future flow shape {future_flow.shape}')
            future_flow = future_flow[0].numpy().astype(
                np.float16, copy=False)
        if flow_change_prior is not None:
            if (flow_change_prior.ndim != 5 or
                    flow_change_prior.shape[0] != 1):
                raise ValueError(
                    'Unexpected flow change prior shape '
                    f'{flow_change_prior.shape}')
            flow_change_prior = flow_change_prior[0].numpy().astype(
                np.float16, copy=False)
        if changed_class_prediction is not None:
            if (changed_class_prediction.ndim != 5 or
                    changed_class_prediction.shape[0] != 1):
                raise ValueError(
                    'Unexpected changed-class prediction shape '
                    f'{changed_class_prediction.shape}')
            changed_class_prediction = (
                changed_class_prediction[0].numpy().astype(
                    np.uint8, copy=False))
        if warped_instance_probability is not None:
            if (warped_instance_probability.ndim != 5 or
                    warped_instance_probability.shape[0] != 1):
                raise ValueError(
                    'Unexpected warped-instance shape '
                    f'{warped_instance_probability.shape}')
            warped_instance_probability = (
                warped_instance_probability[0].numpy().astype(
                    np.float32 if save_query_dynamic_diagnostic
                    else np.float16, copy=False))
        if physical_confidence_probability is not None:
            if (physical_confidence_probability.ndim != 5 or
                    physical_confidence_probability.shape[0] != 1):
                raise ValueError(
                    'Unexpected physical-confidence shape '
                    f'{physical_confidence_probability.shape}')
            physical_confidence_probability = (
                physical_confidence_probability[0].numpy().astype(
                    np.float16, copy=False))
        label_path = dataset.occworld_labels[reference_index]
        output_dir = epoch_root / f'{reference_index:06d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / _prediction_filename(label_path)
        payload = dict(
            reference_index=np.int64(reference_index),
            checkpoint_epoch=np.int64(epoch),
            world_pred_class_3d=prediction,
            world_valid_probability_3d=valid_probability)
        if motion_actor_diagnostic is not None:
            payload.update(motion_actor_diagnostic)
        if motion_actor_arrival is not None:
            expected_arrival = (1, prediction.shape[0] - 1,
                                *prediction.shape[1:])
            if tuple(motion_actor_arrival.shape) != expected_arrival:
                raise ValueError(
                    'Unexpected motion actor arrival shape '
                    f'{tuple(motion_actor_arrival.shape)}')
            payload['motion_actor_arrival_mask_3d'] = (
                motion_actor_arrival[0].numpy().astype(
                    np.bool_, copy=False))
        if raw_prediction is not None:
            payload['raw_world_pred_class_3d'] = raw_prediction
        if query_residual_ablation is not None:
            payload['query_adapter_ablation_world_pred_class_3d'] = (
                query_residual_ablation)
        if query_dynamic_probability is not None:
            payload['query_dynamic_probability_2d'] = (
                query_dynamic_probability)
            payload['observation_class_3d'] = observation_class
            payload['observation_known_3d'] = observation_known
        if anchor_path is not None:
            payload['current_anchor_override'] = np.asarray(True)
            payload['current_anchor_path'] = np.asarray(str(anchor_path))
        if online_input_path is not None:
            payload['online_input_override'] = np.asarray(True)
            payload['online_input_path'] = np.asarray(
                str(online_input_path))
        if save_track_boxes:
            track_result = results[0].get('pts_bbox', {})
            track_boxes, track_instances, track_summary = (
                track_result_to_occworld_instances(
                    track_result,
                    score_threshold=track_score_threshold,
                    track_score_threshold=track_score_threshold,
                    class_count=len(dataset.CLASSES)))
            payload.update(
                track_boxes_3d=track_boxes,
                track_scores_3d=np.asarray([
                    instance['score_3d']
                    for instance in track_instances
                ], dtype=np.float32),
                track_runtime_scores=np.asarray([
                    instance['track_score']
                    for instance in track_instances
                ], dtype=np.float32),
                track_labels_3d=np.asarray([
                    instance['bbox_label_3d']
                    for instance in track_instances
                ], dtype=np.int64),
                track_ids=np.asarray([
                    instance['track_id']
                    for instance in track_instances
                ], dtype=np.int64),
                track_box_z_origin=np.asarray(
                    track_summary['output_z_origin']),
                track_input_count=np.int64(
                    track_summary['input_count']),
                track_score_threshold=np.float32(
                    track_score_threshold))
        if future_change_probability is not None:
            payload['future_change_probability_3d'] = (
                future_change_probability)
        if future_flow is not None:
            payload['future_flow_2d'] = future_flow
        if flow_change_prior is not None:
            payload['flow_change_prior_3d'] = flow_change_prior
        if changed_class_prediction is not None:
            payload['future_changed_class_pred_3d'] = (
                changed_class_prediction)
        if warped_instance_probability is not None:
            payload['warped_instance_probability_3d'] = (
                warped_instance_probability)
        if physical_confidence_probability is not None:
            payload['physical_confidence_probability_3d'] = (
                physical_confidence_probability)
        np.savez_compressed(output_path, **payload)
        output_paths.append(str(output_path))
        del results, occ, prediction, valid_probability, payload
    return {
        'epoch': epoch,
        'checkpoint': str(checkpoint_path),
        'split': split,
        'reference_indices': references,
        'prediction_files': output_paths,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path,
        default=Path(
            'projects/configs/stage2_e2e_lidar/'
            'base_e2e_lidar_occworld_split_v1_eval.py'))
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument('--checkpoint', type=Path)
    checkpoint_group.add_argument('--checkpoint-dir', type=Path)
    parser.add_argument('--checkpoint-glob', default='epoch_*.pth')
    parser.add_argument(
        '--split', choices=(
            'train', 'validation', 'test', 'blind', 'final_holdout',
            'fresh_holdout'),
        default='validation')
    parser.add_argument(
        '--output-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_world_only_v1'))
    parser.add_argument('--workers-per-gpu', type=int, default=0)
    parser.add_argument(
        '--save-track-boxes', action='store_true',
        help='Save filtered current TrackFormer boxes in each prediction.')
    parser.add_argument(
        '--save-raw-world-prediction', action='store_true',
        help='Save pre-overlay argmax semantics for validation diagnostics.')
    parser.add_argument(
        '--save-query-dynamic-diagnostic', action='store_true',
        help='Save query OCC and observation signals for validation audits.')
    parser.add_argument(
        '--save-query-residual-ablation', action='store_true',
        help='Save paired adapter-off semantics from the same forward pass.')
    parser.add_argument(
        '--save-motion-actor-diagnostic', action='store_true',
        help='Save aligned MotionHead actor boxes and future XY centers.')
    parser.add_argument(
        '--save-motion-actor-overlay-diagnostic', action='store_true',
        help='Save the model-side raw-free Motion actor arrival mask.')
    parser.add_argument(
        '--motion-actor-step-seconds', type=float, default=0.5,
        help='Time interval represented by consecutive motion steps.')
    parser.add_argument(
        '--track-score-threshold', type=float, default=0.1)
    parser.add_argument(
        '--current-anchor-root', type=Path,
        help='Override B15 current-world input from saved online anchors.')
    parser.add_argument(
        '--online-input-root', type=Path,
        help='Override current and five-frame history from online replay.')
    parser.add_argument('--shard-count', type=int, default=1)
    parser.add_argument('--shard-index', type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    if (args.current_anchor_root is not None and
            args.online_input_root is not None):
        raise ValueError(
            '--current-anchor-root and --online-input-root are mutually '
            'exclusive')
    cfg = Config.fromfile(str(args.config))
    dataset_key = 'test' if args.split == 'test' else 'val'
    dataset_cfg = cfg.data[dataset_key]
    dataset = build_dataset(dataset_cfg)
    shard_summary = _shard_dataset_by_scene(
        dataset, args.shard_count, args.shard_index)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers_per_gpu,
        dist=False,
        shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = dataset.CLASSES
    model = model.cuda()
    wrapped_model = MMDataParallel(model, device_ids=[0])
    summaries = []
    for checkpoint_path in _discover_checkpoints(args):
        summary = export_checkpoint(
            model=model,
            wrapped_model=wrapped_model,
            loader=loader,
            dataset=dataset,
            checkpoint_path=checkpoint_path,
            split=args.split,
            output_root=args.output_root,
            save_track_boxes=args.save_track_boxes,
            save_raw_world_prediction=args.save_raw_world_prediction,
            save_query_dynamic_diagnostic=(
                args.save_query_dynamic_diagnostic),
            save_query_residual_ablation=(
                args.save_query_residual_ablation),
            save_motion_actor_diagnostic=(
                args.save_motion_actor_diagnostic),
            save_motion_actor_overlay_diagnostic=(
                args.save_motion_actor_overlay_diagnostic),
            motion_actor_step_seconds=args.motion_actor_step_seconds,
            track_score_threshold=args.track_score_threshold,
            current_anchor_root=args.current_anchor_root,
            online_input_root=args.online_input_root)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False))
    print(json.dumps({
        'config': str(args.config),
        'split': args.split,
        'checkpoint_count': len(summaries),
        'dataset_length': len(dataset),
        'shard': shard_summary,
        'output_root': str(args.output_root),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
