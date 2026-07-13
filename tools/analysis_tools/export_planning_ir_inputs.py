#!/usr/bin/env python
"""Export leak-free Planning-IR teacher inputs from a D2 audit eval."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import os.path as osp
import sys

import mmcv
import numpy as np
from mmcv import Config


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from projects.mmdet3d_plugin.uniad.dense_heads.motion_head_plugin.map_lane_encoder import (  # noqa: E501
    MapPlanningCandidateGenerator,
)
from third_party.uniad_mmdet3d.datasets.builder import build_dataset
from tools.analysis_tools.planning_ir_audit_utils import (
    front_obstacle_bucket,
    horizon_collisions,
    horizon_l2,
    motion_bucket,
    planning_mask,
    planning_trajectory,
    to_numpy,
    write_jsonl,
)


INPUT_SCHEMA_VERSION = 'planning-ir-audit-input/v1'
COMMAND_NAMES = {0: 'RIGHT', 1: 'LEFT', 2: 'STRAIGHT'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export online-only Planning-IR inputs and audit labels.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--results-pkl', required=True)
    parser.add_argument('--out-jsonl', required=True)
    parser.add_argument('--semantic-rules', default=None,
                        help='Optional static semantic-rule JSON file.')
    parser.add_argument('--max-actors', type=int, default=16)
    parser.add_argument('--score-threshold', type=float, default=0.05)
    parser.add_argument('--limit', type=int, default=0,
                        help='Export only the first N frames; 0 means all.')
    return parser.parse_args()


def import_cfg_modules(cfg, config_path):
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)
    if cfg.get('plugin', False):
        module_dir = osp.dirname(cfg.get(
            'plugin_dir', osp.dirname(config_path)))
        module_path = module_dir.replace('/', '.').strip('.')
        if module_path:
            importlib.import_module(module_path)


def resolve_path(path):
    if osp.isabs(path) or osp.exists(path):
        return path
    return osp.join(REPO_ROOT, path)


def unwrap_results(loaded):
    if isinstance(loaded, dict):
        loaded = loaded.get('bbox_results', loaded.get('results'))
    if not isinstance(loaded, list):
        raise TypeError('Expected a result list or dict containing bbox_results')
    return loaded


def unwrap_pts_bbox(result):
    if isinstance(result, (list, tuple)) and result:
        result = result[0]
    if not isinstance(result, dict):
        raise TypeError(f'Unexpected result type: {type(result)}')
    return result.get('pts_bbox', result), result


def squeeze_sample(value, ndim):
    array = to_numpy(value)
    if array is None:
        return None
    while array.ndim > ndim and array.shape[0] == 1:
        array = array[0]
    if array.ndim != ndim:
        raise ValueError(
            f'Expected {ndim} dimensions after batch squeeze, got {array.shape}')
    return array


def scalar(value, default=None):
    array = to_numpy(value)
    if array is None or array.size == 0:
        return default
    return array.reshape(-1)[0].item()


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def float_list(value):
    return [finite_float(item) for item in np.asarray(value).reshape(-1)]


def trajectory_list(value):
    array = np.asarray(value, dtype=np.float64)
    return [[finite_float(x), finite_float(y)] for x, y in array[:, :2]]


def load_semantic_rules(path):
    if path is None:
        return []
    with open(resolve_path(path), 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    rules = payload.get('rules', payload) if isinstance(payload, dict) else payload
    if not isinstance(rules, list):
        raise TypeError('Semantic rules must be a list or {"rules": [...]}')
    normalized = []
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict):
            raise TypeError(f'Semantic rule {index} must be an object')
        rule_id = str(rule.get('rule_id', rule.get('id', index)))
        normalized.append(dict(rule, rule_id=rule_id))
    return normalized


def candidate_rules(rules, candidates):
    lane_ids = {
        lane['lane_id']
        for candidate in candidates
        for lane in candidate.get('lane_sequence', [])
    }
    selected = []
    for rule in rules:
        applies_to = rule.get('lane_ids')
        if not applies_to or lane_ids.intersection(map(str, applies_to)):
            selected.append(rule)
    return selected


def class_name(dataset, label):
    classes = getattr(dataset, 'CLASSES', None)
    if classes is None:
        return str(label)
    return str(classes[label]) if 0 <= label < len(classes) else str(label)


def best_motion_trajectories(traj, scores, count):
    traj = to_numpy(traj)
    if traj is None or traj.ndim != 4:
        return [None] * count
    output = [None] * count
    available = min(count, len(traj))
    scores = to_numpy(scores)
    if scores is None or scores.ndim != 2:
        modes = np.zeros((available,), dtype=np.int64)
    else:
        modes = np.argmax(scores[:available], axis=1)
    selected = traj[np.arange(available), modes, :, :2]
    output[:available] = [trajectory_list(item) for item in selected]
    return output


def build_actor_inputs(pts_bbox, dataset, max_actors, score_threshold):
    boxes = to_numpy(pts_bbox.get(
        'track_boxes_3d', pts_bbox.get('boxes_3d')))
    if boxes is None:
        return []
    boxes = boxes.reshape(-1, boxes.shape[-1])
    scores = to_numpy(pts_bbox.get(
        'track_scores', pts_bbox.get('scores_3d')))
    labels = to_numpy(pts_bbox.get(
        'track_labels_3d', pts_bbox.get('labels_3d')))
    track_ids = to_numpy(pts_bbox.get('track_ids'))
    scores = (np.ones(len(boxes), dtype=np.float32) if scores is None
              else scores.reshape(-1))
    labels = (np.zeros(len(boxes), dtype=np.int64) if labels is None
              else labels.reshape(-1).astype(np.int64))
    track_ids = (np.arange(len(boxes), dtype=np.int64)
                 if track_ids is None else track_ids.reshape(-1))
    count = min(len(boxes), len(scores), len(labels), len(track_ids))
    motions = best_motion_trajectories(
        pts_bbox.get('traj'), pts_bbox.get('traj_scores'), count)
    keep = np.argsort(-scores[:count])
    keep = keep[scores[keep] >= score_threshold][:max_actors]
    actors = []
    for index in keep:
        box = boxes[index]
        actor = dict(
            actor_id=str(int(track_ids[index])),
            class_name=class_name(dataset, int(labels[index])),
            confidence=finite_float(scores[index]),
            center_xy_m=float_list(box[:2]),
            size_lwh_m=float_list(box[3:6]) if len(box) >= 6 else [],
            yaw_rad=finite_float(box[6]) if len(box) >= 7 else None,
            velocity_xy_mps=(float_list(box[7:9])
                             if len(box) >= 9 else None),
            predicted_displacements_xy_m=motions[index],
        )
        actors.append(actor)
    return actors


def build_ego_input(pts_bbox, raw_result):
    boxes = to_numpy(pts_bbox.get('sdc_boxes_3d'))
    box = None if boxes is None or boxes.size == 0 else boxes.reshape(
        -1, boxes.shape[-1])[0]
    planning = raw_result.get('planning', {})
    planning_gt = planning.get('planning_gt', {}) \
        if isinstance(planning, dict) else {}
    command = scalar(raw_result.get(
        'command', planning_gt.get('command')), default=-1)
    ego = dict(
        route_command_id=int(command),
        route_command=COMMAND_NAMES.get(int(command), 'UNKNOWN'),
    )
    if box is not None:
        ego.update(
            center_xy_m=float_list(box[:2]),
            size_lwh_m=float_list(box[3:6]) if len(box) >= 6 else [],
            yaw_rad=finite_float(box[6]) if len(box) >= 7 else None,
            velocity_xy_mps=(float_list(box[7:9])
                             if len(box) >= 9 else None),
        )
    return ego


def build_generator(cfg):
    lane_cfg = cfg.model.get('map_lane_encoder')
    if lane_cfg is None or lane_cfg.get('planning_candidates') is None:
        raise ValueError('Config does not enable map planning candidates')
    candidate_cfg = dict(lane_cfg.planning_candidates)
    return MapPlanningCandidateGenerator(
        map_path=resolve_path(lane_cfg.map_path),
        **{key: resolve_path(value) if key.endswith('_path') else value
           for key, value in candidate_cfg.items()})


def build_candidates(result_planning, metadata):
    required = (
        'multimodal_audit_indices',
        'multimodal_audit_valid',
        'multimodal_audit_raw_candidates',
        'multimodal_audit_refined_candidates',
    )
    missing = [key for key in required if result_planning.get(key) is None]
    if missing:
        raise KeyError(
            'Audit eval output is missing fields: ' + ', '.join(missing))
    indices = squeeze_sample(
        result_planning['multimodal_audit_indices'], 1).astype(np.int64)
    valid = squeeze_sample(
        result_planning['multimodal_audit_valid'], 1).astype(bool)
    raw = squeeze_sample(
        result_planning['multimodal_audit_raw_candidates'], 3)
    refined = squeeze_sample(
        result_planning['multimodal_audit_refined_candidates'], 3)
    logits = squeeze_sample(
        result_planning.get('multimodal_audit_logits'), 1)
    probabilities = squeeze_sample(
        result_planning.get('multimodal_audit_selection_probabilities'), 1)
    costs = squeeze_sample(
        result_planning.get('multimodal_audit_predicted_horizon_costs'), 2)
    metadata_by_id = {item['candidate_id']: item for item in metadata}
    fallback_id = len(metadata)
    if (len(indices) == 0 or int(indices[-1]) != fallback_id
            or int(np.sum(indices == fallback_id)) != 1):
        raise ValueError(
            'Audit payload does not end with the exact runtime fallback: '
            f'expected candidate {fallback_id}, got {indices.tolist()}')
    if len(set(indices.tolist())) != len(indices):
        raise ValueError('Audit payload contains duplicate candidate IDs')
    if np.any(indices < 0) or np.any(indices > fallback_id):
        raise ValueError('Audit payload contains an out-of-range candidate ID')
    candidates = []
    for offset, candidate_id in enumerate(indices.tolist()):
        candidate = dict(
            candidate_id=int(candidate_id),
            source=('fallback' if candidate_id == fallback_id else 'map'),
            valid=bool(valid[offset]),
            raw_trajectory_xy_m=trajectory_list(raw[offset]),
            refined_trajectory_xy_m=trajectory_list(refined[offset]),
            scorer_logit=(finite_float(logits[offset])
                           if logits is not None else None),
            selection_probability=(finite_float(probabilities[offset])
                                   if probabilities is not None else None),
        )
        if costs is not None:
            horizon_costs = float_list(costs[offset])
            candidate['predicted_horizon_costs_m'] = horizon_costs
            finite = [value for value in horizon_costs if value is not None]
            candidate['predicted_mean_cost_m'] = (
                float(np.mean(finite)) if finite else None)
        if candidate['source'] == 'map':
            candidate.update(metadata_by_id[candidate_id])
        else:
            candidate.update(
                path_index=None,
                speed_profile_index=None,
                lateral_offset_index=None,
            )
        candidates.append(candidate)
    return candidates


def candidate_audit_labels(candidates, gt_xy, valid, segmentation,
                           pc_range, cell_size):
    metrics = {}
    for candidate in candidates:
        candidate_id = str(candidate['candidate_id'])
        if not candidate['valid'] or gt_xy is None or valid is None:
            metrics[candidate_id] = dict(
                horizon_l2_m=[None, None, None], mean_l2_m=None,
                horizon_collision=[None, None, None])
            continue
        trajectory = np.asarray(
            candidate['refined_trajectory_xy_m'], dtype=np.float32)
        horizon_values, mean_value = horizon_l2(
            trajectory, gt_xy, valid)
        metrics[candidate_id] = dict(
            horizon_l2_m=horizon_values,
            mean_l2_m=mean_value,
            horizon_collision=horizon_collisions(
                trajectory, segmentation, pc_range, cell_size),
        )
    return metrics


def make_record(index, result, dataset, generator, semantic_rules,
                max_actors, score_threshold):
    pts_bbox, raw_result = unwrap_pts_bbox(result)
    planning = raw_result.get('planning', {})
    result_planning = planning.get('result_planning', {})
    planning_gt = planning.get('planning_gt', {})
    ego2global = to_numpy(pts_bbox.get('ego2global'))
    if ego2global is None:
        raise KeyError('pts_bbox.ego2global is required for candidate metadata')
    ego2global = ego2global.reshape(4, 4)
    metadata = generator.build_candidate_metadata(ego2global)
    candidates = build_candidates(result_planning, metadata)

    raw_index = (dataset.valid_data_indices[index]
                 if getattr(dataset, 'valid_data_indices', None) is not None
                 else index)
    info = dataset.data_infos[raw_index]
    sample = dict(
        token=str(info.get('token', index)),
        scene_token=str(info.get('scene_token', info.get('scene', ''))),
        timestamp=int(info.get('timestamp', 0)),
        result_index=index,
        raw_dataset_index=raw_index,
    )
    rules = candidate_rules(semantic_rules, candidates)
    teacher_input = dict(
        sample=sample,
        ego=build_ego_input(pts_bbox, raw_result),
        actors=build_actor_inputs(
            pts_bbox, dataset, max_actors, score_threshold),
        candidates=candidates,
        semantic_rules=rules,
    )

    gt_plan = planning_trajectory(planning_gt.get('sdc_planning'))
    gt_mask = planning_mask(planning_gt.get('sdc_planning_mask'))
    if gt_plan is not None and gt_mask is not None:
        steps = min(len(gt_plan), len(gt_mask))
        gt_plan = gt_plan[:steps]
        gt_mask = gt_mask[:steps]
    segmentation = planning_gt.get('segmentation')
    pc_range = list(dataset._planning_pc_range())
    cell_size = float(dataset._planning_cell_size())
    metrics = candidate_audit_labels(
        candidates, gt_plan, gt_mask, segmentation, pc_range, cell_size)
    valid_metrics = {
        int(candidate_id): values['mean_l2_m']
        for candidate_id, values in metrics.items()
        if values['mean_l2_m'] is not None
    }
    fallback_id = next(
        candidate['candidate_id'] for candidate in candidates
        if candidate['source'] == 'fallback')
    selected_id = int(scalar(
        result_planning.get('multimodal_selected_index'), fallback_id))
    if selected_id not in {
            candidate['candidate_id'] for candidate in candidates}:
        raise ValueError(
            f'D2 selected candidate {selected_id} is absent from audit top-K')
    oracle_id = (min(valid_metrics, key=valid_metrics.get)
                 if valid_metrics else fallback_id)
    audit_labels = dict(
        gt_planning_xy_m=(trajectory_list(gt_plan)
                          if gt_plan is not None else None),
        gt_planning_valid=(gt_mask.astype(bool).tolist()
                           if gt_mask is not None else None),
        motion_bucket=(motion_bucket(gt_plan, gt_mask)
                       if gt_plan is not None and gt_mask is not None
                       else 'invalid'),
        obstacle_bucket=front_obstacle_bucket(
            segmentation, pc_range, cell_size),
        candidate_metrics=metrics,
        d2_selected_candidate_id=selected_id,
        fallback_candidate_id=int(fallback_id),
        audit_oracle_candidate_id=int(oracle_id),
    )
    return dict(
        input_schema_version=INPUT_SCHEMA_VERSION,
        result_index=index,
        sample_token=sample['token'],
        teacher_input=teacher_input,
        audit_labels=audit_labels,
    )


def main():
    args = parse_args()
    if args.max_actors < 0:
        raise ValueError('--max-actors must be non-negative')
    cfg_path = resolve_path(args.config)
    cfg = Config.fromfile(cfg_path)
    import_cfg_modules(cfg, cfg_path)
    dataset = build_dataset(cfg.data.test)
    results = unwrap_results(mmcv.load(resolve_path(args.results_pkl)))
    if len(results) != len(dataset):
        raise ValueError(
            f'Result count {len(results)} != dataset count {len(dataset)}')
    generator = build_generator(cfg)
    semantic_rules = load_semantic_rules(args.semantic_rules)
    count = len(results) if args.limit <= 0 else min(args.limit, len(results))

    def records():
        for index in range(count):
            yield make_record(
                index, results[index], dataset, generator, semantic_rules,
                args.max_actors, args.score_threshold)

    write_jsonl(resolve_path(args.out_jsonl), records())
    print(f'Exported {count} Planning-IR audit records to {args.out_jsonl}')


if __name__ == '__main__':
    main()
