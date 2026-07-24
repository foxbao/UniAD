#!/usr/bin/env python
"""Compare B15, B16 and validation-selected B17A predictions."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)
from tools.analysis_tools.visualize_kl_occworld_change_gate import (
    _change_tile,
)
from tools.analysis_tools.visualize_kl_occworld_exported_predictions import (
    _prediction_arrays,
    _semantic_row,
)
from tools.analysis_tools.visualize_kl_occworld_temporal_predictions import (
    _bev_tile,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_full_train3_final30_manifest_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_expanded70'))
    parser.add_argument(
        '--b15-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b15_full_predicted_history_validation_v1/'
            'validation/epoch_007'))
    parser.add_argument(
        '--b16-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b16_full_predicted_history_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument(
        '--b17-prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b17a_full_history_validation_v1/'
            'validation/epoch_003'))
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b17a_full_history_validation_visuals_v1'))
    parser.add_argument('--animation-reference', type=int)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--seconds-per-horizon', type=float, default=1.0)
    parser.add_argument('--final-hold-seconds', type=float, default=1.5)
    return parser.parse_args()


def _sample_future_miou(target, prediction, known):
    scope = known.copy()
    scope[0] = False
    ious = []
    for state in (1, 2, 3):
        target_mask = scope & (target == state)
        prediction_mask = scope & (prediction == state)
        union = np.count_nonzero(target_mask | prediction_mask)
        if union:
            intersection = np.count_nonzero(
                target_mask & prediction_mask)
            ious.append(intersection / union)
    return float(np.mean(ious)) if ious else 0.0


def _binary_row(name, masks, known, target_times, z_centers, collision_z):
    return np.concatenate([
        _change_tile(
            mask.astype(np.float32), scope, z_centers, collision_z,
            f'{name} | t={time_s:.1f}s')
        for mask, scope, time_s in zip(masks, known, target_times)
    ], axis=1)


def _contact_sheet(rows):
    separator = np.full(
        (6, rows[0].shape[1], 3), 28, dtype=np.uint8)
    contact = rows[0]
    for row in rows[1:]:
        contact = np.concatenate([contact, separator, row], axis=0)
    return contact


def _load_record(reference, label_path, b15_path, b16_path, b17_path):
    with np.load(label_path, allow_pickle=False) as label:
        target = np.asarray(
            label['world_target_state_3d'], dtype=np.uint8)
        target_valid = np.asarray(
            label['world_target_valid_3d'], dtype=np.bool_)
        current = np.asarray(
            label['current_observation_state_3d'], dtype=np.uint8)
        target_times = np.asarray(
            label['target_times_s'], dtype=np.float32)
        pc_range = np.asarray(label['pc_range'], dtype=np.float32)
        occ_size = np.asarray(label['occ_size'], dtype=np.int64)
        collision_z = tuple(
            np.asarray(label['collision_z'], dtype=np.float32))

    known = target_valid & (target != 0)
    persistence = np.broadcast_to(current, target.shape).copy()
    b15, _ = _prediction_arrays(b15_path)
    b16, _ = _prediction_arrays(b16_path)
    b17, _ = _prediction_arrays(b17_path)
    display_arrays = [target, persistence, b15, b16, b17]
    for states in display_arrays:
        states[~known] = 0

    future = known.copy()
    future[0] = False
    b15_correct = future & (b15 == target)
    b16_correct = future & (b16 == target)
    b17_correct = future & (b17 == target)
    fixed_vs_b16 = b17_correct & ~b16_correct
    harmed_vs_b16 = b16_correct & ~b17_correct
    metrics = {
        'b15_future_miou': _sample_future_miou(
            target, b15, known),
        'b16_future_miou': _sample_future_miou(
            target, b16, known),
        'b17_future_miou': _sample_future_miou(
            target, b17, known),
        'b17_only_correct_voxels_vs_b16': int(
            np.count_nonzero(fixed_vs_b16)),
        'b16_only_correct_voxels_vs_b17': int(
            np.count_nonzero(harmed_vs_b16)),
    }
    metrics['b17_minus_b16_future_miou'] = (
        metrics['b17_future_miou'] - metrics['b16_future_miou'])
    metrics['net_correct_voxel_gain_vs_b16'] = (
        metrics['b17_only_correct_voxels_vs_b16']
        - metrics['b16_only_correct_voxels_vs_b17'])
    return {
        'reference_index': int(reference),
        'target': target,
        'persistence': persistence,
        'b15': b15,
        'b16': b16,
        'b17': b17,
        'known': known,
        'fixed_vs_b16': fixed_vs_b16,
        'harmed_vs_b16': harmed_vs_b16,
        'target_times': target_times,
        'z_centers': _voxel_z_centers(pc_range, occ_size),
        'collision_z': collision_z,
        'metrics': metrics,
    }


def _animation_frame(record, horizon):
    reference = record['reference_index']
    time_s = float(record['target_times'][horizon])
    suffix = f'#{reference} | t={time_s:.1f}s'
    z_centers = record['z_centers']
    collision_z = record['collision_z']
    tiles = [
        _bev_tile(
            record['target'][horizon], z_centers, collision_z,
            f'future GT | {suffix}'),
        _bev_tile(
            record['b15'][horizon], z_centers, collision_z,
            f'B15 | {suffix}'),
        _bev_tile(
            record['b16'][horizon], z_centers, collision_z,
            f'B16 | {suffix}'),
        _bev_tile(
            record['b17'][horizon], z_centers, collision_z,
            f'B17A epoch 3 | {suffix}'),
    ]
    panel = np.concatenate([
        np.concatenate(tiles[:2], axis=1),
        np.concatenate(tiles[2:], axis=1),
    ], axis=0)
    panel = cv2.resize(
        panel, (1200, 900), interpolation=cv2.INTER_NEAREST)
    frame = cv2.copyMakeBorder(
        panel, 76, 0, 0, 0, cv2.BORDER_CONSTANT,
        value=(24, 24, 24))
    delta = record['metrics']['b17_minus_b16_future_miou'] * 100
    cv2.putText(
        frame,
        f'Full predicted history validation | B17A-B16 sample mIoU '
        f'{delta:+.2f} pp',
        (24, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.92,
        (238, 238, 238), 2, cv2.LINE_AA)
    return frame


def _write_video(frames, out_dir, stem, fps, seconds_per_horizon,
                 final_hold_seconds):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required for browser outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    source_path = out_dir / f'{stem}_source.mp4'
    writer = cv2.VideoWriter(
        str(source_path), cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer: {source_path}')
    repeats = max(1, int(round(fps * seconds_per_horizon)))
    for frame in frames:
        for _ in range(repeats):
            writer.write(frame)
    for _ in range(max(0, int(round(fps * final_hold_seconds)))):
        writer.write(frames[-1])
    writer.release()

    webm_path = out_dir / f'{stem}.webm'
    h264_path = out_dir / f'{stem}_h264.mp4'
    subprocess.run([
        ffmpeg, '-y', '-loglevel', 'error', '-i', str(source_path),
        '-an', '-c:v', 'libvpx-vp9', '-crf', '31', '-b:v', '0',
        '-row-mt', '1', '-pix_fmt', 'yuv420p', str(webm_path),
    ], check=True)
    subprocess.run([
        ffmpeg, '-y', '-loglevel', 'error', '-i', str(source_path),
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(h264_path),
    ], check=True)
    html_path = out_dir / f'{stem}_player.html'
    html_path.write_text(f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OccWorld B17 validation comparison</title>
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; background: #181818; }}
    body {{ display: grid; place-items: center; overflow: hidden; }}
    video {{ width: 100%; height: 100%; object-fit: contain; background: #181818; }}
  </style>
</head>
<body>
  <video controls autoplay muted loop playsinline preload="metadata">
    <source src="{webm_path.name}" type="video/webm; codecs=vp9">
    <source src="{h264_path.name}" type="video/mp4">
  </video>
</body>
</html>
''')
    return webm_path, h264_path, html_path


def main():
    args = parse_args()
    if args.fps < 1 or args.seconds_per_horizon <= 0:
        raise ValueError('Animation timing must be positive')
    manifest = _load_manifest(args.manifest)
    references = _split_references(manifest, 'validation')
    labels = _sequence_mapping(args.sequence_root)
    b15_predictions = _prediction_mapping(args.b15_prediction_root)
    b16_predictions = _prediction_mapping(args.b16_prediction_root)
    b17_predictions = _prediction_mapping(args.b17_prediction_root)
    expected = set(references)
    for name, mapping in (
            ('labels', labels), ('B15', b15_predictions),
            ('B16', b16_predictions), ('B17', b17_predictions)):
        missing = sorted(expected - set(mapping))
        if missing:
            raise FileNotFoundError(f'{name} is missing references: {missing}')

    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for reference in references:
        record = _load_record(
            reference, labels[reference], b15_predictions[reference],
            b16_predictions[reference], b17_predictions[reference])
        rows = [
            _semantic_row(
                'future GT', record['target'], record['target_times'],
                record['z_centers'], record['collision_z']),
            _semantic_row(
                'constant-current', record['persistence'],
                record['target_times'], record['z_centers'],
                record['collision_z']),
            _semantic_row(
                'B15 full history', record['b15'],
                record['target_times'], record['z_centers'],
                record['collision_z']),
            _semantic_row(
                'B16 full history', record['b16'],
                record['target_times'], record['z_centers'],
                record['collision_z']),
            _semantic_row(
                'B17A epoch 3', record['b17'],
                record['target_times'], record['z_centers'],
                record['collision_z']),
            _binary_row(
                'B17 fixed B16 error', record['fixed_vs_b16'],
                record['known'], record['target_times'],
                record['z_centers'], record['collision_z']),
            _binary_row(
                'B17 introduced error', record['harmed_vs_b16'],
                record['known'], record['target_times'],
                record['z_centers'], record['collision_z']),
        ]
        contact_path = args.out_dir / (
            f'{reference:06d}__b17_compare.png')
        if not cv2.imwrite(str(contact_path), _contact_sheet(rows)):
            raise OSError(f'Failed to write {contact_path}')
        record['contact_path'] = str(contact_path)
        records.append(record)

    ranked = sorted(
        records,
        key=lambda row: row['metrics'][
            'b17_minus_b16_future_miou'],
        reverse=True)
    if args.animation_reference is None:
        animation_record = ranked[0]
        selection_rule = 'largest per-sample B17A minus B16 future mIoU'
    else:
        matches = [
            row for row in records
            if row['reference_index'] == args.animation_reference]
        if not matches:
            raise ValueError(
                '--animation-reference is outside validation')
        animation_record = matches[0]
        selection_rule = 'explicit user-provided validation reference'

    reference = animation_record['reference_index']
    animation_dir = args.out_dir / f'{reference:06d}_animation'
    frames = [
        _animation_frame(animation_record, horizon)
        for horizon in range(len(animation_record['target_times']))
    ]
    frame_dir = animation_dir / 'frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    for horizon, frame in enumerate(frames):
        path = frame_dir / f'horizon_{horizon}.png'
        if not cv2.imwrite(str(path), frame):
            raise OSError(f'Failed to write {path}')
    stem = f'b17_validation_{reference:06d}'
    webm_path, h264_path, html_path = _write_video(
        frames, animation_dir, stem, args.fps,
        args.seconds_per_horizon, args.final_hold_seconds)

    summary = {
        'split': 'validation',
        'new_model_inference_performed': False,
        'ranking_metric': 'per-sample future semantic mIoU',
        'ranking_note': (
            'Per-sample ranking is for visualization only and does not '
            'replace dataset-level checkpoint selection.'),
        'records_ranked_by_b17_minus_b16_future_miou': [
            {
                'reference_index': row['reference_index'],
                **row['metrics'],
                'contact_path': row['contact_path'],
            }
            for row in ranked
        ],
        'animation': {
            'reference_index': reference,
            'selection_rule': selection_rule,
            'webm_path': str(webm_path),
            'h264_mp4_path': str(h264_path),
            'html_player_path': str(html_path),
        },
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'reference_count': len(records),
        'animation_reference': reference,
        'animation_delta_pp': (
            animation_record['metrics'][
                'b17_minus_b16_future_miou'] * 100),
        'webm_path': str(webm_path),
        'html_player_path': str(html_path),
        'summary_path': str(summary_path),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
