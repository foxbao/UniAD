#!/usr/bin/env python
"""Animate sealed B15 final-holdout horizons without new inference."""

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
)
from tools.analysis_tools.visualize_kl_occworld_change_gate import (
    _change_tile,
)
from tools.analysis_tools.visualize_kl_occworld_exported_predictions import (
    _prediction_arrays,
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
            'kl_occworld_b15_final_holdout30_evaluation_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_sequence_b15_final_holdout30_v1'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b15_epoch7_final_holdout30_v1/'
            'final_holdout/epoch_007'))
    parser.add_argument(
        '--visual-summary', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b15_final_holdout30_visuals_v1/summary.json'))
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b15_final_holdout30_visuals_v1'))
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--seconds-per-horizon', type=float, default=1.0)
    parser.add_argument('--final-hold-seconds', type=float, default=1.5)
    return parser.parse_args()


def _load_sample(reference, label_path, prediction_path):
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
    target = target.copy()
    target[~known] = 0
    persistence = np.broadcast_to(current, target.shape).copy()
    persistence[~known] = 0
    prediction, visibility = _prediction_arrays(prediction_path)
    prediction[~known] = 0
    return {
        'reference': int(reference),
        'target': target,
        'persistence': persistence,
        'prediction': prediction,
        'visibility': visibility,
        'target_times': target_times,
        'z_centers': _voxel_z_centers(pc_range, occ_size),
        'collision_z': collision_z,
    }


def _scene_panel(sample, horizon, visibility_threshold, model_label='B15'):
    reference = sample['reference']
    time_s = float(sample['target_times'][horizon])
    suffix = f'#{reference} | t={time_s:.1f}s'
    z_centers = sample['z_centers']
    collision_z = sample['collision_z']
    gt = _bev_tile(
        sample['target'][horizon], z_centers, collision_z,
        f'GT | {suffix}')
    model = _bev_tile(
        sample['prediction'][horizon], z_centers, collision_z,
        f'{model_label} | {suffix}')
    persistence = _bev_tile(
        sample['persistence'][horizon], z_centers, collision_z,
        f'constant-current | {suffix}')
    visibility = _change_tile(
        sample['visibility'][horizon],
        np.ones_like(sample['visibility'][horizon], dtype=np.bool_),
        z_centers, collision_z,
        f'Vis p>={visibility_threshold:.2f} | {suffix}')
    return np.concatenate([
        np.concatenate([gt, model], axis=1),
        np.concatenate([persistence, visibility], axis=1),
    ], axis=0)


def _animation_frame(samples, horizon, visibility_threshold):
    panels = [
        _scene_panel(sample, horizon, visibility_threshold)
        for sample in samples
    ]
    if len(panels) != 6:
        raise ValueError('Animation requires exactly six overview samples')
    body = np.concatenate([
        np.concatenate(panels[:3], axis=1),
        np.concatenate(panels[3:], axis=1),
    ], axis=0)
    header_height = 64
    frame = np.full(
        (body.shape[0] + header_height, body.shape[1], 3),
        24, dtype=np.uint8)
    frame[header_height:] = body
    time_s = float(samples[0]['target_times'][horizon])
    cv2.putText(
        frame,
        f'B15 OccWorld final holdout | frozen epoch 7 | t={time_s:.1f}s',
        (24, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
        (238, 238, 238), 2, cv2.LINE_AA)
    return frame


def _write_gif(mp4_path: Path, gif_path: Path):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return False
    filter_graph = (
        'fps=5,scale=1200:-1:flags=lanczos,split[s0][s1];'
        '[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=bayer')
    subprocess.run([
        ffmpeg, '-y', '-loglevel', 'error', '-i', str(mp4_path),
        '-filter_complex', filter_graph, '-loop', '0', str(gif_path),
    ], check=True)
    return True


def _write_browser_outputs(mp4_path: Path, out_dir: Path):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        return None, None, None
    webm_path = out_dir / 'b15_final_holdout_even6_timeline.webm'
    h264_path = out_dir / 'b15_final_holdout_even6_timeline_h264.mp4'
    subprocess.run([
        ffmpeg, '-y', '-loglevel', 'error', '-i', str(mp4_path),
        '-an', '-c:v', 'libvpx-vp9', '-crf', '31', '-b:v', '0',
        '-row-mt', '1', '-pix_fmt', 'yuv420p', str(webm_path),
    ], check=True)
    subprocess.run([
        ffmpeg, '-y', '-loglevel', 'error', '-i', str(mp4_path),
        '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(h264_path),
    ], check=True)
    html_path = out_dir / 'b15_final_holdout_player.html'
    html_path.write_text('''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>B15 OccWorld Final Holdout</title>
  <style>
    html, body { width: 100%; height: 100%; margin: 0; background: #181818; }
    body { display: grid; place-items: center; overflow: hidden; }
    video { width: 100%; height: 100%; object-fit: contain; background: #181818; }
  </style>
</head>
<body>
  <video controls autoplay muted loop playsinline preload="metadata">
    <source src="b15_final_holdout_even6_timeline.webm" type="video/webm; codecs=vp9">
    <source src="b15_final_holdout_even6_timeline_h264.mp4" type="video/mp4">
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
    if manifest.get('status') != (
            'final_holdout_evaluated_once_no_retuning_allowed'):
        raise ValueError('Final holdout is not sealed')
    protocol = manifest['frozen_model_protocol']
    visibility_threshold = float(protocol['visibility_threshold'])
    visual_summary = json.load(args.visual_summary.open())
    references = [
        int(reference)
        for reference in visual_summary['overview_reference_indices']
    ]
    labels = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    samples = [
        _load_sample(reference, labels[reference], predictions[reference])
        for reference in references
    ]
    horizon_count = len(samples[0]['target_times'])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = args.out_dir / 'animation_frames'
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for horizon in range(horizon_count):
        frame = _animation_frame(
            samples, horizon, visibility_threshold)
        frame_path = frame_dir / f'horizon_{horizon}.png'
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f'Failed to write {frame_path}')
        frames.append(frame)
    mp4_path = args.out_dir / 'b15_final_holdout_even6_timeline.mp4'
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(mp4_path), cv2.VideoWriter_fourcc(*'mp4v'),
        args.fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError('OpenCV could not open the MP4 writer')
    repeats = max(1, int(round(args.fps * args.seconds_per_horizon)))
    final_extra = max(0, int(round(args.fps * args.final_hold_seconds)))
    for frame in frames:
        for _ in range(repeats):
            writer.write(frame)
    for _ in range(final_extra):
        writer.write(frames[-1])
    writer.release()
    if not mp4_path.exists() or mp4_path.stat().st_size == 0:
        raise RuntimeError('MP4 output is empty')
    gif_path = args.out_dir / 'b15_final_holdout_even6_timeline.gif'
    gif_written = _write_gif(mp4_path, gif_path)
    webm_path, h264_path, html_path = _write_browser_outputs(
        mp4_path, args.out_dir)
    summary = {
        'split': 'final_holdout',
        'new_model_inference_performed': False,
        'references': references,
        'target_times_s': samples[0]['target_times'].tolist(),
        'visibility_threshold': visibility_threshold,
        'fps': args.fps,
        'seconds_per_horizon': args.seconds_per_horizon,
        'final_hold_seconds': args.final_hold_seconds,
        'frame_size': [height, width],
        'frame_paths': [
            str(frame_dir / f'horizon_{index}.png')
            for index in range(horizon_count)
        ],
        'mp4_path': str(mp4_path),
        'gif_path': str(gif_path) if gif_written else None,
        'webm_path': None if webm_path is None else str(webm_path),
        'h264_mp4_path': None if h264_path is None else str(h264_path),
        'html_player_path': None if html_path is None else str(html_path),
    }
    with (args.out_dir / 'animation_summary.json').open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
