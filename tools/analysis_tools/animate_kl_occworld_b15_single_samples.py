#!/usr/bin/env python
"""Build one browser animation per sealed B15 final-holdout sample."""

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

from tools.analysis_tools.animate_kl_occworld_b15_final_holdout import (
    _load_sample,
    _scene_panel,
)
from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
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
    parser.add_argument('--references', type=int, nargs='+')
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b15_final_holdout30_visuals_v1/'
            'single_sample_animations'))
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--seconds-per-horizon', type=float, default=1.0)
    parser.add_argument('--final-hold-seconds', type=float, default=1.5)
    return parser.parse_args()


def _frame(sample, horizon, visibility_threshold):
    panel = _scene_panel(sample, horizon, visibility_threshold)
    panel = cv2.resize(
        panel, (1200, 900), interpolation=cv2.INTER_NEAREST)
    header_height = 72
    frame = cv2.copyMakeBorder(
        panel, header_height, 0, 0, 0,
        cv2.BORDER_CONSTANT, value=(24, 24, 24))
    time_s = float(sample['target_times'][horizon])
    cv2.putText(
        frame,
        f"B15 OccWorld final holdout | sample #{sample['reference']} | "
        f't={time_s:.1f}s',
        (24, 46), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
        (238, 238, 238), 2, cv2.LINE_AA)
    return frame


def _write_mp4(frames, path, fps, seconds_per_horizon,
               final_hold_seconds):
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video writer for {path}')
    repeats = max(1, int(round(fps * seconds_per_horizon)))
    final_extra = max(0, int(round(fps * final_hold_seconds)))
    for frame in frames:
        for _ in range(repeats):
            writer.write(frame)
    for _ in range(final_extra):
        writer.write(frames[-1])
    writer.release()
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f'Empty video output: {path}')


def _write_browser_outputs(mp4_path, output_dir, stem):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required for browser video outputs')
    webm_path = output_dir / f'{stem}.webm'
    h264_path = output_dir / f'{stem}_h264.mp4'
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
    html_path = output_dir / f'{stem}_player.html'
    html_path.write_text(f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>B15 OccWorld sample</title>
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
    if manifest.get('status') != (
            'final_holdout_evaluated_once_no_retuning_allowed'):
        raise ValueError('Final holdout is not sealed')
    visibility_threshold = float(
        manifest['frozen_model_protocol']['visibility_threshold'])
    with args.visual_summary.open() as source:
        visual_summary = json.load(source)
    references = (
        [int(value) for value in args.references]
        if args.references is not None else [
            int(value) for value in
            visual_summary['overview_reference_indices']
        ])
    allowed = {
        int(row['reference_index'])
        for row in manifest['splits']['final_holdout']
    }
    if not references or not set(references).issubset(allowed):
        raise ValueError('Requested reference is outside final holdout')
    labels = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for reference in references:
        sample = _load_sample(
            reference, labels[reference], predictions[reference])
        frames = [
            _frame(sample, horizon, visibility_threshold)
            for horizon in range(len(sample['target_times']))
        ]
        output_dir = args.out_dir / f'{reference:06d}'
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_dir = output_dir / 'frames'
        frame_dir.mkdir(parents=True, exist_ok=True)
        for horizon, frame in enumerate(frames):
            path = frame_dir / f'horizon_{horizon}.png'
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f'Could not write {path}')
        stem = f'b15_final_{reference:06d}'
        mp4_path = output_dir / f'{stem}.mp4'
        _write_mp4(
            frames, mp4_path, args.fps,
            args.seconds_per_horizon, args.final_hold_seconds)
        webm_path, h264_path, html_path = _write_browser_outputs(
            mp4_path, output_dir, stem)
        entries.append({
            'reference_index': reference,
            'new_model_inference_performed': False,
            'target_times_s': sample['target_times'].tolist(),
            'frame_size': list(frames[0].shape[:2]),
            'webm_path': str(webm_path),
            'h264_mp4_path': str(h264_path),
            'html_player_path': str(html_path),
        })
    summary = {
        'split': 'final_holdout',
        'layout': 'one reference per animation',
        'selection_rule': (
            'frozen six evenly spaced manifest positions unless overridden'),
        'visibility_threshold': visibility_threshold,
        'fps': args.fps,
        'seconds_per_horizon': args.seconds_per_horizon,
        'final_hold_seconds': args.final_hold_seconds,
        'entries': entries,
    }
    with (args.out_dir / 'summary.json').open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps({
        'animation_count': len(entries),
        'references': references,
        'output_root': str(args.out_dir),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
