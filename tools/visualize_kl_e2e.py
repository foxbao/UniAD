#!/usr/bin/env python
"""Visualize KL UniAD LiDAR E2E tracking and motion results in BEV.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_kl_e2e.py \
        --config projects/configs/stage2_e2e_lidar/base_e2e_lidar.py \
        --checkpoint projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/epoch_1.pth \
        --out-dir projects/work_dirs/vis_e2e_epoch1_scene0 \
        --start-index 0 --max-frames 24 \
        --score-thr 0.0 --motion-top-modes 3
"""

from __future__ import annotations

import argparse
import os
import os.path as osp
import subprocess
import sys
from typing import Dict, List, Optional, Sequence

os.environ.setdefault('MPLCONFIGDIR', '/tmp/matplotlib')
REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmcv
import numpy as np
from mmcv import Config, DictAction

from mmdet3d.datasets import build_dataset

from visualize_kl_track import (build_uniad_model, color_from_track_id,
                                consecutive_scene_indices,
                                draw_gt_boxes, draw_points_and_ego,
                                draw_track_boxes, import_cfg_modules,
                                model_forward_one, raw_info_from_dataset,
                                reset_track_state, resolve_start_index,
                                setup_bev_axis, unwrap_uniad_result)
from visualize_kl_velocity import (boxes_to_numpy, get_class_names,
                                   get_dataset_cfg, get_label_mapping,
                                   lidar_xy_to_display, load_points,
                                   map_label, tensor_to_numpy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Visualize KL UniAD E2E tracking and motion in BEV.')
    parser.add_argument('--config', required=True, help='config file path')
    parser.add_argument('--checkpoint', required=True, help='checkpoint file')
    parser.add_argument('--out-dir', required=True, help='output directory')
    parser.add_argument(
        '--split',
        default='val',
        choices=['val', 'test'],
        help='which dataloader config to use')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument(
        '--token',
        default=None,
        help='start from this sample token instead of --start-index')
    parser.add_argument(
        '--scene-token',
        default=None,
        help='start from the first sample of this scene token')
    parser.add_argument('--max-frames', type=int, default=24)
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='extra visualization threshold after tracker emission')
    parser.add_argument('--topk', type=int, default=120)
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument('--vel-scale', type=float, default=1.2)
    parser.add_argument('--min-vel-draw', type=float, default=0.2)
    parser.add_argument(
        '--motion-top-modes',
        type=int,
        default=3,
        help='number of trajectory modes to draw per tracked object')
    parser.add_argument(
        '--motion-min-prob',
        type=float,
        default=0.02,
        help='skip non-best motion modes below this softmax probability')
    parser.add_argument(
        '--traj-coordinate',
        default='offset',
        choices=['offset', 'absolute'],
        help='motion traj coordinate. KL motion uses offset from current box.')
    parser.add_argument('--annotate', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument(
        '--webm-fps',
        type=float,
        default=3.0,
        help='FPS for generated e2e_vis.webm. Set <= 0 to skip WebM.')
    parser.add_argument(
        '--webm-crf',
        type=int,
        default=34,
        help='VP9 CRF for generated WebM; lower is higher quality.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override config options, e.g. model.score_thresh=0.7')
    args = parser.parse_args()
    if args.max_frames <= 0:
        raise ValueError('--max-frames must be positive.')
    if args.point_stride <= 0:
        raise ValueError('--point-stride must be positive.')
    if args.topk <= 0:
        raise ValueError('--topk must be positive.')
    if args.motion_top_modes <= 0:
        raise ValueError('--motion-top-modes must be positive.')
    return args


def _empty_motion(num_boxes: int = 0) -> Dict[str, np.ndarray]:
    return dict(
        traj=np.zeros((num_boxes, 0, 0, 2), dtype=np.float32),
        traj_scores=np.zeros((num_boxes, 0), dtype=np.float32))


def _normalize_traj_array(traj, num_boxes: int) -> np.ndarray:
    if traj is None:
        return _empty_motion(num_boxes)['traj']
    traj = tensor_to_numpy(traj)
    if traj is None:
        return _empty_motion(num_boxes)['traj']
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[:, None, :, :]
    if traj.ndim != 4 or traj.shape[-1] < 2:
        return _empty_motion(num_boxes)['traj']
    return np.nan_to_num(traj[..., :2], nan=0.0, posinf=0.0, neginf=0.0)


def _normalize_score_array(scores, num_boxes: int, num_modes: int) -> np.ndarray:
    if scores is None:
        return np.zeros((num_boxes, num_modes), dtype=np.float32)
    scores = tensor_to_numpy(scores)
    if scores is None:
        return np.zeros((num_boxes, num_modes), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim == 1:
        scores = scores[:, None]
    if scores.ndim != 2:
        return np.zeros((num_boxes, num_modes), dtype=np.float32)
    return np.nan_to_num(scores, nan=-1e6, posinf=1e6, neginf=-1e6)


def pred_arrays_from_e2e_result(result: dict, score_thr: float,
                                topk: int) -> Dict[str, np.ndarray]:
    result = unwrap_uniad_result(result)
    boxes = boxes_to_numpy(
        result.get('track_boxes_3d', result.get('boxes_3d', None)))
    scores = tensor_to_numpy(
        result.get('track_scores', result.get('scores_3d',
                                              np.zeros((len(boxes),)))))
    labels = tensor_to_numpy(
        result.get('track_labels_3d', result.get('labels_3d',
                                                 np.zeros((len(boxes),)))))
    track_ids = tensor_to_numpy(
        result.get('track_ids', np.full((len(boxes),), -1)))
    traj = _normalize_traj_array(result.get('traj', None), len(boxes))
    traj_scores = _normalize_score_array(
        result.get('traj_scores', None), len(boxes),
        traj.shape[1] if traj.ndim == 4 else 0)

    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    track_ids = np.asarray(track_ids, dtype=np.int64).reshape(-1)
    if len(boxes) == 0:
        out = dict(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0, ), dtype=np.int64),
            scores=np.zeros((0, ), dtype=np.float32),
            track_ids=np.zeros((0, ), dtype=np.int64))
        out.update(_empty_motion(0))
        return out

    num = min(len(boxes), len(scores), len(labels), len(track_ids), len(traj))
    if len(traj_scores):
        num = min(num, len(traj_scores))
    boxes = boxes[:num]
    scores = scores[:num]
    labels = labels[:num]
    track_ids = track_ids[:num]
    traj = traj[:num]
    traj_scores = traj_scores[:num]

    order = np.argsort(-scores)
    keep = order[scores[order] >= score_thr][:topk]
    return dict(
        boxes=boxes[keep],
        labels=labels[keep],
        scores=scores[keep],
        track_ids=track_ids[keep],
        traj=traj[keep],
        traj_scores=traj_scores[keep])


def _stack_gt_trajs(trajs: Sequence[np.ndarray],
                    masks: Sequence[np.ndarray]) -> Dict[str, np.ndarray]:
    if not trajs:
        return _empty_motion(0)
    num_steps = max((traj.shape[0] for traj in trajs), default=0)
    stacked_trajs = np.zeros((len(trajs), num_steps, 2), dtype=np.float32)
    stacked_masks = np.zeros((len(trajs), num_steps), dtype=np.bool_)
    for idx, (traj, mask) in enumerate(zip(trajs, masks)):
        if traj.size == 0 or num_steps == 0:
            continue
        steps = min(num_steps, traj.shape[0])
        stacked_trajs[idx, :steps] = traj[:steps, :2]
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = mask.any(axis=-1)
        mask = mask.astype(np.bool_).reshape(-1)
        if mask.size == 0:
            stacked_masks[idx, :steps] = np.isfinite(traj[:steps, :2]).all(
                axis=-1)
        else:
            valid_steps = min(steps, mask.shape[0])
            stacked_masks[idx, :valid_steps] = mask[:valid_steps]
    return dict(traj=stacked_trajs, traj_mask=stacked_masks)


def gt_arrays_from_info_with_motion(info: dict, dataset_cfg,
                                    class_names: Sequence[str],
                                    label_mapping: Optional[Sequence[int]]
                                    ) -> Dict[str, np.ndarray]:
    use_valid_flag = bool(dataset_cfg.get('use_valid_flag', False))
    boxes = []
    labels = []
    track_ids = []
    trajs = []
    traj_masks = []
    for inst in info.get('instances', []):
        raw_label = inst.get('bbox_label_3d', inst.get('bbox_label', -1))
        label = map_label(raw_label, label_mapping, len(class_names))
        if label < 0:
            continue
        if use_valid_flag:
            keep = bool(inst.get('bbox_3d_isvalid', False))
        else:
            keep = int(inst.get('num_lidar_pts', 0)) > 0
        if not keep:
            continue

        box = np.asarray(inst['bbox_3d'], dtype=np.float32)
        vel = np.asarray(inst.get('velocity', [0.0, 0.0]), dtype=np.float32)
        if box.shape[0] == 7:
            box = np.concatenate([box, vel], axis=0)
        else:
            box = box.copy()
            if box.shape[0] >= 9:
                box[7:9] = vel[:2]
        boxes.append(box[:9])
        labels.append(label)
        track_ids.append(int(inst.get('track_id', -1)))

        traj = np.asarray(
            inst.get('gt_forecasting_locs', inst.get('gt_fut_traj',
                                                     np.zeros((0, 2)))),
            dtype=np.float32)
        mask = np.asarray(
            inst.get('gt_forecasting_mask', inst.get('gt_fut_traj_mask',
                                                     np.zeros((0, )))))
        trajs.append(traj)
        traj_masks.append(mask)

    if not boxes:
        out = dict(
            boxes=np.zeros((0, 9), dtype=np.float32),
            labels=np.zeros((0, ), dtype=np.int64),
            scores=np.ones((0, ), dtype=np.float32),
            track_ids=np.zeros((0, ), dtype=np.int64))
        out.update(dict(
            traj=np.zeros((0, 0, 2), dtype=np.float32),
            traj_mask=np.zeros((0, 0), dtype=np.bool_)))
        return out

    out = dict(
        boxes=np.stack(boxes, axis=0).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        scores=np.ones((len(boxes), ), dtype=np.float32),
        track_ids=np.asarray(track_ids, dtype=np.int64))
    out.update(_stack_gt_trajs(trajs, traj_masks))
    return out


def _future_points(center_xy: np.ndarray, traj_xy: np.ndarray,
                   coordinate: str) -> np.ndarray:
    traj_xy = np.asarray(traj_xy, dtype=np.float32)
    if coordinate == 'offset':
        return center_xy[None, :] + traj_xy
    return traj_xy


def _mode_probabilities(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32)
    scores = np.nan_to_num(scores.astype(np.float32), nan=-1e6)
    shifted = scores - float(np.max(scores))
    exp_scores = np.exp(np.clip(shifted, -50.0, 50.0))
    denom = float(exp_scores.sum())
    if denom <= 0:
        return np.zeros_like(exp_scores, dtype=np.float32)
    return (exp_scores / denom).astype(np.float32)


def draw_gt_future_trajs(ax,
                         boxes: np.ndarray,
                         trajs: np.ndarray,
                         traj_masks: np.ndarray,
                         coordinate: str) -> None:
    if boxes.size == 0 or trajs.size == 0:
        return
    for idx, box in enumerate(boxes):
        if idx >= len(trajs):
            break
        mask = traj_masks[idx] if idx < len(traj_masks) else np.ones(
            (trajs.shape[1], ), dtype=np.bool_)
        if not np.any(mask):
            continue
        center = np.asarray(box[:2], dtype=np.float32)
        future = _future_points(center, trajs[idx, mask, :2], coordinate)
        path = np.concatenate([center[None, :], future], axis=0)
        path_disp = lidar_xy_to_display(path)
        ax.plot(
            path_disp[:, 0],
            path_disp[:, 1],
            color='#9cffbf',
            linewidth=1.1,
            linestyle=':',
            alpha=0.85)
        ax.scatter(
            path_disp[1:, 0],
            path_disp[1:, 1],
            s=5,
            c='#9cffbf',
            alpha=0.7,
            linewidths=0)


def draw_motion_predictions(ax,
                            boxes: np.ndarray,
                            trajs: np.ndarray,
                            traj_scores: np.ndarray,
                            track_ids: np.ndarray,
                            coordinate: str,
                            top_modes: int,
                            min_prob: float) -> None:
    if boxes.size == 0 or trajs.size == 0:
        return
    num = min(len(boxes), len(trajs), len(traj_scores))
    for idx in range(num):
        center = np.asarray(boxes[idx, :2], dtype=np.float32)
        track_id = int(track_ids[idx]) if idx < len(track_ids) else -1
        color = color_from_track_id(track_id)
        probs = _mode_probabilities(traj_scores[idx])
        mode_order = np.argsort(-probs)[:top_modes]
        for rank, mode_idx in enumerate(mode_order):
            prob = float(probs[mode_idx]) if mode_idx < len(probs) else 0.0
            if rank > 0 and prob < min_prob:
                continue
            traj_xy = trajs[idx, mode_idx, :, :2]
            finite = np.isfinite(traj_xy).all(axis=-1)
            if not np.any(finite):
                continue
            future = _future_points(center, traj_xy[finite], coordinate)
            path = np.concatenate([center[None, :], future], axis=0)
            path_disp = lidar_xy_to_display(path)
            linewidth = 1.8 if rank == 0 else 0.9
            alpha = 0.95 if rank == 0 else max(0.18, min(0.55, prob + 0.15))
            linestyle = '-' if rank == 0 else '--'
            ax.plot(
                path_disp[:, 0],
                path_disp[:, 1],
                color=color,
                linewidth=linewidth,
                linestyle=linestyle,
                alpha=alpha)
            ax.scatter(
                path_disp[1:, 0],
                path_disp[1:, 1],
                s=8 if rank == 0 else 4,
                c=[color],
                alpha=alpha,
                linewidths=0)


def render_frame(points: np.ndarray,
                 gt_data: Dict[str, np.ndarray],
                 pred_data: Dict[str, np.ndarray],
                 class_names: Sequence[str],
                 save_path: str,
                 title: str,
                 pc_range: Sequence[float],
                 point_stride: int,
                 vel_scale: float,
                 min_vel_draw: float,
                 annotate: bool,
                 traj_coordinate: str,
                 motion_top_modes: int,
                 motion_min_prob: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 8.8), dpi=160)
    fig.patch.set_facecolor('black')
    left_ax, right_ax = axes

    setup_bev_axis(left_ax, pc_range,
                   f'GT boxes + future ({len(gt_data["boxes"])})')
    setup_bev_axis(
        right_ax, pc_range,
        f'E2E track + motion ({len(pred_data["boxes"])})')
    draw_points_and_ego(left_ax, points, point_stride)
    draw_points_and_ego(right_ax, points, point_stride)
    draw_gt_boxes(
        left_ax,
        gt_data['boxes'],
        gt_data['labels'],
        gt_data['track_ids'],
        class_names,
        vel_scale=vel_scale,
        min_vel_draw=min_vel_draw,
        annotate=annotate)
    draw_gt_future_trajs(left_ax, gt_data['boxes'], gt_data['traj'],
                         gt_data['traj_mask'], traj_coordinate)
    draw_track_boxes(
        right_ax,
        pred_data['boxes'],
        pred_data['labels'],
        pred_data['scores'],
        pred_data['track_ids'],
        class_names,
        vel_scale,
        min_vel_draw,
        annotate)
    draw_motion_predictions(
        right_ax,
        pred_data['boxes'],
        pred_data['traj'],
        pred_data['traj_scores'],
        pred_data['track_ids'],
        coordinate=traj_coordinate,
        top_modes=motion_top_modes,
        min_prob=motion_min_prob)
    fig.suptitle(title, color='white', fontsize=11)
    fig.subplots_adjust(
        left=0.055, right=0.985, bottom=0.07, top=0.91, wspace=0.06)
    fig.savefig(save_path, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_html_player(out_dir: str, frame_files: List[str]) -> None:
    frames = [osp.basename(path) for path in frame_files]
    if not frames:
        return
    frame_items = ',\n      '.join(f'"{name}"' for name in frames)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>KL E2E Motion Sequence</title>
  <style>
    body {{
      margin: 0;
      background: #101114;
      color: #f2f5fa;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    main {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 18px 0 28px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 16px;
      margin-bottom: 12px;
    }}
    h1 {{ margin: 0; font-size: 18px; letter-spacing: 0; }}
    #frameText {{ color: #9aa4b2; font-size: 13px; white-space: nowrap; }}
    .stage {{
      display: grid;
      place-items: center;
      min-height: 360px;
      background: #050608;
      border: 1px solid #343844;
      border-radius: 6px;
      overflow: hidden;
    }}
    #frameImage {{
      display: block;
      max-width: 100%;
      max-height: calc(100vh - 210px);
      width: auto;
      height: auto;
    }}
    .controls {{
      display: grid;
      gap: 10px;
      margin-top: 12px;
      padding: 12px;
      background: #181a20;
      border: 1px solid #343844;
      border-radius: 6px;
    }}
    .row {{ display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }}
    button {{
      height: 34px;
      min-width: 42px;
      padding: 0 12px;
      border: 1px solid #343844;
      border-radius: 5px;
      background: #222631;
      color: #f2f5fa;
      font: inherit;
      cursor: pointer;
    }}
    button.primary {{ background: #113548; border-color: #26637d; }}
    input[type="range"] {{
      flex: 1 1 300px;
      min-width: 160px;
      accent-color: #64d2ff;
    }}
    input[type="number"] {{
      width: 72px;
      height: 32px;
      border: 1px solid #343844;
      border-radius: 5px;
      background: #11141a;
      color: #f2f5fa;
      padding: 0 8px;
      font: inherit;
    }}
    label {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      color: #9aa4b2;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>KL E2E Motion Sequence</h1>
      <div id="frameText"></div>
    </header>
    <section class="stage">
      <img id="frameImage" alt="e2e motion frame">
    </section>
    <section class="controls">
      <div class="row">
        <button id="prevBtn">Prev</button>
        <button id="playBtn" class="primary">Play</button>
        <button id="nextBtn">Next</button>
        <label>FPS <input id="fpsInput" type="number" value="3" min="0.5" max="30" step="0.5"></label>
        <label><input id="loopInput" type="checkbox" checked> Loop</label>
      </div>
      <div class="row">
        <input id="frameSlider" type="range" min="0" max="{len(frames) - 1}" value="0" step="1">
      </div>
    </section>
  </main>
  <script>
    const frames = [
      {frame_items}
    ];
    const image = document.getElementById("frameImage");
    const text = document.getElementById("frameText");
    const slider = document.getElementById("frameSlider");
    const playBtn = document.getElementById("playBtn");
    const fpsInput = document.getElementById("fpsInput");
    const loopInput = document.getElementById("loopInput");
    let index = 0;
    let timer = null;
    function setFrame(nextIndex) {{
      index = Math.max(0, Math.min(frames.length - 1, nextIndex));
      image.src = frames[index];
      slider.value = index;
      text.textContent = `Frame ${{index + 1}} / ${{frames.length}} | ${{frames[index]}}`;
    }}
    function stop() {{
      if (timer !== null) {{
        clearInterval(timer);
        timer = null;
      }}
      playBtn.textContent = "Play";
    }}
    function play() {{
      stop();
      playBtn.textContent = "Pause";
      const fps = Math.max(0.5, Number(fpsInput.value) || 3);
      timer = setInterval(() => {{
        if (index >= frames.length - 1) {{
          if (!loopInput.checked) {{
            stop();
            return;
          }}
          setFrame(0);
          return;
        }}
        setFrame(index + 1);
      }}, 1000 / fps);
    }}
    function togglePlay() {{ timer === null ? play() : stop(); }}
    document.getElementById("prevBtn").addEventListener("click", () => {{ stop(); setFrame(index - 1); }});
    document.getElementById("nextBtn").addEventListener("click", () => {{ stop(); setFrame(index + 1); }});
    playBtn.addEventListener("click", togglePlay);
    fpsInput.addEventListener("change", () => {{ if (timer !== null) play(); }});
    slider.addEventListener("input", () => {{ stop(); setFrame(Number(slider.value)); }});
    document.addEventListener("keydown", (event) => {{
      if (event.key === " ") {{ event.preventDefault(); togglePlay(); }}
      else if (event.key === "ArrowLeft") {{ stop(); setFrame(index - 1); }}
      else if (event.key === "ArrowRight") {{ stop(); setFrame(index + 1); }}
    }});
    setFrame(0);
  </script>
</body>
</html>
"""
    with open(osp.join(out_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(html)


def write_webm_video(out_dir: str, fps: float = 3.0, crf: int = 34) -> None:
    if fps <= 0:
        return
    output_path = osp.join(out_dir, 'e2e_vis.webm')
    input_glob = osp.join(out_dir, '*.png')
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate',
        str(float(fps)),
        '-pattern_type',
        'glob',
        '-i',
        input_glob,
        '-vf',
        'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v',
        'libvpx-vp9',
        '-b:v',
        '0',
        '-crf',
        str(int(crf)),
        output_path,
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        print('[WARN] ffmpeg not found; skipped WebM generation.')
    except subprocess.CalledProcessError as exc:
        print(f'[WARN] ffmpeg failed with exit code {exc.returncode}; '
              'skipped WebM generation.')


def run_online(cfg: Config, args: argparse.Namespace) -> None:
    dataset_cfg = get_dataset_cfg(cfg, args.split)
    dataset_cfg.test_mode = True
    dataset = build_dataset(dataset_cfg)
    class_names = get_class_names(cfg, dataset_cfg)
    label_mapping = get_label_mapping(cfg, dataset_cfg)
    start_index = resolve_start_index(dataset, args)
    indices = consecutive_scene_indices(dataset, start_index, args.max_frames)
    if not indices:
        raise RuntimeError('No frames selected for visualization.')

    model = build_uniad_model(cfg, args.checkpoint, args.device, dataset)
    reset_track_state(model)

    mmcv.mkdir_or_exist(args.out_dir)
    summary = []
    frame_files = []
    for frame_id, idx in enumerate(indices):
        info = raw_info_from_dataset(dataset, idx)
        token = info['token']
        save_path = osp.join(args.out_dir,
                             f'{frame_id:03d}_{idx:06d}_{token}.png')
        if args.skip_existing and osp.exists(save_path):
            print(f'[SKIP] {save_path}')
            frame_files.append(save_path)
            continue

        points = load_points(cfg, dataset_cfg, info)

        result = model_forward_one(model, dataset, idx, args.device)
        pred_data = pred_arrays_from_e2e_result(result, args.score_thr,
                                                args.topk)
        boxes = pred_data['boxes']
        track_ids = pred_data['track_ids']

        gt_data = gt_arrays_from_info_with_motion(info, dataset_cfg,
                                                  class_names, label_mapping)
        scene = str(info.get('scene_token', ''))
        num_motion = int(len(pred_data['traj']))
        title = (
            f'KL UniAD E2E | frame={frame_id} index={idx} '
            f'token={token[:8]} scene={scene[-8:]}\n'
            f'GT={len(gt_data["boxes"])} Track={len(boxes)} '
            f'Motion={num_motion} modes={pred_data["traj"].shape[1] if num_motion else 0}')
        render_frame(
            points=points,
            gt_data=gt_data,
            pred_data=pred_data,
            class_names=class_names,
            save_path=save_path,
            title=title,
            pc_range=cfg.point_cloud_range,
            point_stride=args.point_stride,
            vel_scale=args.vel_scale,
            min_vel_draw=args.min_vel_draw,
            annotate=args.annotate,
            traj_coordinate=args.traj_coordinate,
            motion_top_modes=args.motion_top_modes,
            motion_min_prob=args.motion_min_prob)
        frame_files.append(save_path)
        summary.append(dict(
            frame=int(frame_id),
            index=int(idx),
            token=token,
            scene_token=info.get('scene_token'),
            out_file=save_path,
            num_gt=int(len(gt_data['boxes'])),
            num_track=int(len(boxes)),
            num_motion=int(num_motion),
            track_ids=[int(x) for x in track_ids.tolist()]))
        print(f'[OK] {save_path}')

    mmcv.dump(summary, osp.join(args.out_dir, 'summary.json'))
    write_html_player(args.out_dir, frame_files)
    write_webm_video(args.out_dir, fps=args.webm_fps, crf=args.webm_crf)


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_cfg_modules(cfg, args.config)
    run_online(cfg, args)


if __name__ == '__main__':
    main()
