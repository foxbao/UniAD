#!/usr/bin/env python
"""Visualize one moving-ego B24 example in 3D and diagnostic BEV."""

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.analysis_tools.build_kl_occworld_scene_split import (
    _sequence_mapping,
)
from tools.analysis_tools.audit_kl_occworld_future_reveal import (
    _paint_point_band,
    _point_indices,
)
from tools.analysis_tools.evaluate_kl_occworld import (
    _load_manifest,
    _prediction_mapping,
    _split_references,
)
from tools.analysis_tools.visualize_kl_occworld_3d import (
    COLORS,
    FREE,
    INSTANCE,
    STATIC,
    _crop_rendered_images,
    _font,
    _free_footprint_xyz,
    _surface_mask,
    _voxel_faces,
)
from tools.analysis_tools.visualize_kl_occworld_temporal_predictions import (
    _bev_tile,
)
from tools.data_converter.generate_kl_occworld_history_batch import (
    _voxel_z_centers,
)
from tools.data_converter.generate_kl_occworld_labels import (
    MultiLidarOccLabelBuilder,
    _load_infos,
    _resolve_path,
)


def _yaw_degrees(transform: np.ndarray) -> float:
    return math.degrees(math.atan2(transform[1, 0], transform[0, 0]))


def _angle_difference_degrees(first: float, second: float) -> float:
    difference = math.radians(second - first)
    return math.degrees(math.atan2(math.sin(difference),
                                   math.cos(difference)))


def _ego_path(target_to_reference: np.ndarray) -> np.ndarray:
    return np.asarray(target_to_reference[:, :2, 3], dtype=np.float32)


def _render_state_with_ego(
        state: np.ndarray,
        pc_range: np.ndarray,
        occ_size: np.ndarray,
        output_path: Path,
        ego_path_xy: np.ndarray,
        ego_transforms: np.ndarray,
        horizon: int,
        render_size,
        azimuth: float,
        elevation: float,
        free_surface_z: float) -> dict:
    """Render semantic voxels plus ego path in the reference coordinate."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    dpi = 120
    figure = plt.figure(
        figsize=(render_size[0] / dpi, render_size[1] / dpi), dpi=dpi,
        facecolor=(0.97, 0.975, 0.98))
    axis = figure.add_subplot(111, projection='3d')
    axis.set_facecolor((0.97, 0.975, 0.98))
    counts = {}

    free_points = _free_footprint_xyz(
        state, pc_range, occ_size, free_surface_z)
    counts['free_surface'] = int(len(free_points))
    if len(free_points):
        color = np.asarray(COLORS[FREE], dtype=np.float32) / 255.0
        axis.scatter(
            free_points[:, 0], free_points[:, 1], free_points[:, 2],
            marker='s', s=8.0, c=[color], alpha=0.24,
            linewidths=0, depthshade=False)

    for semantic, name in ((STATIC, 'static'), (INSTANCE, 'instance')):
        mask = state == semantic
        surface = _surface_mask(mask)
        counts[name] = int(surface.sum())
        faces, light = _voxel_faces(mask, pc_range, occ_size)
        counts[f'{name}_faces'] = int(len(faces))
        if not len(faces):
            continue
        base = np.asarray(COLORS[semantic], dtype=np.float32) / 255.0
        facecolors = np.clip(base[None] * light[:, None], 0.0, 1.0)
        collection = Poly3DCollection(
            faces, facecolors=facecolors, edgecolors=facecolors * 0.82,
            linewidths=0.06, alpha=0.98)
        axis.add_collection3d(collection)

    path_z = np.full(len(ego_path_xy), free_surface_z + 0.32,
                     dtype=np.float32)
    if horizon:
        axis.plot(
            ego_path_xy[:horizon + 1, 0], ego_path_xy[:horizon + 1, 1],
            path_z[:horizon + 1], color='#ff9f1c', linewidth=3.2,
            solid_capstyle='round')
    if horizon < len(ego_path_xy) - 1:
        axis.plot(
            ego_path_xy[horizon:, 0], ego_path_xy[horizon:, 1],
            path_z[horizon:], color='#f4c95d', linewidth=2.0,
            linestyle='--', alpha=0.72)
    current_xy = ego_path_xy[horizon]
    axis.scatter(
        [current_xy[0]], [current_xy[1]], [path_z[horizon]],
        marker='D', s=72, c=['#ff7f0e'], edgecolors=['#222222'],
        linewidths=0.8, depthshade=False)
    heading = np.asarray(
        ego_transforms[horizon, :2, 0], dtype=np.float32)
    axis.quiver(
        current_xy[0], current_xy[1], path_z[horizon] + 0.03,
        heading[0], heading[1], 0.0, length=3.6,
        color='#111111', linewidth=2.0, arrow_length_ratio=0.28)
    axis.scatter(
        [ego_path_xy[0, 0]], [ego_path_xy[0, 1]], [path_z[0]],
        marker='s', s=34, c=['#20242a'], depthshade=False)

    axis.set_xlim(float(pc_range[0]), float(pc_range[3]))
    axis.set_ylim(float(pc_range[1]), float(pc_range[4]))
    axis.set_zlim(float(pc_range[2]), float(pc_range[5]))
    axis.set_box_aspect((
        float(pc_range[3] - pc_range[0]),
        float(pc_range[4] - pc_range[1]),
        float((pc_range[5] - pc_range[2]) * 2.5)))
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path, dpi=dpi, facecolor=figure.get_facecolor(),
        bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return counts


def _world_to_tile_xy(path_xy: np.ndarray, pc_range: np.ndarray,
                      occ_size: np.ndarray, tile_size=(300, 225)):
    voxel_size = (
        (pc_range[3:] - pc_range[:3]) / occ_size.astype(np.float32))
    columns = (path_xy[:, 0] - pc_range[0]) / voxel_size[0]
    y_indices = (path_xy[:, 1] - pc_range[1]) / voxel_size[1]
    rows = occ_size[1] - 1 - y_indices
    x = columns / occ_size[0] * tile_size[0]
    y = 30.0 + rows / occ_size[1] * tile_size[1]
    return np.stack([x, y], axis=1).round().astype(np.int32)


def _draw_ego_path(tile: np.ndarray, path_xy: np.ndarray,
                   pc_range: np.ndarray, occ_size: np.ndarray,
                   horizon: int) -> np.ndarray:
    tile = tile.copy()
    pixels = _world_to_tile_xy(path_xy, pc_range, occ_size)
    if horizon < len(pixels) - 1:
        cv2.polylines(
            tile, [pixels[horizon:]], False, (90, 210, 244), 1,
            cv2.LINE_AA)
    if horizon:
        cv2.polylines(
            tile, [pixels[:horizon + 1]], False, (28, 145, 255), 3,
            cv2.LINE_AA)
    cv2.rectangle(
        tile, tuple(pixels[0] - 3), tuple(pixels[0] + 3),
        (32, 36, 42), -1)
    cv2.drawMarker(
        tile, tuple(pixels[horizon]), (0, 110, 255),
        cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
    return tile


def _semantic_tile(state: np.ndarray, title: str,
                   z_centers: np.ndarray, collision_z,
                   path_xy: np.ndarray, pc_range: np.ndarray,
                   occ_size: np.ndarray, horizon: int) -> np.ndarray:
    tile = _bev_tile(state, z_centers, collision_z, title)
    return _draw_ego_path(
        tile, path_xy, pc_range, occ_size, horizon)


def _heatmap_tile(values: np.ndarray, valid: np.ndarray,
                  scope: np.ndarray, title: str,
                  z_centers: np.ndarray, collision_z,
                  path_xy: np.ndarray, pc_range: np.ndarray,
                  occ_size: np.ndarray, horizon: int) -> np.ndarray:
    keep = ((z_centers >= collision_z[0]) &
            (z_centers <= collision_z[1]))
    bev = np.where(valid[keep], values[keep], 0.0).max(axis=0)
    valid_bev = valid[keep].any(axis=0)
    scope_bev = scope[keep].any(axis=0)
    heatmap = cv2.applyColorMap(
        np.clip(bev * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    image = np.full((*valid_bev.shape, 3), (20, 22, 26), dtype=np.uint8)
    image[scope_bev] = (92, 96, 104)
    image[valid_bev] = heatmap[valid_bev]
    image = cv2.resize(image, (300, 225), interpolation=cv2.INTER_NEAREST)
    header = np.full((30, 300, 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (235, 235, 235), 1, cv2.LINE_AA)
    tile = np.concatenate([header, image], axis=0)
    return _draw_ego_path(
        tile, path_xy, pc_range, occ_size, horizon)


def _binary_tile(mask: np.ndarray, scope: np.ndarray, title: str,
                 z_centers: np.ndarray, collision_z,
                 path_xy: np.ndarray, pc_range: np.ndarray,
                 occ_size: np.ndarray, horizon: int,
                 color=(60, 220, 250)) -> np.ndarray:
    keep = ((z_centers >= collision_z[0]) &
            (z_centers <= collision_z[1]))
    mask_bev = mask[keep].any(axis=0)
    scope_bev = scope[keep].any(axis=0)
    image = np.full((*mask_bev.shape, 3), (20, 22, 26), dtype=np.uint8)
    image[scope_bev] = (92, 96, 104)
    image[mask_bev] = color
    image = cv2.resize(image, (300, 225), interpolation=cv2.INTER_NEAREST)
    header = np.full((30, 300, 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (235, 235, 235), 1, cv2.LINE_AA)
    tile = np.concatenate([header, image], axis=0)
    return _draw_ego_path(
        tile, path_xy, pc_range, occ_size, horizon)


def _pointcloud_tile(points: np.ndarray, title: str,
                     collision_z, path_xy: np.ndarray,
                     pc_range: np.ndarray, occ_size: np.ndarray,
                     horizon: int) -> np.ndarray:
    scale = 4
    image_shape = (int(occ_size[1]) * scale,
                   int(occ_size[0]) * scale)
    image = np.full((*image_shape, 3), (20, 22, 26), dtype=np.uint8)
    rows, columns, heights = _point_indices(
        points, pc_range, image_shape)
    low = heights < collision_z[0]
    collision = ((heights >= collision_z[0]) &
                 (heights <= collision_z[1]))
    high = heights > collision_z[1]
    _paint_point_band(image, rows[low], columns[low], (180, 145, 100))
    _paint_point_band(image, rows[high], columns[high], (205, 105, 205))
    _paint_point_band(
        image, rows[collision], columns[collision], (70, 220, 250))
    image = cv2.resize(image, (300, 225), interpolation=cv2.INTER_AREA)
    header = np.full((30, 300, 3), 28, dtype=np.uint8)
    cv2.putText(header, title, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (235, 235, 235), 1, cv2.LINE_AA)
    tile = np.concatenate([header, image], axis=0)
    return _draw_ego_path(
        tile, path_xy, pc_range, occ_size, horizon)


def _diagnostic_tiles(sample: dict, horizon: int) -> dict:
    time_s = float(sample['target_times'][horizon])
    suffix = f't={time_s:.1f}s'
    common = (
        sample['z_centers'], sample['collision_z'], sample['ego_path_xy'],
        sample['pc_range'], sample['occ_size'], horizon)
    return {
        'target': _semantic_tile(
            sample['target'][horizon], f'GT target | {suffix}', *common),
        'raw': _semantic_tile(
            sample['raw'][horizon], f'B17A raw | {suffix}', *common),
        'final': _semantic_tile(
            sample['final'][horizon], f'B24 final | {suffix}', *common),
        'pointcloud': _pointcloud_tile(
            sample['pointclouds'][horizon],
            f'aligned 8-LiDAR | {suffix}', sample['collision_z'],
            sample['ego_path_xy'], sample['pc_range'], sample['occ_size'],
            horizon),
        'event': _binary_tile(
            sample['event_mask'][horizon], sample['future_scope'][horizon],
            f'event candidate | {suffix}', *common, color=(0, 215, 255)),
        'alpha': _heatmap_tile(
            sample['alpha'][horizon], sample['event_mask'][horizon],
            sample['future_scope'][horizon], f'event alpha | {suffix}',
            *common),
        'fixed': _binary_tile(
            sample['improved'][horizon], sample['future_scope'][horizon],
            f'fixed raw error | {suffix}', *common, color=(80, 220, 80)),
        'harmed': _binary_tile(
            sample['harmed'][horizon], sample['future_scope'][horizon],
            f'harmed raw | {suffix}', *common, color=(70, 70, 245)),
    }


def _contact_sheet(sample: dict, output_path: Path) -> None:
    keys = (
        'target', 'raw', 'final', 'pointcloud', 'event', 'alpha',
        'fixed', 'harmed')
    tiles = [
        _diagnostic_tiles(sample, horizon)
        for horizon in range(len(sample['target_times']))
    ]
    rows = []
    for key in keys:
        rows.append(np.concatenate([
            horizon_tiles[key] for horizon_tiles in tiles
        ], axis=1))
    separator = np.full((6, rows[0].shape[1], 3), 28, dtype=np.uint8)
    sheet = rows[0]
    for row in rows[1:]:
        sheet = np.concatenate([sheet, separator, row], axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet):
        raise OSError(f'Failed to write {output_path}')


def _animation_frames(sample: dict, rendered: dict,
                      output_dir: Path) -> list:
    output_dir.mkdir(parents=True, exist_ok=True)
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    title_font = _font(25)
    body_font = _font(17)
    paths = []
    methods = (
        ('target', 'GT target'), ('raw', 'B17A raw'),
        ('final', 'B24 final'))
    for horizon, time_s in enumerate(sample['target_times']):
        canvas = Image.new('RGB', (1600, 850), (248, 249, 251))
        draw = ImageDraw.Draw(canvas)
        displacement = float(np.linalg.norm(
            sample['ego_path_xy'][horizon] - sample['ego_path_xy'][0]))
        draw.text(
            (22, 13),
            f'B24 moving-ego diagnostic | ref {sample["reference"]:06d} '
            f'| t={float(time_s):.1f}s | ego displacement={displacement:.2f}m',
            fill=(28, 32, 39), font=title_font)
        draw.text(
            (22, 48),
            'orange diamond: current ego | orange line: travelled | '
            'yellow dashed: future path',
            fill=(70, 75, 84), font=body_font)
        for column, (key, label) in enumerate(methods):
            x = column * 533
            draw.text((x + 18, 76), label, fill=(42, 47, 55),
                      font=body_font)
            image = Image.open(rendered[key][horizon]).convert('RGB')
            image = image.resize((533, 360), resampling)
            canvas.paste(image, (x, 102))

        diagnostics = _diagnostic_tiles(sample, horizon)
        diagnostic_keys = (
            'pointcloud', 'event', 'alpha', 'fixed', 'harmed')
        for column, key in enumerate(diagnostic_keys):
            tile = cv2.cvtColor(diagnostics[key], cv2.COLOR_BGR2RGB)
            image = Image.fromarray(tile).resize((320, 272), resampling)
            canvas.paste(image, (column * 320, 490))
        draw.text(
            (22, 815),
            'semantic: green free, red static, blue instance | '
            'LiDAR: tan ground, cyan collision-height, purple high | '
            'alpha: blue low, red high',
            fill=(70, 75, 84), font=body_font)
        path = output_dir / f'horizon_{horizon}.png'
        canvas.save(path)
        paths.append(path)
    return paths


def _write_video(frame_paths: list, output_dir: Path, stem: str,
                 fps: int, seconds_per_horizon: float,
                 final_hold_seconds: float):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required for browser outputs')
    concat_path = output_dir / 'frames.ffconcat'
    lines = ['ffconcat version 1.0']
    for path in frame_paths:
        lines.append(f"file '{path.resolve()}'")
        lines.append(f'duration {seconds_per_horizon:.6f}')
    lines.append(f"file '{frame_paths[-1].resolve()}'")
    lines.append(f'duration {final_hold_seconds:.6f}')
    lines.append(f"file '{frame_paths[-1].resolve()}'")
    concat_path.write_text('\n'.join(lines) + '\n')
    webm_path = output_dir / f'{stem}.webm'
    mp4_path = output_dir / f'{stem}_h264.mp4'
    common = [
        ffmpeg, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
        '-i', str(concat_path), '-r', str(fps), '-an']
    subprocess.run(common + [
        '-c:v', 'libvpx-vp9', '-crf', '31', '-b:v', '0', '-row-mt', '1',
        '-pix_fmt', 'yuv420p', str(webm_path)], check=True)
    subprocess.run(common + [
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4_path)],
        check=True)
    html_path = output_dir / f'{stem}_player.html'
    html_path.write_text(f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>B24 moving-ego visualization</title>
  <style>
    html, body {{ width: 100%; height: 100%; margin: 0; background: #111318; }}
    body {{ display: grid; place-items: center; overflow: hidden; }}
    video {{ width: 100%; height: 100%; object-fit: contain; background: #111318; }}
  </style>
</head>
<body>
  <video controls autoplay muted loop playsinline preload="metadata">
    <source src="{webm_path.name}" type="video/webm; codecs=vp9">
    <source src="{mp4_path.name}" type="video/mp4">
  </video>
</body>
</html>
''')
    return webm_path, mp4_path, html_path


def _load_sample(label_path: Path, prediction_path: Path,
                 reference: int) -> dict:
    with np.load(label_path, allow_pickle=False) as label:
        target_state = np.asarray(
            label['world_target_state_3d'], dtype=np.uint8)
        target_valid = np.asarray(
            label['world_target_valid_3d'], dtype=np.bool_)
        target_times = np.asarray(
            label['target_times_s'], dtype=np.float32)
        target_to_reference = np.asarray(
            label['target_to_reference'], dtype=np.float64)
        target_indices = np.asarray(
            label['target_indices'], dtype=np.int64)
        pc_range = np.asarray(label['pc_range'], dtype=np.float32)
        occ_size = np.asarray(label['occ_size'], dtype=np.int64)
        collision_z = tuple(
            np.asarray(label['collision_z'], dtype=np.float32))
    with np.load(prediction_path, allow_pickle=False) as prediction:
        exported_reference = int(prediction['reference_index'])
        checkpoint_epoch = int(prediction['checkpoint_epoch'])
        raw_class = np.asarray(
            prediction['raw_world_pred_class_3d'], dtype=np.uint8)
        final_class = np.asarray(
            prediction['world_pred_class_3d'], dtype=np.uint8)
        alpha_future = np.asarray(
            prediction['event_reliability_alpha_3d'], dtype=np.float32)
        event_mask_future = np.asarray(
            prediction['event_candidate_mask_3d'], dtype=np.bool_)
    if exported_reference != reference or checkpoint_epoch != 3:
        raise ValueError('Prediction identity or checkpoint epoch mismatch')

    known = target_valid & (target_state != 0)
    target_class = np.zeros_like(target_state, dtype=np.uint8)
    target_class[known] = target_state[known] - 1
    raw_correct = known & (raw_class == target_class)
    final_correct = known & (final_class == target_class)
    changed = raw_class != final_class
    improved = changed & ~raw_correct & final_correct
    harmed = changed & raw_correct & ~final_correct
    exported_changed_voxels = int(np.count_nonzero(
        raw_class[1:] != final_class[1:]))
    known_changed_voxels = int(np.count_nonzero(changed[1:] & known[1:]))

    target = target_state.copy()
    target[~known] = 0
    raw = raw_class + 1
    final = final_class + 1
    raw[~known] = 0
    final[~known] = 0
    alpha = np.zeros_like(target_state, dtype=np.float32)
    event_mask = np.zeros_like(known, dtype=np.bool_)
    alpha[1:] = alpha_future
    event_mask[1:] = event_mask_future
    future_scope = known.copy()
    future_scope[0] = False
    ego_path_xy = _ego_path(target_to_reference)
    return {
        'reference': reference,
        'checkpoint_epoch': checkpoint_epoch,
        'target': target,
        'raw': raw,
        'final': final,
        'known': known,
        'alpha': alpha,
        'event_mask': event_mask,
        'future_scope': future_scope,
        'improved': improved,
        'harmed': harmed,
        'exported_changed_voxels': exported_changed_voxels,
        'known_changed_voxels': known_changed_voxels,
        'target_times': target_times,
        'target_to_reference': target_to_reference,
        'target_indices': target_indices,
        'ego_path_xy': ego_path_xy,
        'pc_range': pc_range,
        'occ_size': occ_size,
        'collision_z': collision_z,
        'z_centers': _voxel_z_centers(pc_range, occ_size),
    }


def _load_aligned_pointclouds(ann_file: Path, sample: dict) -> list:
    infos, metainfo = _load_infos(_resolve_path(ann_file))
    target_indices = sample['target_indices']
    if np.any(target_indices < 0) or np.any(target_indices >= len(infos)):
        raise IndexError('A target frame index is outside the annotation file')
    target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    builder = MultiLidarOccLabelBuilder(
        sample['pc_range'],
        (int(sample['occ_size'][1]), int(sample['occ_size'][0])),
        sample['occ_size'], target_frame=target_frame,
        collision_z=sample['collision_z'])
    aligned_pointclouds = []
    for horizon, target_index in enumerate(target_indices):
        result = builder.build(
            infos[int(target_index)], diagnostics=False,
            return_points=True)
        points = np.asarray(result['points'], dtype=np.float32)
        homogeneous = np.concatenate([
            points[:, :3], np.ones((len(points), 1), dtype=np.float32)
        ], axis=1)
        aligned = points.copy()
        aligned[:, :3] = (
            homogeneous @ sample['target_to_reference'][horizon].T
        )[:, :3]
        aligned_pointclouds.append(aligned)
    return aligned_pointclouds


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b24_exploratory_internal_split_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/occworld_sequence_full_train_v1'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b24_internal_epochs1_3_fixed_v1/'
            'internal_dev/epoch_003'))
    parser.add_argument('--split', default='internal_dev')
    parser.add_argument('--reference', type=int, default=25267)
    parser.add_argument(
        '--ann-file', type=Path,
        default=Path('data/kl_8/kl_infos_train.pkl'))
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b24_moving_ego_visuals_v1/025267'))
    parser.add_argument('--render-size', type=int, nargs=2,
                        default=(960, 720))
    parser.add_argument('--azimuth', type=float, default=-128.0)
    parser.add_argument('--elevation', type=float, default=34.0)
    parser.add_argument('--free-surface-z', type=float, default=0.0)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--seconds-per-horizon', type=float, default=1.0)
    parser.add_argument('--final-hold-seconds', type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = _load_manifest(args.manifest)
    allowed = set(_split_references(manifest, args.split))
    if args.reference not in allowed:
        raise ValueError('Reference is outside the requested split')
    scene_rows = {
        int(row['reference_index']): row
        for row in manifest['splits'][args.split]
    }
    labels = _sequence_mapping(args.sequence_root)
    predictions = _prediction_mapping(args.prediction_root)
    if args.reference not in labels or args.reference not in predictions:
        raise FileNotFoundError('Reference label or prediction is missing')
    sample = _load_sample(
        labels[args.reference], predictions[args.reference], args.reference)
    sample['pointclouds'] = _load_aligned_pointclouds(
        args.ann_file, sample)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    contact_path = args.out_dir / 'b24_moving_ego_bev_diagnostics.png'
    _contact_sheet(sample, contact_path)

    render_root = args.out_dir / 'renders'
    rendered = {'target': [], 'raw': [], 'final': []}
    counts = {'target': [], 'raw': [], 'final': []}
    for key in rendered:
        for horizon, state in enumerate(sample[key]):
            path = render_root / key / f'horizon_{horizon}.png'
            count = _render_state_with_ego(
                state, sample['pc_range'], sample['occ_size'], path,
                sample['ego_path_xy'], sample['target_to_reference'],
                horizon, args.render_size, args.azimuth, args.elevation,
                args.free_surface_z)
            rendered[key].append(path)
            counts[key].append(count)
    crop_box = _crop_rendered_images(rendered)

    frame_paths = _animation_frames(
        sample, rendered, args.out_dir / 'animation_frames')
    stem = f'b24_moving_ego_{args.reference:06d}'
    webm_path, mp4_path, html_path = _write_video(
        frame_paths, args.out_dir, stem, args.fps,
        args.seconds_per_horizon, args.final_hold_seconds)

    path_xy = sample['ego_path_xy']
    displacement = float(np.linalg.norm(path_xy[-1] - path_xy[0]))
    start_yaw = _yaw_degrees(sample['target_to_reference'][0])
    end_yaw = _yaw_degrees(sample['target_to_reference'][-1])
    improved = int(np.count_nonzero(sample['improved'][1:]))
    harmed = int(np.count_nonzero(sample['harmed'][1:]))
    summary = {
        'schema_version': 1,
        'reference_index': args.reference,
        'scene_token': scene_rows[args.reference]['scene_token'],
        'timestamp': float(scene_rows[args.reference]['timestamp']),
        'split': args.split,
        'checkpoint_epoch': sample['checkpoint_epoch'],
        'selection_rule': (
            'moving ego with rich B24 event changes; selected before '
            'rendering from causal pose metadata and prediction diagnostics'),
        'selection_uses_gt_accuracy': False,
        'target_times_s': sample['target_times'].tolist(),
        'target_frame_indices': sample['target_indices'].tolist(),
        'aligned_point_counts': [
            int(len(points)) for points in sample['pointclouds']],
        'ego_path_xy_in_reference_m': path_xy.tolist(),
        'ego_displacement_t0_to_t4_m': displacement,
        'ego_yaw_change_t0_to_t4_deg': _angle_difference_degrees(
            start_yaw, end_yaw),
        'event_candidate_voxels': int(np.count_nonzero(
            sample['event_mask'][1:])),
        'changed_voxels_all_exported': sample['exported_changed_voxels'],
        'changed_voxels_in_evaluation_known_scope': (
            sample['known_changed_voxels']),
        'improved_voxels': improved,
        'harmed_voxels': harmed,
        'net_correct_voxels': improved - harmed,
        'scope': 'evaluation-known voxels for GT/raw/final comparison',
        'coordinate_note': (
            'All occupancy horizons, LiDAR points, and the ego path are '
            'expressed in the reference-frame coordinate; ego motion is '
            'not image jitter.'),
        'rendered_voxel_counts': counts,
        'shared_render_crop_box_xyxy': crop_box,
        'new_model_inference_performed': False,
        'bev_contact_sheet': str(contact_path),
        'webm': str(webm_path),
        'h264_mp4': str(mp4_path),
        'html_player': str(html_path),
    }
    summary_path = args.out_dir / 'summary.json'
    with summary_path.open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
