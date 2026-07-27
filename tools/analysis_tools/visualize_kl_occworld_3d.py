#!/usr/bin/env python
"""Render sealed KL OccWorld predictions as OccWorld-style 3D voxels."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


UNKNOWN = 0
FREE = 1
STATIC = 2
INSTANCE = 3
COLORS = {
    FREE: (93, 190, 120),
    STATIC: (225, 76, 68),
    INSTANCE: (52, 108, 224),
}


def _artifact_mapping(root: Path, suffix: str) -> dict:
    mapping = {}
    for path in sorted(root.glob(f'*/*__{suffix}.npz')):
        with np.load(path, allow_pickle=False) as payload:
            reference = int(payload['reference_index'])
        if reference in mapping:
            raise ValueError(f'Duplicate {suffix} for reference {reference}')
        mapping[reference] = path
    return mapping


def _surface_mask(mask: np.ndarray) -> np.ndarray:
    """Keep occupied boundary voxels and remove fully enclosed interiors."""
    if mask.ndim != 3:
        raise ValueError('Surface extraction expects [Z,H,W]')
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    interior = np.ones(mask.shape, dtype=bool)
    for slices in (
            (slice(0, -2), slice(1, -1), slice(1, -1)),
            (slice(2, None), slice(1, -1), slice(1, -1)),
            (slice(1, -1), slice(0, -2), slice(1, -1)),
            (slice(1, -1), slice(2, None), slice(1, -1)),
            (slice(1, -1), slice(1, -1), slice(0, -2)),
            (slice(1, -1), slice(1, -1), slice(2, None))):
        interior &= padded[slices]
    return mask.astype(bool) & ~interior


def _zhw_mask_to_xyz(mask: np.ndarray, pc_range: np.ndarray,
                     occ_size: np.ndarray) -> np.ndarray:
    """Convert image-aligned [Z,H,W] voxels to physical XYZ centers."""
    indices = np.argwhere(mask)
    if not len(indices):
        return np.empty((0, 3), dtype=np.float32)
    z_index = indices[:, 0]
    y_index = mask.shape[1] - 1 - indices[:, 1]
    x_index = indices[:, 2]
    xyz_index = np.stack([x_index, y_index, z_index], axis=1)
    voxel_size = (
        (pc_range[3:] - pc_range[:3]) / occ_size.astype(np.float32))
    return (pc_range[:3] +
            (xyz_index.astype(np.float32) + 0.5) * voxel_size)


def _free_footprint_xyz(state: np.ndarray, pc_range: np.ndarray,
                        occ_size: np.ndarray,
                        surface_z: float = 0.0) -> np.ndarray:
    free = np.any(state == FREE, axis=0)
    occupied = np.any((state == STATIC) | (state == INSTANCE), axis=0)
    rows, columns = np.nonzero(free & ~occupied)
    if not len(rows):
        return np.empty((0, 3), dtype=np.float32)
    voxel_size = (
        (pc_range[3:] - pc_range[:3]) / occ_size.astype(np.float32))
    x = pc_range[0] + (columns.astype(np.float32) + 0.5) * voxel_size[0]
    y_index = state.shape[1] - 1 - rows
    y = pc_range[1] + (y_index.astype(np.float32) + 0.5) * voxel_size[1]
    z = np.full_like(x, float(surface_z))
    return np.stack([x, y, z], axis=1)


def _load_sample(label_path: Path, prediction_path: Path,
                 visibility_threshold: float,
                 evaluation_scope: bool) -> dict:
    with np.load(label_path, allow_pickle=False) as label:
        target = np.asarray(label['world_target_state_3d'], dtype=np.uint8)
        target_valid = np.asarray(
            label['world_target_valid_3d'], dtype=bool)
        current = np.asarray(
            label['current_observation_state_3d'], dtype=np.uint8)
        current_valid = np.asarray(
            label['current_observation_valid_3d'], dtype=bool)
        target_times = np.asarray(label['target_times_s'], dtype=np.float32)
        pc_range = np.asarray(label['pc_range'], dtype=np.float32)
        occ_size = np.asarray(label['occ_size'], dtype=np.int64)
    with np.load(prediction_path, allow_pickle=False) as prediction:
        model = np.asarray(
            prediction['world_pred_class_3d'], dtype=np.uint8) + 1
        model_visible = np.asarray(
            prediction['world_valid_probability_3d'], dtype=np.float32)
    target_known = target_valid & (target != UNKNOWN)
    target = target.copy()
    target[~target_known] = UNKNOWN
    persistence = np.broadcast_to(current, target.shape).copy()
    if evaluation_scope:
        persistence[~target_known] = UNKNOWN
        model[~target_known] = UNKNOWN
        scope_name = 'evaluation-known voxels'
    else:
        persistence_scope = np.broadcast_to(
            current_valid & (current != UNKNOWN), target.shape)
        persistence[~persistence_scope] = UNKNOWN
        model[model_visible < visibility_threshold] = UNKNOWN
        scope_name = 'online visible voxels'
    return {
        'target': target,
        'persistence': persistence,
        'model': model,
        'target_times': target_times,
        'pc_range': pc_range,
        'occ_size': occ_size,
        'scope_name': scope_name,
    }


def _voxel_faces(mask: np.ndarray, pc_range: np.ndarray,
                 occ_size: np.ndarray):
    """Return exposed physical-space cube faces and their light factors."""
    voxel_size = (
        (pc_range[3:] - pc_range[:3]) / occ_size.astype(np.float32))
    faces = []
    light = []
    directions = (
        (-1, 0, 0, 0.64), (1, 0, 0, 1.00),
        (0, -1, 0, 0.76), (0, 1, 0, 0.90),
        (0, 0, -1, 0.70), (0, 0, 1, 0.84))
    depth, height, width = mask.shape
    for z_index, row, column in np.argwhere(mask):
        x0 = pc_range[0] + column * voxel_size[0]
        x1 = x0 + voxel_size[0]
        y_index = height - 1 - row
        y0 = pc_range[1] + y_index * voxel_size[1]
        y1 = y0 + voxel_size[1]
        z0 = pc_range[2] + z_index * voxel_size[2]
        z1 = z0 + voxel_size[2]
        candidates = (
            ((x0, y0, z0), (x1, y0, z0),
             (x1, y1, z0), (x0, y1, z0)),
            ((x0, y0, z1), (x0, y1, z1),
             (x1, y1, z1), (x1, y0, z1)),
            ((x0, y0, z0), (x0, y0, z1),
             (x1, y0, z1), (x1, y0, z0)),
            ((x0, y1, z0), (x1, y1, z0),
             (x1, y1, z1), (x0, y1, z1)),
            ((x0, y0, z0), (x0, y1, z0),
             (x0, y1, z1), (x0, y0, z1)),
            ((x1, y0, z0), (x1, y0, z1),
             (x1, y1, z1), (x1, y1, z0)),
        )
        for candidate, (delta_z, delta_h, delta_w, factor) in zip(
                candidates, directions):
            neighbor = (
                z_index + delta_z, row + delta_h, column + delta_w)
            inside = (
                0 <= neighbor[0] < depth and
                0 <= neighbor[1] < height and
                0 <= neighbor[2] < width)
            if not inside or not mask[neighbor]:
                faces.append(candidate)
                light.append(factor)
    return np.asarray(faces, dtype=np.float32), np.asarray(light)


def _render_state(state: np.ndarray, pc_range: np.ndarray,
                  occ_size: np.ndarray, output_path: Path,
                  render_size, azimuth: float, elevation: float,
                  free_surface_z: float) -> dict:
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

    axis.scatter(
        [0.0], [0.0], [0.4], marker='s', s=24,
        c=[(0.12, 0.14, 0.17)], depthshade=False)
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


def _font(size: int):
    candidates = (
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf')
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _crop_rendered_images(rendered: dict, padding: int = 18):
    """Crop every render to one shared content box to avoid video jitter."""
    paths = [path for method_paths in rendered.values()
             for path in method_paths]
    boxes = []
    for path in paths:
        image = np.asarray(Image.open(path).convert('RGB'), dtype=np.int16)
        background = image[0, 0]
        content = np.max(np.abs(image - background), axis=2) > 12
        rows, columns = np.nonzero(content)
        if len(rows):
            boxes.append((
                int(columns.min()), int(rows.min()),
                int(columns.max()) + 1, int(rows.max()) + 1))
    if not boxes:
        raise ValueError('3D renders contain no visible content')
    sample = Image.open(paths[0])
    width, height = sample.size
    box = (
        max(0, min(value[0] for value in boxes) - padding),
        max(0, min(value[1] for value in boxes) - padding),
        min(width, max(value[2] for value in boxes) + padding),
        min(height, max(value[3] for value in boxes) + padding),
    )
    for path in paths:
        image = Image.open(path).convert('RGB')
        image.crop(box).save(path)
    return list(box)


def _contact_sheet(rendered: dict, target_times: np.ndarray,
                   output_path: Path, reference: int,
                   scope_name: str, tile_size=(480, 360)):
    methods = (
        ('target', 'GT target'),
        ('persistence', 'Persistence'),
        ('model', 'B17A epoch 3'))
    header = 122
    row_label = 180
    width = row_label + tile_size[0] * len(target_times)
    height = header + tile_size[1] * len(methods)
    canvas = Image.new('RGB', (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(26)
    label_font = _font(18)
    small_font = _font(16)
    draw.text(
        (24, 16), f'KL OccWorld 3D | reference {reference} | {scope_name}',
        fill=(28, 32, 39), font=title_font)
    legend_x = 24
    for semantic, name in ((FREE, 'free surface'),
                           (STATIC, 'static'), (INSTANCE, 'instance')):
        draw.rectangle(
            (legend_x, 56, legend_x + 18, 74), fill=COLORS[semantic])
        draw.text(
            (legend_x + 25, 54), name, fill=(58, 64, 74),
            font=small_font)
        legend_x += 145
    for horizon, time_value in enumerate(target_times):
        x = row_label + horizon * tile_size[0]
        draw.text(
            (x + 18, 91), f't={float(time_value):.1f}s',
            fill=(58, 64, 74), font=small_font)
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    for row, (key, label) in enumerate(methods):
        y = header + row * tile_size[1]
        draw.text(
            (16, y + tile_size[1] // 2 - 12), label,
            fill=(28, 32, 39), font=label_font)
        for horizon in range(len(target_times)):
            image = Image.open(rendered[key][horizon]).convert('RGB')
            image = image.resize(tile_size, resampling)
            canvas.paste(image, (row_label + horizon * tile_size[0], y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _timeline_frames(rendered: dict, target_times: np.ndarray,
                     output_dir: Path, reference: int,
                     scope_name: str, tile_size=(480, 360)) -> list:
    methods = (
        ('target', 'GT target'),
        ('persistence', 'Persistence'),
        ('model', 'B17A epoch 3'))
    resampling = getattr(Image, 'Resampling', Image).LANCZOS
    title_font = _font(24)
    label_font = _font(18)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for horizon, time_value in enumerate(target_times):
        width = tile_size[0] * len(methods)
        height = tile_size[1] + 82
        frame = Image.new('RGB', (width, height), (248, 249, 251))
        draw = ImageDraw.Draw(frame)
        draw.text(
            (22, 14),
            f'KL OccWorld 3D | reference {reference} | '
            f't={float(time_value):.1f}s | {scope_name}',
            fill=(28, 32, 39), font=title_font)
        for column, (key, label) in enumerate(methods):
            x = column * tile_size[0]
            draw.text((x + 18, 52), label, fill=(58, 64, 74),
                      font=label_font)
            image = Image.open(rendered[key][horizon]).convert('RGB')
            image = image.resize(tile_size, resampling)
            frame.paste(image, (x, 82))
        path = output_dir / f'horizon_{horizon}.png'
        frame.save(path)
        paths.append(path)
    return paths


def _write_video(frame_paths: list, output_dir: Path, fps: int,
                 seconds_per_horizon: float, final_hold_seconds: float):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        raise RuntimeError('ffmpeg is required for 3D animation output')
    concat_path = output_dir / 'frames.ffconcat'
    lines = ['ffconcat version 1.0']
    for path in frame_paths:
        lines.append(f"file '{path.resolve()}'")
        lines.append(f'duration {seconds_per_horizon:.6f}')
    lines.append(f"file '{frame_paths[-1].resolve()}'")
    lines.append(f'duration {final_hold_seconds:.6f}')
    lines.append(f"file '{frame_paths[-1].resolve()}'")
    concat_path.write_text('\n'.join(lines) + '\n')
    webm_path = output_dir / 'b17a_fresh_3d_timeline.webm'
    mp4_path = output_dir / 'b17a_fresh_3d_timeline_h264.mp4'
    common = [
        ffmpeg, '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
        '-i', str(concat_path), '-r', str(fps), '-an']
    subprocess.run(common + [
        '-c:v', 'libvpx-vp9', '-crf', '31', '-b:v', '0',
        '-pix_fmt', 'yuv420p', str(webm_path)], check=True)
    subprocess.run(common + [
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '22',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4_path)],
        check=True)
    html_path = output_dir / 'b17a_fresh_3d_player.html'
    html_path.write_text(f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>B17A OccWorld 3D</title>
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path,
        default=Path(
            'documents/patent_2026_occ/'
            'kl_occworld_b17a_fresh_holdout_val65_evaluation_v1.json'))
    parser.add_argument(
        '--sequence-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_sequence_b17_fresh_holdout_val65_v1'))
    parser.add_argument(
        '--prediction-root', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_predictions_b17a_fresh_holdout_val65_v1/'
            'fresh_holdout/epoch_003'))
    parser.add_argument('--split', default='fresh_holdout')
    parser.add_argument(
        '--required-status',
        default='fresh_holdout_evaluated_once_no_retuning_allowed')
    parser.add_argument('--reference', type=int, default=3420)
    parser.add_argument('--visibility-threshold', type=float, default=0.7)
    parser.add_argument(
        '--scope', choices=('evaluation', 'online'), default='evaluation')
    parser.add_argument(
        '--out-dir', type=Path,
        default=Path(
            'outputs/patent_2026_occ/'
            'occworld_b17a_fresh_holdout_val65_visuals_v1/'
            '3d_occworld_style/003420'))
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
    manifest = json.load(args.manifest.open())
    if manifest.get('status') != args.required_status:
        raise ValueError('The requested holdout is not sealed')
    protocol_threshold = float(
        manifest['frozen_model_protocol']['visibility_threshold'])
    if args.visibility_threshold != protocol_threshold:
        raise ValueError('Visualization threshold differs from frozen value')
    allowed = {
        int(row['reference_index'])
        for row in manifest['splits'][args.split]
    }
    if args.reference not in allowed:
        raise ValueError('Reference is outside the sealed split')
    labels = _artifact_mapping(args.sequence_root, 'occworld_sequence')
    predictions = _artifact_mapping(
        args.prediction_root, 'occworld_prediction')
    if args.reference not in labels or args.reference not in predictions:
        raise FileNotFoundError('Reference label or prediction is missing')
    sample = _load_sample(
        labels[args.reference], predictions[args.reference],
        args.visibility_threshold, args.scope == 'evaluation')

    render_root = args.out_dir / 'renders'
    rendered = {'target': [], 'persistence': [], 'model': []}
    counts = {'target': [], 'persistence': [], 'model': []}
    for key in rendered:
        for horizon, state in enumerate(sample[key]):
            path = render_root / key / f'horizon_{horizon}.png'
            count = _render_state(
                state, sample['pc_range'], sample['occ_size'], path,
                args.render_size, args.azimuth, args.elevation,
                args.free_surface_z)
            rendered[key].append(path)
            counts[key].append(count)
    crop_box = _crop_rendered_images(rendered)

    contact_path = args.out_dir / 'b17a_fresh_3d_contact_sheet.png'
    _contact_sheet(
        rendered, sample['target_times'], contact_path,
        args.reference, sample['scope_name'])
    frame_paths = _timeline_frames(
        rendered, sample['target_times'], args.out_dir / 'animation_frames',
        args.reference, sample['scope_name'])
    webm_path, mp4_path, html_path = _write_video(
        frame_paths, args.out_dir, args.fps,
        args.seconds_per_horizon, args.final_hold_seconds)
    summary = {
        'schema_version': 1,
        'reference_index': args.reference,
        'split': args.split,
        'source_status': manifest['status'],
        'source_inspiration': (
            '/home/baojiali/Downloads/public_code/OccWorld/'
            'visualize_demo.py'),
        'rendering_backend': (
            'Matplotlib exposed-cube faces; Mayavi visual language '
            'without an X-server dependency'),
        'scope': args.scope,
        'scope_name': sample['scope_name'],
        'visibility_threshold': args.visibility_threshold,
        'target_times_s': sample['target_times'].tolist(),
        'camera': {
            'azimuth': args.azimuth,
            'elevation': args.elevation,
            'projection': 'matplotlib perspective, z display x2.5',
        },
        'rendered_voxel_counts': counts,
        'shared_render_crop_box_xyxy': crop_box,
        'new_model_inference_performed': False,
        'metric_based_sample_selection_performed': False,
        'contact_sheet': str(contact_path),
        'webm': str(webm_path),
        'h264_mp4': str(mp4_path),
        'html_player': str(html_path),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / 'summary.json').open('w') as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write('\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
