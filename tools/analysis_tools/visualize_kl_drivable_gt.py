# Copyright (c) OpenMMLab. All rights reserved.
"""Visualize the fused KL drivable seg-head ground-truth.

Renders the target that ``GenerateKLDrivableMapLabels`` feeds to the
Pansegformer seg-head, for a bounded set of frames. Each frame is a 2x2
grid; every panel has its own title bar and the colours are explained by
two legend strips along the bottom. Panels read input -> output:

* **D: raw LiDAR (colour = height)** -- the raw point cloud coloured by z
  via a JET colormap (top-left); the model's actual input.
* **A: final drivable GT + LiDAR** -- final drivable GT (green) over a
  LiDAR point-density backdrop (gray), with the ego centre marked (red).
* **B: where the GT comes from** -- source decomposition: red = HD-map only,
  green = raycast-ground only, yellow = both agree, blue = raycast obstacle
  (subtracted from the GT).
* **C: final mask (Dice target)** -- the binary target actually used for the
  Dice loss.

Optionally muxes the frames into a ``.webm`` (libvpx-vp9, matching
run_e2e_subset.py) for browser inspection.

Read-only: renders images, never writes labels or touches training.

Example
-------
    PYTHONPATH=$(pwd) python3 tools/analysis_tools/visualize_kl_drivable_gt.py \
        projects/configs/stage1_track_map_lidar/base_track_drivable_lidar.py \
        --split train --num-frames 40 --stride 5 \
        --out-dir /tmp/kl_gt_vis --video /tmp/kl_gt_vis/drivable_gt.webm
"""
import argparse
import copy
import os
import os.path as osp
import subprocess

import cv2
import numpy as np
from mmcv import Config

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (
    build_map_mask, windowed_count)
from tools.analysis_tools.check_kl_drivable_alignment import (
    build_generator, build_split_dataset, find_generator, import_plugin,
    steps_before_generator)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Visualize fused KL drivable seg-head GT')
    parser.add_argument('config', help='train config path')
    parser.add_argument('--split', default='train',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--num-frames', type=int, default=40)
    parser.add_argument('--stride', type=int, default=5)
    parser.add_argument(
        '--indices', type=int, nargs='+', default=None,
        help='explicit raw dataset indices; overrides --num-frames/--stride')
    parser.add_argument('--out-dir', required=True,
                        help='folder for rendered PNG frames')
    parser.add_argument('--video', default=None,
                        help='optional .webm output path')
    parser.add_argument('--scale', type=int, default=3,
                        help='nearest-neighbour upscale factor for legibility')
    parser.add_argument('--fps', type=int, default=4)
    parser.add_argument(
        '--compare-ground-endpoint-modes', action='store_true',
        help=('also render voxel-center baseline vs point-P90 rescue; '
              'newly recovered drivable pixels are cyan'))
    parser.add_argument(
        '--compare-visible-ground-recovery', action='store_true',
        help=('also render the configured visible local-ground recovery '
              'against the same target with recovery disabled'))
    parser.add_argument(
        '--visible-recovery-distance-mode',
        choices=['euclidean', 'geodesic'], default=None,
        help=('override recovery distance for this read-only visualization; '
              'geodesic requires a connected path through valid ground '
              'candidates'))
    parser.add_argument(
        '--visible-recovery-max-distance', type=float, default=None,
        help='override recovery distance in BEV cells for visualization')
    parser.add_argument(
        '--render-ground-evidence', action='store_true',
        help=('also render the local-ground support, fill extension, and '
              'obstacle suppression used to produce the drivable target'))
    parser.add_argument(
        '--ground-fill-variants', nargs='*', default=None,
        help=('read-only fill ablations as RADIUS:MIN_NEIGHBOURS, for '
              'example --ground-fill-variants 2:3 3:5'))
    parser.add_argument(
        '--ground-fill-variants-visible-only', action='store_true',
        help=('for fill ablations, retain only additions crossed by a LiDAR '
              'free ray; the configured target itself remains unchanged'))
    parser.add_argument(
        '--outline-static-top-right', action='store_true',
        help=('outline the nearest connected static-obstacle component in '
              'the image-space upper-right quadrant of the ego centre'))
    parser.add_argument(
        '--outline-static-top-right-rank', type=int, default=0,
        help=('rank among image-space upper-right static components, ordered '
              'by distance from ego; 0 is the nearest'))
    return parser.parse_args()


def lidar_density_bev(pts, pc_range, bev_size):
    """Gray backdrop: clipped LiDAR point count per BEV pixel."""
    h, w = bev_size
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    cols = ((pts[:, 0] - x_min) / (x_max - x_min) * w).astype(int)
    rows = ((y_max - pts[:, 1]) / (y_max - y_min) * h).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    bg = np.zeros((h, w), dtype=np.uint8)
    np.add.at(bg, (rows[ok], cols[ok]), 1)
    return (np.clip(bg, 0, 3) * 70).astype(np.uint8)


def lidar_height_bev(pts, pc_range, bev_size):
    """BGR panel of raw LiDAR coloured by height (z), JET colormap.

    Higher points are warmer; the topmost point per pixel wins so tall
    structures (stacks, gantry) stand out over ground. This is a richer
    view of the raw input than the gray density backdrop in panel A.
    """
    h, w = bev_size
    x_min, y_min, z_min, x_max, y_max, z_max = [float(v) for v in pc_range]
    cols = ((pts[:, 0] - x_min) / (x_max - x_min) * w).astype(int)
    rows = ((y_max - pts[:, 1]) / (y_max - y_min) * h).astype(int)
    ok = (cols >= 0) & (cols < w) & (rows >= 0) & (rows < h)
    cols, rows, z = cols[ok], rows[ok], pts[ok, 2]
    # Keep the highest point per pixel so foreground structure is visible.
    order = np.argsort(z)
    cols, rows, z = cols[order], rows[order], z[order]
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-6), 0, 1)
    color = cv2.applyColorMap((z_norm * 255).astype(np.uint8),
                              cv2.COLORMAP_JET).reshape(-1, 3)
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[rows, cols] = color
    return panel


def draw_ego_axes(panel):
    """Draw only the two ego-frame axes; labels would obscure BEV evidence."""
    h, w = panel.shape[:2]
    origin = (w // 2, h // 2)
    x_end = (origin[0] + min(32, w // 7), origin[1])
    y_end = (origin[0], origin[1] - min(32, h // 7))
    for end, color in ((x_end, (40, 60, 235)),
                       (y_end, (75, 205, 75))):
        cv2.arrowedLine(panel, origin, end, (8, 8, 8), 3, cv2.LINE_AA,
                        tipLength=0.20)
        cv2.arrowedLine(panel, origin, end, color, 1, cv2.LINE_AA,
                        tipLength=0.20)
    cv2.circle(panel, origin, 2, (240, 240, 240), -1, cv2.LINE_AA)


def final_drivable_mask(map_mask, ground, blocked, augment_raycast_ground,
                        keep_raycast_obstacles):
    """Mirror the final binary seg-head target construction exactly."""
    if not augment_raycast_ground:
        return map_mask.astype(np.uint8)
    final = np.maximum(map_mask, ground).astype(np.uint8)
    if not keep_raycast_obstacles:
        final[blocked > 0] = 0
    return final


def _draw_height_colorbar(panel, z_min, z_max):
    """Overlay a small vertical JET colorbar (height key) inside a panel.

    Drawn directly on panel D so the low/high height ramp is scoped to the
    point cloud it describes, instead of leaking into the shared legend.
    """
    h, w = panel.shape[:2]
    bar_h = int(h * 0.5)
    bar_w = 10
    x0 = w - bar_w - 8
    y0 = int(h * 0.25)
    ramp = np.linspace(255, 0, bar_h).astype(np.uint8).reshape(-1, 1)
    ramp = cv2.applyColorMap(ramp, cv2.COLORMAP_JET)
    panel[y0:y0 + bar_h, x0:x0 + bar_w] = ramp
    cv2.rectangle(panel, (x0, y0), (x0 + bar_w, y0 + bar_h),
                  (200, 200, 200), 1)
    cv2.putText(panel, 'high', (x0 - 34, y0 + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1,
                cv2.LINE_AA)
    cv2.putText(panel, 'low', (x0 - 30, y0 + bar_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (235, 235, 235), 1,
                cv2.LINE_AA)
    return panel


def _titled_panel(panel, title, scale):
    """Upscale a panel and stack a dark title bar on top of it."""
    big = cv2.resize(panel, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_NEAREST)
    h, w = big.shape[:2]
    bar = np.full((26, w, 3), 30, dtype=np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.concatenate([bar, big], axis=0)


def _legend_bar(width, items):
    """A bottom strip of colour swatches + labels (BGR colours)."""
    bar = np.full((34, width, 3), 30, dtype=np.uint8)
    x = 8
    for color, label in items:
        cv2.rectangle(bar, (x, 9), (x + 18, 25), color, -1)
        cv2.rectangle(bar, (x, 9), (x + 18, 25), (200, 200, 200), 1)
        x += 24
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(bar, label, (x, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (235, 235, 235), 1, cv2.LINE_AA)
        x += tw + 22
    return bar


def render_frame(map_mask, ground, blocked, pts, pc_range, bev_size,
                 scale=3, frame_idx=None, augment_raycast_ground=True,
                 keep_raycast_obstacles=False):
    """Build the labelled 2x2 panel figure for one frame.

    Panels read input -> output, left-to-right then top-to-bottom:
    D (raw LiDAR by height) | A (final GT on density) on the top row,
    B (source decomposition) | C (final Dice mask) on the bottom row.
    Each panel has its own title bar; legend strips along the bottom
    explain every colour so the figure stands on its own.

    ``augment_raycast_ground`` / ``keep_raycast_obstacles`` mirror the
    generator flags so the rendered ``final`` matches the actual training
    target rather than assuming the default fusion.
    """
    h, w = bev_size
    final = final_drivable_mask(
        map_mask, ground, blocked, augment_raycast_ground,
        keep_raycast_obstacles)

    # Panel D: raw LiDAR coloured by height (the model's actual input).
    panel_d = lidar_height_bev(pts, pc_range, bev_size)
    cv2.drawMarker(panel_d, (w // 2, h // 2), (255, 255, 255),
                   cv2.MARKER_CROSS, 8, 1)
    draw_ego_axes(panel_d)

    # Panel A: final GT over lidar density, ego centre cross.
    bg = lidar_density_bev(pts, pc_range, bev_size)
    panel_a = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)
    panel_a[final > 0] = (60, 200, 60)
    cv2.drawMarker(panel_a, (w // 2, h // 2), (0, 0, 255),
                   cv2.MARKER_CROSS, 8, 1)
    draw_ego_axes(panel_a)

    # Panel B: source decomposition.
    map_b = map_mask > 0
    ground_b = ground > 0
    panel_b = np.zeros((h, w, 3), dtype=np.uint8)
    panel_b[..., 2] = ((map_b & ~ground_b) * 255).astype(np.uint8)  # red
    panel_b[..., 1] = ((ground_b & ~map_b) * 255).astype(np.uint8)  # green
    panel_b[map_b & ground_b] = (0, 255, 255)                       # yellow
    panel_b[blocked > 0] = (255, 80, 0)                             # blue
    draw_ego_axes(panel_b)

    # Panel C: final binary target.
    panel_c = cv2.cvtColor((final * 255).astype(np.uint8),
                           cv2.COLOR_GRAY2BGR)
    draw_ego_axes(panel_c)

    d = _titled_panel(panel_d, 'D: raw LiDAR (colour = height)', scale)
    # Height key lives inside D, scoped to the panel it describes.
    _draw_height_colorbar(d, pc_range[2], pc_range[5])
    a = _titled_panel(panel_a, 'A: final drivable GT + LiDAR', scale)
    b = _titled_panel(panel_b, 'B: where the GT comes from', scale)
    c = _titled_panel(panel_c, 'C: final mask (Dice target)', scale)

    hpad = np.full((d.shape[0], 6, 3), 30, dtype=np.uint8)
    top = np.concatenate([d, hpad, a], axis=1)
    bottom = np.concatenate([b, hpad, c], axis=1)
    vpad = np.full((6, top.shape[1], 3), 30, dtype=np.uint8)
    grid = np.concatenate([top, vpad, bottom], axis=0)

    # Legend rows: A semantics, then B semantics. (D's height ramp is shown
    # by the in-panel colorbar, not here, since it applies only to panel D.)
    legend_da = _legend_bar(grid.shape[1], [
        ((60, 200, 60), 'drivable GT'),
        ((120, 120, 120), 'LiDAR density'),
        ((255, 255, 255), 'ego'),
    ])
    legend_b = _legend_bar(grid.shape[1], [
        ((0, 0, 255), 'HD-map only'),
        ((0, 255, 0), 'raycast ground only'),
        ((0, 255, 255), 'both agree'),
        ((255, 80, 0), 'obstacle (removed)'),
    ])
    out = np.concatenate([grid, legend_da, legend_b], axis=0)
    if frame_idx is not None:
        cv2.putText(out, 'frame %d' % frame_idx,
                    (out.shape[1] - 96, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 220, 255), 1, cv2.LINE_AA)

    stats = dict(
        final=int((final > 0).sum()),
        map=int(map_b.sum()),
        ray=int(ground_b.sum()),
        blocked=int((blocked > 0).sum()))
    return out, stats


def render_ground_endpoint_mode_compare(map_mask, baseline, rescue, pts,
                                        pc_range, bev_size, scale=3,
                                        frame_idx=None,
                                        augment_raycast_ground=True,
                                        keep_raycast_obstacles=False,
                                        baseline_title='old target: voxel-center rule',
                                        rescue_title=(
                                            'new target: P90 rescue '
                                            '(cyan = newly drivable)'),
                                        recovered_label='newly drivable under rescue'):
    """Render the exact target difference between two raycast variants."""
    h, w = bev_size
    baseline_final = final_drivable_mask(
        map_mask, baseline['ground'], baseline['blocked'],
        augment_raycast_ground, keep_raycast_obstacles)
    rescue_final = final_drivable_mask(
        map_mask, rescue['ground'], rescue['blocked'],
        augment_raycast_ground, keep_raycast_obstacles)
    recovered = (rescue_final > 0) & ~(baseline_final > 0)
    removed = (baseline_final > 0) & ~(rescue_final > 0)

    input_panel = lidar_height_bev(pts, pc_range, bev_size)
    draw_ego_axes(input_panel)

    baseline_panel = cv2.cvtColor(
        (baseline_final * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    draw_ego_axes(baseline_panel)

    rescue_panel = cv2.cvtColor(
        (rescue_final * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    rescue_panel[recovered] = (230, 220, 35)  # cyan in RGB
    draw_ego_axes(rescue_panel)

    diff_panel = np.zeros((h, w, 3), dtype=np.uint8)
    diff_panel[recovered] = (230, 220, 35)    # cyan: restored ground
    diff_panel[removed] = (220, 35, 230)      # magenta: lost ground
    draw_ego_axes(diff_panel)

    panels = [
        _titled_panel(input_panel, 'input: fused LiDAR (colour = height)', scale),
        _titled_panel(baseline_panel, baseline_title, scale),
        _titled_panel(rescue_panel, rescue_title, scale),
        _titled_panel(diff_panel, 'target difference (cyan=newly drivable)', scale),
    ]
    hpad = np.full((panels[0].shape[0], 6, 3), 30, dtype=np.uint8)
    top = np.concatenate([panels[0], hpad, panels[1]], axis=1)
    bottom = np.concatenate([panels[2], hpad, panels[3]], axis=1)
    vpad = np.full((6, top.shape[1], 3), 30, dtype=np.uint8)
    grid = np.concatenate([top, vpad, bottom], axis=0)
    legend = _legend_bar(grid.shape[1], [
        ((255, 255, 255), 'drivable target'),
        ((230, 220, 35), recovered_label),
        ((220, 35, 230), 'lost drivable (must stay zero)'),
    ])
    out = np.concatenate([grid, legend], axis=0)
    if frame_idx is not None:
        cv2.putText(out, 'frame %d' % frame_idx,
                    (out.shape[1] - 112, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 220, 255), 1, cv2.LINE_AA)
    stats = dict(
        baseline_final=int((baseline_final > 0).sum()),
        rescue_final=int((rescue_final > 0).sum()),
        recovered=int(recovered.sum()),
        removed=int(removed.sum()))
    return out, stats


def _xy_mask_to_bev(builder, xy_mask):
    """Convert an [x, y] evidence mask into the visualizer's BEV layout."""
    mask = np.zeros((builder.bev_h, builder.bev_w), dtype=np.uint8)
    xys = np.argwhere(xy_mask)
    if len(xys):
        rows = builder.bev_h - 1 - xys[:, 1]
        cols = xys[:, 0]
        valid = ((rows >= 0) & (rows < builder.bev_h) &
                 (cols >= 0) & (cols < builder.bev_w))
        mask[rows[valid], cols[valid]] = 1
    return mask


def _ground_fill_masks(builder, pts, boxes):
    """Expose the pre-blocking 2D ground evidence without changing labels."""
    xyz = np.asarray(pts[:, :3], dtype=np.float32)
    point_voxels = builder.coord_to_index_floor(xyz)
    valid = np.all((point_voxels >= 0) &
                   (point_voxels < builder.occ_size), axis=1)
    xyz = xyz[valid]
    point_voxels = point_voxels[valid]
    if not len(xyz):
        empty = np.zeros((builder.bev_h, builder.bev_w), dtype=np.uint8)
        return empty, empty
    box_mask = builder.box_interior_mask(boxes)
    in_box = box_mask[point_voxels[:, 0], point_voxels[:, 1],
                      point_voxels[:, 2]]
    scene_points = xyz[~in_box]
    scene_voxels = point_voxels[~in_box]
    ground_est, raw_ground_xy = builder.estimate_ground_height(
        scene_points, scene_voxels)
    if builder.fill_ground:
        neighbor_counts = windowed_count(
            raw_ground_xy, builder.ground_fill_radius)
        fill_xy = ((neighbor_counts >= builder.ground_fill_min_neighbors) &
                   np.isfinite(ground_est))
    else:
        fill_xy = raw_ground_xy
    return (_xy_mask_to_bev(builder, raw_ground_xy),
            _xy_mask_to_bev(builder, fill_xy))


def _nearest_upper_right_static_component(static_obstacle, rank=0):
    """Select the blue component described relative to image-space ego centre."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        static_obstacle.astype(np.uint8), connectivity=8)
    h, w = static_obstacle.shape
    ego_x, ego_y = w // 2, h // 2
    candidates = []
    for label in range(1, count):
        component = labels == label
        rows, cols = np.where(component)
        upper_right = (cols > ego_x) & (rows < ego_y)
        upper_right_count = int(upper_right.sum())
        if upper_right_count < 4:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        # Ignore isolated speckles; a fan-like return has a visible footprint.
        if area < 8:
            continue
        min_distance = float(np.sqrt(np.min(
            (cols[upper_right] - ego_x) ** 2 +
            (rows[upper_right] - ego_y) ** 2)))
        candidates.append((min_distance, -upper_right_count, -area, label,
                           component, dict(
                               area=area,
                               upper_right_area=upper_right_count,
                               min_distance_px=min_distance,
                               bbox_xywh=[
                                   int(stats[label, cv2.CC_STAT_LEFT]),
                                   int(stats[label, cv2.CC_STAT_TOP]),
                                   int(stats[label, cv2.CC_STAT_WIDTH]),
                                   int(stats[label, cv2.CC_STAT_HEIGHT])])))
    if not candidates:
        return np.zeros_like(static_obstacle, dtype=bool), None
    candidates.sort(key=lambda item: item[:3])
    rank = min(max(int(rank), 0), len(candidates) - 1)
    _, _, _, _, component, info = candidates[rank]
    info['rank'] = rank
    info['candidate_count'] = len(candidates)
    info['candidate_summary'] = [item[5] for item in candidates]
    return component, info


def render_ground_evidence(raycast_builder, raycast, map_mask, pts, boxes,
                           pc_range, bev_size, scale=3, frame_idx=None,
                           augment_raycast_ground=True,
                           keep_raycast_obstacles=False,
                           outline_static_top_right=False,
                           outline_static_top_right_rank=0):
    """Show why a BEV cell is, or is not, included in drivable GT."""
    h, w = bev_size
    raw_support, filled_ground = _ground_fill_masks(
        raycast_builder, pts, boxes)
    blocked = raycast['blocked'] > 0
    static_obstacle = raycast['obstacle'] > 0
    box_only = blocked & ~static_obstacle
    visible_recovery = raycast.get(
        'visible_ground_recovery', np.zeros_like(blocked, dtype=np.uint8)) > 0
    visible_gap = ((raycast['free'] > 0) & ~(filled_ground > 0) &
                   ~blocked & ~visible_recovery)
    final = final_drivable_mask(
        map_mask, raycast['ground'], raycast['blocked'],
        augment_raycast_ground, keep_raycast_obstacles)

    input_panel = lidar_height_bev(pts, pc_range, bev_size)
    draw_ego_axes(input_panel)

    support_panel = np.zeros((h, w, 3), dtype=np.uint8)
    support_panel[raw_support > 0] = (50, 210, 50)       # green
    support_panel[(filled_ground > 0) & ~(raw_support > 0)] = (
        230, 220, 35)                                     # cyan in RGB
    support_panel[visible_recovery] = (0, 165, 255)        # orange
    draw_ego_axes(support_panel)

    suppression_panel = support_panel.copy()
    # A free ray that crosses a black ground-evidence hole is a useful
    # distinction: it is visible-but-sparse, not an unobserved occlusion.
    suppression_panel[visible_gap] = (30, 190, 240)         # orange in RGB
    suppression_panel[static_obstacle] = (255, 90, 0)     # blue in RGB
    suppression_panel[box_only] = (220, 35, 230)          # magenta in RGB
    suppression_panel[visible_recovery] = (0, 165, 255)   # recovered ground
    outlined_component = None
    if outline_static_top_right:
        outlined_component, outlined_info = _nearest_upper_right_static_component(
            static_obstacle, rank=outline_static_top_right_rank)
        if outlined_info is not None:
            contours, _ = cv2.findContours(
                outlined_component.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(suppression_panel, contours, -1,
                             (8, 8, 8), 3, cv2.LINE_AA)
            cv2.drawContours(suppression_panel, contours, -1,
                             (255, 255, 255), 1, cv2.LINE_AA)
    else:
        outlined_info = None
    draw_ego_axes(suppression_panel)

    target_panel = cv2.cvtColor(
        (final * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    draw_ego_axes(target_panel)

    panels = [
        _titled_panel(input_panel, 'input: fused LiDAR (colour = height)', scale),
        _titled_panel(
            support_panel,
            'ground evidence: green=direct, cyan=fill, orange=recovered',
            scale),
        _titled_panel(
            suppression_panel,
            'suppression: yellow=visible gap, orange=recovered, blue=static',
            scale),
        _titled_panel(target_panel, 'final drivable GT (white)', scale),
    ]
    hpad = np.full((panels[0].shape[0], 6, 3), 30, dtype=np.uint8)
    top = np.concatenate([panels[0], hpad, panels[1]], axis=1)
    bottom = np.concatenate([panels[2], hpad, panels[3]], axis=1)
    vpad = np.full((6, top.shape[1], 3), 30, dtype=np.uint8)
    grid = np.concatenate([top, vpad, bottom], axis=0)
    legend = _legend_bar(grid.shape[1], [
        ((50, 210, 50), 'direct local-ground support'),
        ((230, 220, 35), 'neighbour-filled ground'),
        ((0, 165, 255), 'visible local-ground recovery'),
        ((30, 190, 240), 'free-ray visibility but no ground evidence'),
        ((255, 90, 0), 'suppressed by static obstacle'),
        ((220, 35, 230), 'suppressed by annotated box'),
        ((255, 255, 255), 'final drivable target'),
    ])
    out = np.concatenate([grid, legend], axis=0)
    if frame_idx is not None:
        cv2.putText(out, 'frame %d' % frame_idx,
                    (out.shape[1] - 112, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 220, 255), 1, cv2.LINE_AA)
    stats = dict(
        direct_support=int((raw_support > 0).sum()),
        neighbour_fill=int(((filled_ground > 0) & ~(raw_support > 0)).sum()),
        visible_no_ground=int(visible_gap.sum()),
        visible_ground_recovery=int(visible_recovery.sum()),
        static_obstacle=int(static_obstacle.sum()),
        annotated_box=int(box_only.sum()),
        unobserved_no_ground=int(
            ((filled_ground == 0) & ~blocked & ~(raycast['free'] > 0)).sum()),
        final=int((final > 0).sum()))
    if outlined_info is not None:
        stats['outlined_static_component'] = outlined_info
    return out, stats


def _parse_fill_variant(value):
    try:
        radius_text, count_text = value.split(':')
        radius, count = int(radius_text), int(count_text)
    except ValueError as error:
        raise ValueError(
            'Each --ground-fill-variants value must be RADIUS:MIN_NEIGHBOURS, '
            f'got {value!r}') from error
    if radius < 0 or count < 1:
        raise ValueError(f'Invalid ground-fill variant {value!r}')
    return radius, count


def render_ground_fill_variant_compare(map_mask, baseline, variants, pts,
                                       pc_range, bev_size, scale=3,
                                       frame_idx=None,
                                       augment_raycast_ground=True,
                                       keep_raycast_obstacles=False,
                                       baseline_radius=2,
                                       baseline_min_neighbours=5,
                                       visible_only=False):
    """Render read-only fill ablations against the configured target."""
    baseline_final = final_drivable_mask(
        map_mask, baseline['ground'], baseline['blocked'],
        augment_raycast_ground, keep_raycast_obstacles)
    input_panel = lidar_height_bev(pts, pc_range, bev_size)
    draw_ego_axes(input_panel)
    baseline_panel = cv2.cvtColor(
        (baseline_final * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    draw_ego_axes(baseline_panel)
    panels = [
        _titled_panel(input_panel, 'input: fused LiDAR (colour = height)', scale),
        _titled_panel(
            baseline_panel,
            f'configured target: fill r={baseline_radius}, '
            f'n={baseline_min_neighbours}', scale),
    ]
    stats = []
    for radius, min_neighbours, raycast in variants:
        candidate = final_drivable_mask(
            map_mask, raycast['ground'], raycast['blocked'],
            augment_raycast_ground, keep_raycast_obstacles)
        if visible_only:
            # A local fill is allowed to extend the target only where at
            # least one fused LiDAR ray has line-of-sight through the cell.
            # This retains the configured target and prevents an unobserved
            # fill from silently turning an occlusion into drivable space.
            candidate = baseline_final.copy()
            candidate[(raycast['ground'] > 0) & (baseline['free'] > 0)] = 1
        recovered = (candidate > 0) & ~(baseline_final > 0)
        lost = (baseline_final > 0) & ~(candidate > 0)
        panel = cv2.cvtColor(
            (candidate * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        panel[recovered] = (230, 220, 35)  # cyan in RGB
        panel[lost] = (220, 35, 230)       # magenta in RGB
        draw_ego_axes(panel)
        panels.append(_titled_panel(
            panel,
            f'candidate: fill r={radius}, n={min_neighbours} '
            f'({"visible-only, " if visible_only else ""}cyan = added)',
            scale))
        stats.append(dict(
            radius=radius,
            min_neighbours=min_neighbours,
            final=int((candidate > 0).sum()),
            recovered=int(recovered.sum()),
            lost=int(lost.sum())))

    # Every panel has fixed dimensions. Pad the final row only when needed.
    blank = np.full_like(panels[0], 30)
    if len(panels) % 2:
        panels.append(blank)
    rows = []
    hpad = np.full((panels[0].shape[0], 6, 3), 30, dtype=np.uint8)
    for start in range(0, len(panels), 2):
        rows.append(np.concatenate([panels[start], hpad, panels[start + 1]], axis=1))
    vpad = np.full((6, rows[0].shape[1], 3), 30, dtype=np.uint8)
    grid = rows[0]
    for row in rows[1:]:
        grid = np.concatenate([grid, vpad, row], axis=0)
    legend = _legend_bar(grid.shape[1], [
        ((255, 255, 255), 'configured / candidate drivable'),
        ((230, 220, 35), 'candidate adds drivable'),
        ((220, 35, 230), 'candidate loses drivable'),
    ])
    out = np.concatenate([grid, legend], axis=0)
    if frame_idx is not None:
        cv2.putText(out, 'frame %d' % frame_idx,
                    (out.shape[1] - 112, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (180, 220, 255), 1, cv2.LINE_AA)
    return out, stats

def write_webm(frame_dir, out_path, fps):
    """Mux numbered PNG frames into a VP9 webm (matches run_e2e_subset.py)."""
    if osp.splitext(out_path)[1].lower() != '.webm':
        raise ValueError('--video must end in .webm')
    out_parent = osp.dirname(out_path)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)
    # Frames are already even-sized after the integer upscale; still guard
    # the encoder with a trunc-to-even scale filter.
    cmd = [
        'ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', osp.join(frame_dir, 'gt_%05d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
        '-c:v', 'libvpx-vp9', '-crf', '32', '-b:v', '0',
        '-pix_fmt', 'yuv420p',
        out_path,
    ]
    subprocess.run(cmd, check=True)


def prepare_prefix_input(dataset, index):
    """Return one raw sample with E2E future annotations when required.

    ``KlTrackDataset`` normally adds these fields inside its queue-preparation
    path.  A single-frame diagnostic intentionally bypasses the queue, but
    still has to satisfy ``LoadAnnotations3D_E2E`` before replaying the
    drivable-label transform.
    """
    info = dataset.get_data_info(index)
    if (info is None or 'occ_future_ann_infos' in info or
            not getattr(dataset, 'with_occ_labels', False)):
        return info
    raw_index = (dataset._to_raw_index(index)
                 if hasattr(dataset, '_to_raw_index') else index)
    occ_inputs = dataset._build_occ_inputs(raw_index)
    if occ_inputs is not None:
        info.update(occ_inputs)
    return info


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    import_plugin(cfg)

    pipeline_cfg = cfg.data[args.split].pipeline
    gen = build_generator(find_generator(pipeline_cfg))
    pre = steps_before_generator(pipeline_cfg)
    dataset = build_split_dataset(cfg, args.split)

    num_total = len(dataset)
    if args.indices is None:
        indices = list(range(0, num_total, args.stride))[:args.num_frames]
    else:
        indices = list(args.indices)
        invalid = [idx for idx in indices if idx < 0 or idx >= num_total]
        if invalid:
            raise ValueError(
                f'--indices must be in [0, {num_total}), got {invalid}')
    os.makedirs(args.out_dir, exist_ok=True)

    bev_size = gen.bev_size
    pc_range = gen.point_cloud_range
    if args.visible_recovery_distance_mode is not None:
        gen.raycast_builder.visible_ground_recovery_distance_mode = (
            args.visible_recovery_distance_mode)
    if args.visible_recovery_max_distance is not None:
        if args.visible_recovery_max_distance < 0:
            raise ValueError('--visible-recovery-max-distance must be >= 0')
        gen.raycast_builder.visible_ground_recovery_max_distance = (
            args.visible_recovery_max_distance)
    baseline_raycast_builder = None
    if args.compare_ground_endpoint_modes:
        baseline_raycast_builder = copy.deepcopy(gen.raycast_builder)
        baseline_raycast_builder.ground_endpoint_height_mode = 'voxel_center'
    visible_recovery_baseline_builder = None
    if args.compare_visible_ground_recovery:
        visible_recovery_baseline_builder = copy.deepcopy(gen.raycast_builder)
        visible_recovery_baseline_builder.visible_ground_recovery = False
    fill_variant_builders = []
    for value in args.ground_fill_variants or []:
        radius, min_neighbours = _parse_fill_variant(value)
        builder = copy.deepcopy(gen.raycast_builder)
        builder.ground_fill_radius = radius
        builder.ground_fill_min_neighbors = min_neighbours
        fill_variant_builders.append((radius, min_neighbours, builder))
    seq = 0
    for idx in indices:
        info = prepare_prefix_input(dataset, idx)
        if info is None:
            continue
        dataset.pre_pipeline(info)
        results = pre(info)
        if results is None or 'points' not in results:
            continue

        pts = gen._points_numpy(results['points'])
        boxes = gen._boxes_numpy(results.get('gt_bboxes_3d'))
        map_mask = build_map_mask(
            gen.drivable_global, gen._ego2global(results), pc_range, bev_size)
        raycast = gen.raycast_builder.build(pts, boxes)
        strip, stats = render_frame(
            map_mask, raycast['ground'], raycast['blocked'],
            pts, pc_range, bev_size, scale=args.scale, frame_idx=idx,
            augment_raycast_ground=gen.augment_raycast_ground,
            keep_raycast_obstacles=gen.keep_raycast_obstacles)

        # Sequential filenames so ffmpeg sees a contiguous frame range even
        # when --stride skips dataset indices.
        cv2.imwrite(osp.join(args.out_dir, 'gt_%05d.png' % seq), strip)
        if baseline_raycast_builder is not None:
            baseline_raycast = baseline_raycast_builder.build(pts, boxes)
            comparison, comparison_stats = render_ground_endpoint_mode_compare(
                map_mask, baseline_raycast, raycast, pts, pc_range, bev_size,
                scale=args.scale, frame_idx=idx,
                augment_raycast_ground=gen.augment_raycast_ground,
                keep_raycast_obstacles=gen.keep_raycast_obstacles)
            cv2.imwrite(
                osp.join(args.out_dir, 'ground_endpoint_compare_%05d.png' % seq),
                comparison)
            print(
                '  endpoint mode comparison: old=%d new=%d recovered=%d '
                'lost=%d' % (
                    comparison_stats['baseline_final'],
                    comparison_stats['rescue_final'],
                    comparison_stats['recovered'],
                    comparison_stats['removed']))
        if visible_recovery_baseline_builder is not None:
            visible_recovery_baseline = visible_recovery_baseline_builder.build(
                pts, boxes)
            comparison, comparison_stats = render_ground_endpoint_mode_compare(
                map_mask, visible_recovery_baseline, raycast, pts,
                pc_range, bev_size, scale=args.scale, frame_idx=idx,
                augment_raycast_ground=gen.augment_raycast_ground,
                keep_raycast_obstacles=gen.keep_raycast_obstacles,
                baseline_title='baseline target: no local recovery',
                rescue_title='new target: visible local recovery (cyan)',
                recovered_label='newly drivable under local recovery')
            cv2.imwrite(
                osp.join(args.out_dir,
                         'visible_ground_recovery_compare_%05d.png' % seq),
                comparison)
            print(
                '  visible-ground recovery comparison: old=%d new=%d '
                'recovered=%d lost=%d' % (
                    comparison_stats['baseline_final'],
                    comparison_stats['rescue_final'],
                    comparison_stats['recovered'],
                    comparison_stats['removed']))
        if args.render_ground_evidence:
            evidence, evidence_stats = render_ground_evidence(
                gen.raycast_builder, raycast, map_mask, pts, boxes,
                pc_range, bev_size, scale=args.scale, frame_idx=idx,
                augment_raycast_ground=gen.augment_raycast_ground,
                keep_raycast_obstacles=gen.keep_raycast_obstacles,
                outline_static_top_right=args.outline_static_top_right,
                outline_static_top_right_rank=(
                    args.outline_static_top_right_rank))
            cv2.imwrite(
                osp.join(args.out_dir, 'ground_evidence_%05d.png' % seq),
                evidence)
            print(
                '  ground evidence: direct=%d fill=%d recovery=%d '
                'visible_gap=%d static=%d box=%d unseen_gap=%d final=%d' % (
                    evidence_stats['direct_support'],
                    evidence_stats['neighbour_fill'],
                    evidence_stats['visible_ground_recovery'],
                    evidence_stats['visible_no_ground'],
                    evidence_stats['static_obstacle'],
                    evidence_stats['annotated_box'],
                    evidence_stats['unobserved_no_ground'],
                    evidence_stats['final']))
            if 'outlined_static_component' in evidence_stats:
                print('  outlined upper-right static component: %s' %
                      evidence_stats['outlined_static_component'])
        if fill_variant_builders:
            fill_variants = [
                (radius, min_neighbours, builder.build(pts, boxes))
                for radius, min_neighbours, builder in fill_variant_builders]
            fill_comparison, fill_stats = render_ground_fill_variant_compare(
                map_mask, raycast, fill_variants, pts, pc_range, bev_size,
                scale=args.scale, frame_idx=idx,
                augment_raycast_ground=gen.augment_raycast_ground,
                keep_raycast_obstacles=gen.keep_raycast_obstacles,
                baseline_radius=gen.raycast_builder.ground_fill_radius,
                baseline_min_neighbours=(
                    gen.raycast_builder.ground_fill_min_neighbors),
                visible_only=args.ground_fill_variants_visible_only)
            cv2.imwrite(
                osp.join(args.out_dir, 'ground_fill_variants_%05d.png' % seq),
                fill_comparison)
            for stat in fill_stats:
                print(
                    '  fill variant r=%d n=%d%s: final=%d added=%d lost=%d' % (
                        stat['radius'], stat['min_neighbours'],
                        ' visible-only'
                        if args.ground_fill_variants_visible_only else '',
                        stat['final'],
                        stat['recovered'], stat['lost']))
        seq += 1
        print('frame %5d: final=%d px (map=%d ray=%d blocked=%d)'
              % (idx, stats['final'], stats['map'], stats['ray'],
                 stats['blocked']))

    if seq == 0:
        print('No usable frames; nothing rendered.')
        return
    print('rendered %d frames to %s' % (seq, args.out_dir))

    if args.video:
        write_webm(args.out_dir, args.video, args.fps)
        print('video saved to %s' % args.video)


if __name__ == '__main__':
    main()
