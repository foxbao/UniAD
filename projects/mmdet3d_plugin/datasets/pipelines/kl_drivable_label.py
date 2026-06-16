import pickle
import os.path as osp
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES
from shapely import ops
from shapely.errors import TopologicalError
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box

try:
    from shapely.validation import make_valid as shapely_make_valid
except ImportError:
    shapely_make_valid = None


@dataclass
class MapStats:
    lanes: int = 0
    lane_polygons: int = 0
    roads: int = 0
    road_polygons: int = 0
    junctions: int = 0
    drivable_junctions: int = 0
    blocking_junctions: int = 0
    electric_fences: int = 0
    not_drivable_fences: int = 0
    parking_lots: int = 0
    parking_lot_polygons: int = 0
    positive_area: float = 0.0
    negative_area: float = 0.0
    drivable_area: float = 0.0
    warnings: list = field(default_factory=list)


def find_block_end(text: str, open_brace_idx: int) -> int:
    depth = 0
    for idx in range(open_brace_idx, len(text)):
        ch = text[idx]
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return idx
    raise ValueError('unbalanced braces while parsing map file')


def extract_blocks(text: str, name: str, anchored: bool = False) -> list:
    prefix = r'^' if anchored else r''
    pattern = re.compile(prefix + re.escape(name) + r'\s*\{',
                         re.MULTILINE)
    blocks = []
    for match in pattern.finditer(text):
        open_idx = text.find('{', match.start(), match.end())
        close_idx = find_block_end(text, open_idx)
        blocks.append(text[open_idx + 1:close_idx])
    return blocks


def extract_points(block: str) -> np.ndarray:
    pts = []
    for match in re.finditer(
            r'point\s*\{\s*x:\s*([-\d.eE+]+)\s*y:\s*([-\d.eE+]+)',
            block):
        pts.append((float(match.group(1)), float(match.group(2))))
    return np.asarray(pts, dtype=np.float64)


def extract_type(block: str) -> str:
    match = re.search(r'(?m)^\s*type:\s*([A-Za-z0-9_]+)', block)
    return match.group(1) if match else ''


def repair_geometry(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.is_valid:
        return geom
    if shapely_make_valid is not None:
        try:
            valid_geom = shapely_make_valid(geom)
            if valid_geom is not None and not valid_geom.is_empty:
                geom = valid_geom
        except (ValueError, TopologicalError):
            pass
    if not geom.is_valid:
        try:
            geom = geom.buffer(0)
        except (ValueError, TopologicalError):
            return GeometryCollection()
    return geom


def keep_polygonal(geom):
    geom = repair_geometry(geom)
    if geom is None or geom.is_empty:
        return None
    if isinstance(geom, (Polygon, MultiPolygon)):
        return geom
    if isinstance(geom, GeometryCollection):
        polys = [
            geom_part for geom_part in geom.geoms
            if isinstance(geom_part, (Polygon, MultiPolygon))
            and not geom_part.is_empty
        ]
        if not polys:
            return None
        return repair_geometry(ops.unary_union(polys))
    return None


def polygon_from_points(points: np.ndarray, min_area: float = 0.5):
    if len(points) < 3:
        return None
    if np.linalg.norm(points[0] - points[-1]) > 1e-6:
        points = np.concatenate([points, points[:1]], axis=0)
    poly = Polygon(points)
    if not poly.is_valid:
        poly = poly.buffer(0)
    poly = keep_polygonal(poly)
    if poly is None or poly.is_empty or poly.area < min_area:
        return None
    return poly


def lane_polygon_from_block(block: str):
    left_blocks = extract_blocks(block, 'left_boundary')
    right_blocks = extract_blocks(block, 'right_boundary')
    if not left_blocks or not right_blocks:
        return None
    left = extract_points(left_blocks[0])
    right = extract_points(right_blocks[0])
    if len(left) < 2 or len(right) < 2:
        return None
    points = np.concatenate([left, right[::-1]], axis=0)
    return polygon_from_points(points, min_area=1.0)


def union_or_empty(geoms: Sequence):
    geoms = [
        geom for geom in (repair_geometry(geom) for geom in geoms)
        if geom is not None and not geom.is_empty
    ]
    if not geoms:
        return GeometryCollection()
    try:
        return repair_geometry(ops.unary_union(geoms))
    except (ValueError, TopologicalError):
        merged = GeometryCollection()
        for geom in geoms:
            try:
                merged = repair_geometry(merged.union(geom))
            except (ValueError, TopologicalError):
                continue
        return merged


def build_drivable_geometry(map_file: str,
                            include_lanes: bool = True,
                            include_roads: bool = True,
                            include_junctions: bool = True,
                            include_negative: bool = True,
                            include_parking_lot: bool = False):
    with open(map_file, 'r') as f:
        text = f.read()

    stats = MapStats()
    positives = []
    negatives = []

    lane_blocks = extract_blocks(text, 'lane', anchored=True)
    stats.lanes = len(lane_blocks)
    if include_lanes:
        for block in lane_blocks:
            geom = lane_polygon_from_block(block)
            if geom is not None:
                positives.append(geom)
                stats.lane_polygons += 1

    road_blocks = extract_blocks(text, 'road', anchored=True)
    stats.roads = len(road_blocks)
    if include_roads:
        for block in road_blocks:
            for outer in extract_blocks(block, 'outer_polygon'):
                geom = polygon_from_points(extract_points(outer),
                                           min_area=1.0)
                if geom is not None:
                    positives.append(geom)
                    stats.road_polygons += 1

    junction_blocks = extract_blocks(text, 'junction', anchored=True)
    stats.junctions = len(junction_blocks)
    for block in junction_blocks:
        jtype = extract_type(block)
        geom = polygon_from_points(extract_points(block), min_area=1.0)
        if geom is None:
            continue
        if 'BLOCKING' in jtype or 'NOT_DRIVABLE' in jtype:
            negatives.append(geom)
            stats.blocking_junctions += 1
        elif include_junctions:
            positives.append(geom)
            stats.drivable_junctions += 1

    fence_blocks = extract_blocks(text, 'electric_fence', anchored=True)
    stats.electric_fences = len(fence_blocks)
    for block in fence_blocks:
        geom = polygon_from_points(extract_points(block), min_area=1.0)
        if geom is None:
            continue
        if extract_type(block) == 'NOT_DRIVABLE':
            negatives.append(geom)
            stats.not_drivable_fences += 1

    parking_lot_blocks = extract_blocks(text, 'parking_lot', anchored=True)
    stats.parking_lots = len(parking_lot_blocks)
    if include_parking_lot:
        for block in parking_lot_blocks:
            geom = polygon_from_points(extract_points(block), min_area=1.0)
            if geom is not None:
                positives.append(geom)
                stats.parking_lot_polygons += 1

    positive = keep_polygonal(union_or_empty(positives))
    negative = keep_polygonal(union_or_empty(negatives))
    if positive is None:
        positive = GeometryCollection()
    if negative is None:
        negative = GeometryCollection()

    stats.positive_area = float(positive.area)
    stats.negative_area = float(negative.area)
    if not include_negative or negative.is_empty:
        drivable = positive
    else:
        drivable = keep_polygonal(positive.difference(negative))
        if drivable is None:
            drivable = GeometryCollection()
    if not drivable.is_valid:
        drivable = drivable.buffer(0)
    drivable = keep_polygonal(drivable)
    if drivable is None:
        drivable = GeometryCollection()
    stats.drivable_area = float(drivable.area)

    if stats.road_polygons == 0 and include_roads:
        stats.warnings.append('No valid road polygons were parsed.')
    if stats.drivable_area <= 0:
        stats.warnings.append('Final drivable geometry is empty.')
    return drivable, stats


def load_clean_drivable_geometry(clean_map_file: str):
    with open(clean_map_file, 'rb') as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        if 'drivable_global' in payload:
            geom = payload['drivable_global']
        elif 'drivable' in payload:
            geom = payload['drivable']
        else:
            raise KeyError('Clean KL drivable map must contain '
                           '`drivable_global` or `drivable`.')
    else:
        geom = payload
    geom = keep_polygonal(geom)
    if geom is None:
        return GeometryCollection()
    return geom


def transform_global_geom_to_ego(geom, ego2global: np.ndarray):
    if geom is None or geom.is_empty:
        return geom
    g2e = np.linalg.inv(np.asarray(ego2global, dtype=np.float64))

    def fn(x, y, z=None):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        xe = g2e[0, 0] * x_arr + g2e[0, 1] * y_arr + g2e[0, 3]
        ye = g2e[1, 0] * x_arr + g2e[1, 1] * y_arr + g2e[1, 3]
        return xe, ye

    return ops.transform(fn, geom)


def transform_ego_geom_to_global(geom, ego2global: np.ndarray):
    if geom is None or geom.is_empty:
        return geom
    e2g = np.asarray(ego2global, dtype=np.float64)

    def fn(x, y, z=None):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        xg = e2g[0, 0] * x_arr + e2g[0, 1] * y_arr + e2g[0, 3]
        yg = e2g[1, 0] * x_arr + e2g[1, 1] * y_arr + e2g[1, 3]
        return xg, yg

    return ops.transform(fn, geom)


def iter_polygons(geom) -> Iterable[Polygon]:
    if geom is None or geom.is_empty:
        return
    if isinstance(geom, Polygon):
        yield geom
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            if not poly.is_empty:
                yield poly
    elif isinstance(geom, GeometryCollection):
        for part in geom.geoms:
            yield from iter_polygons(part)


def coords_to_mask_pixels(coords: np.ndarray, pc_range: Sequence[float],
                          bev_size: Sequence[int]) -> np.ndarray:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    bev_h, bev_w = int(bev_size[0]), int(bev_size[1])
    xs = coords[:, 0]
    ys = coords[:, 1]
    cols = (xs - x_min) / (x_max - x_min) * bev_w
    rows = (y_max - ys) / (y_max - y_min) * bev_h
    pix = np.stack([cols, rows], axis=1)
    pix = np.round(pix).astype(np.int32)
    pix[:, 0] = np.clip(pix[:, 0], 0, bev_w - 1)
    pix[:, 1] = np.clip(pix[:, 1], 0, bev_h - 1)
    return pix


def rasterize_ego_geometry(geom, pc_range: Sequence[float],
                           bev_size: Sequence[int]) -> np.ndarray:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    bev_h, bev_w = int(bev_size[0]), int(bev_size[1])
    crop = box(x_min, y_min, x_max, y_max)
    mask = np.zeros((bev_h, bev_w), dtype=np.uint8)
    if geom is None or geom.is_empty:
        return mask
    try:
        clipped = keep_polygonal(geom.intersection(crop))
    except (ValueError, TopologicalError):
        geom = keep_polygonal(repair_geometry(geom))
        if geom is None or geom.is_empty:
            return mask
        try:
            clipped = keep_polygonal(geom.intersection(crop))
        except (ValueError, TopologicalError):
            return mask
    if clipped is None or clipped.is_empty:
        return mask

    for poly in iter_polygons(clipped):
        exterior = np.asarray(poly.exterior.coords, dtype=np.float64)
        if len(exterior) >= 3:
            cv2.fillPoly(
                mask,
                [coords_to_mask_pixels(exterior, pc_range, bev_size)],
                color=1)
        for interior in poly.interiors:
            hole = np.asarray(interior.coords, dtype=np.float64)
            if len(hole) >= 3:
                cv2.fillPoly(
                    mask,
                    [coords_to_mask_pixels(hole, pc_range, bev_size)],
                    color=0)
    return mask


def build_map_mask(drivable_global, ego2global: np.ndarray,
                   pc_range: Sequence[float],
                   bev_size: Sequence[int]) -> np.ndarray:
    x_min, y_min, _, x_max, y_max, _ = [float(v) for v in pc_range]
    ego_crop = box(x_min, y_min, x_max, y_max)
    global_crop = transform_ego_geom_to_global(ego_crop, ego2global)
    try:
        clipped_global = keep_polygonal(drivable_global.intersection(
            global_crop))
    except (ValueError, TopologicalError):
        clipped_global = keep_polygonal(repair_geometry(drivable_global))
        if clipped_global is not None and not clipped_global.is_empty:
            try:
                clipped_global = keep_polygonal(clipped_global.intersection(
                    global_crop))
            except (ValueError, TopologicalError):
                clipped_global = None
    if clipped_global is None or clipped_global.is_empty:
        return np.zeros(tuple(int(v) for v in bev_size), dtype=np.uint8)
    ego_geom = keep_polygonal(transform_global_geom_to_ego(
        clipped_global, ego2global))
    return rasterize_ego_geometry(ego_geom, pc_range, bev_size)


def grouped_linear_percentile(flat_cell: np.ndarray, values: np.ndarray,
                              q: float, num_cells: int):
    """Per-cell percentile equivalent to ``np.percentile(bucket, q*100)``.

    ``flat_cell`` gives the flattened grid index for each value; values
    sharing a cell form that cell's bucket. Returns ``(out, has)`` where
    ``out`` holds the linear-interpolated percentile per cell (NaN where the
    bucket is empty) and ``has`` marks non-empty cells. This reproduces the
    previous per-cell ``np.percentile`` loop exactly (linear interpolation)
    while running as a single vectorised pass.
    """
    out = np.full(num_cells, np.nan, dtype=np.float32)
    has = np.zeros(num_cells, dtype=bool)
    if values.size == 0:
        return out, has
    order = np.lexsort((values, flat_cell))
    cell_sorted = flat_cell[order]
    val_sorted = values[order].astype(np.float64)
    uniq, start, counts = np.unique(
        cell_sorted, return_index=True, return_counts=True)
    has[uniq] = True
    pos = (counts - 1).astype(np.float64) * q
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    frac = pos - lo
    val_lo = val_sorted[start + lo]
    val_hi = val_sorted[start + hi]
    out[uniq] = (val_lo + (val_hi - val_lo) * frac).astype(np.float32)
    return out, has


def windowed_nanpercentile(grid: np.ndarray, radius: int,
                           q: float) -> np.ndarray:
    """Per-cell percentile over a ``(2*radius+1)`` square neighbourhood.

    NaN cells are ignored, matching the previous ``finite``-filtered
    ``np.percentile`` loop. We replace the original ``occ_x*occ_y`` Python
    double loop with a single vectorised pass:

    1. Stack the ``k*k`` neighbour shifts into a ``(occ_x, occ_y, k*k)``
       window tensor (k = 2*radius+1 is tiny, so this loop is cheap).
    2. Sort along the window axis -- ``np.sort`` pushes NaNs to the end, so
       the first ``n`` entries of each row are exactly that cell's finite
       neighbours in ascending order.
    3. Index the linear-interpolation position ``(n-1)*q`` directly. This
       reproduces ``np.percentile(finite_vals, q*100)`` (linear method)
       per cell, but avoids ``np.nanpercentile``'s large per-call overhead
       (~130x faster on a 120x160 grid in benchmarks).
    """
    occ_x, occ_y = grid.shape
    k = 2 * radius + 1
    pad = np.full((occ_x + 2 * radius, occ_y + 2 * radius), np.nan,
                  dtype=np.float32)
    pad[radius:radius + occ_x, radius:radius + occ_y] = grid
    windows = np.empty((occ_x, occ_y, k * k), dtype=np.float32)
    idx = 0
    for di in range(k):
        for dj in range(k):
            windows[:, :, idx] = pad[di:di + occ_x, dj:dj + occ_y]
            idx += 1
    sorted_win = np.sort(windows, axis=-1)  # NaNs sort to the end
    counts = np.isfinite(windows).sum(axis=-1)
    out = np.full((occ_x, occ_y), np.nan, dtype=np.float32)
    valid = counts > 0
    if not np.any(valid):
        return out
    n_valid = counts[valid].astype(np.float64)
    pos = (n_valid - 1.0) * q
    lo = np.floor(pos).astype(np.int64)
    hi = np.ceil(pos).astype(np.int64)
    frac = (pos - lo).astype(np.float32)
    rows = sorted_win[valid]
    val_lo = np.take_along_axis(rows, lo[:, None], axis=1)[:, 0]
    val_hi = np.take_along_axis(rows, hi[:, None], axis=1)[:, 0]
    out[valid] = val_lo + (val_hi - val_lo) * frac
    return out


def windowed_count(mask_bool: np.ndarray, radius: int) -> np.ndarray:
    """Square-neighbourhood true-count via an integral image (exact)."""
    arr = mask_bool.astype(np.int64)
    occ_x, occ_y = arr.shape
    integral = np.zeros((occ_x + 1, occ_y + 1), dtype=np.int64)
    integral[1:, 1:] = np.cumsum(np.cumsum(arr, axis=0), axis=1)
    i = np.arange(occ_x)
    j = np.arange(occ_y)
    x0 = np.maximum(i - radius, 0)[:, None]
    x1 = np.minimum(i + radius + 1, occ_x)[:, None]
    y0 = np.maximum(j - radius, 0)[None, :]
    y1 = np.minimum(j + radius + 1, occ_y)[None, :]
    return (integral[x1, y1] - integral[x0, y1] -
            integral[x1, y0] + integral[x0, y0])


class RaycastDrivableBuilder:
    """Numpy port of the KL raycast OCC ground/obstacle target generator."""

    def __init__(self,
                 pc_range: Sequence[float],
                 bev_size: Sequence[int],
                 occ_size: Optional[Sequence[int]] = None,
                 ground_height_threshold: float = 0.55,
                 ground_smooth_radius: int = 3,
                 fill_ground: bool = True,
                 ground_fill_radius: int = 2,
                 ground_fill_min_neighbors: int = 5,
                 remove_ground_under_obstacle: bool = True,
                 obstacle_min_points_per_voxel: int = 2,
                 obstacle_min_component_voxels: int = 8,
                 obstacle_small_component_keep_min_points: int = 300,
                 obstacle_thin_component_min_major_span: float = 4.0,
                 obstacle_thin_component_max_minor_span: float = 2.4,
                 obstacle_thin_component_max_z_span: float = 1.6,
                 obstacle_thin_component_keep_min_points: int = 300,
                 obstacle_box_ignore_margin: float = 0.8,
                 ego_ignore_range: Optional[Sequence[float]] = (
                     -8.0, -2.0, -2.0, 8.0, 2.0, 6.0)):
        self.pc_range = np.asarray(pc_range, dtype=np.float32)
        bev_h = int(bev_size[0])
        bev_w = int(bev_size[1])
        if occ_size is None:
            occ_size = (bev_w, bev_h, 10)
        self.occ_size = np.asarray(occ_size, dtype=np.int64)
        self.bev_h = bev_h
        self.bev_w = bev_w
        if self.occ_size.shape[0] != 3 or np.any(self.occ_size <= 0):
            raise ValueError(f'Invalid occ_size: {self.occ_size}')
        if self.occ_size[0] != self.bev_w or self.occ_size[1] != self.bev_h:
            raise ValueError(
                'occ_size X/Y must match bev_size W/H, got '
                f'occ={self.occ_size.tolist()} bev={[self.bev_h, self.bev_w]}')

        self.voxel_size = (
            (self.pc_range[3:] - self.pc_range[:3]) /
            self.occ_size.astype(np.float32))
        self.ray_origin = np.zeros(3, dtype=np.float32)
        self.ground_height_threshold = float(ground_height_threshold)
        self.ground_smooth_radius = int(ground_smooth_radius)
        self.fill_ground = bool(fill_ground)
        self.ground_fill_radius = int(ground_fill_radius)
        self.ground_fill_min_neighbors = int(ground_fill_min_neighbors)
        self.remove_ground_under_obstacle = bool(remove_ground_under_obstacle)
        self.obstacle_min_points_per_voxel = int(
            obstacle_min_points_per_voxel)
        self.obstacle_min_component_voxels = int(
            obstacle_min_component_voxels)
        self.obstacle_small_component_keep_min_points = int(
            obstacle_small_component_keep_min_points)
        self.obstacle_thin_component_min_major_span = float(
            obstacle_thin_component_min_major_span)
        self.obstacle_thin_component_max_minor_span = float(
            obstacle_thin_component_max_minor_span)
        self.obstacle_thin_component_max_z_span = float(
            obstacle_thin_component_max_z_span)
        self.obstacle_thin_component_keep_min_points = int(
            obstacle_thin_component_keep_min_points)
        self.filter_thin_obstacle_components = (
            self.obstacle_thin_component_min_major_span > 0 and
            self.obstacle_thin_component_max_minor_span > 0 and
            self.obstacle_thin_component_max_z_span > 0)
        self.obstacle_box_ignore_margin = float(obstacle_box_ignore_margin)
        self.ego_ignore_range = (
            None if ego_ignore_range is None else
            np.asarray(ego_ignore_range, dtype=np.float32))

    def coord_to_index_floor(self, xyz: np.ndarray) -> np.ndarray:
        return np.floor((xyz - self.pc_range[:3]) / self.voxel_size).astype(
            np.int64)

    def coord_to_index_ceil(self, xyz: np.ndarray) -> np.ndarray:
        return np.ceil((xyz - self.pc_range[:3]) / self.voxel_size).astype(
            np.int64)

    def voxel_centers(self, axis: int, indices: np.ndarray) -> np.ndarray:
        return (self.pc_range[axis] +
                (indices.astype(np.float32) + 0.5) * self.voxel_size[axis])

    def box_voxel_indices(self, box_arr: np.ndarray):
        length, width, height = box_arr[3:6]
        if length <= 0 or width <= 0 or height <= 0:
            return None

        yaw = box_arr[6]
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        dx = np.asarray([-0.5, -0.5, 0.5, 0.5], dtype=np.float32) * length
        dy = np.asarray([-0.5, 0.5, -0.5, 0.5], dtype=np.float32) * width
        x_corners = box_arr[0] + dx * cos_yaw - dy * sin_yaw
        y_corners = box_arr[1] + dx * sin_yaw + dy * cos_yaw

        corners_min = np.asarray(
            [x_corners.min(), y_corners.min(), box_arr[2]], dtype=np.float32)
        corners_max = np.asarray(
            [x_corners.max(), y_corners.max(), box_arr[2] + height],
            dtype=np.float32)
        lo = np.maximum(self.coord_to_index_floor(corners_min), 0)
        hi = np.minimum(self.coord_to_index_ceil(corners_max), self.occ_size)
        if np.any(hi <= lo):
            return None

        x_idx = np.arange(lo[0], hi[0], dtype=np.int64)
        y_idx = np.arange(lo[1], hi[1], dtype=np.int64)
        z_idx = np.arange(lo[2], hi[2], dtype=np.int64)
        if x_idx.size == 0 or y_idx.size == 0 or z_idx.size == 0:
            return None

        xs = self.voxel_centers(0, x_idx)
        ys = self.voxel_centers(1, y_idx)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        rel_x = xx - box_arr[0]
        rel_y = yy - box_arr[1]
        local_x = rel_x * cos_yaw + rel_y * sin_yaw
        local_y = -rel_x * sin_yaw + rel_y * cos_yaw
        inside_xy = (
            (np.abs(local_x) <= length * 0.5) &
            (np.abs(local_y) <= width * 0.5))
        if not np.any(inside_xy):
            return None

        grid_x, grid_y = np.meshgrid(x_idx, y_idx, indexing='ij')
        return grid_x[inside_xy], grid_y[inside_xy], z_idx

    def box_interior_mask(self, boxes: np.ndarray) -> np.ndarray:
        mask = np.zeros(tuple(self.occ_size.tolist()), dtype=bool)
        for box_arr in boxes:
            indices = self.box_voxel_indices(box_arr)
            if indices is None:
                continue
            fill_x, fill_y, z_idx = indices
            mask[fill_x[:, None], fill_y[:, None], z_idx[None, :]] = True
        return mask

    def raycast_free_voxels(self, hit_voxels: np.ndarray) -> np.ndarray:
        """Voxels traversed by rays from the origin to each hit voxel.

        Vectorised replacement for the original per-hit Python loop (which
        called ``np.unique`` once per ray -- tens of thousands of calls per
        frame). Every ray uses a different number of samples ``num_steps``,
        so we flatten all rays into one sample array via a "ragged repeat":

        - ``ns[r]`` samples for ray ``r`` -> ``ray`` maps each flat sample
          back to its ray, ``off`` is the per-ray start offset, and
          ``step = global_index - off`` recovers the 0..ns-1 position so
          ``t = step / ns`` matches the old ``arange(ns)/ns`` parametrisation
          exactly.
        - Samples are floored to voxel indices, out-of-range ones dropped,
          and each ray's own hit voxel removed (``notself``).

        A single final ``np.unique`` deduplicates across all rays. Output is
        identical to the old loop (verified element-for-element).
        """
        if hit_voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)

        hit_centers = (
            self.pc_range[:3] +
            (hit_voxels.astype(np.float32) + 0.5) * self.voxel_size)
        deltas = hit_centers - self.ray_origin[None, :]
        max_grid_dist = np.max(np.abs(deltas / self.voxel_size[None, :]),
                               axis=1)
        num_steps = np.ceil(max_grid_dist).astype(np.int64)

        keep = num_steps > 0
        hit_voxels = hit_voxels[keep]
        deltas = deltas[keep]
        num_steps = num_steps[keep]
        if hit_voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)

        total = int(num_steps.sum())
        ray = np.repeat(np.arange(hit_voxels.shape[0]), num_steps)
        ray_start = np.repeat(np.cumsum(num_steps) - num_steps, num_steps)
        step = np.arange(total, dtype=np.int64) - ray_start
        t = (step / num_steps[ray]).astype(np.float32)
        samples = self.ray_origin[None, :] + t[:, None] * deltas[ray]
        voxels = self.coord_to_index_floor(samples)

        in_range = np.all((voxels >= 0) & (voxels < self.occ_size), axis=1)
        voxels = voxels[in_range]
        ray_of = ray[in_range]
        if voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)
        not_self = np.any(voxels != hit_voxels[ray_of], axis=1)
        voxels = voxels[not_self]
        if voxels.shape[0] == 0:
            return np.empty((0, 3), dtype=np.int64)
        return np.unique(voxels, axis=0)

    def ego_ignore_bev_mask(self) -> np.ndarray:
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        if self.ego_ignore_range is None:
            return mask
        lo = np.maximum(self.ego_ignore_range[:3], self.pc_range[:3])
        hi = np.minimum(self.ego_ignore_range[3:], self.pc_range[3:])
        if np.any(hi <= lo):
            return mask
        x_idx = np.arange(self.occ_size[0], dtype=np.int64)
        y_idx = np.arange(self.occ_size[1], dtype=np.int64)
        xs = self.voxel_centers(0, x_idx)
        ys = self.voxel_centers(1, y_idx)
        x_keep = np.flatnonzero((xs >= lo[0]) & (xs <= hi[0]))
        y_keep = np.flatnonzero((ys >= lo[1]) & (ys <= hi[1]))
        if x_keep.size == 0 or y_keep.size == 0:
            return mask
        rows = self.bev_h - 1 - y_keep
        mask[np.ix_(rows, x_keep)] = 1
        return mask

    def estimate_ground_height(self, points_xyz: np.ndarray,
                               point_voxels: np.ndarray):
        # Vectorised port of the original two nested ``occ_x*occ_y`` loops:
        #   (1) per-(x,y) cell 10th-percentile of point z  -> raw ground,
        #   (2) 25th-percentile smoothing over a square neighbourhood.
        # Cells are flattened to a single ``x*occ_y + y`` key so the whole
        # per-cell percentile step is one grouped pass; the smoothing reuses
        # ``windowed_nanpercentile``. Output matches the old loop exactly
        # (linear-interpolation percentile), ~40x faster on a real frame.
        occ_x = int(self.occ_size[0])
        occ_y = int(self.occ_size[1])
        flat_cell = (point_voxels[:, 0].astype(np.int64) * occ_y +
                     point_voxels[:, 1].astype(np.int64))
        raw_flat, has_flat = grouped_linear_percentile(
            flat_cell, points_xyz[:, 2], 0.10, occ_x * occ_y)
        ground_raw = raw_flat.reshape(occ_x, occ_y)
        raw_ground_xy = has_flat.reshape(occ_x, occ_y)

        ground_est = windowed_nanpercentile(
            ground_raw, self.ground_smooth_radius, 0.25)
        return ground_est, raw_ground_xy

    def component_point_count(self, component: Sequence[Tuple[int, int, int]],
                              point_counts: Optional[np.ndarray]) -> int:
        if point_counts is None:
            return 0
        component_idx = np.asarray(component, dtype=np.int64)
        return int(point_counts[component_idx[:, 0], component_idx[:, 1],
                                component_idx[:, 2]].sum())

    def should_keep_dense_small_component(
            self,
            component: Sequence[Tuple[int, int, int]],
            point_counts: Optional[np.ndarray]) -> bool:
        return (
            self.obstacle_small_component_keep_min_points > 0 and
            self.component_point_count(component, point_counts) >=
            self.obstacle_small_component_keep_min_points)

    def is_thin_obstacle_component(
            self,
            component: Sequence[Tuple[int, int, int]],
            point_counts: Optional[np.ndarray] = None) -> bool:
        if not self.filter_thin_obstacle_components:
            return False
        component_idx = np.asarray(component, dtype=np.int64)
        if (self.obstacle_thin_component_keep_min_points > 0 and
                point_counts is not None):
            if (self.component_point_count(component_idx, point_counts) >=
                    self.obstacle_thin_component_keep_min_points):
                return False
        centers = self.pc_range[:3] + (
            component_idx.astype(np.float32) + 0.5) * self.voxel_size
        span = centers.max(axis=0) - centers.min(axis=0)
        major_xy_span = float(max(span[0], span[1]))
        minor_xy_span = float(min(span[0], span[1]))
        z_span = float(span[2])
        eps = 1e-4
        return (
            major_xy_span + eps >=
            self.obstacle_thin_component_min_major_span and
            minor_xy_span <=
            self.obstacle_thin_component_max_minor_span + eps and
            z_span <= self.obstacle_thin_component_max_z_span + eps)

    def filter_obstacle_voxels(self, obstacle_voxels: np.ndarray,
                               scene_point_voxels: np.ndarray) -> np.ndarray:
        if obstacle_voxels.shape[0] == 0:
            return obstacle_voxels

        filtered = obstacle_voxels
        counts = None
        if (self.obstacle_min_points_per_voxel > 1 or
                self.obstacle_small_component_keep_min_points > 0 or
                self.obstacle_thin_component_keep_min_points > 0):
            counts = np.zeros(tuple(self.occ_size.tolist()), dtype=np.uint16)
            if scene_point_voxels.shape[0] > 0:
                np.add.at(counts, tuple(scene_point_voxels.T), 1)
        if self.obstacle_min_points_per_voxel > 1:
            keep = (
                counts[filtered[:, 0], filtered[:, 1], filtered[:, 2]] >=
                self.obstacle_min_points_per_voxel)
            filtered = filtered[keep]
            if filtered.shape[0] == 0:
                return filtered

        if (self.obstacle_min_component_voxels <= 1 and
                not self.filter_thin_obstacle_components):
            return filtered

        obstacle_mask = np.zeros(tuple(self.occ_size.tolist()), dtype=bool)
        obstacle_mask[filtered[:, 0], filtered[:, 1], filtered[:, 2]] = True
        visited = np.zeros_like(obstacle_mask)
        keep_mask = np.zeros_like(obstacle_mask)
        neighbors = (
            (1, 0, 0), (-1, 0, 0), (0, 1, 0),
            (0, -1, 0), (0, 0, 1), (0, 0, -1))

        for start in filtered:
            start_tuple = tuple(int(v) for v in start)
            if visited[start_tuple] or not obstacle_mask[start_tuple]:
                continue
            stack = [start_tuple]
            visited[start_tuple] = True
            component = []
            while stack:
                voxel = stack.pop()
                component.append(voxel)
                for offset in neighbors:
                    nxt = (
                        voxel[0] + offset[0],
                        voxel[1] + offset[1],
                        voxel[2] + offset[2])
                    if (0 <= nxt[0] < self.occ_size[0] and
                            0 <= nxt[1] < self.occ_size[1] and
                            0 <= nxt[2] < self.occ_size[2] and
                            obstacle_mask[nxt] and not visited[nxt]):
                        visited[nxt] = True
                        stack.append(nxt)
            if len(component) < self.obstacle_min_component_voxels:
                if not self.should_keep_dense_small_component(
                        component, counts):
                    continue
            if self.is_thin_obstacle_component(component, counts):
                continue
            for voxel in component:
                keep_mask[voxel] = True

        keep = keep_mask[filtered[:, 0], filtered[:, 1], filtered[:, 2]]
        return filtered[keep]

    def obstacle_near_box_mask(self, obstacle_voxels: np.ndarray,
                               boxes: np.ndarray) -> np.ndarray:
        mask = np.zeros((obstacle_voxels.shape[0],), dtype=bool)
        if (obstacle_voxels.shape[0] == 0 or boxes.shape[0] == 0 or
                self.obstacle_box_ignore_margin <= 0):
            return mask
        margin = self.obstacle_box_ignore_margin
        centers = (
            self.pc_range[:3] +
            (obstacle_voxels.astype(np.float32) + 0.5) * self.voxel_size)
        for box_arr in boxes:
            length, width, height = box_arr[3:6]
            if length <= 0 or width <= 0 or height <= 0:
                continue
            yaw = box_arr[6]
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            rel_x = centers[:, 0] - box_arr[0]
            rel_y = centers[:, 1] - box_arr[1]
            local_x = rel_x * cos_yaw + rel_y * sin_yaw
            local_y = -rel_x * sin_yaw + rel_y * cos_yaw
            inside = (
                (np.abs(local_x) <= length * 0.5 + margin) &
                (np.abs(local_y) <= width * 0.5 + margin) &
                (centers[:, 2] >= box_arr[2] - margin) &
                (centers[:, 2] <= box_arr[2] + height + margin))
            mask |= inside
        return mask

    def split_scene_ground_obstacle(
            self, scene_points_xyz: np.ndarray,
            scene_point_voxels: np.ndarray,
            scene_hit_voxels: np.ndarray,
            semantic_voxels: np.ndarray) -> Tuple[np.ndarray, np.ndarray,
                                                  np.ndarray]:
        if scene_hit_voxels.shape[0] == 0 or scene_points_xyz.shape[0] == 0:
            empty = np.empty((0, 3), dtype=np.int64)
            return empty, empty, empty

        ground_est, raw_ground_xy = self.estimate_ground_height(
            scene_points_xyz, scene_point_voxels)
        center_z = (self.pc_range[2] +
                    (scene_hit_voxels[:, 2].astype(np.float32) + 0.5) *
                    self.voxel_size[2])
        cell_ground = ground_est[scene_hit_voxels[:, 0],
                                 scene_hit_voxels[:, 1]]
        is_ground = (
            np.isfinite(cell_ground) &
            (center_z <= cell_ground + self.ground_height_threshold))
        observed_ground = scene_hit_voxels[is_ground]
        raw_obstacle = scene_hit_voxels[~is_ground]
        obstacle = self.filter_obstacle_voxels(
            raw_obstacle, scene_point_voxels)

        if self.fill_ground:
            # Old code counted finite ground neighbours per cell with an
            # occ_x*occ_y double loop; ``windowed_count`` does the same square
            # neighbourhood count in one integral-image (summed-area) pass.
            neighbor_counts = windowed_count(
                raw_ground_xy, self.ground_fill_radius)
            fill_xy = (
                (neighbor_counts >= self.ground_fill_min_neighbors) &
                np.isfinite(ground_est))
            xy = np.argwhere(fill_xy)
            z_idx = np.floor(
                (ground_est[fill_xy] - self.pc_range[2]) /
                self.voxel_size[2]).astype(np.int64)
            valid = (z_idx >= 0) & (z_idx < self.occ_size[2])
            ground = np.column_stack([xy[valid], z_idx[valid]])
        else:
            ground = observed_ground

        if self.remove_ground_under_obstacle and ground.shape[0] > 0:
            # Drop ground voxels whose (x,y) column is blocked by an obstacle
            # or in-box semantic voxel. Replaces the per-voxel Python ``set``
            # membership test with a flattened ``x*occ_y + y`` key + np.isin.
            occ_y = int(self.occ_size[1])
            blocked_keys = obstacle[:, 0] * occ_y + obstacle[:, 1]
            if semantic_voxels.shape[0] > 0:
                blocked_keys = np.concatenate([
                    blocked_keys,
                    semantic_voxels[:, 0] * occ_y + semantic_voxels[:, 1]])
            ground_keys = ground[:, 0] * occ_y + ground[:, 1]
            keep = ~np.isin(ground_keys, blocked_keys)
            ground = ground[keep]
        return ground, obstacle, raw_obstacle

    def voxels_to_bev(self, voxels: np.ndarray) -> np.ndarray:
        mask = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
        if voxels.shape[0] == 0:
            return mask
        rows = self.bev_h - 1 - voxels[:, 1]
        cols = voxels[:, 0]
        valid = (
            (rows >= 0) & (rows < self.bev_h) &
            (cols >= 0) & (cols < self.bev_w))
        mask[rows[valid], cols[valid]] = 1
        return mask

    def build(self, points: np.ndarray, boxes: np.ndarray) -> dict:
        pts = np.asarray(points[:, :3], dtype=np.float32)
        point_voxels = self.coord_to_index_floor(pts)
        valid = np.all((point_voxels >= 0) & (point_voxels < self.occ_size),
                       axis=1)
        pts = pts[valid]
        point_voxels = point_voxels[valid]
        if point_voxels.shape[0] == 0:
            empty = np.zeros((self.bev_h, self.bev_w), dtype=np.uint8)
            return dict(ground=empty, obstacle=empty, free=empty,
                        blocked=empty)

        hit_voxels = np.unique(point_voxels, axis=0)
        free_voxels = self.raycast_free_voxels(hit_voxels)
        box_mask = self.box_interior_mask(boxes)
        point_in_box = box_mask[
            point_voxels[:, 0], point_voxels[:, 1], point_voxels[:, 2]]
        scene_points = pts[~point_in_box]
        scene_point_voxels = point_voxels[~point_in_box]
        hit_in_box = box_mask[
            hit_voxels[:, 0], hit_voxels[:, 1], hit_voxels[:, 2]]
        scene_hit_voxels = hit_voxels[~hit_in_box]
        semantic_voxels = hit_voxels[hit_in_box]

        ground_voxels, obstacle_voxels, _ = self.split_scene_ground_obstacle(
            scene_points, scene_point_voxels, scene_hit_voxels,
            semantic_voxels)
        if obstacle_voxels.shape[0] > 0:
            near_box = self.obstacle_near_box_mask(obstacle_voxels, boxes)
            obstacle_voxels = obstacle_voxels[~near_box]
            if obstacle_voxels.shape[0] > 0:
                obstacle_voxels = self.filter_obstacle_voxels(
                    obstacle_voxels, scene_point_voxels)

        ground_mask = self.voxels_to_bev(ground_voxels)
        obstacle_mask = self.voxels_to_bev(obstacle_voxels)
        semantic_mask = self.voxels_to_bev(semantic_voxels)
        free_mask = self.voxels_to_bev(free_voxels)
        blocked_mask = np.maximum(obstacle_mask, semantic_mask).astype(
            np.uint8)
        ego_ignore_mask = self.ego_ignore_bev_mask()
        if ego_ignore_mask.any():
            ground_mask[ego_ignore_mask > 0] = 0
            obstacle_mask[ego_ignore_mask > 0] = 0
            free_mask[ego_ignore_mask > 0] = 0
            blocked_mask[ego_ignore_mask > 0] = 0

        return dict(
            ground=ground_mask,
            obstacle=obstacle_mask,
            free=free_mask,
            blocked=blocked_mask)


@PIPELINES.register_module()
class GenerateKLDrivableMapLabels:
    """Generate Pansegformer-style drivable-space labels for KL LiDAR data."""

    def __init__(self,
                 map_file: Optional[str] = None,
                 point_cloud_range: Sequence[float] = None,
                 bev_size: Sequence[int] = None,
                 clean_map_file: Optional[str] = None,
                 use_map: bool = True,
                 drivable_label: int = 3,
                 include_lanes: bool = True,
                 include_roads: bool = True,
                 include_junctions: bool = True,
                 include_negative: bool = True,
                 include_parking_lot: bool = False,
                 augment_raycast_ground: bool = True,
                 keep_raycast_obstacles: bool = False,
                 current_frame_only: bool = True,
                 box_z_origin: str = 'center',
                 raycast_occ_size: Optional[Sequence[int]] = None,
                 raycast_ground_height_threshold: float = 0.55,
                 raycast_ground_smooth_radius: int = 3,
                 raycast_fill_ground: bool = True,
                 raycast_ground_fill_radius: int = 2,
                 raycast_ground_fill_min_neighbors: int = 5,
                 raycast_remove_ground_under_obstacle: bool = True,
                 raycast_obstacle_min_points_per_voxel: int = 2,
                 raycast_obstacle_min_component_voxels: int = 8,
                 raycast_obstacle_small_component_keep_min_points: int = 300,
                 raycast_obstacle_thin_component_min_major_span: float = 4.0,
                 raycast_obstacle_thin_component_max_minor_span: float = 2.4,
                 raycast_obstacle_thin_component_max_z_span: float = 1.6,
                 raycast_obstacle_thin_component_keep_min_points: int = 300,
                 raycast_obstacle_box_ignore_margin: float = 0.8,
                 raycast_ego_ignore_range: Optional[Sequence[float]] = (
                     -8.0, -2.0, -2.0, 8.0, 2.0, 6.0)):
        self.map_file = osp.expanduser(map_file) if map_file else None
        self.clean_map_file = (
            osp.expanduser(clean_map_file) if clean_map_file else None)
        self.point_cloud_range = np.asarray(point_cloud_range,
                                            dtype=np.float32)
        self.bev_size = tuple(int(v) for v in bev_size)
        self.use_map = bool(use_map)
        self.drivable_label = int(drivable_label)
        self.augment_raycast_ground = bool(augment_raycast_ground)
        self.keep_raycast_obstacles = bool(keep_raycast_obstacles)
        self.current_frame_only = bool(current_frame_only)
        self.box_z_origin = box_z_origin
        # When use_map is False the navigation HD-map is dropped entirely:
        # the drivable target is built from the LiDAR raycast ground alone.
        # A navigation map describes route topology, not the true drivable
        # surface, so it can mislabel the seg-head target.
        if not self.use_map:
            self.drivable_global = GeometryCollection()
            self.map_stats = None
        elif self.clean_map_file and osp.exists(self.clean_map_file):
            self.drivable_global = load_clean_drivable_geometry(
                self.clean_map_file)
            self.map_stats = None
        else:
            if not self.map_file:
                raise ValueError(
                    'map_file (or clean_map_file) is required when '
                    'use_map=True.')
            self.drivable_global, self.map_stats = build_drivable_geometry(
                self.map_file,
                include_lanes=include_lanes,
                include_roads=include_roads,
                include_junctions=include_junctions,
                include_negative=include_negative,
                include_parking_lot=include_parking_lot)
        self.raycast_builder = RaycastDrivableBuilder(
            self.point_cloud_range,
            self.bev_size,
            occ_size=raycast_occ_size,
            ground_height_threshold=raycast_ground_height_threshold,
            ground_smooth_radius=raycast_ground_smooth_radius,
            fill_ground=raycast_fill_ground,
            ground_fill_radius=raycast_ground_fill_radius,
            ground_fill_min_neighbors=raycast_ground_fill_min_neighbors,
            remove_ground_under_obstacle=raycast_remove_ground_under_obstacle,
            obstacle_min_points_per_voxel=raycast_obstacle_min_points_per_voxel,
            obstacle_min_component_voxels=raycast_obstacle_min_component_voxels,
            obstacle_small_component_keep_min_points=(
                raycast_obstacle_small_component_keep_min_points),
            obstacle_thin_component_min_major_span=(
                raycast_obstacle_thin_component_min_major_span),
            obstacle_thin_component_max_minor_span=(
                raycast_obstacle_thin_component_max_minor_span),
            obstacle_thin_component_max_z_span=(
                raycast_obstacle_thin_component_max_z_span),
            obstacle_thin_component_keep_min_points=(
                raycast_obstacle_thin_component_keep_min_points),
            obstacle_box_ignore_margin=raycast_obstacle_box_ignore_margin,
            ego_ignore_range=raycast_ego_ignore_range)

    @staticmethod
    def _empty_targets(bev_size: Sequence[int]):
        h, w = int(bev_size[0]), int(bev_size[1])
        return (
            torch.zeros((0,), dtype=torch.long),
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0, h, w), dtype=torch.uint8))

    def _targets_from_mask(self, mask: np.ndarray):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return self._empty_targets(mask.shape)
        labels = torch.tensor([self.drivable_label], dtype=torch.long)
        bboxes = torch.tensor([[xs.min(), ys.min(), xs.max(), ys.max()]],
                              dtype=torch.float32)
        masks = torch.from_numpy(mask[None].astype(np.uint8))
        return labels, bboxes, masks

    @staticmethod
    def _points_numpy(points):
        tensor = points.tensor if hasattr(points, 'tensor') else points
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return np.asarray(tensor, dtype=np.float32)

    def _boxes_numpy(self, boxes):
        if boxes is None:
            return np.zeros((0, 7), dtype=np.float32)
        tensor = boxes.tensor if hasattr(boxes, 'tensor') else boxes
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        arr = np.asarray(tensor, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 7), dtype=np.float32)
        arr = arr[:, :7].copy()
        if self.box_z_origin == 'center':
            arr[:, 2] -= arr[:, 5] * 0.5
        elif self.box_z_origin != 'bottom':
            raise ValueError(f'Unsupported box_z_origin={self.box_z_origin}')
        return arr

    def _ego2global(self, results):
        if 'ego2global' in results:
            return np.asarray(results['ego2global'], dtype=np.float64)
        if 'l2g_r_mat' in results and 'l2g_t' in results:
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = np.asarray(results['l2g_r_mat'], dtype=np.float64)
            mat[:3, 3] = np.asarray(results['l2g_t'], dtype=np.float64)
            return mat
        return np.eye(4, dtype=np.float64)

    def __call__(self, results):
        if (self.current_frame_only and
                not results.get('_kl_is_current_frame', True)):
            labels, bboxes, masks = self._empty_targets(self.bev_size)
            results['gt_lane_labels'] = labels
            results['gt_lane_bboxes'] = bboxes
            results['gt_lane_masks'] = masks
            return results

        if self.use_map:
            map_mask = build_map_mask(
                self.drivable_global,
                self._ego2global(results),
                self.point_cloud_range,
                self.bev_size)
        else:
            map_mask = np.zeros(self.bev_size, dtype=np.uint8)
        final_mask = map_mask
        if self.augment_raycast_ground and 'points' in results:
            raycast = self.raycast_builder.build(
                self._points_numpy(results['points']),
                self._boxes_numpy(results.get('gt_bboxes_3d')))
            final_mask = np.maximum(map_mask, raycast['ground']).astype(
                np.uint8)
            if not self.keep_raycast_obstacles:
                final_mask[raycast['blocked'] > 0] = 0

        labels, bboxes, masks = self._targets_from_mask(final_mask)
        results['gt_lane_labels'] = labels
        results['gt_lane_bboxes'] = bboxes
        results['gt_lane_masks'] = masks
        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(use_map={self.use_map}, '
            f'map_file={self.map_file}, '
            f'clean_map_file={self.clean_map_file}, '
            f'bev_size={self.bev_size}, '
            f'augment_raycast_ground={self.augment_raycast_ground})')
