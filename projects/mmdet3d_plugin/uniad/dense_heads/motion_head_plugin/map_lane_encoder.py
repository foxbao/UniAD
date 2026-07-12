"""HD Map lane encoder for MotionFormer.

Parses a protobuf-text HD map file once at init, then per-frame transforms
lane segments into ego coordinates and encodes them as lane_query tensors
for the existing MapInteraction cross-attention layers.
"""

import hashlib
import os
import math
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points, pos2posemb2d)


class HDMapParser:
    """Parse protobuf-text HD map and cache lane polylines."""

    CACHE_SCHEMA_VERSION = 2

    def __init__(self, map_path: str, num_points_per_lane: int = 20):
        self.map_path = map_path
        self.num_points_per_lane = num_points_per_lane
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f'HD map file not found: {map_path}. The lane encoder cannot '
                'build lane_query without it.')
        self.lanes = self._load_or_parse()
        if not self.lanes:
            raise ValueError(
                f'HD map {map_path} parsed to 0 lanes. Check the file format '
                '(expected protobuf-text with `lane {{ central_curve ... }}` '
                'blocks); otherwise the encoder silently emits all-invalid '
                'lanes and MapLaneEncoder degrades to a no-op.')
        self.lane_by_id = {
            str(lane['id']): lane for lane in self.lanes
            if lane.get('id') is not None
        }

    def _cache_path(self) -> str:
        return f'{self.map_path}.parsed.{self.num_points_per_lane}.npz'

    def _load_or_parse(self) -> List[dict]:
        cache = self._cache_path()
        map_mtime = os.path.getmtime(self.map_path)
        if os.path.exists(cache):
            data = np.load(cache, allow_pickle=True)
            schema_version = int(data.get('schema_version', 0))
            if (float(data.get('mtime', 0)) == map_mtime
                    and schema_version == self.CACHE_SCHEMA_VERSION):
                return list(data['lanes'])
        lanes = self._parse_map()
        np.savez(cache, lanes=np.array(lanes, dtype=object),
                 mtime=map_mtime,
                 schema_version=self.CACHE_SCHEMA_VERSION)
        return lanes

    @staticmethod
    def _extract_id(block: str, field: str) -> Optional[str]:
        match = re.search(
            rf'\b{re.escape(field)}\s*\{{\s*id:\s*"([^"]+)"',
            block)
        return match.group(1) if match else None

    @staticmethod
    def _extract_ids(block: str, field: str) -> List[str]:
        return re.findall(
            rf'\b{re.escape(field)}\s*\{{\s*id:\s*"([^"]+)"',
            block)

    @staticmethod
    def _extract_float(block: str, field: str,
                       default: float = 0.0) -> float:
        match = re.search(
            rf'\b{re.escape(field)}:\s*([-\d.e+]+)', block)
        return float(match.group(1)) if match else float(default)

    @staticmethod
    def _extract_enum(block: str, field: str,
                      default: str = '') -> str:
        match = re.search(
            rf'\b{re.escape(field)}:\s*([A-Za-z0-9_]+)', block)
        return match.group(1) if match else default

    @staticmethod
    def _extract_bool(block: str, field: str,
                      default: bool = False) -> bool:
        match = re.search(
            rf'\b{re.escape(field)}:\s*(true|false)', block,
            flags=re.IGNORECASE)
        return match.group(1).lower() == 'true' if match else default

    def _extract_points(self, block: str) -> np.ndarray:
        pts = []
        for m in re.finditer(
                r'point\s*\{\s*x:\s*([-\d.e+]+)\s*y:\s*([-\d.e+]+)\s*\}',
                block):
            pts.append([float(m.group(1)), float(m.group(2))])
        return np.array(pts, dtype=np.float32) if pts else np.zeros((0, 2))

    def _resample(self, pts: np.ndarray, n: int) -> np.ndarray:
        if len(pts) < 2:
            return np.zeros((n, 2), dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
        total = cum_len[-1]
        if total < 1e-6:
            return np.tile(pts[0], (n, 1)).astype(np.float32)
        targets = np.linspace(0, total, n)
        x_interp = np.interp(targets, cum_len, pts[:, 0])
        y_interp = np.interp(targets, cum_len, pts[:, 1])
        return np.stack([x_interp, y_interp], axis=1).astype(np.float32)

    def _parse_map(self) -> List[dict]:
        with open(self.map_path, 'r') as f:
            content = f.read()
        lanes = []
        n = self.num_points_per_lane
        for block in content.split('lane {')[1:]:
            depth = 0
            end_idx = 0
            for i, ch in enumerate(block):
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth < 0:
                        end_idx = i
                        break
            lane_block = block[:end_idx]

            cc_start = lane_block.find('central_curve')
            lb_start = lane_block.find('left_boundary')
            rb_start = lane_block.find('right_boundary')

            if cc_start < 0:
                continue
            cc_end = lb_start if lb_start > cc_start else len(lane_block)
            central = self._extract_points(lane_block[cc_start:cc_end])
            if len(central) < 2:
                continue

            left = np.zeros((0, 2), dtype=np.float32)
            right = np.zeros((0, 2), dtype=np.float32)
            if lb_start >= 0:
                lb_end = rb_start if rb_start > lb_start else len(lane_block)
                left = self._extract_points(lane_block[lb_start:lb_end])
            if rb_start >= 0:
                right = self._extract_points(lane_block[rb_start:])

            central_resampled = self._resample(central, n)
            left_resampled = self._resample(left, n) if len(left) > 1 else None
            right_resampled = (self._resample(right, n)
                               if len(right) > 1 else None)
            lanes.append(dict(
                id=self._extract_id(lane_block, 'id'),
                central=central_resampled,
                left=left_resampled,
                right=right_resampled,
                predecessor_ids=self._extract_ids(
                    lane_block, 'predecessor_id'),
                successor_ids=self._extract_ids(
                    lane_block, 'successor_id'),
                left_neighbor_forward_ids=self._extract_ids(
                    lane_block, 'left_neighbor_forward_lane_id'),
                right_neighbor_forward_ids=self._extract_ids(
                    lane_block, 'right_neighbor_forward_lane_id'),
                speed_limit=self._extract_float(
                    lane_block, 'speed_limit'),
                turn=self._extract_enum(lane_block, 'turn'),
                direction=self._extract_enum(lane_block, 'direction'),
                lane_type=self._extract_enum(lane_block, 'type'),
                is_reverse_road=self._extract_bool(
                    lane_block, 'is_reverse_road')))
        return lanes

    def _lateral_offset(self, central: np.ndarray,
                        boundary: Optional[np.ndarray]) -> np.ndarray:
        """Compute signed lateral distance from central to boundary."""
        if boundary is None:
            return np.zeros(len(central), dtype=np.float32)
        diff = boundary - central
        return np.linalg.norm(diff, axis=1).astype(np.float32)

    def crop_and_transform(
        self,
        ego2global: np.ndarray,
        pc_range: List[float],
        num_lanes: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transform lanes to ego frame, crop, and build features.

        Returns:
            features: (num_lanes, num_points_per_lane, 7)
            centroids: (num_lanes, 2) midpoint in ego frame
            valid: (num_lanes,) bool
        """
        g2e = np.linalg.inv(ego2global).astype(np.float32)
        R = g2e[:2, :2]
        t = g2e[:2, 3]
        n = self.num_points_per_lane
        xmin, ymin = pc_range[0], pc_range[1]
        xmax, ymax = pc_range[3], pc_range[4]

        candidates = []
        for lane in self.lanes:
            central_ego = lane['central'] @ R.T + t
            cmin = central_ego.min(axis=0)
            cmax = central_ego.max(axis=0)
            if cmax[0] < xmin or cmin[0] > xmax:
                continue
            if cmax[1] < ymin or cmin[1] > ymax:
                continue
            mid = central_ego[n // 2]
            dist = np.sqrt(mid[0]**2 + mid[1]**2)
            candidates.append((dist, central_ego, lane))

        candidates.sort(key=lambda x: x[0])
        candidates = candidates[:num_lanes]

        features = np.zeros((num_lanes, n, 7), dtype=np.float32)
        centroids = np.zeros((num_lanes, 2), dtype=np.float32)
        valid = np.zeros(num_lanes, dtype=bool)

        for i, (dist, central_ego, lane) in enumerate(candidates):
            # tangent direction
            tangent = np.zeros_like(central_ego)
            tangent[:-1] = central_ego[1:] - central_ego[:-1]
            tangent[-1] = tangent[-2]
            norms = np.linalg.norm(tangent, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-6, None)
            tangent = tangent / norms

            # arc length normalized
            seg_lens = np.linalg.norm(
                np.diff(central_ego, axis=0), axis=1)
            cum_len = np.concatenate([[0], np.cumsum(seg_lens)])
            total = cum_len[-1] if cum_len[-1] > 1e-6 else 1.0
            s_norm = (cum_len / total).astype(np.float32)

            # boundary offsets
            left_off = self._lateral_offset(
                central_ego,
                lane['left'] @ R.T + t if lane['left'] is not None else None)
            right_off = self._lateral_offset(
                central_ego,
                lane['right'] @ R.T + t if lane['right'] is not None else None)

            features[i, :, 0:2] = central_ego
            features[i, :, 2:4] = tangent
            features[i, :, 4] = s_norm
            features[i, :, 5] = left_off
            features[i, :, 6] = right_off
            centroids[i] = central_ego[n // 2]
            valid[i] = True

        return features, centroids, valid


class MapPlanningCandidateGenerator:
    """Build topology-path x speed-profile planning candidates at runtime."""

    def __init__(self,
                 map_path: str,
                 profiles_path: str,
                 planning_steps: int = 6,
                 num_points_per_lane: int = 80,
                 num_start_lanes: int = 8,
                 max_paths: int = 16,
                 path_margin: float = 5.0,
                 max_start_distance: float = 12.0,
                 max_heading_error_deg: float = 80.0,
                 max_depth: int = 8,
                 max_join_gap: float = 6.0,
                 lateral_offsets=(-2.0, -1.5, -1.0, -0.5, 0.0,
                                  0.5, 1.0, 1.5, 2.0)):
        if not os.path.exists(profiles_path):
            raise FileNotFoundError(
                f'Planning speed profiles not found: {profiles_path}. '
                'Run tools/data_converter/generate_planning_speed_profiles.py '
                'before enabling D1 candidates.')
        data = np.load(profiles_path, allow_pickle=True)
        profiles = np.asarray(data['profiles'], dtype=np.float64)
        if profiles.ndim != 2 or profiles.shape[1] < planning_steps:
            raise ValueError(
                f'Expected planning profiles [K,T>={planning_steps}], got '
                f'{profiles.shape}')
        self.parser = HDMapParser(map_path, num_points_per_lane)
        self.profiles = profiles[:, :planning_steps]
        self.planning_steps = int(planning_steps)
        self.num_start_lanes = int(num_start_lanes)
        self.max_paths = int(max_paths)
        self.path_margin = float(path_margin)
        self.max_start_distance = float(max_start_distance)
        self.max_heading_error_deg = float(max_heading_error_deg)
        self.max_depth = int(max_depth)
        self.max_join_gap = float(max_join_gap)
        self.lateral_offsets = tuple(float(x) for x in lateral_offsets)

    @staticmethod
    def _polyline_length(points):
        if len(points) < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())

    @staticmethod
    def _nearest_point_and_heading(points, position):
        index = int(np.linalg.norm(points - position[None], axis=1).argmin())
        left = max(0, index - 1)
        right = min(len(points) - 1, index + 1)
        tangent = points[right] - points[left]
        norm = float(np.linalg.norm(tangent))
        tangent = (tangent / norm if norm >= 1e-6
                   else np.array([1.0, 0.0], dtype=np.float64))
        distance = float(np.linalg.norm(points[index] - position))
        return index, distance, tangent

    def _select_start_lanes(self, ego2global):
        position = ego2global[:2, 3]
        heading = ego2global[:2, 0]
        heading = heading / max(float(np.linalg.norm(heading)), 1e-6)
        min_alignment = math.cos(math.radians(self.max_heading_error_deg))
        candidates = []
        nearest = []
        for lane in self.parser.lanes:
            lane_id = str(lane['id'])
            if lane_id not in self.parser.lane_by_id:
                continue
            points = np.asarray(lane['central'], dtype=np.float64)
            index, distance, tangent = self._nearest_point_and_heading(
                points, position)
            alignment = abs(float(np.dot(tangent, heading)))
            row = (distance + 5.0 * (1.0 - alignment), distance,
                   -alignment, lane_id, index)
            nearest.append(row)
            if (distance <= self.max_start_distance
                    and alignment >= min_alignment):
                candidates.append(row)
        if not candidates:
            candidates = nearest
        candidates.sort()
        oriented = []
        for score, distance, neg_alignment, lane_id, index in \
                candidates[:self.num_start_lanes]:
            lane_size = len(self.parser.lane_by_id[lane_id]['central'])
            oriented.append((score, distance, neg_alignment, lane_id,
                             index, False))
            oriented.append((score, distance, neg_alignment, lane_id,
                             lane_size - 1 - index, True))
        return oriented

    def _join_lane_points(self, points, extension):
        if len(points) == 0:
            return extension.copy()
        gap = float(np.linalg.norm(points[-1] - extension[0]))
        if gap > self.max_join_gap:
            return None
        start = 1 if gap < 0.2 else 0
        return np.concatenate([points, extension[start:]], axis=0)

    def _extend_lane_chain(self, lane_id, reverse, points, sequence,
                           required_length, output):
        if (self._polyline_length(points) >= required_length
                or len(sequence) >= self.max_depth):
            output.append((tuple(sequence), points))
            return
        lane = self.parser.lane_by_id[lane_id]
        next_field = 'predecessor_ids' if reverse else 'successor_ids'
        successors = [
            successor for successor in lane.get(next_field, [])
            if (successor in self.parser.lane_by_id
                and (successor, reverse) not in sequence)
        ]
        extended = False
        for successor in successors:
            extension = np.asarray(
                self.parser.lane_by_id[successor]['central'],
                dtype=np.float64)
            if reverse:
                extension = extension[::-1]
            joined = self._join_lane_points(points, extension)
            if joined is None:
                continue
            extended = True
            self._extend_lane_chain(
                successor, reverse, joined,
                sequence + [(successor, reverse)], required_length, output)
        if not extended:
            output.append((tuple(sequence), points))

    def _build_route_paths(self, ego2global):
        required_length = float(self.profiles.max()) + self.path_margin
        paths = []
        for (score, _distance, _neg_alignment, lane_id, start_index,
             reverse) in self._select_start_lanes(ego2global):
            points = np.asarray(
                self.parser.lane_by_id[lane_id]['central'],
                dtype=np.float64)
            if reverse:
                points = points[::-1]
            points = points[start_index:]
            if len(points) < 2:
                continue
            expanded = []
            self._extend_lane_chain(
                lane_id, reverse, points, [(lane_id, reverse)],
                required_length, expanded)
            for sequence, route_points in expanded:
                paths.append((score, sequence, route_points))
        paths.sort(key=lambda row: (row[0], row[1]))
        unique = []
        seen = set()
        for row in paths:
            if row[1] in seen:
                continue
            seen.add(row[1])
            unique.append(row)
            if len(unique) >= self.max_paths:
                break
        return unique

    @staticmethod
    def _interpolate_polyline(points, distances):
        segment = np.linalg.norm(np.diff(points, axis=0), axis=1)
        cumulative = np.concatenate([[0.0], np.cumsum(segment)])
        if cumulative[-1] < 1e-6:
            return np.repeat(points[:1], len(distances), axis=0)
        distances = np.clip(distances, 0.0, cumulative[-1])
        right = np.searchsorted(cumulative, distances, side='left')
        right = np.clip(right, 1, len(points) - 1)
        left = right - 1
        denom = np.maximum(cumulative[right] - cumulative[left], 1e-6)
        alpha = ((distances - cumulative[left]) / denom)[:, None]
        return points[left] * (1.0 - alpha) + points[right] * alpha

    @classmethod
    def _sample_path(cls, points, distances, ego2global):
        samples = cls._interpolate_polyline(points, distances)
        relative_global = samples - points[0][None]
        global2ego = np.linalg.inv(ego2global)
        return relative_global @ global2ego[:2, :2].T

    @staticmethod
    def _apply_lateral_offset(path, offset):
        if abs(offset) < 1e-8:
            return path.copy()
        points = np.concatenate(
            [np.zeros((1, 2), dtype=np.float64), path], axis=0)
        tangent = np.diff(points, axis=0)
        norms = np.linalg.norm(tangent, axis=1, keepdims=True)
        tangent = tangent / np.maximum(norms, 1e-6)
        normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=-1)
        distance = np.cumsum(norms[:, 0])
        progress = distance / max(float(distance[-1]), 1e-6)
        smooth = progress * progress * (3.0 - 2.0 * progress)
        return path + normal * (offset * smooth[:, None])

    def __call__(self, ego2global, device, dtype):
        if isinstance(ego2global, torch.Tensor):
            ego2global = ego2global.detach().cpu().numpy()
        ego2global = np.asarray(ego2global, dtype=np.float64)
        paths = self._build_route_paths(ego2global)
        candidates = []
        for _score, _sequence, points in paths:
            for profile in self.profiles:
                path = self._sample_path(points, profile, ego2global)
                for offset in self.lateral_offsets:
                    candidates.append(self._apply_lateral_offset(path, offset))
        if not candidates:
            empty = torch.zeros(
                (1, 0, self.planning_steps, 2), device=device, dtype=dtype)
            return empty, torch.zeros((1, 0), device=device, dtype=torch.bool)
        candidates = torch.from_numpy(
            np.asarray(candidates, dtype=np.float32)).to(
                device=device, dtype=dtype)
        valid = torch.ones(
            (1, candidates.size(0)), device=device, dtype=torch.bool)
        return candidates[None], valid


class MapLaneEncoder(nn.Module):
    """Encode HD map lanes into lane_query for MotionFormer."""

    def __init__(self,
                 map_path: str,
                 pc_range: List[float],
                 embed_dims: int = 256,
                 num_lanes: int = 64,
                 num_points_per_lane: int = 20,
                 planning_candidates=None):
        super().__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_lanes = num_lanes
        self.num_points_per_lane = num_points_per_lane

        self.parser = HDMapParser(map_path, num_points_per_lane)
        self.planning_candidate_generator = None
        if planning_candidates is not None:
            self.planning_candidate_generator = MapPlanningCandidateGenerator(
                map_path=map_path, **planning_candidates)

        self.point_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128))
        self.lane_proj = nn.Sequential(
            nn.Linear(128, embed_dims),
            nn.LayerNorm(embed_dims))

    def forward(self, ego2global, device, dtype):
        """Encode HD map lanes for current frame.

        Args:
            ego2global: (4, 4) tensor or numpy array.
            device: target torch device.
            dtype: target torch dtype.

        Returns:
            lane_query: (1, num_lanes, embed_dims)
            lane_query_pos: (1, num_lanes, embed_dims)
            lane_valid: (1, num_lanes) bool
            lane_centroids: (1, num_lanes, 2) ego-metric lane midpoints
                (zero for invalid lanes); used for per-agent local map
                attention (MTR-style K-nearest lane selection).
            lane_points: (1, num_lanes, num_points_per_lane, 2) ego-metric
                resampled lane centerlines, zeroed for invalid lanes. This is
                optional downstream geometry for planning-only lane-anchor
                diagnostics; existing query consumers can ignore it.
        """
        if isinstance(ego2global, torch.Tensor):
            e2g_np = ego2global.detach().cpu().numpy()
        else:
            e2g_np = np.asarray(ego2global, dtype=np.float64)

        features, centroids, valid = self.parser.crop_and_transform(
            e2g_np, self.pc_range, self.num_lanes)

        feat_t = torch.from_numpy(features).to(device=device, dtype=dtype)
        centroids_t = torch.from_numpy(centroids).to(
            device=device, dtype=dtype)
        valid_t = torch.from_numpy(valid).to(device=device)

        # Per-point MLP + max-pool
        # feat_t: (num_lanes, num_points, 7)
        point_feat = self.point_mlp(feat_t)  # (num_lanes, num_points, 128)
        lane_feat = point_feat.max(dim=1).values  # (num_lanes, 128)
        lane_query = self.lane_proj(lane_feat)  # (num_lanes, embed_dims)

        # Zero out invalid lanes
        lane_query = lane_query * valid_t[:, None].to(dtype)

        # Positional encoding from centroid
        pc_range_t = torch.tensor(
            self.pc_range, device=device, dtype=dtype)
        centroids_norm = norm_points(
            centroids_t[None], pc_range_t)[0]  # (num_lanes, 2)
        lane_query_pos = pos2posemb2d(centroids_norm)  # (num_lanes, 256)
        lane_query_pos = lane_query_pos * valid_t[:, None].to(dtype)

        # Ego-metric centroids, zeroed for invalid lanes, for K-nearest
        # per-agent lane selection downstream.
        lane_centroids = centroids_t * valid_t[:, None].to(dtype)
        lane_points = feat_t[..., 0:2] * valid_t[:, None, None].to(dtype)

        return (lane_query[None], lane_query_pos[None], valid_t[None],
                lane_centroids[None], lane_points[None])

    def build_planning_candidates(self, ego2global, device, dtype):
        if self.planning_candidate_generator is None:
            return None, None
        return self.planning_candidate_generator(
            ego2global, device=device, dtype=dtype)
