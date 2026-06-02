"""HD Map lane encoder for MotionFormer.

Parses a protobuf-text HD map file once at init, then per-frame transforms
lane segments into ego coordinates and encodes them as lane_query tensors
for the existing MapInteraction cross-attention layers.
"""

import hashlib
import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points, pos2posemb2d)


class HDMapParser:
    """Parse protobuf-text HD map and cache lane polylines."""

    def __init__(self, map_path: str, num_points_per_lane: int = 20):
        self.map_path = map_path
        self.num_points_per_lane = num_points_per_lane
        self.lanes = self._load_or_parse()

    def _cache_path(self) -> str:
        return self.map_path + '.parsed.npz'

    def _load_or_parse(self) -> List[dict]:
        cache = self._cache_path()
        map_mtime = os.path.getmtime(self.map_path)
        if os.path.exists(cache):
            data = np.load(cache, allow_pickle=True)
            if float(data.get('mtime', 0)) == map_mtime:
                return list(data['lanes'])
        lanes = self._parse_map()
        np.savez(cache, lanes=np.array(lanes, dtype=object),
                 mtime=map_mtime)
        return lanes

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

            # PLACEHOLDER_PARSE_CONTINUE
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
                central=central_resampled,
                left=left_resampled,
                right=right_resampled))
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


class MapLaneEncoder(nn.Module):
    """Encode HD map lanes into lane_query for MotionFormer."""

    def __init__(self,
                 map_path: str,
                 pc_range: List[float],
                 embed_dims: int = 256,
                 num_lanes: int = 64,
                 num_points_per_lane: int = 20):
        super().__init__()
        self.pc_range = pc_range
        self.embed_dims = embed_dims
        self.num_lanes = num_lanes
        self.num_points_per_lane = num_points_per_lane

        self.parser = HDMapParser(map_path, num_points_per_lane)

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

        return lane_query[None], lane_query_pos[None]
