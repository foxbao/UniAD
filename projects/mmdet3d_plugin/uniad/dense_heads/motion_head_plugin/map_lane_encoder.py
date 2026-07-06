"""TensorRT-friendly HD-map lane encoder for LiDAR planning deployment."""

import os
import re
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from projects.mmdet3d_plugin.models.utils.functional import (
    norm_points, pos2posemb2d)


class HDMapParser:
    """Parse protobuf-text HD map lanes once during export/model build."""

    def __init__(self, map_path: str, num_points_per_lane: int = 20):
        self.map_path = map_path
        self.num_points_per_lane = num_points_per_lane
        if not os.path.exists(map_path):
            raise FileNotFoundError(f'HD map file not found: {map_path}')
        self.lanes = self._load_or_parse()
        if not self.lanes:
            raise ValueError(f'HD map {map_path} parsed to 0 lanes.')

    def _cache_path(self):
        return self.map_path + '.parsed.npz'

    def _load_or_parse(self):
        cache = self._cache_path()
        if os.path.exists(cache):
            data = np.load(cache, allow_pickle=True)
            return list(data['lanes'])
        return self._parse_map()

    @staticmethod
    def _extract_points(block: str) -> np.ndarray:
        pts = []
        for match in re.finditer(
                r'point\s*\{\s*x:\s*([-\d.e+]+)\s*y:\s*([-\d.e+]+)\s*\}',
                block):
            pts.append([float(match.group(1)), float(match.group(2))])
        return np.asarray(pts, dtype=np.float32) if pts else np.zeros((0, 2))

    @staticmethod
    def _resample(pts: np.ndarray, num_points: int) -> np.ndarray:
        if len(pts) < 2:
            return np.zeros((num_points, 2), dtype=np.float32)
        diffs = np.diff(pts, axis=0)
        seg_lens = np.linalg.norm(diffs, axis=1)
        cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
        total = cum_len[-1]
        if total < 1e-6:
            return np.tile(pts[0], (num_points, 1)).astype(np.float32)
        targets = np.linspace(0.0, total, num_points)
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

            lanes.append(dict(
                central=self._resample(central, n),
                left=self._resample(left, n) if len(left) > 1 else None,
                right=self._resample(right, n) if len(right) > 1 else None))
        return lanes


def _lane_arrays(lanes: List[dict], num_points: int):
    central = []
    left = []
    right = []
    left_valid = []
    right_valid = []
    for lane in lanes:
        central.append(lane['central'])
        if lane['left'] is None:
            left.append(np.zeros((num_points, 2), dtype=np.float32))
            left_valid.append(False)
        else:
            left.append(lane['left'])
            left_valid.append(True)
        if lane['right'] is None:
            right.append(np.zeros((num_points, 2), dtype=np.float32))
            right_valid.append(False)
        else:
            right.append(lane['right'])
            right_valid.append(True)
    return (
        np.asarray(central, dtype=np.float32),
        np.asarray(left, dtype=np.float32),
        np.asarray(right, dtype=np.float32),
        np.asarray(left_valid, dtype=np.bool_),
        np.asarray(right_valid, dtype=np.bool_),
    )


class MapLaneEncoderTRT(nn.Module):
    """Encode static HD-map lanes with ONNX/TensorRT-friendly tensor ops."""

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

        parser = HDMapParser(map_path, num_points_per_lane)
        if len(parser.lanes) < num_lanes:
            raise ValueError(
                f'HD map has {len(parser.lanes)} lanes, fewer than '
                f'num_lanes={num_lanes}.')
        central, left, right, left_valid, right_valid = _lane_arrays(
            parser.lanes, num_points_per_lane)
        # FP16 numerical hygiene: HD-map points live in the global frame
        # (~2000-3340 m here). Subtracting the ego global translation (same
        # magnitude) inside the network is catastrophic cancellation in FP16
        # (ULP ~2 m at 3000), which poisons the whole map branch. We shift both
        # operands by a constant origin (the map centroid) so the tensors that
        # enter the graph are ~100 m instead of ~3000 m. This is math-identical:
        #   (p - t) == (p - o) - (t - o)
        # The map-point side (p - o) is baked here in float64 -> exact; only the
        # 2-element (t - o) happens at runtime, and we keep it in fp32.
        origin = central.reshape(-1, 2).astype(np.float64).mean(axis=0)
        origin = origin.astype(np.float32)
        central = (central - origin).astype(np.float32)
        left = (left - origin).astype(np.float32)
        right = (right - origin).astype(np.float32)
        self.register_buffer('map_central', torch.from_numpy(central))
        self.register_buffer('map_left', torch.from_numpy(left))
        self.register_buffer('map_right', torch.from_numpy(right))
        self.register_buffer('map_left_valid', torch.from_numpy(left_valid))
        self.register_buffer('map_right_valid', torch.from_numpy(right_valid))
        self.register_buffer('map_origin', torch.from_numpy(origin))
        self.register_buffer('pc_range_tensor',
                             torch.tensor(pc_range, dtype=torch.float32))

        self.point_mlp = nn.Sequential(
            nn.Linear(7, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128))
        self.lane_proj = nn.Sequential(
            nn.Linear(128, embed_dims),
            nn.LayerNorm(embed_dims))

    def _transform_to_ego(self, points, l2g_r_mat, l2g_t):
        if l2g_r_mat.dim() == 3:
            rot = l2g_r_mat[0]
        else:
            rot = l2g_r_mat
        if l2g_t.dim() == 2:
            trans = l2g_t[0]
        else:
            trans = l2g_t
        rot2 = rot[:2, :2].to(dtype=points.dtype)
        # `points` already carry the constant origin shift baked in __init__.
        # Compute (trans - origin) in fp32 (both ~3000 m, so this subtraction
        # must not run in fp16) and only then cast to the working dtype. The
        # tensor that feeds the graph is now O(100 m), not O(3000 m).
        trans2 = (trans[:2].to(dtype=self.map_origin.dtype) - self.map_origin)
        trans2 = trans2.to(dtype=points.dtype)
        return (points - trans2).matmul(rot2)

    def forward(self, l2g_r_mat, l2g_t):
        dtype = l2g_r_mat.dtype
        central = self.map_central.to(dtype=dtype)
        left = self.map_left.to(dtype=dtype)
        right = self.map_right.to(dtype=dtype)

        central_ego = self._transform_to_ego(central, l2g_r_mat, l2g_t)
        left_ego = self._transform_to_ego(left, l2g_r_mat, l2g_t)
        right_ego = self._transform_to_ego(right, l2g_r_mat, l2g_t)

        cmin = central_ego.min(dim=1).values
        cmax = central_ego.max(dim=1).values
        xmin = self.pc_range_tensor[0].to(dtype=dtype)
        ymin = self.pc_range_tensor[1].to(dtype=dtype)
        xmax = self.pc_range_tensor[3].to(dtype=dtype)
        ymax = self.pc_range_tensor[4].to(dtype=dtype)
        valid = ((cmax[:, 0] >= xmin) & (cmin[:, 0] <= xmax) &
                 (cmax[:, 1] >= ymin) & (cmin[:, 1] <= ymax))
        mid = central_ego[:, self.num_points_per_lane // 2]
        dist = torch.linalg.norm(mid, dim=-1)
        # Keep the invalid-lane sentinel finite in FP16. 1e8 overflows to inf
        # during TensorRT FP16 build and can poison downstream TopK/attention.
        dist = torch.where(valid, dist, torch.full_like(dist, 1.0e4))
        _, top_idx = torch.topk(
            dist, k=self.num_lanes, dim=0, largest=False, sorted=True)

        central_sel = central_ego.index_select(0, top_idx)
        left_sel = left_ego.index_select(0, top_idx)
        right_sel = right_ego.index_select(0, top_idx)
        valid_sel = valid.index_select(0, top_idx)
        left_valid = self.map_left_valid.index_select(0, top_idx)
        right_valid = self.map_right_valid.index_select(0, top_idx)

        tangent_body = central_sel[:, 1:] - central_sel[:, :-1]
        tangent_last = tangent_body[:, -1:]
        tangent = torch.cat([tangent_body, tangent_last], dim=1)
        tangent = tangent / torch.linalg.norm(
            tangent, dim=-1, keepdim=True).clamp_min(1.0e-6)

        seg_lens = torch.linalg.norm(tangent_body, dim=-1)
        cum_len = torch.cat([
            seg_lens.new_zeros((self.num_lanes, 1)),
            torch.cumsum(seg_lens, dim=1)
        ], dim=1)
        total = cum_len[:, -1:].clamp_min(1.0e-6)
        s_norm = cum_len / total

        left_off = torch.linalg.norm(left_sel - central_sel, dim=-1)
        right_off = torch.linalg.norm(right_sel - central_sel, dim=-1)
        left_off = left_off * left_valid[:, None].to(dtype)
        right_off = right_off * right_valid[:, None].to(dtype)

        features = torch.cat([
            central_sel,
            tangent,
            s_norm.unsqueeze(-1),
            left_off.unsqueeze(-1),
            right_off.unsqueeze(-1),
        ], dim=-1)

        point_feat = self.point_mlp(features)
        lane_feat = point_feat.max(dim=1).values
        lane_query = self.lane_proj(lane_feat)
        valid_f = valid_sel[:, None].to(dtype)
        lane_query = lane_query * valid_f

        centroids = mid.index_select(0, top_idx) * valid_f
        centroids_norm = norm_points(
            centroids[None], self.pc_range_tensor.to(dtype=dtype))[0]
        lane_query_pos = pos2posemb2d(centroids_norm) * valid_f
        lane_centroids = centroids

        return (lane_query[None], lane_query_pos[None], valid_sel[None],
                lane_centroids[None])
