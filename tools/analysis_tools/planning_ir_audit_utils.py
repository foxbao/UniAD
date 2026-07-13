import json
import math
import os

import numpy as np


EVAL_HORIZON_INDICES = (1, 3, 5)


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return to_numpy(value[0])
    if hasattr(value, 'tensor'):
        value = value.tensor
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def read_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f'Invalid JSON at {path}:{line_number}: {exc}') from exc
    return records


def write_jsonl(path, records):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        for record in records:
            handle.write(json.dumps(
                record, ensure_ascii=False, separators=(',', ':')))
            handle.write('\n')


def extract_json_object(text):
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError('model output does not contain a valid JSON object')


def planning_mask(value):
    mask = to_numpy(value)
    if mask is None:
        return None
    if mask.ndim >= 2:
        mask = mask.any(axis=-1).reshape(-1)
    else:
        mask = mask.reshape(-1)
    return mask.astype(bool)


def planning_trajectory(value):
    trajectory = to_numpy(value)
    if trajectory is None:
        return None
    return trajectory.reshape(-1, trajectory.shape[-1])


def _wrap_pi(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _heading_change_deg(points, min_segment_disp=0.05):
    if len(points) < 2:
        return 0.0
    points = np.concatenate([
        np.zeros((1, 2), dtype=np.float64), points[:, :2]], axis=0)
    deltas = np.diff(points, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.where(norms >= min_segment_disp)[0]
    if len(valid) < 2:
        return 0.0
    first = deltas[valid[0]]
    last = deltas[valid[-1]]
    first_yaw = math.atan2(float(first[1]), float(first[0]))
    last_yaw = math.atan2(float(last[1]), float(last[0]))
    return abs(math.degrees(_wrap_pi(last_yaw - first_yaw)))


def _yaw_change_deg(trajectory, valid):
    if trajectory.shape[1] < 3:
        return 0.0
    indices = np.where(valid[:len(trajectory)])[0]
    if len(indices) < 2:
        return 0.0
    yaw = trajectory[indices, 2]
    if not np.isfinite(yaw).all():
        return 0.0
    return abs(math.degrees(_wrap_pi(float(yaw[-1] - yaw[0]))))


def _lateral_ratio(points):
    if len(points) < 2:
        return 0.0
    endpoint = points[-1, :2]
    distance = float(np.linalg.norm(endpoint))
    if distance < 1e-6:
        return 0.0
    direction = endpoint / distance
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    lateral = float(np.max(np.abs(points[:, :2] @ normal)))
    return lateral / distance


def motion_bucket(gt_plan, valid):
    indices = np.where(valid[:len(gt_plan)])[0]
    if len(indices) == 0:
        return 'invalid'
    last = int(indices[-1])
    points = gt_plan[:last + 1][valid[:last + 1]]
    final_disp = float(np.linalg.norm(points[-1, :2]))
    if final_disp < 0.5:
        return 'static'
    if final_disp < 2.0:
        return 'slow'
    turn_angle = max(
        _heading_change_deg(points),
        _yaw_change_deg(gt_plan[:last + 1], valid[:last + 1]))
    if turn_angle >= 15.0 or _lateral_ratio(points) >= 0.15:
        return 'turning'
    return 'moving_straight'


def horizon_l2(trajectory, gt_xy, valid,
               horizon_indices=EVAL_HORIZON_INDICES):
    values = []
    for index in horizon_indices:
        if index >= len(trajectory) or index >= len(gt_xy) or index >= len(valid):
            values.append(None)
        elif not valid[index]:
            values.append(None)
        else:
            values.append(float(np.linalg.norm(
                trajectory[index, :2] - gt_xy[index, :2])))
    finite = [value for value in values if value is not None]
    return values, (float(np.mean(finite)) if finite else None)


def front_obstacle_bucket(segmentation, pc_range, cell_size=0.8,
                          x_range=(0.0, 30.0), y_abs=6.0):
    segmentation = to_numpy(segmentation)
    if segmentation is None:
        return 'unknown'
    if segmentation.ndim == 4:
        segmentation = segmentation[0]
    if segmentation.ndim != 3:
        return 'unknown'
    x0, y0 = float(pc_range[0]), float(pc_range[1])
    height, width = segmentation.shape[-2:]
    x_min, x_max = x_range
    y_min, y_max = -float(y_abs), float(y_abs)
    c0 = max(0, int(math.floor((x_min - x0) / cell_size)))
    c1 = min(width, int(math.ceil((x_max - x0) / cell_size)) + 1)
    r0 = max(0, int(math.floor((y_min - y0) / cell_size)))
    r1 = min(height, int(math.ceil((y_max - y0) / cell_size)) + 1)
    if c0 >= c1 or r0 >= r1:
        return 'front_clear'
    future = segmentation[1:] if len(segmentation) > 1 else segmentation
    return ('front_obstacle' if future[:, r0:r1, c0:c1].any()
            else 'front_clear')


def horizon_collisions(trajectory, segmentation, pc_range, cell_size=0.8,
                       ego_width=3.0, ego_length=14.6,
                       horizon_indices=EVAL_HORIZON_INDICES):
    segmentation = to_numpy(segmentation)
    if segmentation is None:
        return [None] * len(horizon_indices)
    if segmentation.ndim == 4:
        segmentation = segmentation[0]
    if segmentation.ndim != 3:
        return [None] * len(horizon_indices)

    try:
        from skimage.draw import polygon as draw_polygon
    except ImportError as exc:
        raise ImportError('skimage is required for collision audit') from exc

    x0, y0 = float(pc_range[0]), float(pc_range[1])
    height, width = segmentation.shape[-2:]
    corners = np.array([
        [+ego_length / 2.0, -ego_width / 2.0],
        [+ego_length / 2.0, +ego_width / 2.0],
        [-ego_length / 2.0, +ego_width / 2.0],
        [-ego_length / 2.0, -ego_width / 2.0],
    ], dtype=np.float32) / float(cell_size)
    row_offsets, col_offsets = draw_polygon(corners[:, 1], corners[:, 0])

    output = []
    for index in horizon_indices:
        if index >= len(trajectory):
            output.append(None)
            continue
        seg_index = min(index + 1, len(segmentation) - 1)
        x, y = trajectory[index, :2]
        center_col = int(round((float(x) - x0) / cell_size))
        center_row = int(round((float(y) - y0) / cell_size))
        rows = row_offsets + center_row
        cols = col_offsets + center_col
        inside = ((rows >= 0) & (rows < height)
                  & (cols >= 0) & (cols < width))
        collision = bool(
            inside.any() and segmentation[seg_index, rows[inside], cols[inside]].any())
        output.append(collision)
    return output
