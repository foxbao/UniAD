#!/usr/bin/env python
"""Planning final-displacement diagnostics for KL UniAD eval outputs.

The input is a pkl produced by tools/uniad_dist_eval.sh --out. The script
summarizes whether SDC planning is short by command, GT motion bucket, front
occupancy, and GT-distance thresholds.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import os.path as osp
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import mmcv
import numpy as np

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


COMMAND_NAME = {
    0: 'Right',
    1: 'Left',
    2: 'Straight',
}

BUCKET_ORDER = (
    'Overall',
    'Static',
    'Slow',
    'MovingStraight',
    'Turning',
)

FRONT_ORDER = (
    'FrontClear',
    'FrontObstacle',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Diagnose planning final displacement from eval pkl files.')
    parser.add_argument(
        '--results',
        nargs='+',
        required=True,
        help='One or more pkl files. Use NAME=path.pkl to set display names.')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--gt-far-thrs',
        nargs='*',
        type=float,
        default=[2.0, 3.0, 4.0],
        help='GT final displacement thresholds for far-motion splits.')
    parser.add_argument(
        '--ratio-min-gt-disp',
        type=float,
        default=0.5,
        help='Minimum GT final displacement used for ratio statistics.')
    parser.add_argument('--planning-steps', type=int, default=6)
    parser.add_argument(
        '--front-x-range',
        nargs=2,
        type=float,
        default=[0.0, 30.0],
        metavar=('X_MIN', 'X_MAX'))
    parser.add_argument('--front-y-abs', type=float, default=6.0)
    parser.add_argument(
        '--point-cloud-range',
        nargs=6,
        type=float,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument(
        '--min-n-warn',
        type=int,
        default=30,
        help='Mark cells below this N as low-sample diagnostics.')
    return parser.parse_args()


def parse_named_path(text: str) -> Tuple[str, str]:
    if '=' in text:
        name, path = text.split('=', 1)
        return name.strip(), path.strip()
    path = text.strip()
    parent = osp.basename(osp.dirname(path))
    stem = osp.splitext(osp.basename(path))[0]
    name = parent if parent else stem
    return name, path


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        if len(value) == 1:
            return to_numpy(value[0])
    if hasattr(value, 'detach'):
        return value.detach().cpu().numpy()
    if value.__class__.__name__ == 'DataContainer':
        return to_numpy(value.data)
    return np.asarray(value)


def first_plan_array(value, dims: int = 2):
    arr = to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim > dims and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > dims:
        arr = arr.reshape(-1, *arr.shape[-dims:])[0]
    return arr


def normalize_traj(value, channels: int = 2) -> Optional[np.ndarray]:
    arr = first_plan_array(value, dims=2)
    if arr is None or arr.ndim < 2:
        return None
    return np.asarray(arr[:, :channels], dtype=np.float64)


def normalize_mask(value, steps: int) -> np.ndarray:
    if value is None:
        return np.ones((steps, ), dtype=np.bool_)
    arr = first_plan_array(value, dims=2)
    if arr is None:
        return np.ones((steps, ), dtype=np.bool_)
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return np.ones((steps, ), dtype=np.bool_)
    if arr.ndim == 1:
        valid = arr > 0
    elif arr.shape[-1] > 1:
        valid = np.any(arr > 0, axis=-1)
    else:
        valid = arr.reshape(-1) > 0
    return valid.astype(np.bool_)


def normalize_segmentation(value) -> Optional[np.ndarray]:
    arr = to_numpy(value)
    if arr is None:
        return None
    arr = np.asarray(arr)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim > 3:
        arr = arr.reshape(-1, *arr.shape[-3:])[0]
    if arr.ndim != 3:
        return None
    return arr


def unwrap_results(data) -> List[dict]:
    if isinstance(data, dict):
        if 'bbox_results' in data:
            data = data['bbox_results']
        else:
            raise KeyError('Results dict does not contain bbox_results.')
    if not isinstance(data, (list, tuple)):
        raise TypeError(f'Expected list or dict results, got {type(data)}.')
    return list(data)


def get_planning_payload(sample: dict) -> Optional[dict]:
    if not isinstance(sample, dict):
        return None
    if 'planning' in sample:
        return sample['planning']
    if 'pts_bbox' in sample and isinstance(sample['pts_bbox'], dict):
        return sample['pts_bbox'].get('planning')
    return None


def wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def valid_points(traj: np.ndarray, valid: np.ndarray, steps: int) -> np.ndarray:
    steps = min(steps, len(traj), len(valid))
    if steps <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    return traj[:steps, :2][valid[:steps]]


def heading_change_deg(points: np.ndarray, min_segment_disp: float = 0.05) -> float:
    if len(points) < 2:
        return 0.0
    pts = np.concatenate(
        [np.zeros((1, 2), dtype=np.float64), points[:, :2]], axis=0)
    deltas = np.diff(pts, axis=0)
    norms = np.linalg.norm(deltas, axis=1)
    valid = np.where(norms >= min_segment_disp)[0]
    if len(valid) < 2:
        return 0.0
    v0 = deltas[valid[0]]
    v1 = deltas[valid[-1]]
    a0 = math.atan2(float(v0[1]), float(v0[0]))
    a1 = math.atan2(float(v1[1]), float(v1[0]))
    return abs(math.degrees(wrap_pi(a1 - a0)))


def yaw_change_deg(gt_plan: np.ndarray, valid: np.ndarray, steps: int) -> float:
    if gt_plan.shape[1] < 3:
        return 0.0
    steps = min(steps, len(gt_plan), len(valid))
    idx = np.where(valid[:steps])[0]
    if len(idx) < 2:
        return 0.0
    yaw = gt_plan[idx, 2]
    if not np.isfinite(yaw).all():
        return 0.0
    return abs(math.degrees(wrap_pi(float(yaw[-1] - yaw[0]))))


def lateral_ratio(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    end = points[-1, :2]
    net = float(np.linalg.norm(end))
    if net < 1e-6:
        return 0.0
    direction = end / net
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    lateral = float(np.max(np.abs(points[:, :2] @ normal)))
    return lateral / max(net, 1e-6)


def planning_bucket(gt_plan: np.ndarray,
                    valid: np.ndarray,
                    steps: int,
                    static_thr: float = 0.5,
                    slow_thr: float = 2.0,
                    turn_deg: float = 15.0,
                    turn_lat_ratio: float = 0.15) -> str:
    points = valid_points(gt_plan, valid, steps)
    if len(points) == 0:
        return 'UnknownBucket'
    final_disp = float(np.linalg.norm(points[-1, :2]))
    if final_disp < static_thr:
        return 'Static'
    if final_disp < slow_thr:
        return 'Slow'
    turn_angle = max(
        heading_change_deg(points),
        yaw_change_deg(gt_plan, valid, steps))
    if turn_angle >= turn_deg or lateral_ratio(points) >= turn_lat_ratio:
        return 'Turning'
    return 'MovingStraight'


def front_bucket(segmentation: Optional[np.ndarray],
                 point_cloud_range: Iterable[float],
                 x_range: Iterable[float],
                 y_abs: float) -> str:
    if segmentation is None:
        return 'FrontUnknown'
    seg = np.asarray(segmentation)
    if seg.ndim != 3:
        return 'FrontUnknown'
    _, height, width = seg.shape
    pcr = np.asarray(point_cloud_range, dtype=np.float64)
    x0, y0 = float(pcr[0]), float(pcr[1])
    cell_x = float(pcr[3] - pcr[0]) / max(width, 1)
    cell_y = float(pcr[4] - pcr[1]) / max(height, 1)
    x_min, x_max = [float(v) for v in x_range]
    y_min, y_max = -float(y_abs), float(y_abs)
    c0 = max(0, int(math.floor((x_min - x0) / cell_x)))
    c1 = min(width, int(math.ceil((x_max - x0) / cell_x)) + 1)
    r0 = max(0, int(math.floor((y_min - y0) / cell_y)))
    r1 = min(height, int(math.ceil((y_max - y0) / cell_y)) + 1)
    if c0 >= c1 or r0 >= r1:
        return 'FrontClear'
    future_seg = seg[1:] if seg.shape[0] > 1 else seg
    return 'FrontObstacle' if bool(future_seg[:, r0:r1, c0:c1].any()) \
        else 'FrontClear'


def command_name(value) -> str:
    if value is None:
        return 'CommandUnknown'
    arr = to_numpy(value)
    if arr is None:
        return 'CommandUnknown'
    try:
        cmd = int(np.asarray(arr).reshape(-1)[0])
    except Exception:
        return 'CommandUnknown'
    return COMMAND_NAME.get(cmd, f'Command{cmd}')


def extract_records(name: str,
                    samples: List[dict],
                    args: argparse.Namespace) -> List[dict]:
    records = []
    skipped_no_planning = 0
    skipped_invalid = 0
    for sample_idx, sample in enumerate(samples):
        planning = get_planning_payload(sample)
        if not isinstance(planning, dict):
            skipped_no_planning += 1
            continue
        result_planning = planning.get('result_planning', {})
        planning_gt = planning.get('planning_gt', {})
        pred = normalize_traj(result_planning.get('sdc_traj'), channels=2)
        gt = normalize_traj(planning_gt.get('sdc_planning'), channels=3)
        if pred is None:
            pred = normalize_traj(sample.get('planning_traj'), channels=2)
        if gt is None:
            gt = normalize_traj(sample.get('planning_traj_gt'), channels=3)
        if pred is None or gt is None:
            skipped_invalid += 1
            continue
        mask = normalize_mask(planning_gt.get('sdc_planning_mask'),
                              args.planning_steps)
        steps = min(args.planning_steps, len(pred), len(gt), len(mask))
        if steps <= 0 or not np.any(mask[:steps]):
            skipped_invalid += 1
            continue
        valid_idx = np.where(mask[:steps])[0]
        final_idx = int(valid_idx[-1])
        pred_disp = float(np.linalg.norm(pred[final_idx, :2]))
        gt_disp = float(np.linalg.norm(gt[final_idx, :2]))
        err = np.linalg.norm(pred[:steps, :2] - gt[:steps, :2], axis=-1)
        ade = float(np.mean(err[mask[:steps]]))
        fde = float(err[final_idx])
        ratio = pred_disp / max(gt_disp, 1e-6) \
            if gt_disp >= args.ratio_min_gt_disp else np.nan
        seg = normalize_segmentation(planning_gt.get('segmentation'))
        records.append(
            dict(
                model=name,
                sample_idx=sample_idx,
                command=command_name(
                    planning_gt.get('command', sample.get('command'))),
                bucket=planning_bucket(gt, mask, steps),
                front=front_bucket(seg, args.point_cloud_range,
                                   args.front_x_range, args.front_y_abs),
                pred_disp=pred_disp,
                gt_disp=gt_disp,
                ratio=ratio,
                ade=ade,
                fde=fde,
            ))
    if skipped_no_planning:
        print(f'[{name}] skipped {skipped_no_planning} samples without planning payload')
    if skipped_invalid:
        print(f'[{name}] skipped {skipped_invalid} invalid planning samples')
    return records


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float('nan')
    return float(np.percentile(values, q))


def summarize_cell(model: str,
                   group: str,
                   cell: str,
                   records: List[dict],
                   min_n_warn: int) -> dict:
    pred = np.asarray([r['pred_disp'] for r in records], dtype=np.float64)
    gt = np.asarray([r['gt_disp'] for r in records], dtype=np.float64)
    ratio = np.asarray([r['ratio'] for r in records], dtype=np.float64)
    ade = np.asarray([r['ade'] for r in records], dtype=np.float64)
    fde = np.asarray([r['fde'] for r in records], dtype=np.float64)
    ratio = ratio[np.isfinite(ratio)]
    q25 = percentile(ratio, 25)
    q75 = percentile(ratio, 75)
    return dict(
        model=model,
        group=group,
        cell=cell,
        N=len(records),
        N_ratio=int(ratio.size),
        low_N=int(len(records) < min_n_warn),
        pred_final_disp_mean=float(np.mean(pred)) if pred.size else float('nan'),
        gt_final_disp_mean=float(np.mean(gt)) if gt.size else float('nan'),
        final_disp_ratio_mean=float(np.mean(ratio)) if ratio.size else float('nan'),
        final_disp_ratio_median=float(np.median(ratio)) if ratio.size else float('nan'),
        final_disp_ratio_iqr=(q75 - q25) if ratio.size else float('nan'),
        ade_mean=float(np.mean(ade)) if ade.size else float('nan'),
        fde_mean=float(np.mean(fde)) if fde.size else float('nan'),
    )


def add_summary(rows: List[dict],
                model: str,
                group: str,
                cell: str,
                records: List[dict],
                min_n_warn: int) -> None:
    if not records:
        return
    rows.append(summarize_cell(model, group, cell, records, min_n_warn))


def summarize_records(records: List[dict],
                      gt_far_thrs: Iterable[float],
                      min_n_warn: int) -> List[dict]:
    rows = []
    by_model: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_model[rec['model']].append(rec)

    for model, model_records in sorted(by_model.items()):
        add_summary(rows, model, 'overall', 'Overall', model_records,
                    min_n_warn)

        for cmd in ('Left', 'Right', 'Straight'):
            add_summary(rows, model, 'command', cmd,
                        [r for r in model_records if r['command'] == cmd],
                        min_n_warn)

        for bucket in BUCKET_ORDER[1:]:
            add_summary(rows, model, 'bucket', bucket,
                        [r for r in model_records if r['bucket'] == bucket],
                        min_n_warn)

        for front in FRONT_ORDER:
            add_summary(rows, model, 'front', front,
                        [r for r in model_records if r['front'] == front],
                        min_n_warn)

        for bucket in ('MovingStraight', 'Turning'):
            for front in FRONT_ORDER:
                subset = [
                    r for r in model_records
                    if r['bucket'] == bucket and r['front'] == front
                ]
                add_summary(rows, model, 'bucket_front',
                            f'{bucket}/{front}', subset, min_n_warn)

        for thr in gt_far_thrs:
            far = [r for r in model_records if r['gt_disp'] >= thr]
            add_summary(rows, model, 'gt_far', f'GT>={thr:g}m', far,
                        min_n_warn)
            for front in FRONT_ORDER:
                subset = [r for r in far if r['front'] == front]
                add_summary(rows, model, 'front_gt_far',
                            f'{front}/GT>={thr:g}m', subset, min_n_warn)
            for bucket in ('MovingStraight', 'Turning'):
                for front in FRONT_ORDER:
                    subset = [
                        r for r in far
                        if r['bucket'] == bucket and r['front'] == front
                    ]
                    add_summary(rows, model, 'bucket_front_gt_far',
                                f'{bucket}/{front}/GT>={thr:g}m', subset,
                                min_n_warn)
    return rows


def fmt(value, digits=3) -> str:
    if value is None:
        return ''
    try:
        if not np.isfinite(float(value)):
            return 'nan'
        return f'{float(value):.{digits}f}'
    except Exception:
        return str(value)


def write_csv(path: str, rows: List[dict]) -> None:
    fieldnames = [
        'model', 'group', 'cell', 'N', 'N_ratio', 'low_N',
        'pred_final_disp_mean', 'gt_final_disp_mean',
        'final_disp_ratio_mean', 'final_disp_ratio_median',
        'final_disp_ratio_iqr', 'ade_mean', 'fde_mean',
    ]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, '') for key in fieldnames})


def write_markdown(path: str, rows: List[dict]) -> None:
    columns = [
        'model', 'group', 'cell', 'N', 'N_ratio',
        'pred_final_disp_mean', 'gt_final_disp_mean',
        'final_disp_ratio_mean', 'final_disp_ratio_median',
        'final_disp_ratio_iqr', 'ade_mean', 'fde_mean', 'low_N',
    ]
    with open(path, 'w', encoding='utf-8') as f:
        f.write('| ' + ' | '.join(columns) + ' |\n')
        f.write('| ' + ' | '.join(['---'] * len(columns)) + ' |\n')
        for row in rows:
            values = []
            for col in columns:
                value = row.get(col, '')
                if isinstance(value, float):
                    value = fmt(value)
                values.append(str(value))
            f.write('| ' + ' | '.join(values) + ' |\n')


def print_key_rows(rows: List[dict]) -> None:
    key_groups = {
        'overall', 'command', 'bucket', 'front', 'front_gt_far',
        'bucket_front_gt_far',
    }
    shown = [r for r in rows if r['group'] in key_groups]
    header = (
        f'{"model":28s} {"group":22s} {"cell":38s} '
        f'{"N":>6s} {"ratio":>7s} {"med":>7s} {"IQR":>7s} '
        f'{"pred":>7s} {"gt":>7s}')
    print(header)
    print('-' * len(header))
    for row in shown:
        print(
            f'{row["model"][:28]:28s} '
            f'{row["group"][:22]:22s} '
            f'{row["cell"][:38]:38s} '
            f'{int(row["N"]):6d} '
            f'{fmt(row["final_disp_ratio_mean"]):>7s} '
            f'{fmt(row["final_disp_ratio_median"]):>7s} '
            f'{fmt(row["final_disp_ratio_iqr"]):>7s} '
            f'{fmt(row["pred_final_disp_mean"]):>7s} '
            f'{fmt(row["gt_final_disp_mean"]):>7s}')


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    all_records = []
    for item in args.results:
        name, path = parse_named_path(item)
        if not osp.exists(path):
            raise FileNotFoundError(path)
        data = mmcv.load(path)
        samples = unwrap_results(data)
        records = extract_records(name, samples, args)
        print(f'[{name}] loaded {len(records)} planning samples from {path}')
        if not records:
            raise RuntimeError(
                f'No planning payload found in {path}. Re-run eval with a '
                'planning config and --out to save compact planning results.')
        all_records.extend(records)

    rows = summarize_records(all_records, args.gt_far_thrs, args.min_n_warn)
    csv_path = osp.join(args.out_dir, 'planning_disp_diagnostics.csv')
    md_path = osp.join(args.out_dir, 'planning_disp_diagnostics.md')
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)
    print_key_rows(rows)
    print(f'\nWrote: {csv_path}')
    print(f'Wrote: {md_path}')
    if any(row['low_N'] for row in rows):
        print(
            f'Note: low_N=1 means N < {args.min_n_warn}; treat those cells as '
            'directional only.')


if __name__ == '__main__':
    main()
