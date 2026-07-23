#!/usr/bin/env python
"""Render point-cloud comparisons for selected KL dual-label audits."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2

from tools.analysis_tools.audit_kl_occworld_dual_representation import _tile
from tools.analysis_tools.audit_kl_occworld_future_reveal import (
    _save_pointcloud_bev,
)
from tools.data_converter.generate_kl_occworld_labels import (
    MultiLidarOccLabelBuilder,
    _load_infos,
    _output_stem,
    _resolve_path,
)


def _read(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--ann-file', default='data/kl_8/kl_infos_train.pkl')
    parser.add_argument('--dual-dir', required=True)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--pc-range', type=float, nargs=6,
        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size', type=int, nargs=2, default=[120, 160])
    parser.add_argument('--occ-size', type=int, nargs=3,
                        default=[160, 120, 10])
    parser.add_argument('--collision-z', type=float, nargs=2,
                        default=[0.3, 2.5])
    return parser.parse_args()


def main():
    args = parse_args()
    infos, metainfo = _load_infos(_resolve_path(args.ann_file))
    dual_dir = Path(args.dual_dir)
    if args.indices is None:
        with (dual_dir / 'summary.json').open() as f:
            indices = json.load(f)['largest8_frame_indices']
    else:
        indices = list(dict.fromkeys(args.indices))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_frame = str(metainfo.get('lidar_coord_frame', 'FLU'))
    builder = MultiLidarOccLabelBuilder(
        args.pc_range, args.bev_size, args.occ_size,
        target_frame=target_frame, collision_z=args.collision_z)

    rows = []
    for index in indices:
        stem = _output_stem(infos[index])
        current = builder.build(
            infos[index], diagnostics=False, return_points=True)
        point_path = out_dir / f'{stem}__pointcloud_bev.png'
        _save_pointcloud_bev(
            point_path, current['points'], args.pc_range,
            tuple(args.bev_size), args.collision_z)
        paths = [
            dual_dir / f'{stem}__K.png',
            point_path,
            dual_dir / f'{stem}__non_drivable_raw.png',
            dual_dir / f'{stem}__traversability.png',
        ]
        titles = (
            f'#{index} K observation',
            'current 8-LiDAR points',
            'raw opposing-boundary evidence',
            'filtered traversability',
        )
        panels = [
            _tile(_read(path), title, (320, 240))
            for path, title in zip(paths, titles)
        ]
        compare_path = out_dir / f'{stem}__pointcloud_compare.png'
        compare = cv2.hconcat(panels)
        cv2.imwrite(str(compare_path), compare)
        rows.append(compare)
        print(f'[{index}] points={current["points"].shape[0]} '
              f'comparison={compare_path.name}')

    cv2.imwrite(
        str(out_dir / 'largest8_pointcloud_compare.png'),
        cv2.vconcat(rows))
    print(f'contact={out_dir / "largest8_pointcloud_compare.png"}')


if __name__ == '__main__':
    main()
