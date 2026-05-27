#!/usr/bin/env python
import argparse
import hashlib
import json
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from shapely.validation import explain_validity


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.datasets.pipelines.kl_drivable_label import (  # noqa: E402
    build_drivable_geometry, build_map_mask, keep_polygonal)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def geometry_count(geom) -> int:
    if geom is None or geom.is_empty:
        return 0
    if isinstance(geom, Polygon):
        return 1
    if isinstance(geom, MultiPolygon):
        return len(geom.geoms)
    if isinstance(geom, GeometryCollection):
        return sum(geometry_count(part) for part in geom.geoms)
    return 0


def default_output_path(map_file: Path) -> Path:
    return map_file.with_name(f'{map_file.stem}_drivable_clean.pkl')


def load_infos(info_file: Path):
    try:
        import mmcv
        payload = mmcv.load(str(info_file))
    except Exception:
        with info_file.open('rb') as f:
            payload = pickle.load(f)
    if isinstance(payload, dict):
        return payload.get('data_list') or payload.get('infos') or []
    return payload


def validate_pose_samples(geom, info_file: Path, pc_range, bev_size,
                          num_samples: int):
    infos = load_infos(info_file)
    if not infos:
        return dict(enabled=True, checked=0, failures=[],
                    warning='No infos found.')
    indices = np.linspace(0, len(infos) - 1,
                          min(num_samples, len(infos)),
                          dtype=np.int64)
    failures = []
    non_empty = 0
    for idx in indices:
        info = infos[int(idx)]
        ego2global = info.get('ego2global')
        if ego2global is None:
            failures.append(dict(index=int(idx), error='missing ego2global'))
            continue
        try:
            mask = build_map_mask(geom, np.asarray(ego2global,
                                                  dtype=np.float64),
                                  pc_range, bev_size)
            non_empty += int(mask.sum() > 0)
        except Exception as exc:
            failures.append(dict(index=int(idx), error=repr(exc)))
            if len(failures) >= 20:
                break
    return dict(enabled=True,
                info_file=str(info_file),
                checked=int(len(indices)),
                non_empty=int(non_empty),
                failures=failures)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare a cleaned KL drivable-space map artifact.')
    parser.add_argument('--map-file',
                        default='data/kl_8/map/base_map.txt',
                        help='KL base_map.txt path.')
    parser.add_argument('--out',
                        default=None,
                        help='Output clean map pkl path.')
    parser.add_argument('--report',
                        default=None,
                        help='Output JSON report path.')
    parser.add_argument('--include-lanes',
                        action='store_true',
                        default=True,
                        help='Include lane polygons.')
    parser.add_argument('--no-include-lanes',
                        dest='include_lanes',
                        action='store_false')
    parser.add_argument('--include-roads',
                        action='store_true',
                        default=True,
                        help='Include road polygons.')
    parser.add_argument('--no-include-roads',
                        dest='include_roads',
                        action='store_false')
    parser.add_argument('--include-junctions',
                        action='store_true',
                        default=True,
                        help='Include drivable junction polygons.')
    parser.add_argument('--no-include-junctions',
                        dest='include_junctions',
                        action='store_false')
    parser.add_argument('--include-negative',
                        action='store_true',
                        default=True,
                        help='Subtract NOT_DRIVABLE/BLOCKING geometry.')
    parser.add_argument('--no-include-negative',
                        dest='include_negative',
                        action='store_false')
    parser.add_argument('--include-parking-lot',
                        action='store_true',
                        help='Include parking lot polygons.')
    parser.add_argument('--info-file',
                        default=None,
                        help='Optional KL info pkl for pose validation.')
    parser.add_argument('--validate-samples',
                        type=int,
                        default=1000,
                        help='Number of info samples to validate.')
    parser.add_argument('--pc-range',
                        nargs=6,
                        type=float,
                        default=[-64.0, -48.0, -2.0, 64.0, 48.0, 6.0])
    parser.add_argument('--bev-size',
                        nargs=2,
                        type=int,
                        default=[120, 160],
                        help='BEV mask size as H W.')
    return parser.parse_args()


def main():
    args = parse_args()
    map_file = Path(args.map_file)
    out_file = Path(args.out) if args.out else default_output_path(map_file)
    report_file = (
        Path(args.report) if args.report else
        out_file.with_name(f'{out_file.stem}_report.json'))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.parent.mkdir(parents=True, exist_ok=True)

    geom, stats = build_drivable_geometry(
        str(map_file),
        include_lanes=args.include_lanes,
        include_roads=args.include_roads,
        include_junctions=args.include_junctions,
        include_negative=args.include_negative,
        include_parking_lot=args.include_parking_lot)
    geom = keep_polygonal(geom)
    if geom is None:
        geom = GeometryCollection()

    source_hash = file_sha256(map_file)
    payload = dict(
        version=1,
        source_map_file=str(map_file),
        source_sha256=source_hash,
        options=dict(
            include_lanes=args.include_lanes,
            include_roads=args.include_roads,
            include_junctions=args.include_junctions,
            include_negative=args.include_negative,
            include_parking_lot=args.include_parking_lot),
        drivable_global=geom,
        stats=asdict(stats))
    with out_file.open('wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    validation = dict(enabled=False)
    if args.info_file:
        validation = validate_pose_samples(
            geom,
            Path(args.info_file),
            args.pc_range,
            args.bev_size,
            args.validate_samples)

    report = dict(
        output=str(out_file),
        source_map_file=str(map_file),
        source_sha256=source_hash,
        final_valid=bool(geom.is_valid),
        final_validity=explain_validity(geom),
        final_area=float(geom.area),
        final_bounds=[float(v) for v in geom.bounds]
        if not geom.is_empty else [],
        final_polygon_count=geometry_count(geom),
        stats=asdict(stats),
        validation=validation)
    with report_file.open('w') as f:
        json.dump(report, f, indent=2, sort_keys=True)
        f.write('\n')

    print(f'Wrote clean map: {out_file}')
    print(f'Wrote report: {report_file}')
    print(f'valid={report["final_valid"]} '
          f'area={report["final_area"]:.3f} '
          f'polygons={report["final_polygon_count"]}')
    if validation.get('enabled'):
        print(f'validated={validation["checked"]} '
              f'failures={len(validation["failures"])}')


if __name__ == '__main__':
    main()
