#!/usr/bin/env python
"""Run resumable, sharded data preparation for the B15 train split."""

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / (
    'documents/patent_2026_occ/'
    'kl_occworld_full_train3_final30_manifest_v1.json')
DEFAULT_DISK_ROOT = Path(
    '/mnt/disk1/baojiali/UniAD_occworld_full_train_v1')
WORK_DIR = REPO_ROOT / (
    'projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b15_full_train_continuous10')
CHECKPOINT_DIR = Path(
    '/mnt/disk1/baojiali/UniAD_occworld_checkpoints/'
    'projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_b15_full_train_continuous10')


def _load_manifest(path: Path) -> dict:
    with path.open() as source:
        manifest = json.load(source)
    if manifest.get('status') not in {
            'frozen_before_full_gt_generation',
            'ready_after_full_generation_and_audit'}:
        raise ValueError(
            f'Unexpected B15 manifest status: {manifest.get("status")}')
    return manifest


def _safe_directory_link(target: Path, source: Path):
    source.mkdir(parents=True, exist_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise ValueError(f'Wrong symlink target: {target}')
        return
    if target.exists():
        if target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            raise FileExistsError(
                f'Refusing to replace existing path: {target}')
    target.symlink_to(source, target_is_directory=True)


def setup_paths(manifest: dict, disk_root: Path):
    disk_root.mkdir(parents=True, exist_ok=True)
    for relative in manifest['roots'].values():
        target = REPO_ROOT / relative
        source = disk_root / Path(relative).name
        _safe_directory_link(target, source)
    _safe_directory_link(WORK_DIR, CHECKPOINT_DIR)
    for name in ('logs', 'temporal', 'occlusion', 'dual',
                 'observation_cache', 'sequence', 'history'):
        (disk_root / 'shards' / name).mkdir(
            parents=True, exist_ok=True)


def _scene_shards(records: Sequence[dict], count: int):
    """Balance scene groups while keeping each scene in one worker."""
    explicit = [record.get('preparation_shard_index') for record in records]
    if explicit and all(value is not None for value in explicit):
        invalid = sorted({
            int(value) for value in explicit
            if not 0 <= int(value) < count
        })
        if invalid:
            raise ValueError(
                f'Explicit preparation shards are incompatible with '
                f'worker count {count}: {invalid}')
        shards = [[] for _ in range(count)]
        scene_to_shard = {}
        for record, value in zip(records, explicit):
            scene = str(record['scene_token'])
            shard_index = int(value)
            previous = scene_to_shard.setdefault(scene, shard_index)
            if previous != shard_index:
                raise ValueError(
                    f'Scene {scene} spans explicit preparation shards')
            shards[shard_index].append(int(record['reference_index']))
        return [sorted(values) for values in shards]
    grouped = {}
    for record in records:
        grouped.setdefault(str(record['scene_token']), []).append(
            int(record['reference_index']))
    shards = [[] for _ in range(count)]
    loads = [0] * count
    groups = sorted(
        grouped.values(), key=lambda values: (-len(values), min(values)))
    for values in groups:
        shard_index = min(range(count), key=lambda index: (loads[index], index))
        shards[shard_index].extend(sorted(values))
        loads[shard_index] += len(values)
    return [sorted(values) for values in shards]


def _run(command: Sequence[str], log_path: Path, dry_run: bool):
    printable = ' '.join(command)
    if dry_run:
        print(printable)
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w') as log:
        log.write(printable + '\n')
        log.flush()
        result = subprocess.run(
            command, cwd=str(REPO_ROOT), stdout=log,
            stderr=subprocess.STDOUT, env={
                **os.environ,
                'PYTHONPATH': str(REPO_ROOT),
            })
    if result.returncode:
        raise RuntimeError(
            f'Command failed with exit {result.returncode}; see {log_path}')


def _parallel(worker, shard_values, max_workers: int,
              selected_indices=None):
    selected = (
        set(range(len(shard_values))) if selected_indices is None
        else set(selected_indices))
    invalid = sorted(selected.difference(range(len(shard_values))))
    if invalid:
        raise ValueError(f'Invalid shard indices: {invalid}')
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers) as executor:
        futures = [
            executor.submit(worker, shard_index, values)
            for shard_index, values in enumerate(shard_values)
            if values and shard_index in selected
        ]
        for future in concurrent.futures.as_completed(futures):
            future.result()


def _link_files(sources: Iterable[Path], target: Path, pattern: str):
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sources:
        for path in sorted(source.glob(pattern)):
            link = target / path.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(path.resolve())
            count += 1
    return count


def _link_reference_dirs(sources: Iterable[Path], target: Path):
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for source in sources:
        for path in sorted(source.iterdir()):
            if not path.is_dir() or not path.name.isdigit():
                continue
            link = target / path.name
            if link.exists() or link.is_symlink():
                continue
            link.symlink_to(path.resolve(), target_is_directory=True)
            count += 1
    return count


def run_base_labels(manifest: dict, disk_root: Path,
                    worker_count: int, dry_run: bool,
                    selected_indices=None):
    shard_values = _scene_shards(
        manifest['splits']['train'], worker_count)

    def worker(shard_index, indices):
        temporal = disk_root / 'shards/temporal' / f'shard_{shard_index:02d}'
        occlusion = disk_root / 'shards/occlusion' / f'shard_{shard_index:02d}'
        dual = disk_root / 'shards/dual' / f'shard_{shard_index:02d}'
        index_args = [str(index) for index in indices]
        _run([
            sys.executable,
            'tools/data_converter/generate_kl_occworld_temporal_labels.py',
            '--indices', *index_args,
            '--save-per-sensor-diagnostics',
            '--out-dir', str(temporal),
        ], disk_root / f'logs/base_temporal_{shard_index:02d}.log',
            dry_run)
        _run([
            sys.executable,
            'tools/analysis_tools/audit_kl_occworld_occlusion_batch.py',
            '--temporal-dir', str(temporal),
            '--out-dir', str(occlusion),
        ], disk_root / f'logs/base_occlusion_{shard_index:02d}.log',
            dry_run)
        _run([
            sys.executable,
            'tools/analysis_tools/audit_kl_occworld_dual_batch.py',
            '--temporal-dir', str(temporal),
            '--occlusion-dir', str(occlusion),
            '--indices', *index_args,
            '--out-dir', str(dual),
        ], disk_root / f'logs/base_dual_{shard_index:02d}.log', dry_run)

    _parallel(
        worker, shard_values, worker_count,
        selected_indices=selected_indices)
    if dry_run:
        return
    root = lambda key: REPO_ROOT / manifest['roots'][key]
    temporal_shards = [
        disk_root / 'shards/temporal' / f'shard_{index:02d}'
        for index in range(worker_count)
    ]
    occlusion_shards = [
        disk_root / 'shards/occlusion' / f'shard_{index:02d}'
        for index in range(worker_count)
    ]
    dual_shards = [
        disk_root / 'shards/dual' / f'shard_{index:02d}'
        for index in range(worker_count)
    ]
    _link_files(temporal_shards, root('temporal'), '*__temporal.npz')
    _link_files(occlusion_shards, root('occlusion'), '*__occlusion.npz')
    _link_files(dual_shards, root('dual'), '*__dual_audit.npz')


def run_cross_scene(manifest: dict, disk_root: Path, dry_run: bool):
    reference_args = [
        str(record['reference_index'])
        for record in manifest['splits']['train']
    ]
    command = [
        sys.executable,
        'tools/analysis_tools/audit_kl_occworld_dual_cross_scene.py',
        '--dual-dir', str(REPO_ROOT / manifest['roots']['dual']),
        '--indices', *reference_args,
        '--out-dir', str(
            REPO_ROOT / manifest['roots']['dual_cross_scene']),
    ]
    support = manifest.get('cross_scene_support')
    if support is not None:
        command.extend([
            '--support-dual-dir', str(
                REPO_ROOT / support['dual_dir']),
            '--support-indices', *[
                str(index) for index in support['reference_indices']
            ],
        ])
    _run(command, disk_root / 'logs/cross_scene.log', dry_run)


def run_sequence_history(manifest: dict, disk_root: Path,
                         worker_count: int, dry_run: bool,
                         selected_indices=None):
    shard_values = _scene_shards(
        manifest['splits']['train'], worker_count)
    cross_scene = REPO_ROOT / manifest['roots']['dual_cross_scene']

    def worker(shard_index, indices):
        cache = (
            disk_root / 'shards/observation_cache' /
            f'shard_{shard_index:02d}')
        sequence = (
            disk_root / 'shards/sequence' / f'shard_{shard_index:02d}')
        history = (
            disk_root / 'shards/history' / f'shard_{shard_index:02d}')
        index_args = [str(index) for index in indices]
        _run([
            sys.executable,
            'tools/data_converter/generate_kl_occworld_observation_cache.py',
            '--reference-indices', *index_args,
            '--offsets', *[str(index) for index in range(-4, 9)],
            '--out-dir', str(cache),
        ], disk_root / f'logs/cache_{shard_index:02d}.log', dry_run)
        _run([
            sys.executable,
            'tools/data_converter/generate_kl_occworld_sequence_batch.py',
            '--reference-indices', *index_args,
            '--dual-dir', str(cross_scene),
            '--observation-cache-dir', str(cache),
            '--out-dir', str(sequence),
            '--fail-fast',
        ], disk_root / f'logs/sequence_{shard_index:02d}.log', dry_run)
        _run([
            sys.executable,
            'tools/data_converter/generate_kl_occworld_history_batch.py',
            '--reference-indices', *index_args,
            '--observation-cache-dir', str(cache),
            '--out-dir', str(history),
        ], disk_root / f'logs/history_{shard_index:02d}.log', dry_run)

    _parallel(
        worker, shard_values, worker_count,
        selected_indices=selected_indices)
    if dry_run:
        return
    cache_shards = [
        disk_root / 'shards/observation_cache' / f'shard_{index:02d}'
        for index in range(worker_count)
    ]
    sequence_shards = [
        disk_root / 'shards/sequence' / f'shard_{index:02d}'
        for index in range(worker_count)
    ]
    history_shards = [
        disk_root / 'shards/history' / f'shard_{index:02d}'
        for index in range(worker_count)
    ]
    _link_files(
        cache_shards,
        REPO_ROOT / manifest['roots']['observation_cache'],
        '*__observation.npz')
    _link_reference_dirs(
        sequence_shards, REPO_ROOT / manifest['roots']['sequence'])
    _link_reference_dirs(
        history_shards, REPO_ROOT / manifest['roots']['history'])


def run_audit(manifest: dict, disk_root: Path, dry_run: bool):
    sequence_root = REPO_ROOT / manifest['roots']['sequence']
    history_root = REPO_ROOT / manifest['roots']['history']
    sequence_audit = REPO_ROOT / manifest['roots']['sequence_audit']
    history_audit = REPO_ROOT / manifest['roots']['history_audit']
    _run([
        sys.executable,
        'tools/analysis_tools/audit_kl_occworld_sequence_batch.py',
        '--sequence-root', str(sequence_root),
        '--out-dir', str(sequence_audit),
    ], disk_root / 'logs/sequence_audit.log', dry_run)
    _run([
        sys.executable,
        'tools/analysis_tools/audit_kl_occworld_history_batch.py',
        '--history-root', str(history_root),
        '--sequence-root', str(sequence_root),
        '--out-dir', str(history_audit),
    ], disk_root / 'logs/history_audit.log', dry_run)
    if dry_run:
        return
    expected = int(manifest['train_reference_count'])
    sequence_count = len(list(
        sequence_root.glob('*/*__occworld_sequence.npz')))
    history_count = len(list(
        history_root.glob('*/*__occworld_history.npz')))
    if sequence_count != expected or history_count != expected:
        raise ValueError(
            f'Artifact count mismatch: expected={expected}, '
            f'sequence={sequence_count}, history={history_count}')
    manifest['status'] = 'ready_after_full_generation_and_audit'
    manifest['full_generation_audit'] = {
        'sequence_count': sequence_count,
        'history_count': history_count,
        'sequence_audit': manifest['roots']['sequence_audit'],
        'history_audit': manifest['roots']['history_audit'],
        'final_holdout_generated': False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        '--disk-root', type=Path, default=DEFAULT_DISK_ROOT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument(
        '--only-shards', type=int, nargs='+',
        help='Run only selected zero-based shards for resumable recovery.')
    parser.add_argument(
        '--phase', choices=(
            'setup', 'base-labels', 'cross-scene',
            'sequence-history', 'audit', 'all'),
        default='setup')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    manifest_path = args.manifest.resolve()
    manifest = _load_manifest(manifest_path)
    setup_paths(manifest, args.disk_root)
    phases = (
        ['base-labels', 'cross-scene', 'sequence-history', 'audit']
        if args.phase == 'all' else [args.phase])
    for phase in phases:
        if phase == 'setup':
            continue
        if phase == 'base-labels':
            run_base_labels(
                manifest, args.disk_root, args.workers, args.dry_run,
                selected_indices=args.only_shards)
        elif phase == 'cross-scene':
            run_cross_scene(manifest, args.disk_root, args.dry_run)
        elif phase == 'sequence-history':
            run_sequence_history(
                manifest, args.disk_root, args.workers, args.dry_run,
                selected_indices=args.only_shards)
        elif phase == 'audit':
            run_audit(manifest, args.disk_root, args.dry_run)
    if (not args.dry_run and
            manifest.get('status') ==
            'ready_after_full_generation_and_audit'):
        with manifest_path.open('w') as output:
            json.dump(manifest, output, ensure_ascii=False, indent=2)
            output.write('\n')
    print(json.dumps({
        'phase': args.phase,
        'dry_run': args.dry_run,
        'workers': args.workers,
        'train_reference_count': manifest['train_reference_count'],
        'manifest_status': manifest['status'],
        'disk_root': str(args.disk_root),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
