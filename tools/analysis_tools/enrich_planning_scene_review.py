#!/usr/bin/env python3
"""Join kl_8 scenes with chassis/planning context and camera evidence."""

import argparse
import bisect
import csv
import glob
import json
import os
import os.path as osp
import pickle
import re
import tempfile
from collections import Counter, defaultdict

from PIL import Image, ImageDraw, ImageFont, ImageOps


AUTO_MODES = {'COMPLETE_AUTO_DRIVE'}
MANUAL_MODES = {'COMPLETE_MANUAL', 'COMPLETE_MEDIAN'}
CAMERA_ORDER = (
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
)
CONTEXT_FIELDS = (
    'parsed_control_mode', 'control_mode_values',
    'control_mode_switch_count', 'control_mode_sequence',
    'control_mode_match_count', 'control_mode_coverage',
    'control_mode_max_dt_ms', 'control_mode_source',
    'origin_context_status', 'planning_match_count', 'planning_max_dt_ms',
    'planning_scenarios', 'planning_main_tasks', 'planning_stop_reasons',
    'camera_contact_sheet', 'camera_evidence_frames',
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infos', nargs='+', required=True,
        help='Inputs as SPLIT=path_to_with_cam.pkl.')
    parser.add_argument(
        '--origin-root', required=True,
        help='Directory containing PACKAGE_out directories.')
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument(
        '--data-root', default='.',
        help='Root used to resolve relative camera paths from info files.')
    parser.add_argument('--max-chassis-dt', type=float, default=0.10)
    parser.add_argument('--max-planning-dt', type=float, default=0.20)
    parser.add_argument('--camera-frames', type=int, default=3)
    parser.add_argument(
        '--contact-sheet-mode', choices=('none', 'matched', 'all'),
        default='matched')
    parser.add_argument('--cell-width', type=int, default=240)
    parser.add_argument('--cell-height', type=int, default=192)
    return parser.parse_args()


def parse_named_path(value):
    if '=' not in value:
        raise ValueError(f'Expected SPLIT=path, got {value}')
    split, path = value.split('=', 1)
    return split.strip(), path.strip()


def load_info_payload(path):
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if 'data_list' in payload:
            return payload['data_list']
        if 'infos' in payload:
            return payload['infos']
    if isinstance(payload, list):
        return payload
    raise TypeError(f'Unsupported info payload: {type(payload)}')


def load_scene_frames(named_paths):
    grouped = defaultdict(list)
    for split, path in named_paths.items():
        infos = load_info_payload(path)
        for info in infos:
            scene = str(info.get('scene_token', ''))
            cameras = (info.get('sync_info') or {}).get('cameras') or {}
            grouped[(split, scene)].append(dict(
                timestamp=float(info.get('timestamp', 0.0)),
                cameras={
                    name: dict(value) for name, value in cameras.items()
                    if isinstance(value, dict)
                },
            ))
        del infos
    for frames in grouped.values():
        frames.sort(key=lambda frame: frame['timestamp'])
    return grouped


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv_atomic(path, rows, fieldnames):
    directory = osp.dirname(path)
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.' + osp.basename(path) + '.', suffix='.tmp', dir=directory)
    try:
        with os.fdopen(
                descriptor, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if osp.exists(temporary):
            os.unlink(temporary)
        raise


def safe_name(value):
    value = re.sub(r'[^A-Za-z0-9_.-]+', '__', value)
    return value.strip('_') or 'scene'


def timestamped_files(directory):
    rows = []
    for path in glob.glob(osp.join(directory, '*.json')):
        stem = osp.splitext(osp.basename(path))[0]
        try:
            timestamp = float(stem)
        except ValueError:
            continue
        rows.append((timestamp, path))
    rows.sort()
    return rows


def nearest_timestamped_file(files, timestamp):
    if not files:
        return None, None
    index = bisect.bisect_left(files, (timestamp, ''))
    candidates = []
    if index < len(files):
        candidates.append(files[index])
    if index:
        candidates.append(files[index - 1])
    nearest = min(candidates, key=lambda row: abs(row[0] - timestamp))
    return nearest[1], abs(nearest[0] - timestamp)


def load_json_cached(path, cache):
    if path not in cache:
        with open(path, encoding='utf-8') as handle:
            cache[path] = json.load(handle)
    return cache[path]


def normalize_control_mode(raw_mode):
    if raw_mode in AUTO_MODES:
        return 'Auto'
    if raw_mode in MANUAL_MODES:
        return 'Manual'
    return 'Unknown'


def format_counter(counter):
    return ';'.join(
        f'{key}:{value}' for key, value in sorted(
            counter.items(), key=lambda item: str(item[0]))
        if key not in (None, ''))


def run_length_encode(values):
    runs = []
    for value in values:
        if not runs or runs[-1][0] != value:
            runs.append([value, 1])
        else:
            runs[-1][1] += 1
    return '|'.join(f'{value}:{count}' for value, count in runs)


def summarize_origin_context(frames, scene_dir, args):
    output = {field: '' for field in CONTEXT_FIELDS}
    output.update(
        parsed_control_mode='Unknown',
        control_mode_match_count=0,
        control_mode_coverage=0.0,
        planning_match_count=0,
        control_mode_source='chassis.driving_mode.name@kl_timestamp',
    )
    if not osp.isdir(osp.dirname(scene_dir)):
        output['origin_context_status'] = 'missing_package_out'
        return output
    if not osp.isdir(scene_dir):
        output['origin_context_status'] = 'missing_scene_out'
        return output

    chassis_files = timestamped_files(osp.join(scene_dir, 'chassis'))
    if not chassis_files:
        output['origin_context_status'] = 'missing_chassis'
        return output
    chassis_cache = {}
    raw_modes = Counter()
    normalized_modes = Counter()
    normalized_sequence = []
    chassis_deltas = []
    for frame in frames:
        path, delta = nearest_timestamped_file(
            chassis_files, frame['timestamp'])
        if path is None or delta > args.max_chassis_dt:
            continue
        payload = load_json_cached(path, chassis_cache)
        raw_mode = (payload.get('driving_mode') or {}).get('name') \
            or payload.get('autonomous_state')
        raw_modes[raw_mode or 'MISSING'] += 1
        normalized_mode = normalize_control_mode(raw_mode)
        normalized_modes[normalized_mode] += 1
        normalized_sequence.append(normalized_mode)
        chassis_deltas.append(delta)

    matched = sum(normalized_modes.values())
    known_modes = {
        mode for mode, count in normalized_modes.items()
        if count and mode != 'Unknown'
    }
    if len(known_modes) > 1:
        parsed_mode = 'Mixed'
    elif len(known_modes) == 1 and normalized_modes.get('Unknown', 0) == 0:
        parsed_mode = next(iter(known_modes))
    elif len(known_modes) == 1 and normalized_modes.get('Unknown', 0) > 0:
        parsed_mode = 'Mixed'
    else:
        parsed_mode = 'Unknown'
    output.update(
        parsed_control_mode=parsed_mode,
        control_mode_values=format_counter(raw_modes),
        control_mode_switch_count=sum(
            left != right for left, right in zip(
                normalized_sequence, normalized_sequence[1:])),
        control_mode_sequence=run_length_encode(normalized_sequence),
        control_mode_match_count=matched,
        control_mode_coverage=matched / max(len(frames), 1),
        control_mode_max_dt_ms=(
            max(chassis_deltas) * 1000.0 if chassis_deltas else ''),
        origin_context_status=(
            'ok' if matched == len(frames) else 'partial_chassis_match'),
    )

    planning_files = timestamped_files(osp.join(scene_dir, 'planning'))
    planning_cache = {}
    planning_deltas = []
    scenarios = Counter()
    main_tasks = Counter()
    stop_reasons = Counter()
    for frame in frames:
        path, delta = nearest_timestamped_file(
            planning_files, frame['timestamp'])
        if path is None or delta > args.max_planning_dt:
            continue
        payload = load_json_cached(path, planning_cache)
        scenarios[(payload.get('trajectory_scenario') or {}).get('name')] += 1
        main = ((payload.get('decision') or {}).get('main_decision') or {})
        main_tasks[main.get('task')] += 1
        stop_reasons[main.get('reason_code_name')] += 1
        planning_deltas.append(delta)
    output.update(
        planning_match_count=len(planning_deltas),
        planning_max_dt_ms=(
            max(planning_deltas) * 1000.0 if planning_deltas else ''),
        planning_scenarios=format_counter(scenarios),
        planning_main_tasks=format_counter(main_tasks),
        planning_stop_reasons=format_counter(stop_reasons),
    )
    return output


def representative_indices(length, count):
    if length <= 0 or count <= 0:
        return []
    if count == 1:
        return [length // 2]
    return sorted({
        int(round(index * (length - 1) / (count - 1)))
        for index in range(count)
    })


def camera_available_count(frame, data_root):
    return sum(
        osp.isfile(resolve_data_path(
            (frame['cameras'].get(camera) or {}).get('path'), data_root)
                   or '')
        for camera in CAMERA_ORDER)


def representative_camera_indices(frames, count, data_root):
    targets = representative_indices(len(frames), count)
    if not targets:
        return []
    boundaries = [0]
    boundaries.extend(
        (left + right) // 2 + 1
        for left, right in zip(targets, targets[1:]))
    boundaries.append(len(frames))
    selected = []
    for target, start, end in zip(targets, boundaries, boundaries[1:]):
        candidates = range(start, max(start + 1, end))
        selected.append(max(
            candidates,
            key=lambda index: (
                camera_available_count(frames[index], data_root),
                -abs(index - target)),
        ))
    return selected


def resolve_data_path(path, data_root):
    if not path:
        return None
    if osp.isabs(path):
        return path
    return osp.realpath(osp.join(data_root, path))


def fit_image(path, size):
    if path is None or not osp.isfile(path):
        return None
    with Image.open(path) as source:
        source = ImageOps.exif_transpose(source).convert('RGB')
        return ImageOps.contain(source, size, method=Image.Resampling.LANCZOS)


def render_contact_sheet(path, scene_token, frames, indices, args):
    cell_width = args.cell_width
    cell_height = args.cell_height
    column_header = 28
    row_header = 26
    width = cell_width * len(CAMERA_ORDER)
    height = column_header + len(indices) * (cell_height + row_header)
    sheet = Image.new('RGB', (width, height), (28, 32, 35))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, camera in enumerate(CAMERA_ORDER):
        x = column * cell_width
        draw.text((x + 7, 8), camera, fill=(235, 238, 240), font=font)
    for row_index, frame_index in enumerate(indices):
        frame = frames[frame_index]
        y_header = column_header + row_index * (cell_height + row_header)
        draw.text(
            (7, y_header + 7),
            f'{scene_token}  frame={frame_index}  t={frame["timestamp"]:.3f}',
            fill=(235, 238, 240), font=font)
        y = y_header + row_header
        for column, camera in enumerate(CAMERA_ORDER):
            x = column * cell_width
            metadata = frame['cameras'].get(camera) or {}
            source_path = resolve_data_path(
                metadata.get('path'), args.data_root)
            image = fit_image(source_path, (cell_width, cell_height))
            if image is None:
                draw.rectangle(
                    (x, y, x + cell_width - 1, y + cell_height - 1),
                    outline=(100, 105, 110))
                draw.text(
                    (x + 7, y + 7), 'missing', fill=(200, 120, 120),
                    font=font)
                continue
            paste_x = x + (cell_width - image.width) // 2
            paste_y = y + (cell_height - image.height) // 2
            sheet.paste(image, (paste_x, paste_y))
    os.makedirs(osp.dirname(path), exist_ok=True)
    sheet.save(path, quality=88, optimize=True)


def main():
    args = parse_args()
    named_paths = dict(parse_named_path(value) for value in args.infos)
    scene_frames = load_scene_frames(named_paths)
    manifest_path = osp.join(args.audit_dir, 'scene_manifest.csv')
    manifest_rows, manifest_fields = read_csv(manifest_path)
    for field in CONTEXT_FIELDS:
        if field not in manifest_fields:
            manifest_fields.append(field)

    context_rows = []
    status_counter = Counter()
    mode_counter = Counter()
    for index, row in enumerate(manifest_rows, 1):
        key = (row.get('split', ''), row.get('scene_token', ''))
        frames = scene_frames.get(key, [])
        package, _, short_scene = key[1].partition('/')
        package_out = osp.join(args.origin_root, package + '_out')
        scene_dir = osp.join(package_out, short_scene)
        context = summarize_origin_context(frames, scene_dir, args)

        should_render = args.contact_sheet_mode == 'all' or (
            args.contact_sheet_mode == 'matched'
            and context['origin_context_status'] in (
                'ok', 'partial_chassis_match'))
        indices = representative_camera_indices(
            frames, args.camera_frames, args.data_root)
        if should_render and indices:
            relative = osp.join(
                'camera_contact_sheets', key[0], safe_name(key[1]) + '.jpg')
            render_contact_sheet(
                osp.join(args.audit_dir, relative), key[1], frames,
                indices, args)
            context['camera_contact_sheet'] = relative
            context['camera_evidence_frames'] = ';'.join(
                str(value) for value in indices)
        elif row.get('camera_contact_sheet'):
            context['camera_contact_sheet'] = row['camera_contact_sheet']
            context['camera_evidence_frames'] = row.get(
                'camera_evidence_frames', '')

        row.update(context)
        context_rows.append(dict(
            split=key[0], scene_token=key[1], frame_count=len(frames),
            **context))
        status_counter[context['origin_context_status']] += 1
        mode_counter[context['parsed_control_mode']] += 1
        if index % 50 == 0 or index == len(manifest_rows):
            print(f'Processed {index}/{len(manifest_rows)} scenes')

    write_csv_atomic(manifest_path, manifest_rows, manifest_fields)
    context_path = osp.join(args.audit_dir, 'scene_origin_context.csv')
    write_csv_atomic(
        context_path, context_rows,
        ['split', 'scene_token', 'frame_count', *CONTEXT_FIELDS])
    summary = dict(
        sources=named_paths,
        origin_root=osp.realpath(args.origin_root),
        scenes=len(manifest_rows),
        dataset_packages=len({
            row.get('scene_token', '').split('/', 1)[0]
            for row in manifest_rows}),
        available_dataset_package_out=len({
            row.get('scene_token', '').split('/', 1)[0]
            for row in manifest_rows
            if osp.isdir(osp.join(
                args.origin_root,
                row.get('scene_token', '').split('/', 1)[0] + '_out'))
        }),
        context_status=dict(status_counter),
        parsed_control_modes=dict(mode_counter),
        mode_mapping=dict(
            Auto=sorted(AUTO_MODES), Manual=sorted(MANUAL_MODES)),
    )
    summary_path = osp.join(args.audit_dir, 'origin_context_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')
    print(json.dumps(summary, indent=2))
    print(f'Wrote {context_path}')


if __name__ == '__main__':
    main()
