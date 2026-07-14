#!/usr/bin/env python3
"""Generate planning-specific info files from a reviewed scene manifest."""

import argparse
import csv
import json
import os
import os.path as osp
import pickle


LABELS = ('NaturalRun', 'OperationalStop', 'DetectionProbe', 'Uncertain')
CONTROL_MODES = ('Auto', 'Manual', 'Mixed', 'Unknown')
PROXY_TO_CONTROL_MODE = {
    'LikelyAuto': 'Auto',
    'LikelyManual': 'Manual',
    'MixedOrUnknown': 'Unknown',
    'StoppedOrUnknown': 'Unknown',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--infos', nargs='+', required=True,
        help='Inputs as SPLIT=path.pkl.')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument(
        '--label-source', choices=('human', 'auto'), default='human')
    parser.add_argument(
        '--allow-unreviewed', action='store_true',
        help='Place missing human labels in Uncertain instead of failing.')
    return parser.parse_args()


def parse_named_path(value):
    if '=' not in value:
        raise ValueError(f'Expected SPLIT=path, got {value}')
    split, path = value.split('=', 1)
    return split.strip(), path.strip()


def load_payload(path):
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict):
        if 'data_list' in payload:
            return payload, 'data_list', payload['data_list']
        if 'infos' in payload:
            return payload, 'infos', payload['infos']
    if isinstance(payload, list):
        return payload, None, payload
    raise TypeError(f'Unsupported info payload in {path}: {type(payload)}')


def subset_payload(payload, key, infos):
    if key is None:
        return list(infos)
    output = dict(payload)
    output[key] = list(infos)
    return output


def dump_payload(path, payload):
    os.makedirs(osp.dirname(path), exist_ok=True)
    with open(path, 'wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_manifest(path, label_source, allow_unreviewed):
    label_key = 'human_label' if label_source == 'human' else 'auto_label'
    output = {}
    missing_labels = []
    missing_usable = []
    missing_control_modes = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            key = (row.get('split', ''), row.get('scene_token', ''))
            label = row.get(label_key, '').strip()
            if not label:
                if allow_unreviewed:
                    label = 'Uncertain'
                else:
                    missing_labels.append(key)
                    continue
            if label not in LABELS:
                raise ValueError(f'Unknown label {label} for {key}')
            if label_source == 'auto':
                usable = label in ('NaturalRun', 'OperationalStop')
                control_mode = PROXY_TO_CONTROL_MODE.get(
                    row.get('control_mode_proxy', '').strip(), 'Unknown')
            else:
                raw_usable = row.get('planning_usable', '').strip().lower()
                if raw_usable in ('1', 'true', 'yes'):
                    usable = True
                elif raw_usable in ('0', 'false', 'no'):
                    usable = False
                elif allow_unreviewed:
                    usable = False
                else:
                    missing_usable.append(key)
                    continue
                control_mode = row.get('human_control_mode', '').strip()
                if not control_mode:
                    if allow_unreviewed:
                        control_mode = 'Unknown'
                    else:
                        missing_control_modes.append(key)
                        continue
                if control_mode not in CONTROL_MODES:
                    raise ValueError(
                        f'Unknown human_control_mode {control_mode} for {key}')
            output[key] = dict(
                label=label, usable=usable, control_mode=control_mode)
    if missing_labels:
        preview = ', '.join(
            f'{split}:{scene}' for split, scene in missing_labels[:5])
        raise ValueError(
            f'{len(missing_labels)} scenes have no {label_key}; '
            f'first: {preview}')
    if missing_usable:
        preview = ', '.join(
            f'{split}:{scene}' for split, scene in missing_usable[:5])
        raise ValueError(
            f'{len(missing_usable)} scenes have no valid planning_usable; '
            f'first: {preview}')
    if missing_control_modes:
        preview = ', '.join(
            f'{split}:{scene}' for split, scene
            in missing_control_modes[:5])
        raise ValueError(
            f'{len(missing_control_modes)} scenes have no '
            f'human_control_mode; first: {preview}')
    return output


def main():
    args = parse_args()
    named_paths = dict(parse_named_path(value) for value in args.infos)
    labels = load_manifest(
        args.manifest, args.label_source, args.allow_unreviewed)
    os.makedirs(args.out_dir, exist_ok=True)
    report = {}
    for split, path in named_paths.items():
        payload, key, infos = load_payload(path)
        grouped = {label: [] for label in LABELS}
        usable_grouped = {label: [] for label in LABELS}
        control_grouped = {mode: [] for mode in CONTROL_MODES}
        rejected = []
        unknown_scenes = set()
        for info in infos:
            scene = str(info.get('scene_token', '') or 'unknown_scene')
            decision = labels.get((split, scene))
            if decision is None:
                if not args.allow_unreviewed:
                    unknown_scenes.add(scene)
                    continue
                decision = dict(
                    label='Uncertain', usable=False, control_mode='Unknown')
            label = decision['label']
            grouped[label].append(info)
            control_grouped[decision['control_mode']].append(info)
            if decision['usable']:
                usable_grouped[label].append(info)
            else:
                rejected.append(info)
        if unknown_scenes:
            raise ValueError(
                f'{split} contains {len(unknown_scenes)} scenes absent from '
                f'the manifest; first: {sorted(unknown_scenes)[:5]}')
        clean = (usable_grouped['NaturalRun']
                 + usable_grouped['OperationalStop'])
        outputs = {
            'planning_clean': clean,
            'planning_natural': usable_grouped['NaturalRun'],
            'planning_operational': usable_grouped['OperationalStop'],
            'planning_detection_probe': grouped['DetectionProbe'],
            'planning_uncertain': grouped['Uncertain'],
            'planning_rejected': rejected,
            'planning_control_auto': control_grouped['Auto'],
            'planning_control_manual': control_grouped['Manual'],
            'planning_control_mixed': control_grouped['Mixed'],
            'planning_control_unknown': control_grouped['Unknown'],
        }
        report[split] = {}
        source_stem = osp.splitext(osp.basename(path))[0]
        for name, subset in outputs.items():
            output_path = osp.join(
                args.out_dir, f'{source_stem}_{name}.pkl')
            dump_payload(output_path, subset_payload(payload, key, subset))
            report[split][name] = dict(
                frames=len(subset),
                scenes=len({str(info.get('scene_token', ''))
                            for info in subset}),
                path=output_path,
            )
    report_path = osp.join(args.out_dir, 'split_report.json')
    with open(report_path, 'w', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))
    print(f'Wrote {report_path}')


if __name__ == '__main__':
    main()
