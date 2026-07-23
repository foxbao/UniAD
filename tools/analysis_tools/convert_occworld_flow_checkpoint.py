#!/usr/bin/env python
"""Convert an incremental OccWorld flow head to cumulative initialization."""

import argparse
from pathlib import Path

import torch


FLOW_SUFFIXES = (
    'occ_head.world_decoder.future_flow_head.weight',
    'occ_head.world_decoder.future_flow_head.bias',
)


def convert_state_dict(state_dict, future_count):
    converted = dict(state_dict)
    matched = []
    for key, value in state_dict.items():
        if not key.endswith(FLOW_SUFFIXES):
            continue
        if value.shape[0] != future_count * 2:
            raise ValueError(
                f'{key} has {value.shape[0]} outputs, expected '
                f'{future_count * 2}')
        step_shape = (future_count, 2, *value.shape[1:])
        converted[key] = value.reshape(step_shape).cumsum(dim=0).reshape_as(
            value)
        matched.append(key)
    if len(matched) != 2:
        raise ValueError(
            f'Expected one flow weight and bias, found {matched}')
    return converted, matched


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--future-count', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.future_count < 1:
        raise ValueError('future-count must be positive')
    checkpoint = torch.load(args.input, map_location='cpu')
    if 'state_dict' not in checkpoint:
        raise ValueError('Input checkpoint has no state_dict')
    state_dict, matched = convert_state_dict(
        checkpoint['state_dict'], args.future_count)
    meta = dict(checkpoint.get('meta', {}))
    meta['occworld_flow_parameterization'] = 'cumulative_current'
    meta['occworld_flow_initialization_source'] = str(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'meta': meta, 'state_dict': state_dict}, args.output)
    print(f'converted_keys={matched}')
    print(f'output={args.output}')


if __name__ == '__main__':
    main()
