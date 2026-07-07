#!/usr/bin/env python3
"""PTQ-calibrate the LiDAR sparse encoder for libspconv INT8, then write the
per-layer dynamic ranges into the exported FP16 sparse ONNX as node attributes.

libspconv reads INT8 config straight from ONNX SparseConvolution/Add attributes:
  SparseConvolution: precision/output_precision="int8",
                     weight_dynamic_ranges=[per-out-channel max|w|],
                     input_dynamic_range=<per-tensor max|activation|>
  Add:               precision/output_precision="int8",
                     input0_dynamic_range, input1_dynamic_range

So the recipe (no QAT, no pytorch_quantization):
  1. weight_dynamic_ranges: static, computed from ONNX weights (per out-channel).
  2. input_dynamic_range: PTQ — run N real frames through the PyTorch sparse
     backbone with forward hooks on each spconv layer, record max|input|.
  3. Write both into the FP16 ONNX via onnx (attributes), flip precision to int8.

Reuses the export script's model/voxelize helpers so it runs identical real
forward. Standalone: does NOT modify the export script or the FP16 path.
"""
import argparse
import os
import sys

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper

_TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _TOOLS)
sys.path.insert(0, os.path.dirname(_TOOLS))  # UniAD root for projects.*

import export_bevformer_lidar_sparse_onnx as exp  # noqa: E402

try:
    import spconv.pytorch as spconv  # noqa: E402
except Exception:  # pragma: no cover
    spconv = None


def collect_input_ranges(encoder, model, cfg, split, indices, device):
    """Run N frames; forward-hook every SparseConvolution to record max|input|.
    Layers fire in the same order the export script assigns conv{ilayer}, so we
    key the result by that firing order."""
    order = []            # firing sequence of conv modules (per frame)
    per_layer_max = {}    # ilayer -> running max|input feature|

    frame_counter = {'n': 0}
    seq = {'i': 0}

    def hook(module, inputs):
        # inputs[0] is the input SparseConvTensor (has .features)
        x = inputs[0]
        feats = getattr(x, 'features', None)
        if feats is None:
            return
        idx = seq['i']
        m = feats.detach().abs().max().item()
        if idx not in per_layer_max or m > per_layer_max[idx]:
            per_layer_max[idx] = m
        seq['i'] += 1

    handles = []
    for mod in encoder.modules():
        if spconv is not None and isinstance(mod, spconv.conv.SparseConvolution):
            handles.append(mod.register_forward_pre_hook(hook))

    with torch.no_grad():
        for index in indices:
            seq['i'] = 0  # reset firing counter each frame
            vf, coors, meta = exp.make_sparse_input_from_dataset(
                model, cfg, split, index, device)
            bs = 1
            encoder(vf, coors, bs)
            frame_counter['n'] += 1

    for h in handles:
        h.remove()
    return per_layer_max, frame_counter['n']


def weight_ranges_from_onnx(g):
    """Per-output-channel max|w| for each SparseConvolution, keyed by conv name."""
    inits = {i.name: i for i in g.initializer}
    out = {}
    for nd in g.node:
        if nd.op_type != 'SparseConvolution':
            continue
        wname = [x for x in nd.input if 'weight' in x.lower()]
        if not wname:
            continue
        w = numpy_helper.to_array(inits[wname[0]]).astype(np.float32)
        oc = w.shape[0]
        out[nd.name] = np.abs(w.reshape(oc, -1)).max(axis=1).astype(np.float32)
    return out


def set_attr(node, name, value):
    """Replace or add a node attribute (protobuf repeated field safe)."""
    keep = [a for a in node.attribute if a.name != name]
    del node.attribute[:]
    node.attribute.extend(keep)
    if isinstance(value, str):
        node.attribute.append(helper.make_attribute(name, value))
    elif isinstance(value, (list, np.ndarray)):
        node.attribute.append(
            helper.make_attribute(name, [float(v) for v in value]))
    else:
        node.attribute.append(helper.make_attribute(name, float(value)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--fp16-onnx', required=True,
                    help='exported FP16 sparse encoder ONNX to convert')
    ap.add_argument('--out', required=True, help='output INT8 ONNX')
    ap.add_argument('--split', default='test')
    ap.add_argument('--calib-indices', default='0,1,2,3,4,5,6,7',
                    help='comma-separated dataset indices for calibration')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cfg-options', default=None,
                    help='(unused; present for export-script load_config compat)')
    args = ap.parse_args()

    device = torch.device(args.device)
    cfg = exp.load_config(args)
    model, _ = exp.build_loaded_model(cfg, args.checkpoint, device)
    encoder = model.pts_middle_encoder.eval()

    indices = [int(x) for x in args.calib_indices.split(',') if x != '']
    print(f'[calib] {len(indices)} frames, indices={indices}')
    input_max, n = collect_input_ranges(
        encoder, model, cfg, args.split, indices, device)
    print(f'[calib] collected input ranges over {n} frames, '
          f'{len(input_max)} spconv layers')

    m = onnx.load(args.fp16_onnx)
    g = m.graph
    wr = weight_ranges_from_onnx(g)

    # spconv nodes fire in conv{ilayer} order; input_max is keyed by that order.
    conv_nodes = [nd for nd in g.node if nd.op_type == 'SparseConvolution']
    n_int8 = 0
    for i, nd in enumerate(conv_nodes):
        if i not in input_max or nd.name not in wr:
            print(f'  skip {nd.name}: no range')
            continue
        set_attr(nd, 'precision', 'int8')
        set_attr(nd, 'output_precision', 'int8')
        set_attr(nd, 'weight_dynamic_ranges', wr[nd.name])
        set_attr(nd, 'input_dynamic_range', input_max[i])
        n_int8 += 1

    onnx.save(m, args.out)
    print(f'[write] {n_int8}/{len(conv_nodes)} SparseConvolution -> int8')
    print(f'[write] saved {args.out}')


if __name__ == '__main__':
    main()
