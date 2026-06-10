import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from mmcv import Config

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_bevformer_lidar_onnx import (  # noqa: E402
    BEVFormerLidarTRTWrapper,
    build_lidar_head,
    head_state_dict_from_checkpoint,
    import_plugins,
)


OUTPUT_NAMES = ('bev_embed', 'all_cls_scores', 'all_bbox_preds')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare LiDAR dense-BEV PyTorch outputs with a TRT engine.'
    )
    parser.add_argument('config', help='LiDAR TRT config file path.')
    parser.add_argument('checkpoint', help='Checkpoint used for ONNX export.')
    parser.add_argument('engine', help='TensorRT engine to validate.')
    parser.add_argument(
        '--plugin',
        required=True,
        help='TensorRT plugin library, e.g. enqueueV3/build/libuniad_plugin.so.')
    parser.add_argument(
        '--trtexec',
        default='trtexec',
        help='Path to TensorRT trtexec binary.')
    parser.add_argument(
        '--out-dir',
        default='./dumped_inputs/dense_bev_epoch1_validation',
        help='Directory for inputs, PyTorch outputs, TRT outputs and metrics.')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_raw_input(name, tensor, out_dir):
    array = tensor.detach().cpu().numpy().astype(np.float32)
    path = out_dir / f'{name}.bin'
    array.tofile(path)
    np.save(out_dir / f'{name}.npy', array)
    return path


def build_wrapper(config_path, checkpoint_path):
    cfg = Config.fromfile(config_path)
    import_plugins(cfg, config_path)
    head = build_lidar_head(cfg)
    head_state = head_state_dict_from_checkpoint(checkpoint_path)
    missing, unexpected = head.load_state_dict(head_state, strict=False)
    print(f'Loaded {len(head_state)} pts_bbox_head tensors.')
    print(f'Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}')
    return BEVFormerLidarTRTWrapper(head).cuda().eval()


def generate_inputs(wrapper):
    head = wrapper.head
    batch_size = 1
    return {
        'lidar_bev': torch.randn(
            batch_size,
            head.lidar_in_channels,
            head.bev_h,
            head.bev_w,
            device='cuda',
            dtype=torch.float32),
        'prev_bev': torch.randn(
            batch_size,
            head.bev_embed_dims,
            head.bev_h,
            head.bev_w,
            device='cuda',
            dtype=torch.float32),
        'shift': torch.zeros(batch_size, 2, device='cuda',
                             dtype=torch.float32),
        'use_prev_bev': torch.ones(batch_size, device='cuda',
                                   dtype=torch.float32),
    }


def run_pytorch(wrapper, inputs, out_dir):
    with torch.no_grad():
        outputs = wrapper(inputs['lidar_bev'], inputs['prev_bev'],
                          inputs['shift'], inputs['use_prev_bev'])
    output_map = {}
    for name, tensor in zip(OUTPUT_NAMES, outputs):
        array = tensor.detach().cpu().numpy().astype(np.float32)
        np.save(out_dir / f'pytorch_{name}.npy', array)
        output_map[name] = array
    print('PyTorch output shapes:',
          {name: value.shape for name, value in output_map.items()})
    return output_map


def infer_trt_lib_dir(trtexec):
    trtexec = Path(trtexec).resolve()
    candidate = trtexec.parent.parent / 'lib'
    return str(candidate) if candidate.exists() else ''


def run_trtexec(args, input_paths, out_dir):
    output_json = out_dir / 'trt_outputs.json'
    trtexec_log = out_dir / 'trtexec.log'
    load_inputs = ','.join(
        f'{name}:{path}' for name, path in input_paths.items())
    cmd = [
        args.trtexec,
        f'--loadEngine={args.engine}',
        f'--staticPlugins={args.plugin}',
        f'--loadInputs={load_inputs}',
        f'--exportOutput={output_json}',
        '--iterations=1',
        '--warmUp=0',
        '--duration=0',
    ]
    env = os.environ.copy()
    trt_lib_dir = infer_trt_lib_dir(args.trtexec)
    if trt_lib_dir:
        env['LD_LIBRARY_PATH'] = (
            trt_lib_dir + ':' + env.get('LD_LIBRARY_PATH', ''))
    print('Running:', ' '.join(cmd))
    with open(trtexec_log, 'w') as log_file:
        subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, stdout=log_file,
                       stderr=subprocess.STDOUT, check=True)
    return output_json, trtexec_log


def parse_trtexec_outputs(output_json):
    data = json.load(open(output_json))
    outputs = {}
    for item in data:
        dims = tuple(int(dim) for dim in item['dimensions'].split('x'))
        outputs[item['name']] = np.asarray(
            item['values'], dtype=np.float32).reshape(dims)
    return outputs


def compare_outputs(pytorch_outputs, trt_outputs):
    metrics = {}
    for name in OUTPUT_NAMES:
        ref = pytorch_outputs[name]
        got = trt_outputs[name]
        if ref.shape != got.shape:
            raise ValueError(f'{name} shape mismatch: {ref.shape} vs '
                             f'{got.shape}')
        diff = np.abs(ref - got)
        denom = np.maximum(np.abs(ref), 1e-6)
        rel = diff / denom
        metrics[name] = {
            'shape': list(ref.shape),
            'ref_abs_max': float(np.abs(ref).max()),
            'trt_abs_max': float(np.abs(got).max()),
            'max_abs': float(diff.max()),
            'mean_abs': float(diff.mean()),
            'p95_abs': float(np.percentile(diff, 95)),
            'p99_abs': float(np.percentile(diff, 99)),
            'max_rel': float(rel.max()),
            'mean_rel': float(rel.mean()),
            'p95_rel': float(np.percentile(rel, 95)),
        }
    return metrics


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for PyTorch dense-BEV export.')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    wrapper = build_wrapper(args.config, args.checkpoint)
    inputs = generate_inputs(wrapper)
    input_paths = {
        name: save_raw_input(name, tensor, out_dir)
        for name, tensor in inputs.items()
    }
    pytorch_outputs = run_pytorch(wrapper, inputs, out_dir)

    output_json, trtexec_log = run_trtexec(args, input_paths, out_dir)
    trt_outputs = parse_trtexec_outputs(output_json)
    for name, array in trt_outputs.items():
        np.save(out_dir / f'trt_{name}.npy', array)

    metrics = compare_outputs(pytorch_outputs, trt_outputs)
    metrics_path = out_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))
    print(f'Wrote metrics: {metrics_path}')
    print(f'Wrote trtexec log: {trtexec_log}')


if __name__ == '__main__':
    main()
