import argparse
import os
import random
import warnings

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch import nn
from torch.onnx import OperatorExportTypes

from export_e2e_lidar_onnx import (  # noqa: E402
    INPUT_NAMES as E2E_INPUT_NAMES,
    OUTPUT_NAMES as E2E_OUTPUT_NAMES,
    dynamic_axes as e2e_dynamic_axes,
    dump_inputs as dump_e2e_inputs,
    import_plugins,
    make_dummy_inputs as make_e2e_dummy_inputs,
    repair_reshape_allowzero,
)
from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402

warnings.filterwarnings('ignore')

INPUT_NAMES = E2E_INPUT_NAMES + ['command']
OUTPUT_NAMES = E2E_OUTPUT_NAMES + ['sdc_traj']


class E2ELidarPlanTRTWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.forward_e2e_lidar_plan_trt(*inputs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export base_e2e_lidar_plan TensorRT boundary to ONNX.')
    parser.add_argument('config', help='LiDAR e2e plan TRT config path.')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Optional stage2_e2e_lidar_plan checkpoint.')
    parser.add_argument(
        '--onnx-file',
        default='./onnx/base_e2e_lidar_plan_trt.onnx',
        help='Output ONNX path.')
    parser.add_argument(
        '--track-state-len',
        type=int,
        default=601,
        help='Dummy previous track state length used for tracing.')
    parser.add_argument(
        '--command',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Dummy planning command: 0=right, 1=left, 2=forward.')
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--dump-input-dir',
        default=None,
        help='Optional directory to dump tracing inputs as raw .dat files.')
    return parser.parse_args()


def make_dummy_inputs(model, track_state_len, command):
    inputs = list(make_e2e_dummy_inputs(model, track_state_len))
    device = next(model.parameters()).device
    command_tensor = torch.full(
        (1, ), float(command), device=device, dtype=torch.float32)
    inputs.append(command_tensor)
    return tuple(inputs)


def dynamic_axes():
    return e2e_dynamic_axes()


def dump_inputs(inputs, output_dir):
    dump_e2e_inputs(inputs[:-1], output_dir)
    command = inputs[-1].detach().cpu().numpy()
    command.tofile(os.path.join(output_dir, 'command.dat'))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('LiDAR e2e plan ONNX export requires CUDA because '
                           'TensorRT plugin symbolic functions require CUDA '
                           'inputs.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)
    model_cfg = cfg.model.copy()
    model_cfg.train_cfg = None
    model = build_model(model_cfg, test_cfg=cfg.get('test_cfg'))
    if args.checkpoint is not None:
        checkpoint = load_checkpoint(
            model, args.checkpoint, map_location='cpu', strict=False)
        meta = checkpoint.get('meta', {})
        if 'CLASSES' in meta:
            model.CLASSES = meta['CLASSES']

    model.cuda().eval()
    wrapper = E2ELidarPlanTRTWrapper(model).cuda().eval()
    inputs = make_dummy_inputs(model, args.track_state_len, args.command)

    onnx_dir = os.path.dirname(args.onnx_file)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)
    if args.dump_input_dir is not None:
        os.makedirs(args.dump_input_dir, exist_ok=True)
        dump_inputs(inputs, args.dump_input_dir)

    with torch.no_grad():
        outputs = wrapper(*inputs)
    print('PyTorch output shapes:',
          [tuple(output.shape) for output in outputs])

    torch.onnx.export(
        wrapper,
        inputs,
        args.onnx_file,
        verbose=False,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=INPUT_NAMES,
        output_names=OUTPUT_NAMES,
        opset_version=args.opset,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        dynamic_axes=dynamic_axes())
    repaired = repair_reshape_allowzero(args.onnx_file)
    print(f'Exported ONNX: {args.onnx_file}')
    print(f'Repaired ONNX: {repaired}')


if __name__ == '__main__':
    main()
