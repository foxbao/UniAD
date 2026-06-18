import argparse
import os
import random
import sys
import warnings

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch import nn
from torch.onnx import OperatorExportTypes

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402
from export_track_lidar_onnx import (  # noqa: E402
    INPUT_NAMES, OUTPUT_NAMES as TRACK_OUTPUT_NAMES, TRACK_INPUT_NAMES,
    dump_inputs, import_plugins, make_dummy_inputs, repair_reshape_allowzero)


OUTPUT_NAMES = TRACK_OUTPUT_NAMES + ['drivable_score']


class TrackDrivableLidarTRTWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.forward_track_drivable_lidar_trt(*inputs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export base_track_drivable_lidar TensorRT boundary to '
        'ONNX.')
    parser.add_argument('config', help='LiDAR track+drivable TRT config path.')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Optional base_track_drivable_lidar checkpoint.')
    parser.add_argument(
        '--onnx-file',
        default='./onnx/base_track_drivable_lidar_trt.onnx',
        help='Output ONNX path.')
    parser.add_argument(
        '--track-state-len',
        type=int,
        default=601,
        help='Dummy previous track state length used for tracing.')
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--dump-input-dir',
        default=None,
        help='Optional directory to dump tracing inputs as raw .dat files.')
    return parser.parse_args()


def dynamic_axes():
    axes = {}
    for name in TRACK_INPUT_NAMES:
        axes[name] = [0]
    for name in TRACK_OUTPUT_NAMES:
        if name.startswith('prev_track_intances'):
            axes[name] = [0]
    for name in [
            'bboxes_dict_bboxes', 'scores', 'labels', 'bbox_index',
            'obj_idxes'
    ]:
        axes[name] = [0]
    return axes


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('LiDAR track+drivable ONNX export requires CUDA '
                           'because TensorRT plugin symbolic functions '
                           'require CUDA inputs.')

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
    wrapper = TrackDrivableLidarTRTWrapper(model).cuda().eval()
    inputs = make_dummy_inputs(model, args.track_state_len)

    onnx_dir = os.path.dirname(args.onnx_file)
    if onnx_dir:
        os.makedirs(onnx_dir, exist_ok=True)
    if args.dump_input_dir is not None:
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
