import argparse
import importlib
import os
import random
import sys
import warnings

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch import nn
from torch.onnx import OperatorExportTypes

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402


TRACK_INPUT_NAMES = [
    'prev_track_intances0',
    'prev_track_intances1',
    'prev_track_intances2',
    'prev_track_intances3',
    'prev_track_intances4',
    'prev_track_intances5',
    'prev_track_intances6',
    'prev_track_intances7',
    'prev_track_intances8',
    'prev_track_intances9',
    'prev_track_intances10',
    'prev_track_intances11',
    'prev_track_intances12',
    'prev_track_intances13',
]

INPUT_NAMES = TRACK_INPUT_NAMES + [
    'prev_timestamp',
    'prev_l2g_r_mat',
    'prev_l2g_t',
    'prev_bev',
    'lidar_bev',
    'shift',
    'timestamp',
    'l2g_r_mat',
    'l2g_t',
    'use_prev_bev',
    'max_obj_id',
]

TRACK_OUTPUT_NAMES = [
    'prev_track_intances0_out',
    'prev_track_intances1_out',
    'prev_track_intances3_out',
    'prev_track_intances4_out',
    'prev_track_intances5_out',
    'prev_track_intances6_out',
    'prev_track_intances8_out',
    'prev_track_intances9_out',
    'prev_track_intances11_out',
    'prev_track_intances12_out',
    'prev_track_intances13_out',
    'prev_timestamp_out',
    'prev_l2g_t_out',
    'prev_l2g_r_mat_out',
    'bev_embed',
    'bboxes_dict_bboxes',
    'scores',
    'labels',
    'bbox_index',
    'obj_idxes',
    'max_obj_id_out',
]

MOTION_OUTPUT_NAMES = [
    'traj_scores_0',
    'traj_0',
    'traj_scores_1',
    'traj_1',
    'traj_scores',
    'traj',
    'valid_traj_masks',
]

OUTPUT_NAMES = TRACK_OUTPUT_NAMES + MOTION_OUTPUT_NAMES


class E2ELidarTRTWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *inputs):
        return self.model.forward_e2e_lidar_trt(*inputs)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export base_e2e_lidar TensorRT boundary to ONNX.')
    parser.add_argument('config', help='LiDAR e2e TRT config path.')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Optional stage2_e2e_lidar checkpoint.')
    parser.add_argument(
        '--onnx-file',
        default='./onnx/base_e2e_lidar_trt.onnx',
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


def import_plugins(cfg, config_path):
    custom_imports = cfg.get('custom_imports')
    if custom_imports:
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**custom_imports)

    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = cfg.plugin_dir
    else:
        module_dir = os.path.dirname(config_path)
    module_path = module_dir.rstrip('/').replace('/', '.')
    if module_path:
        importlib.import_module(module_path)


def repair_reshape_allowzero(onnx_file):
    graph = gs.import_onnx(onnx.load(onnx_file))
    for node in graph.nodes:
        if node.op == 'Reshape':
            node.attrs['allowzero'] = 1
    repaired_file = onnx_file[:-5] + '.repaired.onnx'
    onnx.save(gs.export_onnx(graph), repaired_file)
    return repaired_file


def pad_first_dim(tensor, length, fill_value=0):
    tensor = tensor.detach().clone()
    if tensor.shape[0] == length:
        return tensor
    if tensor.shape[0] > length:
        return tensor[:length].contiguous()
    pad_shape = (length - tensor.shape[0], ) + tuple(tensor.shape[1:])
    padding = tensor.new_full(pad_shape, fill_value)
    return torch.cat([tensor, padding], dim=0).contiguous()


def make_dummy_track_inputs(model, track_state_len):
    empty_tracks = model._generate_empty_tracks_trt()
    base_len = empty_tracks[0].shape[0]
    if track_state_len < base_len:
        raise ValueError(
            f'--track-state-len must be >= {base_len}, got '
            f'{track_state_len}.')

    tracks = [
        pad_first_dim(item, track_state_len, -1 if idx in (3, 4) else 0)
        for idx, item in enumerate(empty_tracks)
    ]
    device = tracks[0].device
    tracks[3] = torch.full(
        (track_state_len, ), -1, dtype=torch.int32, device=device)
    tracks[4] = torch.full(
        (track_state_len, ), -1, dtype=torch.int32, device=device)
    tracks[5] = torch.zeros(
        (track_state_len, ), dtype=torch.int32, device=device)
    tracks[6] = torch.zeros(
        (track_state_len, ), dtype=torch.float32, device=device)
    tracks[7] = torch.zeros(
        (track_state_len, ), dtype=torch.float32, device=device)
    tracks[8] = torch.zeros(
        (track_state_len, ), dtype=torch.float32, device=device)
    tracks[9] = torch.zeros(
        (track_state_len, 10), dtype=torch.float32, device=device)
    tracks[10] = torch.zeros(
        (track_state_len, model.num_classes),
        dtype=torch.float32,
        device=device)
    tracks[11] = torch.zeros(
        (track_state_len, model.mem_bank_len, model.embed_dims),
        dtype=torch.float32,
        device=device)
    tracks[12] = torch.zeros(
        (track_state_len, model.mem_bank_len),
        dtype=torch.int32,
        device=device)
    tracks[13] = torch.zeros(
        (track_state_len, ), dtype=torch.float32, device=device)
    return tracks


def make_dummy_inputs(model, track_state_len):
    device = next(model.parameters()).device
    dtype = torch.float32
    batch_size = 1
    tracks = make_dummy_track_inputs(model, track_state_len)
    prev_timestamp = torch.zeros(batch_size, device=device, dtype=dtype)
    timestamp = torch.full((batch_size, ), 0.5, device=device, dtype=dtype)
    l2g_r_mat = torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    l2g_t = torch.zeros(batch_size, 3, device=device, dtype=dtype)
    prev_bev = torch.zeros(
        batch_size,
        model.pts_bbox_head.bev_embed_dims,
        model.bev_h,
        model.bev_w,
        device=device,
        dtype=dtype)
    lidar_bev = torch.randn(
        batch_size,
        model.pts_bbox_head.lidar_in_channels,
        model.bev_h,
        model.bev_w,
        device=device,
        dtype=dtype)
    shift = torch.zeros(batch_size, 2, device=device, dtype=dtype)
    use_prev_bev = torch.ones(batch_size, device=device, dtype=dtype)
    max_obj_id = torch.zeros(batch_size, device=device, dtype=torch.int32)
    return tuple(tracks + [
        prev_timestamp,
        l2g_r_mat,
        l2g_t,
        prev_bev,
        lidar_bev,
        shift,
        timestamp,
        l2g_r_mat,
        l2g_t,
        use_prev_bev,
        max_obj_id,
    ])


def dynamic_axes():
    axes = {}
    for name in TRACK_INPUT_NAMES:
        axes[name] = [0]
    for name in TRACK_OUTPUT_NAMES:
        if name.startswith('prev_track_intances'):
            axes[name] = [0]
    for name in [
            'bboxes_dict_bboxes', 'scores', 'labels', 'bbox_index',
            'obj_idxes', 'traj_scores_0', 'traj_0', 'traj_scores_1',
            'traj_1', 'traj_scores', 'traj', 'valid_traj_masks'
    ]:
        axes[name] = [0]
    return axes


def dump_inputs(inputs, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for name, value in zip(INPUT_NAMES, inputs):
        value.detach().cpu().numpy().tofile(
            os.path.join(output_dir, f'{name}.dat'))


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('LiDAR e2e ONNX export requires CUDA because '
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
    wrapper = E2ELidarTRTWrapper(model).cuda().eval()
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
