import argparse
import importlib
import os
import random
import sys
import warnings

import mmcv
import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
from mmcv import Config
from mmdet.models import build_head
from torch import nn
from torch.onnx import OperatorExportTypes

warnings.filterwarnings('ignore')

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


class BEVFormerLidarTRTWrapper(nn.Module):

    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, lidar_bev, prev_bev, shift, use_prev_bev):
        bev_embed = self.head.get_bev_features_trt(
            lidar_bev,
            prev_bev=prev_bev,
            shift=shift,
            use_prev_bev=use_prev_bev)
        preds = self.head.get_detections_trt(bev_embed)
        return bev_embed, preds['all_cls_scores'], preds['all_bbox_preds']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export LiDAR BEVFormer dense-BEV boundary to ONNX.')
    parser.add_argument('config', help='LiDAR TRT config file path')
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Optional checkpoint. Only pts_bbox_head.* weights are loaded.')
    parser.add_argument(
        '--onnx-file',
        default='./onnx/bevformer_lidar_bev_trt.onnx',
        help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = cfg.plugin_dir
    else:
        module_dir = os.path.dirname(config_path)
    module_path = module_dir.rstrip('/').replace('/', '.')
    importlib.import_module(module_path)


def head_state_dict_from_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    head_state = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            key = key[len('module.'):]
        if key.startswith('pts_bbox_head.'):
            head_state[key[len('pts_bbox_head.'):]] = value
    return head_state


def build_lidar_head(cfg):
    head_cfg = cfg.model.pts_bbox_head.copy()
    if cfg.model.get('train_cfg') is not None:
        head_cfg.setdefault('train_cfg', cfg.model.train_cfg.get('pts'))
    if cfg.model.get('test_cfg') is not None:
        head_cfg.setdefault('test_cfg', cfg.model.test_cfg.get('pts'))
    return build_head(head_cfg)


def repair_reshape_allowzero(onnx_file):
    graph = gs.import_onnx(onnx.load(onnx_file))
    for node in graph.nodes:
        if node.op == 'Reshape':
            node.attrs['allowzero'] = 1
    repaired_file = onnx_file[:-5] + '.repaired.onnx'
    onnx.save(gs.export_onnx(graph), repaired_file)
    return repaired_file


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('LiDAR TRT ONNX export requires CUDA because the '
                           'TensorRT plugin symbolic path asserts CUDA inputs.')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    cfg = Config.fromfile(args.config)
    import_plugins(cfg, args.config)

    head = build_lidar_head(cfg)
    if args.checkpoint is not None:
        head_state = head_state_dict_from_checkpoint(args.checkpoint)
        missing, unexpected = head.load_state_dict(head_state, strict=False)
        print(f'Loaded {len(head_state)} pts_bbox_head tensors.')
        print(f'Missing keys: {len(missing)}, unexpected keys: '
              f'{len(unexpected)}')
    head.cuda().eval()

    wrapper = BEVFormerLidarTRTWrapper(head).cuda().eval()
    batch_size = 1
    lidar_bev = torch.randn(
        batch_size,
        head.lidar_in_channels,
        head.bev_h,
        head.bev_w,
        device='cuda',
        dtype=torch.float32)
    prev_bev = torch.zeros(
        batch_size,
        head.bev_embed_dims,
        head.bev_h,
        head.bev_w,
        device='cuda',
        dtype=torch.float32)
    shift = torch.zeros(batch_size, 2, device='cuda', dtype=torch.float32)
    use_prev_bev = torch.ones(batch_size, device='cuda', dtype=torch.float32)

    os.makedirs(os.path.dirname(args.onnx_file), exist_ok=True)
    with torch.no_grad():
        outputs = wrapper(lidar_bev, prev_bev, shift, use_prev_bev)
    print('PyTorch output shapes:',
          [tuple(output.shape) for output in outputs])

    torch.onnx.export(
        wrapper,
        (lidar_bev, prev_bev, shift, use_prev_bev),
        args.onnx_file,
        verbose=False,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=['lidar_bev', 'prev_bev', 'shift', 'use_prev_bev'],
        output_names=['bev_embed', 'all_cls_scores', 'all_bbox_preds'],
        opset_version=args.opset,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH)
    repaired = repair_reshape_allowzero(args.onnx_file)
    print(f'Exported ONNX: {args.onnx_file}')
    print(f'Repaired ONNX: {repaired}')


if __name__ == '__main__':
    main()
