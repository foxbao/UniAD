import argparse
import importlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402


class LidarBackboneNeckWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.backbone = model.pts_backbone
        self.neck = model.pts_neck if model.with_pts_neck else None

    @staticmethod
    def unwrap_single_bev(pts_feats):
        if not isinstance(pts_feats, (list, tuple)) or len(pts_feats) != 1:
            raise ValueError('Expected one BEV feature level, got '
                             f'{type(pts_feats)}.')
        return pts_feats[0]

    def forward(self, middle_bev):
        feats = self.backbone(middle_bev)
        if self.neck is not None:
            feats = self.neck(feats)
        return self.unwrap_single_bev(feats)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export UniAD LiDAR 2D backbone+neck to ONNX.')
    parser.add_argument('config', help='LiDAR BEVFormer config path.')
    parser.add_argument('checkpoint', help='PyTorch checkpoint path.')
    parser.add_argument(
        '--golden-dir',
        default=None,
        help='Optional golden dump directory with current/middle_bev.npy.')
    parser.add_argument(
        '--onnx-file',
        default='./onnx/bevformer_lidar_backbone_neck.onnx',
        help='Output ONNX path.')
    parser.add_argument('--opset', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def import_plugins(cfg, config_path):
    if not getattr(cfg, 'plugin', False):
        return
    if hasattr(cfg, 'plugin_dir'):
        module_dir = cfg.plugin_dir
    else:
        module_dir = os.path.dirname(config_path)
    module_path = module_dir.rstrip('/').replace('/', '.')
    importlib.import_module(module_path)


def build_loaded_model(config_path, checkpoint_path, device):
    cfg = Config.fromfile(config_path)
    import_plugins(cfg, config_path)
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device).eval()
    return model


def load_middle_bev(args):
    if args.golden_dir is not None:
        middle_path = Path(args.golden_dir) / 'current/middle_bev.npy'
        middle_bev = torch.from_numpy(np.load(middle_path)).to(args.device)
        return middle_bev.float()
    return torch.randn(1, 256, 120, 160, device=args.device,
                       dtype=torch.float32)


def compare_with_golden(output, golden_dir):
    if golden_dir is None:
        return None
    ref_path = Path(golden_dir) / 'current/lidar_bev.npy'
    ref = torch.from_numpy(np.load(ref_path)).to(output.device).float()
    got = output.float()
    diff = (got - ref).abs()
    return {
        'max_abs': float(diff.max().item()),
        'mean_abs': float(diff.mean().item()),
        'p99_abs': float(torch.quantile(diff.flatten(), 0.99).item()),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for ONNX export.')

    set_seed(args.seed)
    model = build_loaded_model(args.config, args.checkpoint, args.device)
    wrapper = LidarBackboneNeckWrapper(model).to(args.device).eval()
    middle_bev = load_middle_bev(args)

    with torch.no_grad():
        lidar_bev = wrapper(middle_bev)
    print('Input shape:', tuple(middle_bev.shape))
    print('Output shape:', tuple(lidar_bev.shape))

    onnx_file = Path(args.onnx_file)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (middle_bev, ),
        str(onnx_file),
        verbose=False,
        export_params=True,
        keep_initializers_as_inputs=False,
        do_constant_folding=True,
        input_names=['middle_bev'],
        output_names=['lidar_bev'],
        opset_version=args.opset)
    onnx.checker.check_model(str(onnx_file))

    metrics = compare_with_golden(lidar_bev, args.golden_dir)
    summary = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'onnx_file': str(onnx_file.resolve()),
        'input_shape': list(middle_bev.shape),
        'output_shape': list(lidar_bev.shape),
        'golden_dir': (str(Path(args.golden_dir).resolve())
                       if args.golden_dir else None),
        'fp32_vs_golden': metrics,
    }
    summary_path = onnx_file.with_suffix('.summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Exported ONNX: {onnx_file}')
    print(f'Wrote summary: {summary_path}')
    if metrics is not None:
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
