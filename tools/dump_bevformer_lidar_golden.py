import argparse
import copy
import importlib
import json
import os
import random
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint, wrap_fp16_model

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.uniad_mmdet3d.datasets.builder import build_dataset  # noqa: E402
from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description='Dump BEVFormer LiDAR raw-points PyTorch golden tensors.')
    parser.add_argument('config', help='LiDAR BEVFormer config path.')
    parser.add_argument('checkpoint', help='PyTorch checkpoint path.')
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split from cfg.data to dump.')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Dataset index after any valid-frame filtering.')
    parser.add_argument(
        '--out-dir',
        default='./dumped_inputs/bevformer_lidar_raw_golden',
        help='Output root for dumped tensors.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device used to run the PyTorch model.')
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed for deterministic PyTorch paths.')
    parser.add_argument(
        '--no-history',
        action='store_true',
        help='Skip history BEV computation and feed no prev_bev.')
    parser.add_argument(
        '--dump-history-frontends',
        action='store_true',
        help='Dump voxel/sparse/backbone tensors for each history frame too.')
    parser.add_argument(
        '--no-sparse-stages',
        action='store_true',
        help='Only dump the sparse encoder dense output, not sparse stages.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options, e.g. data.test.data_root=/path.')
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


def unwrap_data(value):
    return value.data if isinstance(value, DataContainer) else value


def tensor_sequence(value):
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        return list(value)
    raise TypeError(f'Expected tensor sequence, got {type(value)}.')


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def array_stats(array):
    if array.size == 0:
        return {}
    if not np.issubdtype(array.dtype, np.number):
        return {}
    flat = array.reshape(-1)
    stats = {
        'min': float(np.min(flat)),
        'max': float(np.max(flat)),
    }
    if np.issubdtype(array.dtype, np.floating):
        finite = np.isfinite(flat)
        stats['finite'] = bool(np.all(finite))
        if np.any(finite):
            finite_flat = flat[finite].astype(np.float64)
            stats['mean'] = float(np.mean(finite_flat))
            stats['std'] = float(np.std(finite_flat))
    return stats


def save_array(name, array, out_dir, manifest, save_bin=True):
    array = np.asarray(array)
    npy_path = out_dir / f'{name}.npy'
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, array)

    record = {
        'shape': list(array.shape),
        'dtype': str(array.dtype),
        'npy': str(npy_path.relative_to(out_dir)),
    }
    if save_bin:
        bin_path = out_dir / f'{name}.bin'
        array.tofile(bin_path)
        record['bin'] = str(bin_path.relative_to(out_dir))
    record.update(array_stats(array))
    manifest['tensors'][name] = record
    return array


def save_tensor(name, tensor, out_dir, manifest, save_bin=True):
    if tensor is None:
        manifest['tensors'][name] = {'value': None}
        return None
    array = tensor.detach().cpu().numpy()
    return save_array(name, array, out_dir, manifest, save_bin=save_bin)


def save_sparse_tensor(name, sparse_tensor, out_dir, manifest):
    sparse_record = {
        'spatial_shape': to_jsonable(getattr(sparse_tensor, 'spatial_shape',
                                             None)),
        'batch_size': to_jsonable(getattr(sparse_tensor, 'batch_size', None)),
    }
    manifest.setdefault('sparse_tensors', {})[name] = sparse_record
    save_tensor(f'{name}/features', sparse_tensor.features, out_dir, manifest)
    save_tensor(f'{name}/indices', sparse_tensor.indices, out_dir, manifest)


def load_config(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugins(cfg, args.config)
    return cfg


def build_sample(cfg, split, index):
    data_cfg = copy.deepcopy(cfg.data[split])
    data_cfg.test_mode = split != 'train'
    data_cfg.pop('samples_per_gpu', None)
    dataset = build_dataset(data_cfg)
    if index < 0:
        index += len(dataset)
    if index < 0 or index >= len(dataset):
        raise IndexError(f'Index {index} out of range for {split} dataset '
                         f'with length {len(dataset)}.')
    return dataset, dataset[index]


def build_loaded_model(cfg, checkpoint_path, device):
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16', None) is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device).eval()
    return model, checkpoint


def points_to_device(points, device):
    if torch.is_tensor(points):
        return [points.to(device)]
    return [point.to(device) for point in points]


def run_sparse_encoder(encoder, voxel_features, coors, batch_size, out_dir,
                       manifest, prefix, dump_stages):
    if not dump_stages:
        middle_bev = encoder(voxel_features, coors, batch_size)
        save_tensor(f'{prefix}/middle_bev', middle_bev, out_dir, manifest)
        return middle_bev

    if not all(hasattr(encoder, attr) for attr in (
            'conv_input', 'encoder_layers', 'conv_out', 'sparse_shape')):
        middle_bev = encoder(voxel_features, coors, batch_size)
        save_tensor(f'{prefix}/middle_bev', middle_bev, out_dir, manifest)
        manifest.setdefault('notes', []).append(
            'Sparse encoder stages were not dumped because the encoder does '
            'not expose SparseEncoderSpconv2 internals.')
        return middle_bev

    try:
        sparse_module = importlib.import_module(
            'projects.mmdet3d_plugin.models.middle_encoders.'
            'sparse_encoder_spconv2')
        spconv = sparse_module.spconv
        if spconv is None:
            raise ImportError('spconv backend is not available')
    except Exception as exc:
        middle_bev = encoder(voxel_features, coors, batch_size)
        save_tensor(f'{prefix}/middle_bev', middle_bev, out_dir, manifest)
        manifest.setdefault('notes', []).append(
            f'Sparse encoder stage dump fell back to forward(): {exc}')
        return middle_bev

    coors = coors.int()
    x = spconv.SparseConvTensor(voxel_features, coors, encoder.sparse_shape,
                                batch_size)
    x = encoder.conv_input(x)
    save_sparse_tensor(f'{prefix}/sparse/conv_input', x, out_dir, manifest)

    for layer_name, encoder_layer in encoder.encoder_layers.named_children():
        x = encoder_layer(x)
        save_sparse_tensor(
            f'{prefix}/sparse/{layer_name}', x, out_dir, manifest)

    out = encoder.conv_out(x)
    save_sparse_tensor(f'{prefix}/sparse/conv_out', out, out_dir, manifest)
    spatial_features = out.dense()
    save_tensor(f'{prefix}/sparse/dense_5d', spatial_features, out_dir,
                manifest)
    n, c, d, h, w = spatial_features.shape
    middle_bev = spatial_features.view(n, c * d, h, w)
    save_tensor(f'{prefix}/middle_bev', middle_bev, out_dir, manifest)
    return middle_bev


def run_lidar_frontend(model, points, img_metas, out_dir, manifest, prefix,
                       dump_sparse_stages=True):
    point_list = points_to_device(points, next(model.parameters()).device)
    for idx, point in enumerate(point_list):
        save_tensor(f'{prefix}/raw_points_{idx}', point, out_dir, manifest)

    voxels, num_points, coors = model.voxelize(point_list)
    save_tensor(f'{prefix}/voxels', voxels, out_dir, manifest)
    save_tensor(f'{prefix}/voxel_num_points', num_points, out_dir, manifest)
    save_tensor(f'{prefix}/voxel_coors', coors, out_dir, manifest)

    try:
        voxel_features = model.pts_voxel_encoder(
            voxels, num_points, coors, None, img_metas)
    except TypeError:
        voxel_features = model.pts_voxel_encoder(voxels, num_points, coors)
    save_tensor(f'{prefix}/voxel_features', voxel_features, out_dir, manifest)

    middle_dtype = next(model.pts_middle_encoder.parameters()).dtype
    voxel_features = voxel_features.to(dtype=middle_dtype)
    if coors.numel() == 0:
        raise RuntimeError('Voxelization produced no voxels.')
    batch_size = int(coors[-1, 0].item()) + 1
    manifest.setdefault('frontend', {})[prefix] = {
        'batch_size': batch_size,
        'num_voxels': int(coors.shape[0]),
    }

    middle_bev = run_sparse_encoder(
        model.pts_middle_encoder, voxel_features, coors, batch_size, out_dir,
        manifest, prefix, dump_sparse_stages)

    backbone_feats = model.pts_backbone(middle_bev)
    for idx, feat in enumerate(tensor_sequence(backbone_feats)):
        save_tensor(f'{prefix}/backbone_{idx}', feat, out_dir, manifest)

    pts_feats = model.pts_neck(backbone_feats) if model.with_pts_neck else \
        backbone_feats
    for idx, feat in enumerate(tensor_sequence(pts_feats)):
        save_tensor(f'{prefix}/neck_{idx}', feat, out_dir, manifest)

    lidar_bev = model._unwrap_single_bev(pts_feats)
    save_tensor(f'{prefix}/lidar_bev', lidar_bev, out_dir, manifest)
    return lidar_bev


def build_history_prev_bev(model, history_points, img_metas, out_dir, manifest,
                           dump_frontends, dump_sparse_stages):
    if history_points is None or len(history_points) == 0:
        return None

    batch_size = len(img_metas)
    history_by_sample = model._normalize_history_points(
        history_points, batch_size)
    if not history_by_sample or len(history_by_sample[0]) == 0:
        return None
    num_history = len(history_by_sample[0])
    queue_metas = [meta['queue_metas'] for meta in img_metas]
    prev_bev = None

    for step in range(num_history):
        step_points = [sample_history[step]
                       for sample_history in history_by_sample]
        for batch_idx, point in enumerate(step_points):
            save_tensor(f'history/{step}/raw_points_{batch_idx}', point,
                        out_dir, manifest)
        if dump_frontends:
            step_lidar_bev = run_lidar_frontend(
                model, step_points, img_metas, out_dir, manifest,
                f'history/{step}/frontend',
                dump_sparse_stages=dump_sparse_stages)
        else:
            step_lidar_bev = model.extract_lidar_bev_from_points(
                points_to_device(step_points, next(model.parameters()).device),
                img_metas)

        step_meta = None
        if prev_bev is not None:
            step_meta = [
                sample_queue_metas[step]
                for sample_queue_metas in queue_metas
            ]
            prev_bev = model.valid_prev_bev(prev_bev, step_meta)
            if prev_bev is None:
                step_meta = None
        prev_bev = model.encode_bev(
            step_lidar_bev, prev_bev, queue_meta=step_meta)
        save_tensor(f'history/{step}/bev_embed', prev_bev, out_dir, manifest)

    return prev_bev


def run_dense_bev_head(model, lidar_bev, prev_bev, img_metas, out_dir,
                       manifest):
    current_meta = model.current_queue_meta(img_metas)
    prev_bev = model.valid_prev_bev(prev_bev, current_meta)
    transformer = model.pts_bbox_head.transformer
    dense_prev_bev = transformer.rotate_prev_bev_if_needed(
        prev_bev, current_meta)
    shift = transformer.shift_from_queue_meta(
        current_meta, lidar_bev.device, lidar_bev.dtype)
    if shift is None:
        shift = lidar_bev.new_zeros((lidar_bev.size(0), 2))
    use_prev_bev = lidar_bev.new_full(
        (lidar_bev.size(0), ), 1.0 if dense_prev_bev is not None else 0.0)
    if dense_prev_bev is None:
        dense_prev_bev = lidar_bev.new_zeros(
            lidar_bev.size(0), model.pts_bbox_head.bev_embed_dims,
            model.pts_bbox_head.bev_h, model.pts_bbox_head.bev_w)

    save_tensor('dense/prev_bev', dense_prev_bev, out_dir, manifest)
    save_tensor('dense/shift', shift, out_dir, manifest)
    save_tensor('dense/use_prev_bev', use_prev_bev, out_dir, manifest)

    bev_embed = model.encode_bev(
        lidar_bev, prev_bev, queue_meta=current_meta)
    save_tensor('dense/bev_embed', bev_embed, out_dir, manifest)

    head = model.pts_bbox_head
    encoder_lidar_bev = head.lidar_input_proj(lidar_bev)
    bev_queries = head.bev_embedding.weight.to(
        dtype=lidar_bev.dtype, device=lidar_bev.device)
    bev_pos = head.positional_encoding(
        lidar_bev.size(0), lidar_bev.device, lidar_bev.dtype)
    boundary_prev = dense_prev_bev if prev_bev is not None else None
    bev_embed_boundary = transformer.encoder(
        encoder_lidar_bev,
        bev_queries=bev_queries,
        bev_pos=bev_pos,
        prev_bev=boundary_prev,
        shift=shift)
    save_tensor('dense/bev_embed_tensor_boundary', bev_embed_boundary,
                out_dir, manifest)
    diff = (bev_embed - bev_embed_boundary).abs()
    manifest.setdefault('comparisons', {})[
        'bev_embed_vs_tensor_boundary'] = {
            'max_abs': float(diff.max().detach().cpu()),
            'mean_abs': float(diff.mean().detach().cpu()),
        }

    preds = head.get_detections(model._wrap_single_bev(bev_embed))
    for name, value in preds.items():
        if torch.is_tensor(value):
            save_tensor(f'dense/{name}', value, out_dir, manifest)
    return bev_embed, preds


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for LiDAR golden dump.')

    set_seed(args.seed)
    cfg = load_config(args)
    dataset, sample = build_sample(cfg, args.split, args.index)

    out_root = Path(args.out_dir)
    run_dir = out_root / f'{args.split}_{args.index:06d}'
    mmcv.mkdir_or_exist(str(run_dir))

    manifest = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'split': args.split,
        'index': args.index,
        'dataset_type': type(dataset).__name__,
        'dataset_length': len(dataset),
        'device': str(device),
        'seed': args.seed,
        'tensors': {},
    }

    points = unwrap_data(sample['points'])
    history_points = unwrap_data(sample.get('history_points'))
    img_metas = unwrap_data(sample['img_metas'])
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    manifest['sample_idx'] = to_jsonable(img_metas[0].get('sample_idx'))
    current_meta = img_metas[0].get('queue_metas', {})
    if current_meta:
        manifest['token'] = to_jsonable(current_meta[max(current_meta)].get(
            'token'))
        manifest['scene_token'] = to_jsonable(
            current_meta[max(current_meta)].get('scene_token'))
    with open(run_dir / 'img_metas.json', 'w') as f:
        json.dump(to_jsonable(img_metas), f, indent=2)

    model, checkpoint = build_loaded_model(cfg, args.checkpoint, device)
    if 'meta' in checkpoint:
        manifest['checkpoint_meta'] = to_jsonable(checkpoint['meta'])

    with torch.no_grad():
        lidar_bev = run_lidar_frontend(
            model, points, img_metas, run_dir, manifest, 'current',
            dump_sparse_stages=not args.no_sparse_stages)
        prev_bev = None
        if not args.no_history:
            prev_bev = build_history_prev_bev(
                model, history_points, img_metas, run_dir, manifest,
                dump_frontends=args.dump_history_frontends,
                dump_sparse_stages=not args.no_sparse_stages)
            save_tensor('dense/raw_prev_bev', prev_bev, run_dir, manifest)
        run_dense_bev_head(model, lidar_bev, prev_bev, img_metas, run_dir,
                           manifest)

    manifest_path = run_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f'Wrote golden dump: {run_dir}')
    print(f'Wrote manifest: {manifest_path}')
    for name in [
            'current/lidar_bev', 'dense/prev_bev', 'dense/shift',
            'dense/bev_embed', 'dense/all_cls_scores',
            'dense/all_bbox_preds']:
        if name in manifest['tensors']:
            record = manifest['tensors'][name]
            print(f'{name}: shape={record.get("shape")} '
                  f'dtype={record.get("dtype")}')
    comparison = manifest.get('comparisons', {}).get(
        'bev_embed_vs_tensor_boundary')
    if comparison:
        print('bev_embed_vs_tensor_boundary:', comparison)


if __name__ == '__main__':
    main()
