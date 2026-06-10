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
import onnx
import onnx.helper as helper
import torch
import torch.nn as nn
from cumm import tensorview as tv
from mmcv import Config, DictAction
from mmcv.parallel import DataContainer
from mmcv.runner import load_checkpoint
from torch.nn import Parameter

import spconv.pytorch as spconv
from spconv.core import ConvAlgo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.mmdet3d_plugin.models.middle_encoders.sparse_encoder_spconv2 import (  # noqa: E501,E402
    SparseBasicBlockSpconv2,
)
from third_party.uniad_mmdet3d.datasets.builder import build_dataset  # noqa: E402
from third_party.uniad_mmdet3d.models.builder import build_model  # noqa: E402


avoid_reuse_container = []
obj_to_tensor_id = {}
nodes = []
initializers = []
enable_trace = False


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export UniAD LiDAR SparseEncoderSpconv2 to libspconv ONNX.'
    )
    parser.add_argument('config', help='LiDAR BEVFormer config path.')
    parser.add_argument('checkpoint', help='PyTorch checkpoint path.')
    parser.add_argument(
        '--golden-dir',
        default=None,
        help='Optional golden dump directory containing current/voxel_features.npy '
        'and current/voxel_coors.npy.')
    parser.add_argument(
        '--split',
        default='test',
        choices=['train', 'val', 'test'],
        help='Dataset split used when --golden-dir is not provided.')
    parser.add_argument(
        '--index',
        type=int,
        default=0,
        help='Dataset index used when --golden-dir is not provided.')
    parser.add_argument(
        '--onnx-file',
        default='./onnx/bevformer_lidar_sparse_encoder.onnx',
        help='Output sparse ONNX path for libspconv.')
    parser.add_argument(
        '--tensor-prefix',
        default='./dumped_inputs/bevformer_lidar_sparse_encoder/infer',
        help='Prefix for libspconv-format input/output tensors.')
    parser.add_argument(
        '--device',
        default='cuda:0',
        help='Torch device used for export tracing.')
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed for deterministic data paths.')
    parser.add_argument(
        '--output-bound',
        type=int,
        default=200000,
        help='Max sparse output points recorded on each SparseConvolution.')
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


def load_config(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    import_plugins(cfg, args.config)
    return cfg


def unwrap_data(value):
    return value.data if isinstance(value, DataContainer) else value


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


def build_loaded_model(cfg, checkpoint_path, device):
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
    model.to(device).eval()
    return model, checkpoint


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


def points_to_device(points, device):
    if torch.is_tensor(points):
        return [points.to(device)]
    return [point.to(device) for point in points]


def make_sparse_input_from_dataset(model, cfg, split, index, device):
    _, sample = build_sample(cfg, split, index)
    points = unwrap_data(sample['points'])
    img_metas = unwrap_data(sample['img_metas'])
    if isinstance(img_metas, dict):
        img_metas = [img_metas]
    point_list = points_to_device(points, device)
    with torch.no_grad():
        voxels, num_points, coors = model.voxelize(point_list)
        try:
            voxel_features = model.pts_voxel_encoder(
                voxels, num_points, coors, None, img_metas)
        except TypeError:
            voxel_features = model.pts_voxel_encoder(
                voxels, num_points, coors)
    return voxel_features, coors, {
        'source': 'dataset',
        'split': split,
        'index': index,
        'sample_idx': to_jsonable(img_metas[0].get('sample_idx')),
    }


def make_sparse_input_from_golden(golden_dir, device):
    golden_dir = Path(golden_dir)
    features = np.load(golden_dir / 'current/voxel_features.npy')
    coors = np.load(golden_dir / 'current/voxel_coors.npy')
    return (
        torch.from_numpy(features).to(device),
        torch.from_numpy(coors).to(device),
        {
            'source': 'golden',
            'golden_dir': str(golden_dir.resolve()),
        },
    )


def fuse_bn_weights(conv_w_oki, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    ndim = conv_w_oki.ndim - 2
    permute_to_oik = [0, ndim + 1] + [idx + 1 for idx in range(ndim)]
    conv_w_oik = conv_w_oki.permute(*permute_to_oik)
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)
    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)
    conv_w_oik = conv_w_oik * (
        bn_w * bn_var_rsqrt).reshape(
            [-1] + [1] * (conv_w_oik.ndim - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b
    permute_to_oki = [0] + [idx + 2 for idx in range(ndim)] + [1]
    conv_w_oki = conv_w_oik.permute(*permute_to_oki).contiguous()
    return Parameter(conv_w_oki), Parameter(conv_b)


def fuse_bn(conv, bn):
    if conv.training or bn.training:
        raise RuntimeError('Conv/BN fusion requires eval mode.')
    conv.weight, conv.bias = fuse_bn_weights(
        conv.weight, conv.bias, bn.running_mean, bn.running_var, bn.eps,
        bn.weight, bn.bias)


def set_attr_by_path(module, path, value):
    parts = path.split('.')
    parent = module
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], value)


def new_sparse_basic_block_forward(block):
    def forward(x):
        identity = x
        out = block.conv1(x)
        out = block.conv2(out)
        if block.downsample is not None:
            identity = block.downsample(x)
        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(block.relu(out.features))
        return out
    return forward


def fuse_sparse_basic_block(block):
    block.forward = new_sparse_basic_block_forward(block)
    block.conv1.act_type = tv.gemm.Activation.ReLU
    block.conv2.act_type = tv.gemm.Activation.None_
    fuse_bn(block.conv1, block.bn1)
    fuse_bn(block.conv2, block.bn2)
    delattr(block, 'bn1')
    delattr(block, 'bn2')


def is_sparse_conv(module):
    return isinstance(module, spconv.conv.SparseConvolution)


def fuse_sparse_encoder_layers(encoder):
    for name, module in list(encoder.named_modules()):
        if isinstance(module, spconv.SparseSequential) and len(module) >= 3:
            if (is_sparse_conv(module[0]) and
                    isinstance(module[1], nn.BatchNorm1d) and
                    isinstance(module[2], nn.ReLU)):
                conv = module[0]
                fuse_bn(conv, module[1])
                conv.act_type = tv.gemm.Activation.ReLU
                if len(module) == 3:
                    new_module = conv
                else:
                    new_module = spconv.SparseSequential(
                        *([conv] + [module[idx]
                                    for idx in range(3, len(module))]))
                set_attr_by_path(encoder, name, new_module)
        elif isinstance(module, SparseBasicBlockSpconv2):
            fuse_sparse_basic_block(module)
        elif isinstance(module, nn.ReLU):
            module.inplace = False
    return encoder


def set_output_bound(encoder, output_bound):
    for module in encoder.modules():
        if is_sparse_conv(module):
            module.output_bound = int(output_bound)


def register_node(fn):
    fnnames = fn.split('.')
    fn_module = eval('.'.join(fnnames[:-1]))
    fn_name = fnnames[-1]
    oldfn = getattr(fn_module, fn_name)

    def make_hook(bind_fn):
        ilayer = 0

        def internal_forward(self, *args):
            global enable_trace
            if not enable_trace:
                return oldfn(self, *args)

            global avoid_reuse_container
            nonlocal ilayer

            enable_trace = False
            y = oldfn(self, *args)
            bind_fn(self, ilayer, y, *args)
            enable_trace = True

            avoid_reuse_container.extend(list(args) + [y])
            ilayer += 1
            return y

        setattr(fn_module, fn_name, internal_forward)

    return make_hook


def __obj_to_id(obj):
    obj_id = id(obj)
    if isinstance(obj, spconv.SparseConvTensor):
        obj_id = id(obj.features)
    return obj_id


def register_tensor(obj):
    obj_to_tensor_id[__obj_to_id(obj)] = str(len(obj_to_tensor_id))


def get_tensor_id(obj):
    obj_id = __obj_to_id(obj)
    if obj_id not in obj_to_tensor_id:
        raise RuntimeError(
            'Cannot find traced tensor id. An operator may be missing from '
            'the sparse ONNX tracer.')
    return obj_to_tensor_id[obj_id]


def activation_name(conv):
    act_type = getattr(conv, 'act_type', tv.gemm.Activation.None_)
    if isinstance(act_type, str):
        return act_type
    mapping = {
        tv.gemm.Activation.ReLU: 'ReLU',
        tv.gemm.Activation.None_: 'None',
        tv.gemm.Activation.Sigmoid: 'Sigmoid',
        tv.gemm.Activation.LeakyReLU: 'LeakyReLU',
    }
    return mapping.get(act_type, 'None')


def algo_name(algo):
    mapping = {
        ConvAlgo.MaskImplicitGemm: 'MaskImplicitGemm',
        ConvAlgo.MaskSplitImplicitGemm: 'MaskSplitImplicitGemm',
        ConvAlgo.Native: 'Native',
    }
    return mapping.get(algo, 'Native')


def append_initializer(value, name):
    value = value.detach().cpu().numpy().astype(np.float16)
    initializers.append(
        helper.make_tensor(
            name=name,
            data_type=helper.TensorProto.DataType.FLOAT16,
            dims=list(value.shape),
            vals=value.tobytes(),
            raw=True))
    return name


@register_node('spconv.conv.SparseConvolution.forward')
def symbolic_sparse_convolution(conv, ilayer, y, x):
    register_tensor(y)
    inputs = [
        get_tensor_id(x),
        append_initializer(conv.weight.data, f'spconv{ilayer}.weight'),
    ]
    if conv.bias is not None:
        inputs.append(
            append_initializer(conv.bias.data, f'spconv{ilayer}.bias'))
    else:
        raise RuntimeError(f'SparseConvolution {ilayer} has no fused bias.')

    nodes.append(
        helper.make_node(
            'SparseConvolution',
            inputs,
            [get_tensor_id(y)],
            f'conv{ilayer}',
            ndim=conv.ndim,
            input_spatial_shape=x.spatial_shape,
            output_spatial_shape=y.spatial_shape,
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            output_bound=int(getattr(conv, 'output_bound', 200000)),
            stride=conv.stride,
            dilation=conv.dilation,
            padding=conv.padding,
            transposed=conv.transposed,
            inverse=conv.inverse,
            output_padding=conv.output_padding,
            groups=conv.groups,
            subm=conv.subm,
            rulebook=conv.indice_key,
            activation=activation_name(conv),
            algo=algo_name(conv.algo),
            input_shape=list(x.features.shape),
            output_shape=list(y.features.shape),
            precision='fp16',
            output_precision='fp16'))


@register_node('torch.nn.ReLU.forward')
def symbolic_relu(relu, ilayer, y, x):
    register_tensor(y)
    nodes.append(
        helper.make_node(
            'Relu', [get_tensor_id(x)], [get_tensor_id(y)], f'relu{ilayer}'))


@register_node('torch.Tensor.__add__')
def symbolic_add(a, ilayer, y, b):
    register_tensor(y)
    nodes.append(
        helper.make_node(
            'Add',
            [get_tensor_id(a), get_tensor_id(b)],
            [get_tensor_id(y)],
            f'add{ilayer}',
            precision='fp16',
            output_precision='fp16'))


@register_node('spconv.core.SparseConvTensor.dense')
def symbolic_sparse_dense(sparse_tensor, ilayer, y):
    register_tensor(y)
    nodes.append(
        helper.make_node(
            'ScatterDense',
            [get_tensor_id(sparse_tensor)],
            [get_tensor_id(y)],
            f'scatter{ilayer}',
            input_spatial_shape=sparse_tensor.spatial_shape,
            format='zyx',
            output_shape=list(y.size()),
            output_layout='NCHW',
            precision='fp16',
            output_precision='fp16'))


def symbolic_reshape(tensor, ilayer, y, *dims):
    register_tensor(y)
    nodes.append(
        helper.make_node(
            'Reshape',
            [get_tensor_id(tensor)],
            [get_tensor_id(y)],
            f'reshape{ilayer}',
            dims=list(dims)))


register_node('torch.Tensor.view')(symbolic_reshape)
register_node('torch.Tensor.reshape')(symbolic_reshape)


def make_encoder_forward(encoder):
    def forward(voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(
            voxel_features, coors, encoder.sparse_shape, int(batch_size))
        x = encoder.conv_input(input_sp_tensor)
        encode_features = []
        for encoder_layer in encoder.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)
        out = encoder.conv_out(encode_features[-1])
        spatial_features = out.dense()
        n, c, d, h, w = spatial_features.shape
        return spatial_features.view(n, c * d, h, w)
    return forward


def export_sparse_onnx(encoder, voxel_features, coors, batch_size, onnx_file):
    global avoid_reuse_container
    global obj_to_tensor_id
    global nodes
    global initializers
    global enable_trace

    avoid_reuse_container = []
    obj_to_tensor_id = {}
    nodes = []
    initializers = []
    encoder.forward = make_encoder_forward(encoder)

    with torch.no_grad():
        register_tensor(voxel_features)
        enable_trace = True
        dense_bev = encoder(voxel_features, coors, batch_size)
        enable_trace = False

    graph_inputs = [
        helper.make_value_info(
            name='0',
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=list(voxel_features.size())))
    ]
    graph_outputs = [
        helper.make_value_info(
            name=get_tensor_id(dense_bev),
            type_proto=helper.make_tensor_type_proto(
                elem_type=helper.TensorProto.DataType.FLOAT16,
                shape=list(dense_bev.size())))
    ]
    graph = helper.make_graph(
        name='bevformer_lidar_sparse_encoder',
        inputs=graph_inputs,
        outputs=graph_outputs,
        nodes=nodes,
        initializer=initializers)
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid('ai.onnx', 11)],
        producer_name='uniad_train',
        producer_version='spconv2')

    onnx_file = Path(onnx_file)
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(onnx_file))
    return dense_bev


DTYPE_TO_ID = {
    'int32': 1,
    'float16': 2,
    'float32': 3,
    'int64': 4,
    'uint64': 5,
    'uint32': 6,
    'int8': 7,
    'uint8': 8,
}


def save_tensor_file(tensor, file):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()
    else:
        tensor = np.asarray(tensor)
    dtype = str(tensor.dtype)
    if dtype not in DTYPE_TO_ID:
        raise RuntimeError(f'Unsupported tensor dtype for libspconv file: '
                           f'{dtype}')
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, 'wb') as f:
        header = np.asarray(
            [0x33ff1101, tensor.ndim, DTYPE_TO_ID[dtype]],
            dtype=np.int32)
        f.write(header.tobytes())
        f.write(np.asarray(tensor.shape, dtype=np.int32).tobytes())
        f.write(np.ascontiguousarray(tensor).tobytes())


def save_numpy(tensor, file):
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().numpy()
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    np.save(file, tensor)


def compare_with_golden_output(dense_bev, golden_dir):
    if golden_dir is None:
        return None
    middle_path = Path(golden_dir) / 'current/middle_bev.npy'
    if not middle_path.exists():
        return None
    golden = torch.from_numpy(np.load(middle_path)).to(
        dense_bev.device, dtype=torch.float32)
    got = dense_bev.float()
    diff = (got - golden).abs()
    return {
        'max_abs': float(diff.max().detach().cpu()),
        'mean_abs': float(diff.mean().detach().cpu()),
        'p99_abs': float(torch.quantile(diff.reshape(-1), 0.99).detach().cpu()),
    }


def check_onnx_status(onnx_file):
    try:
        onnx.checker.check_model(onnx.load(str(onnx_file)))
    except Exception as exc:
        return {
            'ok': False,
            'reason': str(exc),
            'note': ('libspconv sparse ONNX uses custom nodes parsed by '
                     'op_type, so standard ONNX checker failures are '
                     'expected for this graph.'),
        }
    return {'ok': True}


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for sparse ONNX export.')
    set_seed(args.seed)

    cfg = load_config(args)
    model, checkpoint = build_loaded_model(cfg, args.checkpoint, device)
    if args.golden_dir:
        voxel_features, coors, input_meta = make_sparse_input_from_golden(
            args.golden_dir, device)
    else:
        voxel_features, coors, input_meta = make_sparse_input_from_dataset(
            model, cfg, args.split, args.index, device)

    if coors.numel() == 0:
        raise RuntimeError('Sparse export input has no voxel coordinates.')
    batch_size = int(coors[:, 0].max().item()) + 1
    voxel_features = voxel_features.half().contiguous()
    coors = coors.int().contiguous()

    encoder = model.pts_middle_encoder.eval()
    fuse_sparse_encoder_layers(encoder)
    set_output_bound(encoder, args.output_bound)
    encoder.half()

    with torch.no_grad():
        dense_bev = export_sparse_onnx(
            encoder, voxel_features, coors, batch_size, args.onnx_file)

    tensor_prefix = Path(args.tensor_prefix)
    save_tensor_file(voxel_features, str(tensor_prefix) + '.voxels')
    save_tensor_file(coors, str(tensor_prefix) + '.coors')
    save_tensor_file(dense_bev, str(tensor_prefix) + '.dense')
    save_tensor_file(
        np.asarray([batch_size] + list(encoder.sparse_shape), dtype=np.int32),
        str(tensor_prefix) + '.info')
    save_numpy(voxel_features, str(tensor_prefix) + '.voxels.npy')
    save_numpy(coors, str(tensor_prefix) + '.coors.npy')
    save_numpy(dense_bev, str(tensor_prefix) + '.dense.npy')

    summary = {
        'config': str(Path(args.config).resolve()),
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'onnx_file': str(Path(args.onnx_file).resolve()),
        'tensor_prefix': str(tensor_prefix.resolve()),
        'input': input_meta,
        'batch_size': batch_size,
        'sparse_shape': list(encoder.sparse_shape),
        'voxel_features_shape': list(voxel_features.shape),
        'coors_shape': list(coors.shape),
        'dense_bev_shape': list(dense_bev.shape),
        'output_bound': args.output_bound,
        'num_nodes': len(nodes),
        'num_initializers': len(initializers),
        'onnx_checker': check_onnx_status(args.onnx_file),
        'checkpoint_meta': to_jsonable(checkpoint.get('meta', {})),
        'fp16_vs_golden_fp32': compare_with_golden_output(
            dense_bev, args.golden_dir),
    }
    summary_file = tensor_prefix.parent / 'sparse_export_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'Exported sparse ONNX: {args.onnx_file}')
    print(f'Wrote libspconv tensors: {tensor_prefix}.[voxels|coors|dense|info]')
    print(f'Wrote summary: {summary_file}')
    print(f'Nodes: {len(nodes)}, initializers: {len(initializers)}')
    print(f'Input features: {list(voxel_features.shape)}, coors: '
          f'{list(coors.shape)}, dense: {list(dense_bev.shape)}')
    if summary['fp16_vs_golden_fp32'] is not None:
        print('FP16 sparse output vs golden FP32:',
              summary['fp16_vs_golden_fp32'])


if __name__ == '__main__':
    main()
