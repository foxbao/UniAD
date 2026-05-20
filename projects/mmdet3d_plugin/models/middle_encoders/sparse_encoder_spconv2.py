from mmcv.cnn import build_norm_layer
from mmcv.runner import BaseModule, auto_fp16
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn

from mmdet3d.models.builder import MIDDLE_ENCODERS

try:
    import spconv.pytorch as spconv
except ImportError as exc:
    spconv = None
    _SPCONV_IMPORT_ERROR = exc
else:
    _SPCONV_IMPORT_ERROR = None


def _require_spconv2():
    if spconv is None:
        raise ImportError(
            'SparseEncoderSpconv2 requires spconv 2.x. Install a matching '
            'package such as spconv-cu113==2.3.6.') from _SPCONV_IMPORT_ERROR


def replace_feature(out, new_features):
    return out.replace_feature(new_features)


class SparseBasicBlockSpconv2(BasicBlock, spconv.SparseModule if spconv else nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 indice_key=None,
                 conv_cfg=None,
                 norm_cfg=None):
        _require_spconv2()
        BaseModule.__init__(self)
        if conv_cfg is None:
            conv_cfg = dict(type='SubMConv3d')
        if norm_cfg is None:
            norm_cfg = dict(type='BN1d')

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        conv_type = conv_cfg.get('type', 'SubMConv3d')
        self.conv1 = _build_spconv_layer(
            conv_type,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = _build_spconv_layer(
            conv_type,
            planes,
            planes,
            3,
            padding=1,
            bias=False,
            indice_key=indice_key)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x.features

        out = self.conv1(x)
        out = replace_feature(out, self.norm1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.norm2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x).features

        out = replace_feature(out, out.features + identity)
        out = replace_feature(out, self.relu(out.features))

        return out


def _build_spconv_layer(conv_type,
                        in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=0,
                        bias=False,
                        indice_key=None):
    _require_spconv2()
    conv_cls = getattr(spconv, conv_type)
    if conv_type.startswith('SparseInverseConv'):
        return conv_cls(
            in_channels,
            out_channels,
            kernel_size,
            bias=bias,
            indice_key=indice_key)
    return conv_cls(
        in_channels,
        out_channels,
        kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        indice_key=indice_key)


def make_sparse_convmodule_spconv2(in_channels,
                                   out_channels,
                                   kernel_size,
                                   indice_key,
                                   stride=1,
                                   padding=0,
                                   conv_type='SubMConv3d',
                                   norm_cfg=None,
                                   order=('conv', 'norm', 'act')):
    _require_spconv2()
    assert isinstance(order, tuple) and len(order) <= 3
    assert set(order) | {'conv', 'norm', 'act'} == {'conv', 'norm', 'act'}
    if norm_cfg is None:
        norm_cfg = dict(type='BN1d')

    layers = []
    for layer in order:
        if layer == 'conv':
            layers.append(
                _build_spconv_layer(
                    conv_type,
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    indice_key=indice_key))
        elif layer == 'norm':
            layers.append(build_norm_layer(norm_cfg, out_channels)[1])
        elif layer == 'act':
            layers.append(nn.ReLU(inplace=True))

    return spconv.SparseSequential(*layers)


@MIDDLE_ENCODERS.register_module()
class SparseEncoderSpconv2(nn.Module):
    """SECOND sparse encoder backed by spconv 2.x.

    This mirrors mmdet3d 0.17's SparseEncoder but keeps the backend opt-in so
    existing configs can still use the vendored legacy sparse conv.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_cfg=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64),
                                   (64, 64, 64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1),
                                   ((0, 1, 1), 1, 1)),
                 block_type='conv_module'):
        _require_spconv2()
        super().__init__()
        assert block_type in ['conv_module', 'basicblock']
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}

        if self.order[0] != 'conv':
            self.conv_input = make_sparse_convmodule_spconv2(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:
            self.conv_input = make_sparse_convmodule_spconv2(
                in_channels,
                self.base_channels,
                3,
                norm_cfg=norm_cfg,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            make_sparse_convmodule_spconv2,
            norm_cfg,
            self.base_channels,
            block_type=block_type)

        self.conv_out = make_sparse_convmodule_spconv2(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_cfg=norm_cfg,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        n, c, d, h, w = spatial_features.shape
        spatial_features = spatial_features.view(n, c * d, h, w)

        return spatial_features

    def make_encoder_layers(self,
                            make_block,
                            norm_cfg,
                            in_channels,
                            block_type='conv_module',
                            conv_cfg=dict(type='SubMConv3d')):
        assert block_type in ['conv_module', 'basicblock']
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                if i != 0 and j == 0 and block_type == 'conv_module':
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                elif block_type == 'basicblock':
                    if j == len(blocks) - 1 and i != len(
                            self.encoder_channels) - 1:
                        blocks_list.append(
                            make_block(
                                in_channels,
                                out_channels,
                                3,
                                norm_cfg=norm_cfg,
                                stride=2,
                                padding=padding,
                                indice_key=f'spconv{i + 1}',
                                conv_type='SparseConv3d'))
                    else:
                        blocks_list.append(
                            SparseBasicBlockSpconv2(
                                out_channels,
                                out_channels,
                                norm_cfg=norm_cfg,
                                conv_cfg=conv_cfg,
                                indice_key=f'subm{i + 1}'))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_cfg=norm_cfg,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels
