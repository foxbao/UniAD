_base_ = ['./base_track_lidar_trt_p.py']

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 120
bev_w_ = 160
canvas_size = (bev_h_, bev_w_)
point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]

model = dict(
    type='UniADTrackDrivableLidarTRT',
    seg_head=dict(
        type='LidarDrivableHeadTRTP',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        pc_range=point_cloud_range,
        in_channels=_dim_,
        embed_dims=_dim_,
        num_stuff_classes=1,
        stuff_label_offset=3,
        eval_drivable_only=True,
        transformer=dict(
            type='SegDeformableEncoderTRTP',
            num_feature_levels=_num_levels_,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttentionTRTP',
                        embed_dims=_dim_,
                        num_levels=_num_levels_),
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_pos_dim_,
            normalize=True,
            offset=-0.5),
        stuff_transformer_head=dict(
            type='SegMaskHead',
            d_model=_dim_,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True)))

work_dir = (
    './projects/work_dirs/stage1_track_map_lidar/'
    'base_track_drivable_lidar_trt')
