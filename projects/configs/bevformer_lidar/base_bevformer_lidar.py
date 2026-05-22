_base_ = ['../_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

dataset_type = 'KlBEVFormerDataset'
data_root = 'data/kl_8/'
data_prefix = dict(
    pts='v1.0-trainval/samples',
    img='v1.0-trainval/sample',
    sweeps='v1.0-trainval/samples')
file_client_args = dict(backend='disk')

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]
# Original KL labels are 15-way.  Collapse Lorry and ConstructionVehicle
# into OtherVehicle without rewriting the annotation pkl files.
label_mapping = [
    0,   # Pedestrian
    1,   # Car
    2,   # IGV-Full
    3,   # Truck
    4,   # Trailer-Empty
    5,   # Trailer-Full
    6,   # IGV-Empty
    7,   # Crane
    8,   # OtherVehicle
    9,   # Cone
    10,  # ContainerForklift
    11,  # Forklift
    8,   # Lorry -> OtherVehicle
    8,   # ConstructionVehicle -> OtherVehicle
    12,  # WheelCrane
]
num_classes = len(class_names)
input_modality = dict(use_lidar=True, use_camera=False)

point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
voxel_size = [0.1, 0.1, 0.2]
sparse_shape = [41, 960, 1600]
queue_length = 4
_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 120
bev_w_ = 200

model = dict(
    type='BEVFormerLidar',
    point_cloud_range=point_cloud_range,
    num_query=600,
    embed_dims=_dim_,
    video_test_mode=True,
    return_query_feats=False,
    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoderSpconv2',
        in_channels=4,
        sparse_shape=sparse_shape,
        order=('conv', 'norm', 'act'),
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, (0, 1, 1)),
                          (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=_dim_,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[128, 128],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='BEVFormerLidarHead',
        in_channels=_dim_,
        num_classes=num_classes,
        num_query=600,
        embed_dims=_dim_,
        code_size=10,
        with_box_refine=True,
        as_two_stage=False,
        sync_cls_avg_factor=True,
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-80.0, -48.0, -10.0, 80.0, 48.0, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=num_classes),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_),
        bev_h=bev_h_,
        bev_w=bev_w_,
        transformer=dict(
            type='LidarPerceptionTransformer',
            point_cloud_range=point_cloud_range,
            use_shift=True,
            rotate_prev_bev=True,
            encoder=dict(
                type='LidarBEVFormerEncoder',
                num_layers=6,
                num_heads=8,
                temporal_num_points=4,
                spatial_num_points=8,
                ffn_channels=_ffn_dim_,
                dropout=0.1),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='CustomMSDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_),
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm')))),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    train_cfg=dict(
        pts=dict(
            pi_symmetric_class_indices=[2, 6, 12],
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0),
                pc_range=point_cloud_range))),
    test_cfg=dict(
        pts=dict(
            max_num=300,
            score_threshold=0.05,
            post_center_range=[-80.0, -48.0, -10.0, 80.0, 48.0, 10.0])))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d']),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points']),
]

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        label_mapping=label_mapping,
        modality=input_modality,
        test_mode=False,
        data_prefix=data_prefix,
        queue_length=queue_length,
        use_valid_flag=True,
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        label_mapping=label_mapping,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kl_infos_val.pkl',
        pipeline=test_pipeline,
        classes=class_names,
        label_mapping=label_mapping,
        modality=input_modality,
        test_mode=True,
        data_prefix=data_prefix,
        queue_length=queue_length,
        box_type_3d='LiDAR'),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'))

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 6
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, pipeline=test_pipeline)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# The mmcv-1.4 deformable-attention CUDA op in this environment does not
# implement Half kernels, so keep this LiDAR BEVFormer config in FP32.
fp16 = None
find_unused_parameters = False
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './projects/work_dirs/bevformer_lidar/base_bevformer_lidar'
