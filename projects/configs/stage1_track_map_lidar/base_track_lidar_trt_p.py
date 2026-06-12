_base_ = ['../_base_/default_runtime.py']

plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]
num_classes = len(class_names)

point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]
voxel_size = [0.1, 0.1, 0.2]
queue_length = 5
past_steps = 4
fut_steps = 4
train_gt_iou_threshold = 0.3

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 120
bev_w_ = 160

model = dict(
    type='UniADTrackLidarTRT',
    point_cloud_range=point_cloud_range,
    num_query=600,
    embed_dims=_dim_,
    num_classes=num_classes,
    video_test_mode=True,
    queue_length=queue_length,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    gt_iou_threshold=train_gt_iou_threshold,
    with_sdc=True,
    qim_args=dict(
        qim_type='QIMBase',
        merger_dropout=0,
        update_query_pos=True,
        fp_ratio=0.3,
        random_drop=0.1),
    mem_args=dict(
        memory_bank_type='MemoryBank',
        memory_bank_score_thresh=0.0,
        memory_bank_len=4),
    bbox_coder=dict(
        type='DETRTrack3DCoder',
        post_center_range=[-64.0, -48.0, -10.0, 64.0, 48.0, 10.0],
        pc_range=point_cloud_range,
        max_num=300,
        num_classes=num_classes,
        score_threshold=0.0,
        with_nms=False,
        iou_thres=0.3),
    loss_cfg=None,
    pts_bbox_head=dict(
        type='BEVFormerLidarTrackHeadTRTP',
        in_channels=_dim_,
        num_classes=num_classes,
        num_query=600,
        embed_dims=_dim_,
        code_size=10,
        with_box_refine=True,
        as_two_stage=False,
        sync_cls_avg_factor=True,
        past_steps=past_steps,
        fut_steps=fut_steps,
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-64.0, -48.0, -10.0, 64.0, 48.0, 10.0],
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
            type='LidarPerceptionTransformerTRTP',
            point_cloud_range=point_cloud_range,
            use_shift=True,
            rotate_prev_bev=True,
            encoder=dict(
                type='LidarBEVFormerEncoderTRTP',
                num_layers=6,
                num_heads=8,
                temporal_num_points=4,
                spatial_num_points=8,
                ffn_channels=_ffn_dim_,
                dropout=0.1,
                transformerlayers=dict(
                    type='LidarBEVFormerLayerTRTP',
                    attn_cfgs=[
                        dict(
                            type='LidarTemporalSelfAttentionTRTP',
                            embed_dims=_dim_,
                            num_heads=8,
                            num_levels=1,
                            num_points=4),
                        dict(
                            type='LidarSpatialCrossAttentionTRTP',
                            embed_dims=_dim_,
                            num_heads=8,
                            num_levels=1,
                            num_points=8),
                    ],
                    ffn_channels=_ffn_dim_,
                    dropout=0.1)),
            decoder=dict(
                type='DetectionTransformerDecoderTRTP',
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
                            type='CustomMSDeformableAttentionTRTP',
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
            post_center_range=[-64.0, -48.0, -10.0, 64.0, 48.0, 10.0])))

fp16 = None
find_unused_parameters = False
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = './projects/work_dirs/stage1_track_map_lidar/base_track_lidar_trt'
