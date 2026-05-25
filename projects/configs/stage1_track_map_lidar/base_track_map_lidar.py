_base_ = ['../bevformer_lidar/base_bevformer_lidar.py']

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]
label_mapping = [
    0, 1, 2, 3, 4,
    5, 6, 7, 8, 9,
    10, 11, 8, 8, 12,
]
num_classes = len(class_names)
point_cloud_range = [-80.0, -48.0, -2.0, 80.0, 48.0, 6.0]
file_client_args = dict(backend='disk')
dataset_type = 'KlTrackDataset'
queue_length = 4
past_steps = 0
fut_steps = 6
train_gt_iou_threshold = 0.3

model = dict(
    type='UniADTrackLidar',
    return_query_feats=True,
    queue_length=queue_length,
    freeze_lidar_backbone=True,
    freeze_bev_encoder=False,
    score_thresh=0.4,
    filter_score_thresh=0.35,
    gt_iou_threshold=train_gt_iou_threshold,
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
    loss_cfg=dict(
        type='ClipMatcher',
        num_classes=num_classes,
        weight_dict=None,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0,
                      1.0, 1.0, 1.0, 0.2, 0.2],
        loss_past_traj_weight=0.0,
        with_sdc=False,
        assigner=dict(
            type='HungarianAssigner3DTrack',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            pc_range=point_cloud_range),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25)),
    pts_bbox_head=dict(
        with_track_branch=True,
        past_steps=past_steps,
        fut_steps=fut_steps))

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
    dict(type='ObjectRangeFilterTrack', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilterTrack', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds',
            'gt_past_traj', 'gt_past_traj_mask'
        ]),
]

data = dict(
    samples_per_gpu=1,
    train=dict(
        type=dataset_type,
        pipeline=train_pipeline,
        queue_length=queue_length,
        label_mapping=label_mapping),
    val=dict(queue_length=queue_length, label_mapping=label_mapping),
    test=dict(queue_length=queue_length, label_mapping=label_mapping))

total_epochs = 12
runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

load_from = './projects/work_dirs/bevformer_lidar/base_bevformer_lidar/latest.pth'
resume_from = None
work_dir = './projects/work_dirs/stage1_track_map_lidar/base_track_map_lidar'
find_unused_parameters = True
