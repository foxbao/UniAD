_base_ = ['../stage1_track_map_lidar/base_track_lidar.py']

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
point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]
file_client_args = dict(backend='disk')
num_classes = len(class_names)
_dim_ = 256
_ffn_dim_ = _dim_ * 2
bev_h_ = 120
bev_w_ = 160
predict_steps = 12
predict_modes = 6
use_nonlinear_optimizer = False
pedestrian_id_list = [0]
vehicle_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
group_id_list = [pedestrian_id_list, vehicle_id_list]

model = dict(
    type='UniADMotionLidar',
    task_loss_weight=dict(track=1.0, motion=1.0),
    freeze_lidar_backbone=True,
    freeze_bev_encoder=True,
    motion_head=dict(
        type='MotionHeadLidar',
        bev_h=bev_h_,
        bev_w=bev_w_,
        num_query=600,
        num_classes=num_classes,
        predict_steps=predict_steps,
        predict_modes=predict_modes,
        embed_dims=_dim_,
        loss_traj=dict(
            type='TrajLoss',
            use_variance=True,
            cls_loss_weight=0.5,
            nll_loss_weight=0.5,
            loss_weight_minade=0.,
            loss_weight_minfde=0.25),
        num_cls_fcs=3,
        pc_range=point_cloud_range,
        group_id_list=group_id_list,
        vehicle_id_list=vehicle_id_list,
        num_anchor=6,
        use_nonlinear_optimizer=use_nonlinear_optimizer,
        anchor_info_path='data/others/motion_anchor_infos_kl.pkl',
        transformerlayers=dict(
            type='MotionTransformerDecoder',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,
            transformerlayers=dict(
                type='MotionTransformerAttentionLayer',
                batch_first=True,
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttention',
                        num_steps=predict_steps,
                        embed_dims=_dim_,
                        num_levels=1,
                        num_heads=8,
                        num_points=4,
                        sample_index=-1),
                ],
                feedforward_channels=_ffn_dim_,
                ffn_dropout=0.1,
                operation_order=('cross_attn', 'norm', 'ffn', 'norm')))))

# LiDAR-only stage-2 entry point: tracking plus MotionHead. It keeps the
# inherited SDC branch so MotionHead can supervise the ego/SDC trajectory.
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
            'gt_past_traj', 'gt_past_traj_mask', 'gt_fut_traj',
            'gt_fut_traj_mask',
            'gt_sdc_bbox', 'gt_sdc_label',
            'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask'
        ]),
]

data = dict(
    samples_per_gpu=1,
    train=dict(
        pipeline=train_pipeline,
        label_mapping=label_mapping,
        point_cloud_range=point_cloud_range))

optimizer = dict(type='AdamW', lr=2e-4, weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

load_from = './projects/work_dirs/stage1_track_map_lidar/base_track_lidar/latest.pth'
resume_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar'
find_unused_parameters = True
