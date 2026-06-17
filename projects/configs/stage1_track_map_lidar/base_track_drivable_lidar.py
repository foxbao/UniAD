_base_ = ['./base_track_lidar.py']

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 120
bev_w_ = 160
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

# Inherits class_names / label_mapping / point_cloud_range / dataset_type /
# queue_length / past_steps / fut_steps / train_gt_iou_threshold /
# file_client_args from base_track_lidar.py.
class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]
point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]
file_client_args = dict(backend='disk')

model = dict(
    task_loss_weight=dict(track=1.0, map=1.0),
    seg_head=dict(
        # Drivable is a single static, dense, current-frame mask. Instead of
        # the PansegformerHead (HD-map vector panoptic head whose things
        # detection branch ran dead with no GT), use a small occ-style dense
        # head: BEV -> single-channel drivable logit, Dice + sigmoid-BCE.
        # No queries / transformer / future-time dim. Drop-in: same forward_
        # train/forward_test contract, so the detector is unchanged.
        type='DrivableOccHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        in_channels=_dim_,
        proj_channels=_dim_,
        num_conv=4,
        eval_drivable_only=True,
        loss_dice=dict(type='DiceLoss', loss_weight=2.0),
        pos_weight=1.0))

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
    dict(
        type='GenerateKLDrivableMapLabels',
        use_map=False,
        point_cloud_range=point_cloud_range,
        bev_size=canvas_size,
        augment_raycast_ground=True,
        keep_raycast_obstacles=False,
        box_z_origin='bottom'),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds',
            'gt_past_traj', 'gt_past_traj_mask', 'gt_fut_traj',
            'gt_fut_traj_mask',
            'gt_sdc_bbox', 'gt_sdc_label',
            'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask',
            'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks'
        ]),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
    # LoadAnnotations3D supplies gt_bboxes_3d so GenerateKLDrivableMapLabels'
    # raycast obstacle suppression matches train-time behaviour. Without
    # it, drivable masks at eval time include voxels inside obstacles,
    # inflating lane IoU relative to training.
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='GenerateKLDrivableMapLabels',
        use_map=False,
        point_cloud_range=point_cloud_range,
        bev_size=canvas_size,
        augment_raycast_ground=True,
        keep_raycast_obstacles=False,
        box_z_origin='bottom'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(
        type='Collect3D',
        keys=['points', 'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks']),
]

data = dict(
    train=dict(
        pipeline=train_pipeline,
        point_cloud_range=point_cloud_range),
    val=dict(
        pipeline=test_pipeline,
        point_cloud_range=point_cloud_range),
    test=dict(
        pipeline=test_pipeline,
        point_cloud_range=point_cloud_range))

work_dir = './projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar'
