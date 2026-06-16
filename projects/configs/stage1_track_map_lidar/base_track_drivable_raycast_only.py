_base_ = ['./base_track_drivable_lidar.py']

# Raycast-only drivable target.
#
# The KL HD-map is a *navigation* map (route topology / connectivity), not a
# geometric description of the true drivable surface, so feeding it into the
# seg-head target can mislabel drivable space. This variant drops the map
# entirely (`use_map=False`) and builds the drivable ground-truth purely from
# the per-frame LiDAR raycast ground estimate (minus obstacles). Everything
# else is inherited from base_track_drivable_lidar.py.
#
# Diagnostics (alignment / sweep / GT visualiser under tools/analysis_tools)
# all still work: pass this config and the map panels simply show empty.

bev_h_ = 120
bev_w_ = 160
canvas_size = (bev_h_, bev_w_)
point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]
file_client_args = dict(backend='disk')

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
        box_z_origin='center'),
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
        box_z_origin='center'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(
        type='Collect3D',
        keys=['points', 'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks']),
]

data = dict(
    train=dict(pipeline=train_pipeline, point_cloud_range=point_cloud_range),
    val=dict(pipeline=test_pipeline, point_cloud_range=point_cloud_range),
    test=dict(pipeline=test_pipeline, point_cloud_range=point_cloud_range))

work_dir = ('./projects/work_dirs/stage1_track_map_lidar/'
            'base_track_drivable_lidar_raycast_only')
