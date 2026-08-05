_base_ = ['./base_track_lidar.py']

# Drivable-only training with the dedicated LidarDrivableHead. Same compute that
# produces the ~0.78 baseline (deformable BEV encoder + stuff SegMaskHead +
# stuff query/cls), with the dead things branch (decoder, query_embedding,
# cls/reg branches, things_mask_head, Hungarian assigners, focal/bbox/iou
# losses) physically removed. Parallel to base_track_drivable_lidar_panseg.py
# (the original PansegformerHead variant, kept for A/B comparison); both inherit
# base_track_lidar.py directly and carry their own seg_head + pipeline.

_dim_ = 256
_pos_dim_ = _dim_ // 2
_ffn_dim_ = _dim_ * 2
_num_levels_ = 1
bev_h_ = 120
bev_w_ = 160
_feed_dim_ = _ffn_dim_
_dim_half_ = _pos_dim_
canvas_size = (bev_h_, bev_w_)

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]
point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]
file_client_args = dict(backend='disk')

model = dict(
    # 'map' is UniAD's upstream task name for the seg-head branch; it is the
    # prefix the detector stamps on the seg loss (logs as map.loss_mask_stuff).
    # Here that branch is drivable-only (LidarDrivableHead, a DiceLoss on the
    # drivable stuff mask), NOT an HD-map task: the drivable GT comes from LiDAR
    # geometry + raycast ground, not from a map. Distinct from the stage-2
    # map_lane_encoder, which IS a real surveyed HD-map prior. The key is kept
    # as 'map' to stay aligned with upstream and avoid breaking historical
    # loss curves / the prefix<->task_loss_weight contract.
    task_loss_weight=dict(track=1.0, map=1.0),
    seg_head=dict(
        type='LidarDrivableHead',
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
            type='SegDeformableEncoder',
            num_feature_levels=_num_levels_,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention',
                        embed_dims=_dim_,
                        num_levels=_num_levels_),
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5),
        stuff_transformer_head=dict(
            type='SegMaskHead',
            d_model=_dim_,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0)))

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
        raycast_ground_endpoint_height_mode='point_percentile_rescue',
        raycast_ground_endpoint_point_percentile=0.90,
        raycast_visible_ground_recovery=True,
        raycast_visible_ground_recovery_radius=3,
        raycast_visible_ground_recovery_min_neighbors=5,
        # Follow connected, visible candidate ground for up to 2.5 BEV
        # cells. This handles a continuous, slightly raised road surface
        # without bridging an unseen or blocked gap.
        raycast_visible_ground_recovery_distance_mode='geodesic',
        raycast_visible_ground_recovery_max_distance=2.5,
        raycast_visible_ground_recovery_min_component_cells=8,
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
        raycast_ground_endpoint_height_mode='point_percentile_rescue',
        raycast_ground_endpoint_point_percentile=0.90,
        raycast_visible_ground_recovery=True,
        raycast_visible_ground_recovery_radius=3,
        raycast_visible_ground_recovery_min_neighbors=5,
        raycast_visible_ground_recovery_distance_mode='geodesic',
        raycast_visible_ground_recovery_max_distance=2.5,
        raycast_visible_ground_recovery_min_component_cells=8,
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
