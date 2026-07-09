_base_ = ['./base_e2e_lidar_plan.py']

# Goal-conditioned planning (feat/plan-goal).
#
# Motivation: the inherited PlanningHeadSingleMode is pure trajectory imitation
# conditioned only on a coarse 3-class command (right/left/forward), which is
# itself derived from the GT trajectory. There is no way to say "drive toward
# THIS point". This config adds a continuous goal-point input:
#   * data:  with_sdc_goal=True makes the dataset sample sdc_goal (x,y in the
#            initial ego frame) from the ego's own future, farther than the
#            6-step planning horizon (NuScenesTraj.get_sdc_goal).
#   * model: planning_head.use_goal=True encodes sdc_goal into a conditioning
#            token concatenated into plan_query (fuser_dim 3 -> 4).
#   * loss:  UNCHANGED. The goal is only a directional prior; the near 6-step
#            L2 ADE + collision losses still supervise the trajectory. "Reaching
#            a far goal" is realised at deployment via receding-horizon replan
#            with a fixed global goal, not by extending the horizon here.
#
# Ablation: set model.planning_head.use_goal=False (or zero sdc_goal) to get a
# clean goal-on/off comparison and confirm the goal branch is actually learned
# (guarding against the same no-op collapse seen in base_e2e_lidar_plan_mapfuse).
#
# Start from base_e2e_lidar_occ (NOT a fully-trained planner): the planner and
# the goal branch co-adapt while the planner is still forming, so the goal gets
# a real gradient from step 0. This mirrors the base_e2e_lidar_plan_mapfuse_v2
# starting-point fix. Consequently the planning_head is trained fresh; the
# widened mlp_fuser.0 (embed_dims*4) and the new goal_encoder have no counterpart
# in the checkpoint and are initialised from scratch (load_checkpoint strict=False
# skips the shape-mismatched / missing keys).

model = dict(
    planning_head=dict(
        use_goal=True,
    ))

point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]
bev_h_ = 120
bev_w_ = 160
occ_n_future = 6
occ_receptive_field = 3

occflow_grid_conf = {
    'xbound': [-64.0, 64.0, 0.8],
    'ybound': [-48.0, 48.0, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
}

class_names = [
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'WheelCrane',
]

# Collect3D key list = the inherited base list + the two goal tensors. Kept in
# one place so train/test stay in sync.
collect_keys = [
    'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds',
    'gt_past_traj', 'gt_past_traj_mask', 'gt_fut_traj',
    'gt_fut_traj_mask',
    'gt_sdc_bbox', 'gt_sdc_label',
    'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask',
    'gt_lane_labels', 'gt_lane_bboxes', 'gt_lane_masks',
    'sdc_planning', 'sdc_planning_mask', 'command',
    'sdc_goal', 'sdc_goal_mask',
    'gt_future_boxes', 'gt_future_labels',
    'gt_segmentation', 'gt_instance',
    'gt_centerness', 'gt_offset', 'gt_flow', 'gt_backward_flow',
    'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid',
]

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D_E2E',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=True),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=occflow_grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_cls_ids=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        filter_invisible=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilterTrack', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilterTrack', classes=class_names),
    dict(
        type='GenerateKLDrivableMapLabels',
        use_map=False,
        point_cloud_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        augment_raycast_ground=True,
        keep_raycast_obstacles=False,
        box_z_origin='bottom'),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=collect_keys),
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=dict(backend='disk')),
    dict(
        type='LoadAnnotations3D_E2E',
        with_bbox_3d=True,
        with_label_3d=True,
        with_attr_label=False,
        with_future_anns=True,
        with_ins_inds_3d=True,
        ins_inds_add_1=True),
    dict(
        type='GenerateOccFlowLabels',
        grid_conf=occflow_grid_conf,
        ignore_index=255,
        only_vehicle=True,
        filter_cls_ids=[1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12],
        filter_invisible=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilterTrack', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilterTrack', classes=class_names),
    dict(
        type='GenerateKLDrivableMapLabels',
        use_map=False,
        point_cloud_range=point_cloud_range,
        bev_size=(bev_h_, bev_w_),
        augment_raycast_ground=True,
        keep_raycast_obstacles=False,
        box_z_origin='bottom'),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=collect_keys),
]

data = dict(
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline,
               with_sdc_goal=True,
               occ_receptive_field=occ_receptive_field,
               occ_n_future=occ_n_future),
    val=dict(pipeline=test_pipeline,
             with_sdc_goal=True,
             occ_receptive_field=occ_receptive_field,
             occ_n_future=occ_n_future),
    test=dict(pipeline=test_pipeline,
              with_sdc_goal=True,
              occ_receptive_field=occ_receptive_field,
              occ_n_future=occ_n_future))

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_goal'
