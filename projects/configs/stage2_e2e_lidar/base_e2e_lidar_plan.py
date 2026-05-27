_base_ = ['./base_e2e_lidar_occ.py']

# Planning consumes the current frame plus planning_steps future frames
# (collision loss uses indices [1:planning_steps+1]). Bump occ_n_future
# from 4 to 6 so GenerateOccFlowLabels emits enough future boxes.
occ_n_future = 6
occ_receptive_field = 3
planning_steps = 6

point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]
bev_h_ = 120
bev_w_ = 160

occflow_grid_conf = {
    'xbound': [-64.0, 64.0, 0.8],
    'ybound': [-48.0, 48.0, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
}

# Disable the optional non-linear collision optimizer — its inner
# IPOPT solver hard-codes nuScenes' 0.5 m / 5-frame BEV grid and would
# not match our 0.8 m / 6-step setup. Keep collision constraints in the
# differentiable loss only.
model = dict(
    type='UniADMotionLidar',
    task_loss_weight=dict(track=1.0, motion=1.0, occ=1.0, planning=1.0),
    planning_head=dict(
        type='PlanningHeadSingleMode',
        bev_h=bev_h_,
        bev_w=bev_w_,
        embed_dims=256,
        planning_steps=planning_steps,
        with_adapter=True,
        use_col_optim=False,
        loss_planning=dict(type='PlanningLoss'),
        loss_collision=[
            dict(type='CollisionLoss', delta=0.0, weight=2.5,
                 ego_width=3.0, ego_length=14.6),
            dict(type='CollisionLoss', delta=0.5, weight=1.0,
                 ego_width=3.0, ego_length=14.6),
            dict(type='CollisionLoss', delta=1.0, weight=0.25,
                 ego_width=3.0, ego_length=14.6),
        ]))

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
    dict(type='ObjectNameFilterTrack',
         classes=[
             'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
             'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
             'ContainerForklift', 'Forklift', 'WheelCrane',
         ]),
    dict(type='PointShuffle'),
    dict(type='DefaultFormatBundle3D',
         class_names=[
             'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
             'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
             'ContainerForklift', 'Forklift', 'WheelCrane',
         ]),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds',
            'gt_past_traj', 'gt_past_traj_mask', 'gt_fut_traj',
            'gt_fut_traj_mask',
            'gt_sdc_bbox', 'gt_sdc_label',
            'gt_sdc_fut_traj', 'gt_sdc_fut_traj_mask',
            'sdc_planning', 'sdc_planning_mask', 'command',
            'gt_future_boxes', 'gt_future_labels',
            'gt_segmentation', 'gt_instance',
            'gt_centerness', 'gt_offset', 'gt_flow', 'gt_backward_flow',
            'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid',
        ]),
]

data = dict(
    samples_per_gpu=1,
    train=dict(pipeline=train_pipeline,
               occ_receptive_field=occ_receptive_field,
               occ_n_future=occ_n_future))

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan'
