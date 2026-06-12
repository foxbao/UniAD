_base_ = ['./base_e2e_lidar.py']

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
bev_h_ = 120
bev_w_ = 160
vehicle_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]

occ_n_future = 4
occ_receptive_field = 3
occflow_grid_conf = {
    'xbound': [-64.0, 64.0, 0.8],
    'ybound': [-48.0, 48.0, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
}

model = dict(
    type='UniADMotionLidar',
    task_loss_weight=dict(track=1.0, motion=1.0, occ=1.0),
    occ_head=dict(
        type='OccHead',
        receptive_field=occ_receptive_field,
        n_future=occ_n_future,
        grid_conf=occflow_grid_conf,
        bev_grid_conf=occflow_grid_conf,
        bev_size=(bev_h_, bev_w_),
        ignore_index=255,
        bev_proj_dim=256,
        bev_proj_nlayers=4,
        attn_mask_thresh=0.3,
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=5,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=2048,
                operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        query_dim=256,
        query_mlp_layers=3,
        aux_loss_weight=1.,
        loss_mask=dict(
            type='FieryBinarySegmentationLoss',
            use_top_k=True,
            top_k_ratio=0.25,
            future_discount=0.95,
            loss_weight=5.0,
            ignore_index=255),
        loss_dice=dict(
            type='DiceLossWithMasks',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            ignore_index=255,
            loss_weight=1.0),
        pan_eval=True,
        test_seg_thresh=0.1,
        test_with_track_score=True))

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
        file_client_args=file_client_args),
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
        filter_cls_ids=vehicle_id_list,
        filter_invisible=False),
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
            'gt_fut_traj_mask', 'gt_segmentation', 'gt_instance',
            'gt_centerness', 'gt_offset', 'gt_flow', 'gt_backward_flow',
            'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid'
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
        filter_cls_ids=vehicle_id_list,
        filter_invisible=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilterTrack', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilterTrack', classes=class_names),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'points', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_inds',
            'gt_past_traj', 'gt_past_traj_mask', 'gt_fut_traj',
            'gt_fut_traj_mask', 'gt_segmentation', 'gt_instance',
            'gt_centerness', 'gt_offset', 'gt_flow', 'gt_backward_flow',
            'gt_occ_has_invalid_frame', 'gt_occ_img_is_valid'
        ]),
]

data = dict(
    samples_per_gpu=1,
    train=dict(
        type='KlTrackDataset',
        pipeline=train_pipeline,
        label_mapping=label_mapping,
        occ_receptive_field=occ_receptive_field,
        occ_n_future=occ_n_future,
        occ_filter_invalid_sample=False),
    val=dict(
        type='KlTrackDataset',
        pipeline=test_pipeline,
        label_mapping=label_mapping,
        occ_receptive_field=occ_receptive_field,
        occ_n_future=occ_n_future,
        occ_filter_invalid_sample=False),
    test=dict(
        type='KlTrackDataset',
        pipeline=test_pipeline,
        label_mapping=label_mapping,
        occ_receptive_field=occ_receptive_field,
        occ_n_future=occ_n_future,
        occ_filter_invalid_sample=False))

evaluation = dict(interval=1, pipeline=test_pipeline)
load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar/latest.pth'
resume_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_occ'
