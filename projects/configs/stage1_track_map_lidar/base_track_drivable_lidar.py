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
        type='PansegformerHead',
        bev_h=bev_h_,
        bev_w=bev_w_,
        canvas_size=canvas_size,
        pc_range=point_cloud_range,
        eval_drivable_only=True,
        # Drivable is a single stuff-class mask; the things detection branch
        # gets no GT (things_ratio=0) and runs dead. num_query only feeds that
        # branch (stuff uses an independent stuff_query of size num_stuff_
        # classes), so shrinking it from the original 600 cuts ~5/6 of the
        # dead things-decoder cost with zero effect on the drivable output.
        # NOTE: changing this changes query_embedding's shape -> cannot
        # resume/finetune from a 600-query checkpoint; train from scratch.
        num_query=100,
        num_classes=4,
        num_things_classes=3,
        num_stuff_classes=1,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        with_box_refine=True,
        transformer=dict(
            type='SegDeformableTransformer',
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
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
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
                            type='MultiScaleDeformableAttention',
                            embed_dims=_dim_,
                            num_levels=_num_levels_)
                    ],
                    feedforward_channels=_feed_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn',
                                     'norm', 'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=_dim_half_,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
        loss_mask=dict(type='DiceLoss', loss_weight=2.0),
        thing_transformer_head=dict(
            type='SegMaskHead',
            d_model=_dim_,
            nhead=8,
            num_decoder_layers=4),
        stuff_transformer_head=dict(
            type='SegMaskHead',
            d_model=_dim_,
            nhead=8,
            num_decoder_layers=6,
            self_attn=True),
        train_cfg=dict(
            assigner=dict(
                type='HungarianAssigner',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0)),
            assigner_with_mask=dict(
                type='HungarianAssigner_multi_info',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
                mask_cost=dict(type='DiceCost', weight=2.0)),
            sampler=dict(type='PseudoSampler'),
            sampler_with_mask=dict(type='PseudoSampler_segformer'))))

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
