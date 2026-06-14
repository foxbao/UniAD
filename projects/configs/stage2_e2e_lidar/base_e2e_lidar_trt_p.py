_base_ = ['../stage1_track_map_lidar/base_track_lidar_trt_p.py']

point_cloud_range = [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]

num_classes = 13
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
    type='UniADMotionLidarTRT',
    task_loss_weight=dict(track=1.0, motion=1.0),
    motion_head=dict(
        type='MotionHeadLidarTRTP',
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
            type='MotionTransformerDecoderTRTP',
            pc_range=point_cloud_range,
            embed_dims=_dim_,
            num_layers=3,
            transformerlayers=dict(
                type='MotionTransformerAttentionLayerTRTP',
                batch_first=True,
                attn_cfgs=[
                    dict(
                        type='MotionDeformableAttentionTRTP',
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

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_trt'
