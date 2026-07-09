_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# v3 trainable lane-anchor fusion. Starts from the measured-best eval recipe
# (best-endpoint, bidirectional, local-relative lane anchor, alpha ~= 0.3), but
# replaces the fixed alpha with a small learned gate and zero-initialized
# residual-to-anchor head. The initial forward is therefore close to the
# successful blend03 eval run, while training can learn when/how much to trust
# map geometry.
model = dict(
    task_loss_weight=dict(track=0.0, map=0.0, motion=0.0, occ=0.0,
                          planning=1.0),
    freeze_except_prefixes=[
        'planning_head.lane_anchor_gate_head',
        'planning_head.lane_anchor_residual_head',
    ],
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='learned_blend',
        lane_anchor_scale=1.0,
        lane_anchor_offset_scale=0.0,
        lane_anchor_sample_mode='local_forward',
        lane_anchor_reference='relative_start',
        lane_anchor_select_mode='best_endpoint',
        lane_anchor_candidate_k=16,
        lane_anchor_direction_mode='bidirectional',
        lane_anchor_init_alpha=0.3,
        lane_anchor_static_gate_loss_weight=0.1,
        lane_anchor_static_disp_thresh=0.5,
    ))

optimizer = dict(type='AdamW', lr=1e-4, weight_decay=0.01)
total_epochs = 2
runner = dict(type='EpochBasedRunner', max_epochs=2)

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2/epoch_4.pth'
resume_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train'
