_base_ = [
    './base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train.py'
]

# B2 moves map fusion after the frozen BEV planning decoder so its residual
# reaches trajectory regression directly. The modules use new post_map_ names,
# preventing the ineffective pre-BEV weights in the A checkpoint from loading.
# A small random delta and a mostly closed gate preserve the A trajectory at
# initialization while keeping gradients non-degenerate from the first step.
# A 20-iteration smoke run measured grad_norm=0.04-0.17 and only 0.13%-0.16%
# relative map injection, versus grad_norm around 1e-5 for the B1.1 placement.
model = dict(
    freeze_except_prefixes=[
        'planning_head.post_map_attn_module',
        'planning_head.post_map_delta_proj',
        'planning_head.post_map_gate',
    ],
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        map_fusion_position='post_bev',
        map_gate_init=-4.0,
        map_delta_init='small',
        lane_anchor_mode='learned_blend',
    ))

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v3_b2_postattn_mapfusion_train')
