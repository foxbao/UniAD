_base_ = [
    './base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train.py'
]

# B1.1 isolates the latent map-fusion path. The A-route lane-anchor gate and
# residual stay frozen at their best epoch-1 values, so validation changes can
# be attributed to map attention rather than continued lane-anchor drift.
# Diagnostic result (2026-07-10): stopped at iter 350 because map-only
# grad_norm stayed around 1e-5, versus roughly 1-2 for A/B1. Keep this config
# as a reproducible negative ablation, not as the recommended training route.
model = dict(
    freeze_except_prefixes=[
        'planning_head.map_attn_module',
        'planning_head.map_delta_proj',
        'planning_head.map_gate',
    ],
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
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
    'base_e2e_lidar_plan_mapfuse_v3_b11_mapfusion_only_train')
