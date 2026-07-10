_base_ = ['./base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train.py']

# B1: lightly unfreeze the latent map-fusion path on top of the A-route
# learned lane-anchor fusion. This keeps the end-to-end detector/motion stack
# frozen, but lets the planner's map attention/gating layers adapt jointly with
# the learned lane-anchor gate/residual heads.
model = dict(
    freeze_except_prefixes=[
        'planning_head.lane_anchor_gate_head',
        'planning_head.lane_anchor_residual_head',
        'planning_head.map_attn_module',
        'planning_head.map_delta_proj',
        'planning_head.map_gate',
    ],
    planning_head=dict(
        use_map_lane=True,
        map_fusion_mode='gated_residual',
        lane_anchor_mode='learned_blend',
    ))

# Use a lower LR than A because B1 updates pre-existing latent fusion layers,
# not only newly initialized heads.
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
total_epochs = 1
# Validate the B1 direction after one epoch before extending training.
runner = dict(type='EpochBasedRunner', max_epochs=1)

# Intended starting point after A-route training finishes. Override this with
# epoch_1.pth/epoch_2.pth if you want an explicit checkpoint.
# A epoch 1 is the best checkpoint; epoch 2 regressed on planning avg.L2.
load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train/epoch_1.pth'
resume_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_b1_mapfusion_train'
