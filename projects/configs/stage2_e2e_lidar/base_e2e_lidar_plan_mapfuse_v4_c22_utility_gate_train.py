_base_ = ['./base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_train.py']

# C2.2 keeps C2.1's discrete selector, anchor, and residual fixed. It trains
# only the scalar blend gate with a direct per-sample target: the least-squares
# alpha on the line from the frozen base trajectory to the map trajectory.
model = dict(
    freeze_except_eval=True,
    freeze_except_prefixes=[
        'planning_head.lane_anchor_gate_head',
    ],
    planning_head=dict(
        lane_anchor_static_gate_loss_weight=0.0,
        lane_anchor_selector_loss_weight=0.0,
        lane_anchor_utility_gate_loss_weight=1.0,
    ))

optimizer = dict(type='AdamW', lr=5e-5, weight_decay=0.01)
total_epochs = 1
runner = dict(type='EpochBasedRunner', max_epochs=1)

load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c22_utility_gate_train')
