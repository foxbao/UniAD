_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d1_multimodal_train.py']

# D1.1 separates map-mode ranking from the fallback decision. The map scorer
# is supervised only among map candidates; a binary utility gate decides
# whether its current top-1 beats the untouched C2.3 fallback by at least 1 cm.
model = dict(planning_head=dict(multimodal_planner=dict(
    use_utility_gate=True,
    utility_gate_init=-2.0,
    utility_gate_loss_weight=1.0,
    utility_min_improvement=0.01,
)))

# Start D1.1 fresh: the D1 scorer learned the fallback-source shortcut.
load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v4_c23_eval_gate_pilot_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v5_d11_utility_train')
