_base_ = ['./base_e2e_lidar_plan_mapfuse_v5_d11_utility_train.py']

# D2 removes the post-hoc utility gate. A shared head predicts 1s/2s/3s L2
# cost for every refined map candidate and the exact C2.3 fallback, then makes
# a threshold-free argmin decision on their mean predicted cost.
model = dict(
    freeze_except_prefixes=[
        'planning_head.map_multimodal_planner.candidate_cost_head',
    ],
    planning_head=dict(multimodal_planner=dict(
        use_candidate_cost=True,
        candidate_cost_loss_weight=1.0,
        candidate_cost_ranking_weight=0.2,
        candidate_cost_temperature=0.25,
        candidate_cost_target_clip=10.0,
        candidate_cost_background_weight=0.05,
        candidate_cost_oracle_weight=2.0,
        candidate_cost_fallback_weight=2.0,
        candidate_cost_hard_weight=1.0,
        candidate_cost_hard_count=16,
        candidate_cost_near_margin=0.1,
        candidate_cost_fallback_tiebreak=1e-4,
    )))

# Reuse D1.1 because its map-only scorer has useful top-1 candidates. D2.0
# intentionally freezes that representation to isolate cost calibration.
load_from = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v5_d11_utility_pilot_train/'
    'epoch_1.pth')
resume_from = None

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train')
