_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# Diagnostic-eval config (ego-status ablation, map ON). Reuse the v2 checkpoint
# but zero the sdc_track_query at inference, severing the ego position/velocity
# pathway into plan_query. Together with the plain v2 (ego-on, map-on),
# v2_nomap_eval (ego-on, map-off), and v2_egooff_nomap_eval (ego-off, map-off),
# this forms the 2x2 matrix that disambiguates:
#   - "map fusion too weak"  vs  "ego-status dominates the plan"
# Read: does the map-on/off L2 delta GROW once ego-status is removed? If yes,
# the kinematic prior was capping the map. If it stays ~0, the map is genuinely
# redundant with BEV in this ODD. See documents/mapfuse_planning_analysis.md
# §9c/§9d/§11. Eval-only; no retrain, weights loaded non-strict.
model = dict(
    planning_head=dict(
        use_map_lane=True,
        ablate_ego_status='zero',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2_egooff_eval'
