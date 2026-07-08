_base_ = ['./base_e2e_lidar_plan_mapfuse_v2.py']

# Diagnostic-eval config (ego-status ablation, map OFF). Fourth cell of the 2x2
# ego x map ablation matrix on the v2 checkpoint: sdc_track_query zeroed AND the
# planning-head map-lane attention disabled. Comparing this with
# v2_egooff_eval (ego-off, map-on) gives the map delta WITHOUT ego-status;
# comparing that delta to the ego-on map delta (v2 vs v2_nomap_eval) answers
# whether ego-status dominance was capping the map. Eval-only; no retrain.
model = dict(
    planning_head=dict(
        use_map_lane=False,
        ablate_ego_status='zero',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2_egooff_nomap_eval'
