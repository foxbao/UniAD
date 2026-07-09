_base_ = ['../base_e2e_lidar_plan_mapfuse_v2.py']

# Diagnostic-eval: ego-status ablation mode='traj', map OFF.
# Part of the ego x map ablation matrix on the v2 checkpoint. mode='traj' zeros
# sdc_traj_query (the motion-predicted ego kinematic trend — the suspected true
# ego-status pathway, since the ep2 'zero' mode on sdc_track_query had ~0 effect).
# mode='both' zeros both ego pathways. Eval-only; no retrain, weights non-strict.
model = dict(
    planning_head=dict(
        use_map_lane=False,
        ablate_ego_status='traj',
    ))

load_from = None
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2_trajoff_nomap_eval'
