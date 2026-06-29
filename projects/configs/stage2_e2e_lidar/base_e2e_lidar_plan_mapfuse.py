_base_ = ['./base_e2e_lidar_plan.py']

# Planning-specific HD-map fusion. Unlike base_e2e_lidar_HDMap, this config
# does not let every actor's motion query attend to the navigation map. The map
# is encoded once by the detector and fused only inside PlanningHeadSingleMode,
# which matches our use case: the surveyed map is a useful prior for the ego
# vehicle's planned path, but it is not a hard constraint for port actors.
model = dict(
    map_lane_encoder=dict(
        map_path='data/kl_8/map/base_map.txt',
        num_lanes=64,
        num_points_per_lane=20,
    ),
    motion_head=dict(
        map_agent_scope='none',
    ),
    planning_head=dict(
        use_map_lane=True,
        map_local_k=16,
        map_attn_layers=1,
        map_gate_init=-2.0,
    ))

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse'
