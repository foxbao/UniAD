_base_ = ['./base_e2e_lidar_HDMap.py']

# Weak navigation-map prior. Port actors are not strictly constrained by this
# map: cranes/forklifts can rotate or move freely, and even IGVs may be
# remote-controlled. Still, ordinary long-range movement often follows the same
# ordered corridors. Treat surveyed lanes as a weak learned prior instead of a
# hard constraint.
model = dict(
    motion_head=dict(
        # Each agent sees nearby lanes, but MotionFormer starts almost exactly
        # from the no-map behavior and learns how much lane context to use.
        map_agent_scope='all',
        transformerlayers=dict(map_gate_init=-4.0)))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap_weak'
