_base_ = [
    './base_e2e_lidar_occworld_b15_full_train_continuous10_eval.py'
]

model = dict(
    occ_head=dict(
        world_use_query_occupancy_adapter=True,
        world_query_occupancy_adapter_scale=1.0))
