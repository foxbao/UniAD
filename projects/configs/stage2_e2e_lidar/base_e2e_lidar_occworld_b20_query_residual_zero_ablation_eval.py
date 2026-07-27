_base_ = [
    './base_e2e_lidar_occworld_b20_query_residual_eval.py'
]

# Load the same B20 checkpoint but suppress its query input. Because the
# adapter has no bias, this must reproduce the B17A logits exactly.
model = dict(
    occ_head=dict(
        world_query_occupancy_adapter_scale=0.0))
