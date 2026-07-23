_base_ = [
    './base_e2e_lidar_occworld_b13_dense_incremental_local_continuous10_eval.py'
]

# B14 keeps the raw B13 semantic prediction everywhere except flow-derived
# instance arrival/departure events. Threshold 0.9 is selected once on the
# frozen validation split; no trainable parameters are added.
model = dict(
    occ_head=dict(
        world_local_flow_overlay_threshold=0.9))
