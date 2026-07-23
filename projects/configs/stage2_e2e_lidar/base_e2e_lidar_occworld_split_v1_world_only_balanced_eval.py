_base_ = ['./base_e2e_lidar_occworld_split_v1_eval.py']

model = dict(
    occ_head=dict(
        world_class_weights=[
            0.357565,
            1.470832,
            1.171603,
        ]))
