_base_ = ['./base_e2e_lidar_occworld_split_v1_world_only.py']

# Computed only from the ten train scenes using inverse-sqrt frequency,
# normalized to mean one. Validation and test labels are not used.
model = dict(
    occ_head=dict(
        world_class_weights=[
            0.357565,
            1.470832,
            1.171603,
        ]))

work_dir = (
    './projects/work_dirs/stage2_e2e_lidar/'
    'base_e2e_lidar_occworld_split_v1_world_only_balanced')
