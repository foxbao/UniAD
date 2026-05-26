_base_ = ['./base_e2e_lidar.py']

# Enable HD map lane encoder for MotionFormer cross-attention.
# Encodes surveyed lane centerlines directly as lane_query tensors.
model = dict(
    motion_head=dict(
        map_lane_encoder=dict(
            map_path='data/kl_8/map/base_map.txt',
            num_lanes=64,
            num_points_per_lane=20,
        )))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_map/'
