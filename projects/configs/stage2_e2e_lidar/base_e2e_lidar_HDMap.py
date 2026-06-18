_base_ = ['./base_e2e_lidar_plan.py']

# Adds one thing on top of base_e2e_lidar_plan (track + drivable + motion +
# occ + planning): an HD map lane encoder for MotionFormer. It parses surveyed
# lane centerlines from the protobuf-text map once, then per frame transforms
# them into ego coordinates (via img_metas' ego2global) and encodes them as
# lane_query tensors that feed MotionFormer's MapInteraction cross-attention.
# num_lanes=64 caps how many nearest lanes are kept per frame (sorted by
# distance to ego); num_points_per_lane=20 is the polyline resample length.
model = dict(
    motion_head=dict(
        map_lane_encoder=dict(
            map_path='data/kl_8/map/base_map.txt',
            num_lanes=64,
            num_points_per_lane=20,
        )))

data = dict(samples_per_gpu=1)

load_from = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap'
