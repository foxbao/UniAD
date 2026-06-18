_base_ = ['./base_e2e_lidar.py']

# Ablation config for the HD-map lane prior: base_e2e_lidar (track + drivable +
# motion) PLUS map_lane_encoder, and nothing else. It deliberately inherits
# base_e2e_lidar -- NOT _plan -- so the only variable vs the base_e2e_lidar
# control is map_lane_encoder. occ_head / planning_head are intentionally left
# out: they are orthogonal to the lane prior (which only feeds MotionFormer),
# and including them would mix occ/plan noise into the motion-gain measurement.
# Both this config and base_e2e_lidar load_from the SAME stage-1 drivable
# checkpoint, so experiment and control start from identical weights.
#
# map_lane_encoder is a detector-level module (UniADMotionLidar) -- it is an
# external survey input, not a perception output, so it lives beside the heads
# rather than inside one. It parses surveyed lane centerlines from the
# protobuf-text map once, then per frame transforms them into ego coordinates
# (via img_metas' ego2global) and encodes them as lane_query tensors. The
# detector passes these to MotionFormer's MapInteraction cross-attention
# through an outs_map dict (mirroring how camera UniAD feeds the seg head's
# outs_seg). num_lanes=64 caps how many nearest lanes are kept per frame
# (sorted by distance to ego); num_points_per_lane=20 is the resample length.
model = dict(
    map_lane_encoder=dict(
        map_path='data/kl_8/map/base_map.txt',
        num_lanes=64,
        num_points_per_lane=20,
    ),
    # MTR-style local map collection: each agent attends only its K-nearest
    # valid lanes instead of the global lane set. Unset = global behavior.
    motion_head=dict(map_local_k=32))

data = dict(samples_per_gpu=1)

# Same checkpoint as the base_e2e_lidar control, so the ablation isolates
# map_lane_encoder as the only difference.
load_from = './projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap'
