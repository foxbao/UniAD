_base_ = ['./base_e2e_lidar.py']

# Ablation config for the HD-map lane prior: base_e2e_lidar (track + drivable +
# motion) PLUS map_lane_encoder. This is the original "hard map prior" variant:
# all agents may attend to nearby map lanes, and map_local_k=32 restricts each
# agent to its K-nearest valid lanes. Keep this file unchanged as the direct
# map-prior ablation baseline.
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
    # Turn-aware motion anchors (tools/generate_kl_motion_anchors.py): the
    # original kmeans anchors were all straight (max ~13deg heading change)
    # because straight samples dominate; this pkl up-weights turning samples so
    # the vehicle modes cover real turns (~58-76deg).
    # Cone (final class id 9) is split into its OWN anchor group: it is ~99.5%
    # static, and was previously falling back to the pedestrian group (it was
    # absent from group_id_list, and MotionHead defaults unlisted classes to
    # group 0). The 3-group pkl + group_id_list below give it dedicated
    # near-static anchors and stop it polluting the pedestrian anchors.
    # group_id_list / anchor pkl must agree on group count (here 3).
    motion_head=dict(
        map_local_k=32,
        group_id_list=[[0], [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], [9]],
        anchor_info_path='data/others/motion_anchor_infos_kl_turnaware_3grp.pkl'))

data = dict(samples_per_gpu=1)

# Same checkpoint as the base_e2e_lidar control, so the ablation isolates
# map_lane_encoder as the only difference.
load_from = './projects/work_dirs/stage1_track_map_lidar/base_track_drivable_lidar/latest.pth'
work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap'
