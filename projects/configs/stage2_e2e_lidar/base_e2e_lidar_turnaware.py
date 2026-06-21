_base_ = ['./base_e2e_lidar.py']

# No-map control with turn-aware anchors. Completes the 2x2 ablation matrix:
#
#                  original anchors (2grp)        turn-aware + Cone (3grp)
#   no map         base_e2e_lidar                 base_e2e_lidar_turnaware (this)
#   + HD map       base_e2e_lidar_HDMap_old_anchor base_e2e_lidar_HDMap
#
# Inherits base_e2e_lidar (track + drivable + motion, NO map_lane_encoder and
# NO map_local_k) and ONLY swaps in the 3-group turn-aware anchors -- identical
# to the anchor/group override in base_e2e_lidar_HDMap, minus the map prior.
# This lets the two ablation axes be isolated:
#   - map effect:    this vs base_e2e_lidar_HDMap   (anchors fixed at turn3grp)
#   - anchor effect: base_e2e_lidar vs this         (no-map fixed)
# and exposes the interaction (does the lane prior only help once turn anchors
# exist to land on?).
#
# group_id_list group-count and the anchor pkl group-count must agree (both 3;
# num_anchor_group = len(group_id_list)). Cone (id 9) gets its own static group,
# matching base_e2e_lidar_HDMap. load_from is inherited (same stage-1 drivable
# checkpoint as every other arm), so all four arms start from identical weights.
model = dict(
    motion_head=dict(
        group_id_list=[[0], [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12], [9]],
        anchor_info_path='data/others/motion_anchor_infos_kl_turnaware_3grp.pkl'))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_turnaware'
