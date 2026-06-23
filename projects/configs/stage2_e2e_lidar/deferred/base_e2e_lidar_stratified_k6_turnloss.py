_base_ = ['../base_e2e_lidar_turnaware_turnloss.py']

# No-map variant that keeps the turn-aware loss objective but replaces the
# weighted-kmeans anchors with stratified K=6 anchors:
# static, short-straight, long-straight, negative turn, positive turn, maneuver.
model = dict(
    motion_head=dict(
        anchor_info_path='data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl'))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_stratified_k6_turnloss'
