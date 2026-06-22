_base_ = ['./base_e2e_lidar_HDMap_weak_loss.py']

# Weak-map variant with the same turn-aware loss and stratified K=6 anchors.
# This is the most realistic map-prior candidate for port actors that only
# sometimes follow the surveyed navigation map.
model = dict(
    motion_head=dict(
        anchor_info_path='data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl'))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap_weak_stratified_k6_loss'
