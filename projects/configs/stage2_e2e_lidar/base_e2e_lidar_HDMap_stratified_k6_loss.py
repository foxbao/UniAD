_base_ = ['./base_e2e_lidar_HDMap_loss.py']

# Hard-map variant with the same turn-aware loss and stratified K=6 anchors.
# Compare against base_e2e_lidar_stratified_k6_loss to isolate hard-map effect.
model = dict(
    motion_head=dict(
        anchor_info_path='data/others/motion_anchor_infos_kl_stratified_k6_3grp.pkl'))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap_stratified_k6_loss'
