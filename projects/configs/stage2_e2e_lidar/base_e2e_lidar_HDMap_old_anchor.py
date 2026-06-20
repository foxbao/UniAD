_base_ = ['./base_e2e_lidar_HDMap.py']

# Control config for the turn-aware motion anchor ablation: uses the ORIGINAL
# kmeans anchors (motion_anchor_infos_kl.pkl, all <=13deg heading change --
# straight only, 2 groups). Pair with base_e2e_lidar_HDMap to isolate the
# effect of turn-aware anchors on turning-vehicle motion prediction.
#
# Must also restore the original 2-group group_id_list, because HDMap now uses
# a 3-group split (Cone broken out); the original 2-group pkl here would
# mismatch a 3-group group_id_list. Both load_from the same stage-1 drivable
# checkpoint. See tools/generate_kl_motion_anchors.py and
# tools/analysis_tools/anchor_viz_out/ for the anchor comparison.
model = dict(
    motion_head=dict(
        group_id_list=[[0], [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12]],
        anchor_info_path='data/others/motion_anchor_infos_kl.pkl'))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_HDMap_old_anchor'
