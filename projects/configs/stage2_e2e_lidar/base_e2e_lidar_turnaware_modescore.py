_base_ = ['./base_e2e_lidar_turnaware_turnloss.py']

# Follow-up to base_e2e_lidar_turnaware_turnloss:
# keep the same turn-aware anchors and regression/sample weighting, but further
# up-weight the motion mode classification term for turning samples. This targets
# cases where a good candidate trajectory exists (low minFDE) but is not selected
# as top1 at inference time.
model = dict(
    motion_head=dict(
        loss_traj=dict(
            turn_cls_loss_weights=dict(
                static_slow=1.0,
                straight=1.0,
                mild_turn=1.5,
                sharp_turn=2.5))))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_turnaware_modescore'
