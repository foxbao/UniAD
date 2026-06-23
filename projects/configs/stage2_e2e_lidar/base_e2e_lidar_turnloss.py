_base_ = ['./base_e2e_lidar.py']

# Pure loss ablation: keep the original old-anchor no-map baseline unchanged,
# and only switch the motion objective to the turn-aware loss variant.
model = dict(
    motion_head=dict(
        loss_traj=dict(
            type='TrajLoss',
            use_variance=True,
            cls_loss_weight=0.5,
            nll_loss_weight=0.5,
            loss_weight_minade=0.,
            loss_weight_minfde=0.25,
            best_mode_metric='ade_fde',
            fde_weight=0.5,
            turn_loss_weights=dict(
                static_slow=0.7,
                straight=1.0,
                mild_turn=1.5,
                sharp_turn=2.0))))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_turnloss'
