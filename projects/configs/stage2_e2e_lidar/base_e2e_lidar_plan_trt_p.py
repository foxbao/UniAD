_base_ = ['./base_e2e_lidar_trt_p.py']

planning_steps = 6
bev_h_ = 120
bev_w_ = 160

model = dict(
    type='UniADMotionLidarTRT',
    task_loss_weight=dict(track=1.0, motion=1.0, planning=1.0),
    planning_head=dict(
        type='PlanningHeadSingleModeTRTP',
        bev_h=bev_h_,
        bev_w=bev_w_,
        embed_dims=256,
        planning_steps=planning_steps,
        with_adapter=True,
        use_col_optim=False,
        loss_planning=dict(type='PlanningLoss'),
        loss_collision=[
            dict(type='CollisionLoss', delta=0.0, weight=2.5),
            dict(type='CollisionLoss', delta=0.5, weight=1.0),
            dict(type='CollisionLoss', delta=1.0, weight=0.25),
        ]))

work_dir = './projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_trt'
