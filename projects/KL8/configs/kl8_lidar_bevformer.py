gt_annotation_filter = dict(
    enable=True,
    min_points_by_class={
        'Pedestrian': 10,
        'Car': 50,
        'IGV-Full': 50,
        'Truck': 50,
        'Trailer-Empty': 50,
        'Trailer-Full': 50,
        'IGV-Empty': 50,
        'Crane': 50,
        'OtherVehicle': 50,
        'Cone': 5,
        'ContainerForklift': 50,
        'Forklift': 50,
        'Lorry': 50,
        'ConstructionVehicle': 50,
        'WheelCrane': 200,
    })


lidar_selection = dict(
    enable=False,
    use_lidars=[
        'bp_front_left', 'bp_front_right',
        'bp_rear_left', 'bp_rear_right',
        'helios_front_left', 'helios_rear_right',
        'm1_front', 'm1_rear',
    ])


camera_selection = dict(
    enable=True,
    use_cameras=[
        'front',
        'left_front',
        'left_rear',
        'rear',
        'right_front',
        'right_rear',
    ])


# Current KL LiDAR configs only consume point clouds. Keep camera selection
# above for future image/multimodal prep, but skip expensive undistort/resize
# work for the default LiDAR-only conversion.
camera_processing_cfg = dict(
    enable=False,
    img_scale=1.0 / 3.0)


# Only used by tools/data_converter/kl_converter.py when create_data.py does
# not pass --workers. The CLI --workers value takes precedence.
worker_cfg = dict(
    num_workers=8)


sensor_sync_cfg = dict(
    lidar_max_diff=0.05,
    camera_max_diff=0.05,
    localization_max_diff=0.15,
    require_valid_localization=True,
    sensor_time_offsets={})


temporal_chain_cfg = dict(
    enable=True,
    min_adj_time_diff=0.35,
    max_adj_time_diff=0.75)


forecast_cfg = dict(
    enable=False,
    forecast_steps=6)


velocity_cfg = dict(
    enable=True,
    min_dt=1e-3,
    max_time_diff=1.5,
    max_speed=60.0)


gt_processing_cfg = dict(
    # Use GPU by default because KL point counting is materially faster, and
    # GPU 0 is usually reserved for data prep / quick tests on this machine.
    # Switch to 'cpu' when all GPUs are occupied by training.
    device='cuda')


sync_cfg = dict(sensor_sync_cfg)
if temporal_chain_cfg.get('enable', True):
    sync_cfg.update(
        min_adj_time_diff=temporal_chain_cfg['min_adj_time_diff'],
        max_adj_time_diff=temporal_chain_cfg['max_adj_time_diff'])
