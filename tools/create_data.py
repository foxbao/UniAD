import argparse
from os import path as osp
import sys
from mmcv import Config
from data_converter import uniad_nuscenes_converter as nuscenes_converter
sys.path.append('.')


def nuscenes_data_prep(root_path,
                       can_bus_root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int): Number of input consecutive frames. Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, out_dir, can_bus_root_path, info_prefix, version=version, max_sweeps=max_sweeps)

    if version == 'v1.0-test':
        info_test_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_test.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_test_path, version=version)
    else:
        info_train_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_train.pkl')
        info_val_path = osp.join(
            out_dir, f'{info_prefix}_infos_temporal_val.pkl')
        nuscenes_converter.export_2d_annotation(
            root_path, info_train_path, version=version)
        nuscenes_converter.export_2d_annotation(
            root_path, info_val_path, version=version)


def kl_data_prep(root_path,
                 info_prefix,
                 version,
                 out_dir,
                 cfg=None,
                 workers=None):
    """Prepare KL info files for LiDAR-only BEVFormer experiments."""
    from data_converter import kl_converter
    from data_converter.kl_update_infos import update_kl_infos

    kl_converter.create_kl_infos(
        root_path, info_prefix, version=version, cfg=cfg, workers=workers)

    info_train_path = osp.join(out_dir, f'{info_prefix}_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_infos_val.pkl')
    update_kl_infos(info_train_path, out_dir=out_dir)
    update_kl_infos(info_val_path, out_dir=out_dir)

    forecast_cfg = {}
    if cfg is not None and hasattr(cfg, 'forecast_cfg'):
        forecast_cfg = dict(cfg.forecast_cfg)
    if bool(forecast_cfg.get('enable', False)):
        from data_converter.add_forecasting import add_forecasting_to_pkl
        forecast_steps = int(forecast_cfg.get('forecast_steps', 6))
        add_forecasting_to_pkl(info_train_path, forecast_steps=forecast_steps)
        add_forecasting_to_pkl(info_val_path, forecast_steps=forecast_steps)

    velocity_cfg = {}
    if cfg is not None and hasattr(cfg, 'velocity_cfg'):
        velocity_cfg = dict(cfg.velocity_cfg)
    if bool(velocity_cfg.get('enable', True)):
        from data_converter.add_velocity import add_velocity_to_pkl
        add_velocity_to_pkl(
            info_train_path,
            in_place=True,
            min_dt=float(velocity_cfg.get('min_dt', 1e-3)),
            max_time_diff=float(velocity_cfg.get('max_time_diff', 1.5)),
            max_speed=float(velocity_cfg.get('max_speed', 60.0)))
        add_velocity_to_pkl(
            info_val_path,
            in_place=True,
            min_dt=float(velocity_cfg.get('min_dt', 1e-3)),
            max_time_diff=float(velocity_cfg.get('max_time_diff', 1.5)),
            max_speed=float(velocity_cfg.get('max_speed', 60.0)))

    sdc_cfg = {}
    if cfg is not None and hasattr(cfg, 'sdc_cfg'):
        sdc_cfg = dict(cfg.sdc_cfg)
    if bool(sdc_cfg.get('enable', False)):
        from data_converter.add_sdc import add_sdc_to_pkl
        sdc_kwargs = dict(
            future_steps=int(sdc_cfg.get('future_steps', 6)),
            sdc_label_name=sdc_cfg.get('sdc_label_name', 'Car'),
            sdc_label_id=sdc_cfg.get('sdc_label_id', None),
            sdc_size=tuple(sdc_cfg.get('sdc_size', (4.08, 1.73, 1.56))),
            sdc_z=float(sdc_cfg.get('sdc_z', 0.0)),
            sdc_yaw=float(sdc_cfg.get('sdc_yaw', 0.0)),
            min_dt=float(sdc_cfg.get('min_dt', 1e-3)),
            max_time_diff=float(sdc_cfg.get('max_time_diff', 1.5)),
            max_step_time_diff=float(
                sdc_cfg.get('max_step_time_diff', 1.5)),
            max_speed=float(sdc_cfg.get('max_speed', 60.0)),
            max_displacement=float(sdc_cfg.get('max_displacement', 100.0)),
            require_valid_localization=bool(
                sdc_cfg.get('require_valid_localization', True)))
        add_sdc_to_pkl(info_train_path, in_place=True, **sdc_kwargs)
        add_sdc_to_pkl(info_val_path, in_place=True, **sdc_kwargs)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='dataset', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--canbus',
    type=str,
    default='./data',
    help='specify the root path of nuScenes canbus')
parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--cfg',
    type=str,
    default=None,
    help='optional dataset preparation config')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument(
    '--skip-test',
    action='store_true',
    help='skip generating nuScenes v1.0-test infos')
args = parser.parse_args()

if __name__ == '__main__':
    cfg = Config.fromfile(args.cfg) if args.cfg is not None else None
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
        if not args.skip_test:
            test_version = f'{args.version}-test'
            nuscenes_data_prep(
                root_path=args.root_path,
                can_bus_root_path=args.canbus,
                info_prefix=args.extra_tag,
                version=test_version,
                dataset_name='NuScenesDataset',
                out_dir=args.out_dir,
                max_sweeps=args.max_sweeps)
    elif args.dataset == 'kl':
        kl_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=args.version,
            out_dir=args.out_dir,
            cfg=cfg,
            workers=args.workers)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.canbus,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)
