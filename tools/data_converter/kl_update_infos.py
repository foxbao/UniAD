# Copyright (c) OpenMMLab. All rights reserved.
"""Convert KL raw info files to the mmdet3d v2-style info schema.

The KL converter first writes legacy ``infos`` dictionaries.  The LiDAR
BEVFormer code we are porting expects the newer ``data_list/metainfo`` schema,
so this module carries only the KL-specific conversion logic needed here.
"""

import copy
from os import path as osp
from pathlib import Path

import mmcv
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm


KL_CLASSES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
)


def convert_quaternion_to_matrix(quaternion, translation=None):
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = Quaternion(quaternion).rotation_matrix
    if translation is not None:
        transform[:3, 3] = np.asarray(translation, dtype=np.float32)
    return transform.tolist()


def normalize_kl_timestamp(timestamp):
    if timestamp is None:
        return None
    timestamp = float(timestamp)
    return timestamp / 1e6 if abs(timestamp) > 1e12 else timestamp


def get_empty_instance():
    return dict(
        bbox=None,
        bbox_label=None,
        bbox_3d=None,
        bbox_3d_isvalid=None,
        bbox_label_3d=None,
        depth=None,
        center_2d=None,
        attr_label=None,
        num_lidar_pts=None,
        num_radar_pts=None,
        difficulty=None,
        unaligned_bbox_3d=None)


def get_empty_lidar_points():
    return dict(num_pts_feats=None, lidar_path=None, lidar2ego=None)


def get_empty_radar_points():
    return dict(num_pts_feats=None, radar_path=None, radar2ego=None)


def get_empty_img_info():
    return dict(
        img_path=None,
        height=None,
        width=None,
        depth_map=None,
        cam2img=None,
        lidar2img=None,
        cam2ego=None)


def get_single_lidar_sweep():
    return dict(
        timestamp=None,
        ego2global=None,
        lidar_points=get_empty_lidar_points())


def get_empty_standard_data_info():
    return dict(
        sample_idx=None,
        token=None,
        ego2global=None,
        images={},
        lidar_points=get_empty_lidar_points(),
        radar_points=get_empty_radar_points(),
        image_sweeps=[],
        lidar_sweeps=[],
        instances=[],
        instances_ignore=[],
        pts_semantic_mask_path=None,
        pts_instance_mask_path=None)


def clear_instance_unused_keys(instance):
    for key in list(instance.keys()):
        if instance[key] is None:
            del instance[key]
    return instance


def clear_data_info_unused_keys(data_info):
    empty_flag = True
    for key in list(data_info.keys()):
        if key in ['instances', 'cam_sync_instances', 'cam_instances']:
            empty_flag = False
            continue
        if isinstance(data_info[key], list):
            if len(data_info[key]) == 0:
                del data_info[key]
            else:
                empty_flag = False
        elif data_info[key] is None:
            del data_info[key]
        elif isinstance(data_info[key], dict):
            _, sub_empty_flag = clear_data_info_unused_keys(data_info[key])
            if sub_empty_flag:
                del data_info[key]
            else:
                empty_flag = False
        else:
            empty_flag = False
    return data_info, empty_flag


def _relative_image_path(path):
    path = Path(path)
    parts = path.parts
    for version in ('v1.0-trainval', 'v1.0-mini'):
        if version in parts:
            return str(Path(*parts[parts.index(version) + 1:]))
    return path.name


def generate_kl_camera_instances(info):
    cam_instances = {}
    for cam in info.get('cams', {}):
        cam_instances[cam] = []
    return cam_instances


def update_kl_infos(pkl_path, out_dir):
    print(f'{pkl_path} will be modified.')
    if out_dir in pkl_path:
        print(f'Warning, you may overwriting the original data {pkl_path}.')
    print(f'Reading from input file: {pkl_path}.')

    data = mmcv.load(pkl_path)
    raw_infos = data['infos']
    metadata = data.get('metadata', {})

    converted_list = []
    ignore_class_name = set()
    for i, ori_info in enumerate(tqdm(raw_infos, desc='Updating KL infos')):
        data_info = get_empty_standard_data_info()
        data_info['sample_idx'] = i
        data_info['token'] = ori_info['token']
        data_info['ego2global'] = convert_quaternion_to_matrix(
            ori_info['ego2global_rotation'],
            ori_info['ego2global_translation'])
        data_info['prev'] = ori_info.get('prev', '')
        data_info['next'] = ori_info.get('next', '')
        data_info['scene_token'] = ori_info.get('scene_token', '')
        data_info['timestamp'] = normalize_kl_timestamp(ori_info['timestamp'])
        if 'sync_info' in ori_info:
            data_info['sync_info'] = ori_info['sync_info']

        data_info['lidar_points']['num_pts_feats'] = ori_info.get(
            'num_features', 5)
        data_info['lidar_points']['lidar_path'] = Path(
            ori_info['lidar_path']).name
        data_info['lidar_points']['lidar2ego'] = convert_quaternion_to_matrix(
            ori_info['lidar2ego_rotation'],
            ori_info['lidar2ego_translation'])

        for ori_sweep in ori_info.get('sweeps', []):
            lidar_sweep = get_single_lidar_sweep()
            lidar_sweep['lidar_points']['lidar2ego'] = (
                convert_quaternion_to_matrix(
                    ori_sweep['sensor2ego_rotation'],
                    ori_sweep['sensor2ego_translation']))
            lidar_sweep['ego2global'] = convert_quaternion_to_matrix(
                ori_sweep['ego2global_rotation'],
                ori_sweep['ego2global_translation'])
            lidar2sensor = np.eye(4)
            rot = ori_sweep['sensor2lidar_rotation']
            trans = ori_sweep['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(
                rot.T, trans.reshape(3, 1))
            lidar_sweep['lidar_points']['lidar2sensor'] = (
                lidar2sensor.astype(np.float32).tolist())
            lidar_sweep['timestamp'] = normalize_kl_timestamp(
                ori_sweep['timestamp'])
            lidar_sweep['lidar_points']['lidar_path'] = ori_sweep[
                'data_path']
            lidar_sweep['sample_data_token'] = ori_sweep['sample_data_token']
            data_info['lidar_sweeps'].append(lidar_sweep)

        for cam, cam_info in ori_info.get('cams', {}).items():
            img_info = get_empty_img_info()
            img_info['img_path'] = _relative_image_path(cam_info['data_path'])
            img_info['cam2img'] = cam_info['cam_intrinsic'].tolist()
            img_info['sample_data_token'] = cam_info['sample_data_token']
            img_info['timestamp'] = normalize_kl_timestamp(
                cam_info['timestamp'])
            img_info['cam2ego'] = convert_quaternion_to_matrix(
                cam_info['sensor2ego_rotation'],
                cam_info['sensor2ego_translation'])
            lidar2sensor = np.eye(4)
            rot = cam_info['sensor2lidar_rotation']
            trans = cam_info['sensor2lidar_translation']
            lidar2sensor[:3, :3] = rot.T
            lidar2sensor[:3, 3:4] = -1 * np.matmul(
                rot.T, trans.reshape(3, 1))
            img_info['lidar2cam'] = lidar2sensor.astype(np.float32).tolist()
            data_info['images'][cam] = img_info

        if 'gt_boxes' in ori_info:
            for j in range(ori_info['gt_boxes'].shape[0]):
                instance = get_empty_instance()
                instance['bbox_3d'] = ori_info['gt_boxes'][j, :].tolist()
                gt_name = ori_info['gt_names'][j]
                if gt_name in KL_CLASSES:
                    instance['bbox_label'] = KL_CLASSES.index(gt_name)
                else:
                    ignore_class_name.add(gt_name)
                    instance['bbox_label'] = -1
                instance['bbox_label_3d'] = copy.deepcopy(
                    instance['bbox_label'])
                instance['velocity'] = ori_info['gt_velocity'][j, :].tolist()
                instance['num_lidar_pts'] = ori_info['num_lidar_pts'][j]
                instance['num_radar_pts'] = ori_info['num_radar_pts'][j]
                instance['bbox_3d_isvalid'] = ori_info['valid_flag'][j]
                if 'track_ids' in ori_info:
                    instance['track_id'] = ori_info['track_ids'][j]
                if 'gt_forecasting_locs' in ori_info:
                    instance['gt_forecasting_locs'] = ori_info[
                        'gt_forecasting_locs'][j]
                if 'gt_forecasting_mask' in ori_info:
                    instance['gt_forecasting_mask'] = ori_info[
                        'gt_forecasting_mask'][j]
                data_info['instances'].append(
                    clear_instance_unused_keys(instance))
            data_info['cam_instances'] = generate_kl_camera_instances(
                ori_info)

        if 'pts_semantic_mask_path' in ori_info:
            data_info['pts_semantic_mask_path'] = Path(
                ori_info['pts_semantic_mask_path']).name

        data_info, _ = clear_data_info_unused_keys(data_info)
        converted_list.append(data_info)

    metainfo = dict(
        categories={name: i for i, name in enumerate(KL_CLASSES)},
        dataset='kl',
        version=metadata.get('version', 'v1.0-trainval'),
        info_version='1.1',
        lidar_coord_frame=metadata.get('lidar_coord_frame', 'FLU'))
    for ignore_class in ignore_class_name:
        metainfo['categories'][ignore_class] = -1

    out_path = osp.join(out_dir, Path(pkl_path).name)
    print(f'Writing to output file: {out_path}.')
    print(f'ignore classes: {ignore_class_name}')
    mmcv.dump(dict(metainfo=metainfo, data_list=converted_list), out_path)
