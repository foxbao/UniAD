import copy
import math
import time
from os import path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from mmdet.datasets import DATASETS
from third_party.uniad_mmdet3d.core.bbox import LiDARInstance3DBoxes
from third_party.uniad_mmdet3d.datasets.custom_3d import Custom3DDataset
from terminaltables import AsciiTable


KL_CLASSES = (
    'Pedestrian', 'Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
    'Trailer-Full', 'IGV-Empty', 'Crane', 'OtherVehicle', 'Cone',
    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle',
    'WheelCrane',
)


@DATASETS.register_module()
class KlDataset(Custom3DDataset):
    CLASSES = KL_CLASSES

    @staticmethod
    def _format_point_cloud_range(point_cloud_range, name):
        if point_cloud_range is None:
            return None
        point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        if point_cloud_range.shape != (6, ):
            raise ValueError(
                f'{name} must have shape (6,), got '
                f'{point_cloud_range.shape}.')
        return point_cloud_range

    def __init__(self,
                 data_root,
                 ann_file,
                 pipeline=None,
                 classes=None,
                 metainfo=None,
                 modality=None,
                 data_prefix=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 with_velocity=True,
                 use_valid_flag=False,
                 label_mapping=None,
                 track_past_steps=None,
                 track_fut_steps=None,
                 point_cloud_range=None,
                 eval_point_cloud_range=None,
                 filter_eval_by_range=True,
                 pi_symmetric_classes=('IGV-Full', 'IGV-Empty',
                                       'WheelCrane'),
                 **kwargs):
        if classes is None and metainfo is not None:
            classes = metainfo.get('classes', None)
        self.data_prefix = data_prefix or {}
        self.with_velocity = with_velocity
        self.use_valid_flag = use_valid_flag
        self.track_past_steps = (
            None if track_past_steps is None else int(track_past_steps))
        self.track_fut_steps = (
            None if track_fut_steps is None else int(track_fut_steps))
        if self.track_past_steps is None and self.track_fut_steps is None:
            self.track_traj_steps = None
        else:
            self.track_traj_steps = (
                int(self.track_past_steps or 0) +
                int(self.track_fut_steps or 0))
        self.point_cloud_range = self._format_point_cloud_range(
            point_cloud_range, 'point_cloud_range')
        if eval_point_cloud_range is None:
            eval_point_cloud_range = self.point_cloud_range
        self.eval_point_cloud_range = self._format_point_cloud_range(
            eval_point_cloud_range, 'eval_point_cloud_range')
        self.filter_eval_by_range = bool(filter_eval_by_range)
        self.label_mapping = None
        if label_mapping is not None:
            self.label_mapping = {
                int(old_label): int(new_label)
                for old_label, new_label in enumerate(label_mapping)
            }
        self.pi_symmetric_classes = set(pi_symmetric_classes or [])
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)

    def load_annotations(self, ann_file):
        ann_path = ann_file
        if not osp.isabs(ann_path):
            ann_path = osp.join(self.data_root, ann_path)
        data = mmcv.load(ann_path)
        if isinstance(data, dict) and 'data_list' in data:
            self.metainfo = data.get('metainfo', {})
            return data['data_list']
        if isinstance(data, dict) and 'infos' in data:
            self.metainfo = data.get('metadata', {})
            return data['infos']
        return data

    def _map_label(self, label):
        label = int(label)
        if self.label_mapping is not None:
            label = self.label_mapping.get(label, -1)
        if label < 0 or label >= len(self.CLASSES):
            return -1
        return label

    def _resolve_path(self, key, filename):
        if filename is None:
            return None
        if osp.isabs(filename) or osp.exists(filename):
            return filename
        prefix = self.data_prefix.get(key, '')
        return osp.join(self.data_root, prefix, filename)

    def get_data_info(self, index):
        info = self.data_infos[index]
        lidar_info = info.get('lidar_points', {})
        lidar_path = lidar_info.get('lidar_path', info.get('lidar_path'))
        pts_filename = self._resolve_path('pts', lidar_path)
        sample_idx = info.get('sample_idx', index)

        input_dict = dict(
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            file_name=pts_filename,
            token=info.get('token', str(sample_idx)),
            scene_token=info.get('scene_token', ''),
            prev=info.get('prev', None),
            next=info.get('next', None),
            timestamp=float(info.get('timestamp', 0.0)),
            ego2global=np.asarray(
                info.get('ego2global', np.eye(4)), dtype=np.float64))

        if not self.test_mode:
            annos = KlDataset.get_ann_info(self, index)
            input_dict['ann_info'] = annos
            if self.filter_empty_gt and len(annos['gt_labels_3d']) == 0:
                return None
        else:
            # Test pipeline ops like LoadAnnotations3D still need ann_info
            # available (e.g. drivable map raycast uses gt_bboxes_3d to
            # mask obstacles). filter_empty_gt only applies during train.
            input_dict['ann_info'] = KlDataset.get_ann_info(self, index)
        return input_dict

    def get_ann_info(self, index):
        info = self.data_infos[index]
        instances = info.get('instances', [])
        gt_bboxes = []
        gt_labels = []
        gt_names = []
        gt_inds = []
        forecasting_locs = []
        forecasting_mask = []
        track_traj_locs = []
        track_traj_mask = []

        for inst in instances:
            label = inst.get('bbox_label_3d', inst.get('bbox_label', -1))
            if label is None or int(label) < 0:
                continue
            label = self._map_label(label)
            if label < 0:
                continue
            if self.use_valid_flag:
                valid = bool(inst.get('bbox_3d_isvalid', True))
            else:
                valid = inst.get('num_lidar_pts', 1) > 0
            if not valid:
                continue

            bbox = list(inst['bbox_3d'])
            if self.with_velocity and len(bbox) == 7:
                bbox.extend(inst.get('velocity', [0.0, 0.0]))
            gt_bboxes.append(bbox)
            gt_labels.append(label)
            gt_names.append(self.CLASSES[label])
            gt_inds.append(int(inst.get('track_id', -1)))
            inst_fut_traj_locs = np.asarray(
                inst.get('gt_fut_traj_locs',
                         inst.get('gt_forecasting_locs',
                                  np.zeros((0, 2)))),
                dtype=np.float32)
            inst_fut_traj_mask = np.asarray(
                inst.get('gt_fut_traj_mask',
                         inst.get('gt_forecasting_mask', np.zeros((0, )))),
                dtype=np.bool_)
            forecasting_locs.append(
                np.asarray(
                    inst_fut_traj_locs,
                    dtype=np.float32))
            forecasting_mask.append(
                np.asarray(
                    inst_fut_traj_mask,
                    dtype=np.bool_))
            track_traj_locs.append(
                np.asarray(
                    inst.get('gt_track_traj_locs', inst_fut_traj_locs),
                    dtype=np.float32))
            track_traj_mask.append(
                np.asarray(
                    inst.get('gt_track_traj_mask', inst_fut_traj_mask),
                    dtype=np.bool_))

        box_dim = 9 if self.with_velocity else 7
        if gt_bboxes:
            gt_bboxes = np.asarray(gt_bboxes, dtype=np.float32)
        else:
            gt_bboxes = np.zeros((0, box_dim), dtype=np.float32)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        gt_inds = np.asarray(gt_inds, dtype=np.int64)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes,
            box_dim=gt_bboxes.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        gt_fut_traj, gt_fut_traj_mask = self._stack_track_trajs(
            forecasting_locs, forecasting_mask)
        gt_track_traj, gt_track_traj_mask = self._stack_track_trajs(
            track_traj_locs, track_traj_mask, num_steps=self.track_traj_steps)
        ann_info = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels,
            gt_names_3d=np.asarray(gt_names),
            gt_inds=gt_inds,
            gt_fut_traj=gt_fut_traj,
            gt_fut_traj_mask=gt_fut_traj_mask,
            # UniAD stage-1 names this branch "past_traj"; for KL we store
            # the configured track trajectory labels there.
            gt_past_traj=gt_track_traj,
            gt_past_traj_mask=gt_track_traj_mask)

        if 'gt_sdc_bbox' in info:
            sdc_bbox = np.asarray(info['gt_sdc_bbox'], dtype=np.float32)
            sdc_bbox_3d = LiDARInstance3DBoxes(
                sdc_bbox,
                box_dim=sdc_bbox.shape[-1],
                origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)
            ann_info['gt_sdc_bbox'] = sdc_bbox_3d
            ann_info['gt_sdc_label'] = np.asarray(
                info['gt_sdc_label'], dtype=np.int64)
            ann_info['gt_sdc_fut_traj'] = np.asarray(
                info['gt_sdc_fut_traj'], dtype=np.float32)
            ann_info['gt_sdc_fut_traj_mask'] = np.asarray(
                info['gt_sdc_fut_traj_mask'], dtype=np.float32)
        if 'sdc_planning' in info:
            ann_info['sdc_planning'] = np.asarray(
                info['sdc_planning'], dtype=np.float32)
            ann_info['sdc_planning_mask'] = np.asarray(
                info['sdc_planning_mask'], dtype=np.float32)
            ann_info['command'] = np.asarray(
                info['command'], dtype=np.int64)
        return ann_info

    @staticmethod
    def _stack_track_trajs(trajs, masks, num_steps=None):
        if len(trajs) == 0:
            steps = int(num_steps or 0)
            return (np.zeros((0, steps, 2), dtype=np.float32),
                    np.zeros((0, steps, 2), dtype=np.float32))

        if num_steps is None:
            num_steps = max((traj.shape[0] for traj in trajs), default=0)
        num_steps = int(num_steps)
        stacked_trajs = np.zeros((len(trajs), num_steps, 2), dtype=np.float32)
        stacked_masks = np.zeros((len(trajs), num_steps, 2), dtype=np.float32)
        for idx, (traj, mask) in enumerate(zip(trajs, masks)):
            if traj.size == 0:
                continue
            steps = min(num_steps, traj.shape[0])
            stacked_trajs[idx, :steps] = traj[:steps, :2]
            mask = np.asarray(mask[:steps], dtype=np.float32)
            stacked_masks[idx, :steps] = mask[:, None]
        return stacked_trajs, stacked_masks

    @staticmethod
    def _center_distance(gt_box, pred_box):
        return float(np.linalg.norm(gt_box[:2] - pred_box[:2]))

    @staticmethod
    def _scale_error(gt_box, pred_box):
        gt_dims = np.maximum(gt_box[3:6], 0.0)
        pred_dims = np.maximum(pred_box[3:6], 0.0)
        inter = np.prod(np.minimum(gt_dims, pred_dims))
        union = np.prod(gt_dims) + np.prod(pred_dims) - inter
        if union <= 0:
            return 1.0
        return float(1.0 - inter / union)

    @staticmethod
    def _yaw_error(gt_yaw, pred_yaw, period):
        diff = (pred_yaw - gt_yaw + period / 2) % period - period / 2
        return float(abs(diff))

    @staticmethod
    def _velocity_error(gt_box, pred_box):
        if gt_box.shape[-1] <= 8 or pred_box.shape[-1] <= 8:
            return 1.0
        return float(np.linalg.norm(gt_box[7:9] - pred_box[7:9]))

    @staticmethod
    def _cummean(values):
        values = np.asarray(values, dtype=np.float64)
        return np.cumsum(values) / np.arange(1, len(values) + 1)

    @staticmethod
    def _calc_nusc_ap(precision, min_recall=0.1, min_precision=0.1):
        first = round(100 * min_recall) + 1
        clipped = np.copy(precision[first:])
        clipped -= min_precision
        clipped[clipped < 0] = 0
        return float(np.mean(clipped)) / (1.0 - min_precision)

    @staticmethod
    def _calc_nusc_tp(metric_data, metric_name, min_recall=0.1):
        first = round(100 * min_recall) + 1
        confidence = metric_data['confidence']
        non_zero = np.nonzero(confidence)[0]
        last = int(non_zero[-1]) if len(non_zero) > 0 else 0
        if last < first:
            return 1.0
        return float(np.mean(metric_data[metric_name][first:last + 1]))

    def _accumulate_nusc_style(self, gt_by_sample, pred_list, class_name,
                               dist_th):
        npos = sum(len(boxes) for boxes in gt_by_sample.values())
        if npos == 0:
            return None

        recall_grid = np.linspace(0, 1, 101)
        if len(pred_list) == 0:
            ones = np.ones_like(recall_grid)
            return dict(
                recall=recall_grid,
                precision=np.zeros_like(recall_grid),
                confidence=np.zeros_like(recall_grid),
                trans_err=ones,
                scale_err=ones,
                orient_err=ones,
                vel_err=ones,
                attr_err=ones)

        pred_list = sorted(
            pred_list, key=lambda item: item['score'], reverse=True)
        tp = []
        fp = []
        conf = []
        match_data = dict(
            trans_err=[],
            scale_err=[],
            orient_err=[],
            vel_err=[],
            attr_err=[],
            conf=[])
        taken = set()
        orient_period = (
            np.pi if class_name in self.pi_symmetric_classes else 2 * np.pi)

        for pred in pred_list:
            sample_idx = pred['sample_idx']
            pred_box = pred['box']
            min_dist = np.inf
            match_gt_idx = None
            for gt_idx, gt_box in enumerate(gt_by_sample.get(sample_idx, [])):
                if (sample_idx, gt_idx) in taken:
                    continue
                distance = self._center_distance(gt_box, pred_box)
                if distance < min_dist:
                    min_dist = distance
                    match_gt_idx = gt_idx

            is_match = min_dist < dist_th
            if is_match:
                taken.add((sample_idx, match_gt_idx))
                tp.append(1)
                fp.append(0)
                gt_box = gt_by_sample[sample_idx][match_gt_idx]
                match_data['trans_err'].append(
                    self._center_distance(gt_box, pred_box))
                match_data['scale_err'].append(
                    self._scale_error(gt_box, pred_box))
                match_data['orient_err'].append(
                    self._yaw_error(gt_box[6], pred_box[6], orient_period))
                match_data['vel_err'].append(
                    self._velocity_error(gt_box, pred_box))
                match_data['attr_err'].append(0.0)
                match_data['conf'].append(pred['score'])
            else:
                tp.append(0)
                fp.append(1)
            conf.append(pred['score'])

        if len(match_data['trans_err']) == 0:
            ones = np.ones_like(recall_grid)
            return dict(
                recall=recall_grid,
                precision=np.zeros_like(recall_grid),
                confidence=np.zeros_like(recall_grid),
                trans_err=ones,
                scale_err=ones,
                orient_err=ones,
                vel_err=ones,
                attr_err=ones)

        tp = np.cumsum(tp).astype(np.float64)
        fp = np.cumsum(fp).astype(np.float64)
        conf = np.asarray(conf, dtype=np.float64)
        precision = tp / np.maximum(tp + fp, 1e-12)
        recall = tp / float(npos)
        precision = np.interp(recall_grid, recall, precision, right=0)
        confidence = np.interp(recall_grid, recall, conf, right=0)

        metric_data = dict(
            recall=recall_grid,
            precision=precision,
            confidence=confidence)
        match_conf = np.asarray(match_data['conf'], dtype=np.float64)
        for key in ('trans_err', 'scale_err', 'orient_err', 'vel_err',
                    'attr_err'):
            err = self._cummean(match_data[key])
            metric_data[key] = np.interp(
                confidence[::-1], match_conf[::-1], err[::-1]).astype(
                    np.float64)[::-1]
        return metric_data

    def _evaluate_nusc_style(self,
                             results,
                             gt_ann_infos,
                             logger=None,
                             dist_ths=(0.5, 1.0, 2.0, 4.0),
                             dist_th_tp=2.0,
                             min_recall=0.1,
                             min_precision=0.1,
                             max_boxes_per_sample=500,
                             mean_ap_weight=5):
        start_time = time.time()
        label2cat = {i: cat for i, cat in enumerate(self.CLASSES)}
        gt_by_class = {i: {} for i in label2cat}
        pred_by_class = {i: [] for i in label2cat}
        eval_range = self._active_eval_point_cloud_range()
        filtered_gt = 0
        filtered_pred = 0

        for sample_idx, ann_info in enumerate(gt_ann_infos):
            gt_boxes = ann_info['gt_bboxes_3d'].tensor.detach().cpu().numpy()
            gt_labels = np.asarray(ann_info['gt_labels_3d'])
            if eval_range is not None:
                gt_mask = self._box_bev_range_mask(gt_boxes, eval_range)
                filtered_gt += int(len(gt_mask) - gt_mask.sum())
                gt_boxes = gt_boxes[gt_mask]
                gt_labels = gt_labels[gt_mask]
            for box, label in zip(gt_boxes, gt_labels):
                label = int(label)
                if label not in gt_by_class:
                    continue
                gt_by_class[label].setdefault(sample_idx, []).append(box)

        for sample_idx, result in enumerate(results):
            boxes = result.get('boxes_3d', None)
            if boxes is None:
                continue
            box_tensor = boxes.tensor.detach().cpu().numpy()
            scores = result['scores_3d'].detach().cpu().numpy()
            labels = result['labels_3d'].detach().cpu().numpy()
            if eval_range is not None:
                pred_mask = self._box_bev_range_mask(box_tensor, eval_range)
                filtered_pred += int(len(pred_mask) - pred_mask.sum())
                valid_indices = np.nonzero(pred_mask)[0]
            else:
                valid_indices = np.arange(len(box_tensor))
            if len(valid_indices) == 0:
                continue
            order = valid_indices[
                np.argsort(scores[valid_indices])[::-1]
                [:max_boxes_per_sample]]
            for pred_idx in order:
                label = int(labels[pred_idx])
                if label not in pred_by_class:
                    continue
                pred_by_class[label].append(
                    dict(
                        sample_idx=sample_idx,
                        box=box_tensor[pred_idx],
                        score=float(scores[pred_idx])))

        if eval_range is not None:
            print_log(
                'KL bbox evaluation range filter: '
                f'x=[{eval_range[0]:.1f}, {eval_range[3]:.1f}], '
                f'y=[{eval_range[1]:.1f}, {eval_range[4]:.1f}], '
                f'filtered_gt={filtered_gt}, '
                f'filtered_pred={filtered_pred}.',
                logger=logger)

        ret_dict = {}
        table_data = [[
            'classes', 'AP_dist', 'ATE', 'ASE', 'AOE', 'AVE', 'AAE'
        ]]
        mean_aps = []
        class_tp_errors = dict(
            trans_err=[],
            scale_err=[],
            orient_err=[],
            vel_err=[],
            attr_err=[])

        for label, class_name in label2cat.items():
            class_aps = []
            tp_metric_data = None
            has_gt = sum(len(v) for v in gt_by_class[label].values()) > 0
            for dist_th in dist_ths:
                metric_data = self._accumulate_nusc_style(
                    gt_by_class[label],
                    pred_by_class[label],
                    class_name,
                    dist_th)
                if metric_data is None:
                    ap = np.nan
                else:
                    ap = self._calc_nusc_ap(
                        metric_data['precision'],
                        min_recall=min_recall,
                        min_precision=min_precision)
                ret_dict[f'{class_name}_AP_dist_{dist_th:.1f}'] = ap
                class_aps.append(ap)
                if abs(dist_th - dist_th_tp) < 1e-6:
                    tp_metric_data = metric_data

            class_ap = float(np.nanmean(class_aps)) if has_gt else np.nan
            ret_dict[f'{class_name}_AP_dist'] = class_ap
            if np.isfinite(class_ap):
                mean_aps.append(class_ap)

            class_errors = {}
            for metric_name, short_name in (
                    ('trans_err', 'ATE'),
                    ('scale_err', 'ASE'),
                    ('orient_err', 'AOE'),
                    ('vel_err', 'AVE'),
                    ('attr_err', 'AAE')):
                if not has_gt:
                    value = np.nan
                elif tp_metric_data is None:
                    value = 1.0
                else:
                    value = self._calc_nusc_tp(
                        tp_metric_data,
                        metric_name,
                        min_recall=min_recall)
                class_errors[metric_name] = value
                ret_dict[f'{class_name}_{short_name}'] = value
                ret_dict[f'{class_name}_{metric_name}'] = value
                if np.isfinite(value):
                    class_tp_errors[metric_name].append(value)

            table_data.append([
                class_name,
                'nan' if not np.isfinite(class_ap) else f'{class_ap:.4f}',
                'nan' if not np.isfinite(class_errors['trans_err']) else
                f'{class_errors["trans_err"]:.4f}',
                'nan' if not np.isfinite(class_errors['scale_err']) else
                f'{class_errors["scale_err"]:.4f}',
                'nan' if not np.isfinite(class_errors['orient_err']) else
                f'{class_errors["orient_err"]:.4f}',
                'nan' if not np.isfinite(class_errors['vel_err']) else
                f'{class_errors["vel_err"]:.4f}',
                'nan' if not np.isfinite(class_errors['attr_err']) else
                f'{class_errors["attr_err"]:.4f}',
            ])

        mean_ap = float(np.mean(mean_aps)) if mean_aps else 0.0
        ret_dict['mAP'] = mean_ap
        metric_to_key = dict(
            trans_err='mATE',
            scale_err='mASE',
            orient_err='mAOE',
            vel_err='mAVE',
            attr_err='mAAE')
        tp_scores = []
        for metric_name, key in metric_to_key.items():
            values = class_tp_errors[metric_name]
            value = float(np.mean(values)) if values else 1.0
            ret_dict[key] = value
            ret_dict[metric_name] = value
            tp_scores.append(max(0.0, 1.0 - value))
        ret_dict['NDS'] = (
            mean_ap_weight * mean_ap + sum(tp_scores)) / (
                mean_ap_weight + len(tp_scores))
        ret_dict['nd_score'] = ret_dict['NDS']
        ret_dict['eval_time'] = time.time() - start_time

        table_data.append([
            'Overall',
            f'{ret_dict["mAP"]:.4f}',
            f'{ret_dict["mATE"]:.4f}',
            f'{ret_dict["mASE"]:.4f}',
            f'{ret_dict["mAOE"]:.4f}',
            f'{ret_dict["mAVE"]:.4f}',
            f'{ret_dict["mAAE"]:.4f}',
        ])
        print_log('\n' + AsciiTable(table_data).table, logger=logger)
        return ret_dict

    def _get_raw_info(self, index):
        if hasattr(self, '_to_raw_index'):
            index = self._to_raw_index(index)
        return self.data_infos[index]

    @staticmethod
    def _to_numpy(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _active_eval_point_cloud_range(self):
        if not self.filter_eval_by_range:
            return None
        return self.eval_point_cloud_range

    @staticmethod
    def _box_bev_range_mask(boxes, point_cloud_range):
        boxes = np.asarray(boxes)
        num_boxes = int(boxes.shape[0])
        if num_boxes == 0:
            return np.zeros((0, ), dtype=np.bool_)
        point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
        return (
            (boxes[:, 0] >= point_cloud_range[0]) &
            (boxes[:, 0] <= point_cloud_range[3]) &
            (boxes[:, 1] >= point_cloud_range[1]) &
            (boxes[:, 1] <= point_cloud_range[4]))

    @staticmethod
    def _tracking_track_key(scene_token, track_id):
        return f'{scene_token}:{int(track_id)}'

    def _collect_tracking_records(self, results, gt_ann_infos):
        records = []
        eval_range = self._active_eval_point_cloud_range()
        for sample_idx, (result, ann_info) in enumerate(
                zip(results, gt_ann_infos)):
            info = self._get_raw_info(sample_idx)
            scene_token = str(
                info.get('scene_token',
                         info.get('scene_id', info.get('log_id', ''))))
            if not scene_token:
                scene_token = 'default_scene'
            timestamp = float(info.get('timestamp', sample_idx))

            gt_records = []
            gt_boxes = ann_info['gt_bboxes_3d'].tensor.detach().cpu().numpy()
            gt_labels = np.asarray(ann_info['gt_labels_3d'])
            gt_track_ids = np.asarray(ann_info.get('gt_inds', []))
            if eval_range is not None:
                gt_mask = self._box_bev_range_mask(gt_boxes, eval_range)
            else:
                gt_mask = np.ones((len(gt_boxes), ), dtype=np.bool_)
            for gt_idx, (box, label, track_id) in enumerate(
                    zip(gt_boxes, gt_labels, gt_track_ids)):
                if not gt_mask[gt_idx]:
                    continue
                if int(track_id) < 0:
                    continue
                gt_records.append(
                    dict(
                        box=box,
                        label=int(label),
                        track_id=self._tracking_track_key(
                            scene_token, track_id)))

            pred_records = []
            boxes = result.get('track_boxes_3d', result.get('boxes_3d', None))
            scores = result.get('track_scores', result.get('scores_3d', None))
            labels = result.get('track_labels_3d',
                                result.get('labels_3d', None))
            track_ids = result.get('track_ids', None)
            if boxes is not None and track_ids is not None:
                box_tensor = boxes.tensor.detach().cpu().numpy()
                scores = self._to_numpy(scores)
                labels = self._to_numpy(labels)
                track_ids = self._to_numpy(track_ids)
                if eval_range is not None:
                    pred_mask = self._box_bev_range_mask(
                        box_tensor, eval_range)
                else:
                    pred_mask = np.ones((len(box_tensor), ),
                                        dtype=np.bool_)
                num_preds = len(box_tensor)
                if scores is not None:
                    num_preds = min(num_preds, len(scores))
                if labels is not None:
                    num_preds = min(num_preds, len(labels))
                if track_ids is not None:
                    num_preds = min(num_preds, len(track_ids))
                for pred_idx in range(num_preds):
                    if not pred_mask[pred_idx]:
                        continue
                    track_id = int(track_ids[pred_idx])
                    if track_id < 0:
                        continue
                    label = int(labels[pred_idx])
                    if label < 0 or label >= len(self.CLASSES):
                        continue
                    pred_records.append(
                        dict(
                            box=box_tensor[pred_idx],
                            label=label,
                            score=float(scores[pred_idx]),
                            track_id=self._tracking_track_key(
                                scene_token, track_id)))

            records.append(
                dict(
                    sample_idx=sample_idx,
                    scene_token=scene_token,
                    timestamp=timestamp,
                    gts=gt_records,
                    preds=pred_records))

        records.sort(
            key=lambda item: (item['scene_token'], item['timestamp'],
                              item['sample_idx']))
        return records

    @staticmethod
    def _match_tracking_frame(gts, preds, dist_th):
        candidates = []
        for gt_idx, gt in enumerate(gts):
            for pred_idx, pred in enumerate(preds):
                distance = float(np.linalg.norm(gt['box'][:2] -
                                                pred['box'][:2]))
                if distance < dist_th:
                    candidates.append((distance, gt_idx, pred_idx))
        candidates.sort(key=lambda item: item[0])

        matches = []
        used_gts = set()
        used_preds = set()
        for distance, gt_idx, pred_idx in candidates:
            if gt_idx in used_gts or pred_idx in used_preds:
                continue
            used_gts.add(gt_idx)
            used_preds.add(pred_idx)
            matches.append((gt_idx, pred_idx, distance))
        return matches

    def _accumulate_tracking_class(self, records, label, score_thr, dist_th):
        total_gt = 0
        tp = 0
        fp = 0
        fn = 0
        ids = 0
        frag = 0
        dist_sum = 0.0
        track_total = {}
        track_matched = {}
        last_pred_by_gt = {}
        was_tracked = {}
        ever_tracked = {}
        current_scene = None

        for frame in records:
            scene_token = frame['scene_token']
            if scene_token != current_scene:
                current_scene = scene_token
                last_pred_by_gt = {}
                was_tracked = {}
                ever_tracked = {}

            gts = [gt for gt in frame['gts'] if gt['label'] == label]
            preds = [
                pred for pred in frame['preds']
                if pred['label'] == label and pred['score'] >= score_thr
            ]

            total_gt += len(gts)
            for gt in gts:
                track_total[gt['track_id']] = (
                    track_total.get(gt['track_id'], 0) + 1)

            matches = self._match_tracking_frame(gts, preds, dist_th)
            matched_gt_indices = {match[0] for match in matches}
            matched_pred_indices = {match[1] for match in matches}

            tp += len(matches)
            fp += len(preds) - len(matched_pred_indices)
            fn += len(gts) - len(matched_gt_indices)

            for gt_idx, pred_idx, distance in matches:
                gt_id = gts[gt_idx]['track_id']
                pred_id = preds[pred_idx]['track_id']
                if gt_id in last_pred_by_gt and last_pred_by_gt[
                        gt_id] != pred_id:
                    ids += 1
                if ever_tracked.get(gt_id, False) and not was_tracked.get(
                        gt_id, False):
                    frag += 1
                last_pred_by_gt[gt_id] = pred_id
                was_tracked[gt_id] = True
                ever_tracked[gt_id] = True
                track_matched[gt_id] = track_matched.get(gt_id, 0) + 1
                dist_sum += distance

            for gt_idx, gt in enumerate(gts):
                if gt_idx not in matched_gt_indices:
                    was_tracked[gt['track_id']] = False

        if total_gt == 0:
            return None

        recall = tp / max(total_gt, 1)
        mota = 1.0 - (fn + fp + ids) / max(total_gt, 1)
        motp = dist_sum / tp if tp > 0 else np.nan
        motar = max(0.0, 1.0 - (fp + ids) / max(tp, 1)) if tp > 0 else 0.0
        mt = 0
        ml = 0
        for track_id, count in track_total.items():
            ratio = track_matched.get(track_id, 0) / max(count, 1)
            if ratio >= 0.8:
                mt += 1
            if ratio <= 0.2:
                ml += 1

        return dict(
            recall=float(recall),
            motar=float(motar),
            mota=float(mota),
            motp=float(motp),
            gt=int(total_gt),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn),
            ids=int(ids),
            frag=int(frag),
            mt=int(mt),
            ml=int(ml),
            faf=float(fp / max(len(records), 1)))

    def _evaluate_tracking(self,
                           results,
                           gt_ann_infos,
                           logger=None,
                           dist_th=2.0,
                           score_thresholds=None):
        records = self._collect_tracking_records(results, gt_ann_infos)
        if score_thresholds is None:
            score_thresholds = np.linspace(0.0, 1.0, 41)
        else:
            score_thresholds = np.asarray(score_thresholds, dtype=np.float32)

        label2cat = {i: cat for i, cat in enumerate(self.CLASSES)}
        table_data = [[
            'classes', 'AMOTA', 'AMOTP', 'MOTA', 'MOTP', 'Recall', 'IDS',
            'FP', 'FN'
        ]]
        ret_dict = {}
        class_metrics = []
        count_keys = ('gt', 'tp', 'fp', 'fn', 'ids', 'frag', 'mt', 'ml')
        count_sums = {key: 0 for key in count_keys}

        for label, class_name in label2cat.items():
            class_score_thresholds = score_thresholds
            per_threshold = []
            for score_thr in class_score_thresholds:
                metric_data = self._accumulate_tracking_class(
                    records, label, float(score_thr), dist_th)
                if metric_data is not None:
                    per_threshold.append(metric_data)

            if len(per_threshold) == 0:
                class_metric = dict(
                    amota=np.nan,
                    amotp=np.nan,
                    recall=np.nan,
                    motar=np.nan,
                    mota=np.nan,
                    motp=np.nan,
                    gt=0,
                    tp=0,
                    fp=0,
                    fn=0,
                    ids=0,
                    frag=0,
                    mt=0,
                    ml=0,
                    faf=np.nan)
            else:
                amota_values = []
                amotp_values = []
                for min_recall in np.linspace(0.1, 1.0, 40):
                    candidates = [
                        item for item in per_threshold
                        if item['recall'] >= min_recall
                    ]
                    if len(candidates) == 0:
                        amota_values.append(0.0)
                        continue
                    selected = max(
                        candidates,
                        key=lambda item: (item['motar'], item['mota'],
                                          item['recall']))
                    amota_values.append(selected['motar'])
                    if np.isfinite(selected['motp']):
                        amotp_values.append(selected['motp'])
                best = max(
                    per_threshold,
                    key=lambda item: (item['mota'], item['recall'],
                                      item['motar']))
                class_metric = dict(best)
                class_metric['amota'] = float(np.mean(amota_values))
                class_metric['amotp'] = (
                    float(np.mean(amotp_values)) if amotp_values else np.nan)
                class_metrics.append(class_metric)
                for key in count_keys:
                    count_sums[key] += int(best[key])

            for metric_name, value in (
                    ('amota', class_metric['amota']),
                    ('amotp', class_metric['amotp']),
                    ('mota', class_metric['mota']),
                    ('motp', class_metric['motp']),
                    ('recall', class_metric['recall']),
                    ('motar', class_metric['motar']),
                    ('faf', class_metric['faf'])):
                ret_dict[f'{class_name}_{metric_name}'] = value
            for metric_name in count_keys:
                ret_dict[f'{class_name}_{metric_name}'] = class_metric[
                    metric_name]

            table_data.append([
                class_name,
                'nan' if not np.isfinite(class_metric['amota']) else
                f'{class_metric["amota"]:.4f}',
                'nan' if not np.isfinite(class_metric['amotp']) else
                f'{class_metric["amotp"]:.4f}',
                'nan' if not np.isfinite(class_metric['mota']) else
                f'{class_metric["mota"]:.4f}',
                'nan' if not np.isfinite(class_metric['motp']) else
                f'{class_metric["motp"]:.4f}',
                'nan' if not np.isfinite(class_metric['recall']) else
                f'{class_metric["recall"]:.4f}',
                str(class_metric['ids']),
                str(class_metric['fp']),
                str(class_metric['fn']),
            ])

        mean_keys = ('amota', 'amotp', 'recall', 'motar', 'mota', 'motp',
                     'faf')
        overall = {}
        for key in mean_keys:
            values = [
                item[key] for item in class_metrics
                if np.isfinite(item[key])
            ]
            overall[key] = float(np.mean(values)) if values else np.nan
            ret_dict[key] = overall[key]

        for key in count_keys:
            ret_dict[key] = count_sums[key]

        ret_dict.update(
            AMOTA=overall['amota'],
            AMOTP=overall['amotp'],
            MOTA=overall['mota'],
            MOTP=overall['motp'],
            MOTAR=overall['motar'],
            Recall=overall['recall'],
            IDS=count_sums['ids'],
            FRAG=count_sums['frag'],
            FP=count_sums['fp'],
            FN=count_sums['fn'],
            TP=count_sums['tp'])

        table_data.append([
            'Overall',
            'nan' if not np.isfinite(overall['amota']) else
            f'{overall["amota"]:.4f}',
            'nan' if not np.isfinite(overall['amotp']) else
            f'{overall["amotp"]:.4f}',
            'nan' if not np.isfinite(overall['mota']) else
            f'{overall["mota"]:.4f}',
            'nan' if not np.isfinite(overall['motp']) else
            f'{overall["motp"]:.4f}',
            'nan' if not np.isfinite(overall['recall']) else
            f'{overall["recall"]:.4f}',
            str(count_sums['ids']),
            str(count_sums['fp']),
            str(count_sums['fn']),
        ])
        print_log('\n' + AsciiTable(table_data).table, logger=logger)
        return ret_dict

    @staticmethod
    def _motion_valid_mask(mask):
        mask = np.asarray(mask)
        if mask.ndim == 2:
            mask = np.all(mask > 0, axis=-1)
        else:
            mask = mask > 0
        return mask.astype(np.bool_)

    @staticmethod
    def _motion_errors(pred_traj,
                       gt_traj,
                       gt_mask,
                       miss_threshold=2.0):
        pred_traj = np.asarray(pred_traj, dtype=np.float64)[..., :2]
        gt_traj = np.asarray(gt_traj, dtype=np.float64)[..., :2]
        gt_mask = KlDataset._motion_valid_mask(gt_mask)
        if pred_traj.ndim == 2:
            pred_traj = pred_traj[None]
        if pred_traj.ndim != 3 or gt_traj.ndim != 2:
            return None

        steps = min(pred_traj.shape[1], gt_traj.shape[0], gt_mask.shape[0])
        if steps <= 0:
            return None
        pred_traj = np.nan_to_num(pred_traj[:, :steps], nan=0.0,
                                  posinf=0.0, neginf=0.0)
        gt_traj = np.nan_to_num(gt_traj[:steps], nan=0.0,
                                posinf=0.0, neginf=0.0)
        valid = gt_mask[:steps]
        if not np.any(valid):
            return None

        dists = np.linalg.norm(pred_traj[:, valid] - gt_traj[None, valid],
                               axis=-1)
        ade = dists.mean(axis=-1)
        final_step = np.where(valid)[0][-1]
        fde = np.linalg.norm(pred_traj[:, final_step] - gt_traj[final_step],
                             axis=-1)
        min_ade = float(np.min(ade))
        min_fde = float(np.min(fde))
        return min_ade, min_fde, float(min_fde > miss_threshold)

    def _evaluate_motion(self,
                         results,
                         gt_ann_infos,
                         logger=None,
                         dist_th=2.0,
                         miss_threshold=2.0,
                         score_thr=0.0):
        label2cat = {i: cat for i, cat in enumerate(self.CLASSES)}
        class_stats = {
            label: dict(ade=[], fde=[], mr=[], gt=0, matched=0)
            for label in label2cat
        }

        for result, ann_info in zip(results, gt_ann_infos):
            gt_boxes = ann_info['gt_bboxes_3d'].tensor.detach().cpu().numpy()
            gt_labels = np.asarray(ann_info['gt_labels_3d'])
            gt_trajs = np.asarray(
                ann_info.get('gt_fut_traj', np.zeros((0, 0, 2))),
                dtype=np.float32)
            gt_masks = np.asarray(
                ann_info.get('gt_fut_traj_mask', np.zeros((0, 0, 2))),
                dtype=np.float32)
            valid_gt = np.array(
                [self._motion_valid_mask(mask).any() for mask in gt_masks],
                dtype=np.bool_)
            eval_range = self._active_eval_point_cloud_range()
            if eval_range is not None:
                valid_gt &= self._box_bev_range_mask(gt_boxes, eval_range)

            for label in gt_labels[valid_gt]:
                class_stats[int(label)]['gt'] += 1

            if 'traj' not in result:
                continue
            boxes = result.get('track_boxes_3d', result.get('boxes_3d', None))
            labels = result.get('track_labels_3d', result.get('labels_3d', None))
            scores = result.get('track_scores', result.get('scores_3d', None))
            if boxes is None or labels is None:
                continue

            pred_boxes = boxes.tensor.detach().cpu().numpy()
            pred_labels = self._to_numpy(labels)
            pred_scores = self._to_numpy(scores)
            pred_trajs = self._to_numpy(result['traj'])
            if pred_trajs is None:
                continue
            if eval_range is not None:
                pred_range_mask = self._box_bev_range_mask(
                    pred_boxes, eval_range)
            else:
                pred_range_mask = np.ones((len(pred_boxes), ),
                                          dtype=np.bool_)

            num_preds = min(len(pred_boxes), len(pred_labels),
                            len(pred_trajs))
            if pred_scores is not None:
                num_preds = min(num_preds, len(pred_scores))
            candidates = []
            for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes,
                                                            gt_labels)):
                if not valid_gt[gt_idx]:
                    continue
                for pred_idx in range(num_preds):
                    if not pred_range_mask[pred_idx]:
                        continue
                    if int(pred_labels[pred_idx]) != int(gt_label):
                        continue
                    if (pred_scores is not None
                            and float(pred_scores[pred_idx]) < score_thr):
                        continue
                    distance = self._center_distance(gt_box,
                                                     pred_boxes[pred_idx])
                    if distance < dist_th:
                        candidates.append((distance, gt_idx, pred_idx))
            candidates.sort(key=lambda item: item[0])

            used_gts = set()
            used_preds = set()
            for _, gt_idx, pred_idx in candidates:
                if gt_idx in used_gts or pred_idx in used_preds:
                    continue
                errors = self._motion_errors(
                    pred_trajs[pred_idx],
                    gt_trajs[gt_idx],
                    gt_masks[gt_idx],
                    miss_threshold=miss_threshold)
                if errors is None:
                    continue
                used_gts.add(gt_idx)
                used_preds.add(pred_idx)
                label = int(gt_labels[gt_idx])
                min_ade, min_fde, mr = errors
                class_stats[label]['ade'].append(min_ade)
                class_stats[label]['fde'].append(min_fde)
                class_stats[label]['mr'].append(mr)
                class_stats[label]['matched'] += 1

        table_data = [[
            'classes', 'minADE', 'minFDE', 'MR', 'Recall', 'GT', 'Match'
        ]]
        ret_dict = {}
        all_ade = []
        all_fde = []
        all_mr = []
        total_gt = 0
        total_matched = 0
        for label, class_name in label2cat.items():
            stats = class_stats[label]
            total_gt += stats['gt']
            total_matched += stats['matched']
            if stats['ade']:
                min_ade = float(np.mean(stats['ade']))
                min_fde = float(np.mean(stats['fde']))
                mr = float(np.mean(stats['mr']))
                all_ade.extend(stats['ade'])
                all_fde.extend(stats['fde'])
                all_mr.extend(stats['mr'])
            else:
                min_ade = np.nan
                min_fde = np.nan
                mr = np.nan
            recall = (
                float(stats['matched'] / stats['gt'])
                if stats['gt'] > 0 else np.nan)
            ret_dict[f'{class_name}_motion_min_ade'] = min_ade
            ret_dict[f'{class_name}_motion_min_fde'] = min_fde
            ret_dict[f'{class_name}_motion_mr'] = mr
            ret_dict[f'{class_name}_motion_recall'] = recall
            ret_dict[f'{class_name}_motion_gt'] = stats['gt']
            ret_dict[f'{class_name}_motion_matched'] = stats['matched']
            table_data.append([
                class_name,
                'nan' if not np.isfinite(min_ade) else f'{min_ade:.4f}',
                'nan' if not np.isfinite(min_fde) else f'{min_fde:.4f}',
                'nan' if not np.isfinite(mr) else f'{mr:.4f}',
                'nan' if not np.isfinite(recall) else f'{recall:.4f}',
                str(stats['gt']),
                str(stats['matched']),
            ])

        overall_ade = float(np.mean(all_ade)) if all_ade else np.nan
        overall_fde = float(np.mean(all_fde)) if all_fde else np.nan
        overall_mr = float(np.mean(all_mr)) if all_mr else np.nan
        overall_recall = (
            float(total_matched / total_gt) if total_gt > 0 else np.nan)
        ret_dict.update(
            motion_min_ade=overall_ade,
            motion_min_fde=overall_fde,
            motion_mr=overall_mr,
            motion_recall=overall_recall,
            motion_gt=total_gt,
            motion_matched=total_matched)
        table_data.append([
            'Overall',
            'nan' if not np.isfinite(overall_ade) else f'{overall_ade:.4f}',
            'nan' if not np.isfinite(overall_fde) else f'{overall_fde:.4f}',
            'nan' if not np.isfinite(overall_mr) else f'{overall_mr:.4f}',
            'nan' if not np.isfinite(overall_recall) else
            f'{overall_recall:.4f}',
            str(total_gt),
            str(total_matched),
        ])
        print_log('\n' + AsciiTable(table_data).table, logger=logger)
        return ret_dict

    def _evaluate_occ_results(self, occ_results_computed, logger=None):
        occ_metrics = [
            key for key in ('iou', 'pq', 'sq', 'rq')
            if key in occ_results_computed
        ]
        if not occ_metrics:
            return {}

        max_ranges = max(len(occ_results_computed[key]) for key in occ_metrics)
        if max_ranges == 2:
            range_names = ['center30m', 'full']
        else:
            range_names = [f'range_{idx}' for idx in range(max_ranges)]

        table_data = [['metric'] + range_names]
        ret_dict = {}
        for metric in occ_metrics:
            values = [float(v) for v in occ_results_computed[metric]]
            table_data.append([
                metric.upper(),
                *[f'{value:.4f}' for value in values],
            ])
            for range_name, value in zip(range_names, values):
                ret_dict[f'occ_{range_name}_{metric}'] = value

        print_log('\nOcc-flow Val Results:', logger=logger)
        print_log('\n' + AsciiTable(table_data).table, logger=logger)

        if 'num_occ' in occ_results_computed:
            ret_dict['occ_num_occ'] = int(occ_results_computed['num_occ'])
        if 'ratio_occ' in occ_results_computed:
            ret_dict['occ_ratio_occ'] = float(occ_results_computed['ratio_occ'])
        if 'occ_num_occ' in ret_dict and 'occ_ratio_occ' in ret_dict:
            print_log(
                f"num occ evaluated: {ret_dict['occ_num_occ']}, "
                f"ratio: {ret_dict['occ_ratio_occ'] * 100:.1f}%",
                logger=logger)
        return ret_dict

    @staticmethod
    def _map_scalar(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().reshape(-1)
            return float(value[0]) if value.numel() > 0 else 0.0
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)
            return float(value[0]) if value.size > 0 else 0.0
        return float(value)

    def _evaluate_map_results(self, results, logger=None):
        pairs = [
            ('drivable', 'drivable_intersection', 'drivable_union'),
            ('lanes', 'lanes_intersection', 'lanes_union'),
            ('divider', 'divider_intersection', 'divider_union'),
            ('crossing', 'crossing_intersection', 'crossing_union'),
            ('contour', 'contour_intersection', 'contour_union'),
        ]
        totals = {
            name: dict(intersection=0.0, union=0.0)
            for name, _, _ in pairs
        }
        count = 0
        for result in results:
            ret_iou = result.get('ret_iou')
            if not ret_iou:
                continue
            count += 1
            for name, inter_key, union_key in pairs:
                totals[name]['intersection'] += self._map_scalar(
                    ret_iou.get(inter_key, 0.0))
                totals[name]['union'] += self._map_scalar(
                    ret_iou.get(union_key, 0.0))

        if count == 0:
            return {}

        table_data = [['class', 'IoU', 'intersection', 'union']]
        ret_dict = {}
        for name, _, _ in pairs:
            intersection = totals[name]['intersection']
            union = totals[name]['union']
            iou = intersection / union if union > 0 else 0.0
            ret_dict[f'map_{name}_iou'] = float(iou)
            table_data.append([
                name,
                f'{iou:.4f}',
                f'{intersection:.0f}',
                f'{union:.0f}',
            ])

        print_log('\nMap Segmentation Val Results:', logger=logger)
        print_log('\n' + AsciiTable(table_data).table, logger=logger)
        return ret_dict

    def _evaluate_planning(self,
                           results,
                           logger=None,
                           ego_width=3.0,
                           ego_length=14.6,
                           eval_steps=(1, 3, 5)):
        """Compute L2 / collision metrics for SDC planning.

        Mirrors UniAD's PlanningMetric but parameterised for the KL
        IGV (3.0 x 14.6 m) and our BEV grid (0.8 m / cell over the
        config-supplied point_cloud_range). Collision is checked
        against gt_segmentation, the rasterised future-object map
        produced by GenerateOccFlowLabels in the test pipeline.

        eval_steps is a tuple of 0-indexed planning steps used for
        per-time slices. With dt = 0.5 s, (1, 3, 5) corresponds to
        the 1 s / 2 s / 3 s reporting points used by camera UniAD.
        """
        pcr = np.asarray(self._planning_pc_range(), dtype=np.float32)
        cell = float(self._planning_cell_size())
        # bev grid order: [H_y, W_x] (rows are y, cols are x)
        bev_w = int(round((pcr[3] - pcr[0]) / cell))
        bev_h = int(round((pcr[4] - pcr[1]) / cell))
        x0, y0 = float(pcr[0]), float(pcr[1])

        # Ego footprint in cell units, ego-centric (x forward, y left).
        # Polygon corners: front-right, front-left, rear-left, rear-right.
        ego_pts_m = np.array([
            [+ego_length / 2.0, -ego_width / 2.0],
            [+ego_length / 2.0, +ego_width / 2.0],
            [-ego_length / 2.0, +ego_width / 2.0],
            [-ego_length / 2.0, -ego_width / 2.0],
        ], dtype=np.float32)

        try:
            from skimage.draw import polygon as draw_polygon
        except ImportError as exc:  # skimage is already a dep
            raise ImportError(
                'skimage required for planning collision rate') from exc

        def ego_footprint_offsets():
            # Cell offsets covered by the ego footprint when centred
            # at (0, 0); reused per trajectory step.
            pts_cell = ego_pts_m / cell
            # polygon expects (row, col) -> (y, x)
            rr, cc = draw_polygon(pts_cell[:, 1], pts_cell[:, 0])
            return rr.astype(np.int32), cc.astype(np.int32)

        rr0, cc0 = ego_footprint_offsets()

        def collision_at(traj_xy, segmentation):
            """traj_xy: [T, 2] in metres (lidar frame, ego-relative).
            segmentation: [T+1, H_bev, W_bev] uint8/long. We index
            t -> segmentation[t + 1] (skip current frame)."""
            T = traj_xy.shape[0]
            collisions = np.zeros(T, dtype=np.bool_)
            seg_T = segmentation.shape[0]
            for t in range(T):
                seg_idx = min(t + 1, seg_T - 1)
                seg_t = segmentation[seg_idx]
                cx = int(round((float(traj_xy[t, 0]) - x0) / cell))
                cy = int(round((float(traj_xy[t, 1]) - y0) / cell))
                rr = rr0 + cy
                cc = cc0 + cx
                m = (rr >= 0) & (rr < bev_h) & (cc >= 0) & (cc < bev_w)
                if not m.any():
                    continue
                if seg_t[rr[m], cc[m]].any():
                    collisions[t] = True
            return collisions

        T_max = max(eval_steps) + 1
        # Per-step accumulators
        l2_sum = np.zeros(T_max, dtype=np.float64)
        l2_n = np.zeros(T_max, dtype=np.int64)
        col_sum = np.zeros(T_max, dtype=np.int64)
        col_n = np.zeros(T_max, dtype=np.int64)

        # Per-command split (Right=0, Left=1, Straight=2)
        per_cmd = {c: dict(l2_sum=np.zeros(T_max),
                           l2_n=np.zeros(T_max, dtype=np.int64),
                           col_sum=np.zeros(T_max, dtype=np.int64),
                           col_n=np.zeros(T_max, dtype=np.int64))
                   for c in (0, 1, 2)}

        bucket_names = ('static', 'slow', 'moving_straight', 'turning')
        per_bucket = {
            name: dict(l2_sum=np.zeros(T_max),
                       l2_n=np.zeros(T_max, dtype=np.int64),
                       col_sum=np.zeros(T_max, dtype=np.int64),
                       col_n=np.zeros(T_max, dtype=np.int64))
            for name in bucket_names
        }

        def wrap_pi(angle):
            return (angle + math.pi) % (2.0 * math.pi) - math.pi

        def heading_change_deg(points, min_segment_disp=0.05):
            if len(points) < 2:
                return 0.0
            pts = np.concatenate([np.zeros((1, 2), dtype=np.float64),
                                  points[:, :2]], axis=0)
            deltas = np.diff(pts, axis=0)
            norms = np.linalg.norm(deltas, axis=1)
            valid = np.where(norms >= min_segment_disp)[0]
            if len(valid) < 2:
                return 0.0
            v0 = deltas[valid[0]]
            v1 = deltas[valid[-1]]
            a0 = math.atan2(float(v0[1]), float(v0[0]))
            a1 = math.atan2(float(v1[1]), float(v1[0]))
            return abs(math.degrees(wrap_pi(a1 - a0)))

        def yaw_change_deg(traj, valid):
            if traj.shape[1] < 3:
                return 0.0
            idx = np.where(valid[:len(traj)])[0]
            if len(idx) < 2:
                return 0.0
            yaw = traj[idx, 2]
            if not np.isfinite(yaw).all():
                return 0.0
            return abs(math.degrees(wrap_pi(float(yaw[-1] - yaw[0]))))

        def lateral_ratio(points):
            if len(points) < 2:
                return 0.0
            end = points[-1, :2]
            net = float(np.linalg.norm(end))
            if net < 1e-6:
                return 0.0
            direction = end / net
            normal = np.array([-direction[1], direction[0]],
                              dtype=np.float64)
            lateral = float(np.max(np.abs(points[:, :2] @ normal)))
            return lateral / max(net, 1e-6)

        def planning_bucket(gt_plan, valid):
            idx = np.where(valid[:len(gt_plan)])[0]
            if len(idx) == 0:
                return None
            last = int(idx[-1])
            points = gt_plan[:last + 1][valid[:last + 1]]
            if len(points) == 0:
                return None
            final_disp = float(np.linalg.norm(points[-1, :2]))
            if final_disp < 0.5:
                return 'static'
            if final_disp < 2.0:
                return 'slow'
            turn_angle = max(
                heading_change_deg(points),
                yaw_change_deg(gt_plan[:last + 1], valid[:last + 1]))
            if turn_angle >= 15.0 or lateral_ratio(points) >= 0.15:
                return 'turning'
            return 'moving_straight'

        def add_l2(agg, err, valid, T):
            for t in range(T):
                if not valid[t]:
                    continue
                agg['l2_sum'][t] += float(err[t])
                agg['l2_n'][t] += 1

        def add_collision(agg, cols, valid, T):
            for t in range(T):
                if not valid[t]:
                    continue
                agg['col_sum'][t] += int(cols[t])
                agg['col_n'][t] += 1

        for result in results:
            plan = result.get('planning')
            if plan is None:
                continue
            pred_blob = plan.get('result_planning', {})
            gt_blob = plan.get('planning_gt', {})
            sdc_traj = pred_blob.get('sdc_traj')
            sdc_planning = gt_blob.get('sdc_planning')
            sdc_planning_mask = gt_blob.get('sdc_planning_mask')
            segmentation = gt_blob.get('segmentation')
            command = gt_blob.get('command')
            if sdc_traj is None or sdc_planning is None:
                continue
            pred_xy = self._planning_to_numpy(sdc_traj)[..., :2]
            gt_plan = self._planning_to_numpy(sdc_planning)
            gt_xy = gt_plan[..., :2]
            mask = self._planning_to_numpy(sdc_planning_mask)
            # Shapes are [1, T, 2 or 3] -> drop the leading 1.
            pred_xy = pred_xy.reshape(-1, pred_xy.shape[-1])
            gt_plan = gt_plan.reshape(-1, gt_plan.shape[-1])
            gt_xy = gt_xy.reshape(-1, gt_xy.shape[-1])
            if mask.ndim == 3:
                mask = mask[0, :, 0]
            elif mask.ndim == 2:
                mask = mask[:, 0]
            else:
                mask = mask.reshape(-1)
            mask = mask.astype(bool)

            cmd_id = None
            if command is not None:
                cmd_id = int(np.asarray(
                    self._planning_to_numpy(command)).reshape(-1)[0])

            T = min(T_max, pred_xy.shape[0], gt_xy.shape[0], len(mask))
            err = np.linalg.norm(pred_xy[:T, :2] - gt_xy[:T, :2], axis=-1)
            bucket_name = planning_bucket(gt_plan, mask)
            add_l2(dict(l2_sum=l2_sum, l2_n=l2_n), err, mask, T)
            if cmd_id in per_cmd:
                add_l2(per_cmd[cmd_id], err, mask, T)
            if bucket_name in per_bucket:
                add_l2(per_bucket[bucket_name], err, mask, T)

            # Collision needs segmentation; skip frames missing it.
            if segmentation is None:
                continue
            seg = self._planning_to_numpy(segmentation)
            # Expected shape [1, T_seg, H, W] or [T_seg, H, W].
            if seg.ndim == 4:
                seg = seg[0]
            if seg.ndim != 3:
                continue
            if seg.shape[-2:] != (bev_h, bev_w):
                continue
            cols = collision_at(pred_xy[:T], seg)
            add_collision(dict(col_sum=col_sum, col_n=col_n), cols, mask, T)
            if cmd_id in per_cmd:
                add_collision(per_cmd[cmd_id], cols, mask, T)
            if bucket_name in per_bucket:
                add_collision(per_bucket[bucket_name], cols, mask, T)

        ret_dict = {}

        def _format(steps, l2sum, l2n, csum, cnum):
            l2_at = []
            col_at = []
            for s in steps:
                l2 = (l2sum[s] / l2n[s]) if l2n[s] > 0 else float('nan')
                cl = (csum[s] / cnum[s]) if cnum[s] > 0 else float('nan')
                l2_at.append(l2)
                col_at.append(cl)
            return l2_at, col_at

        l2_at, col_at = _format(eval_steps, l2_sum, l2_n,
                                col_sum, col_n)
        labels = [f'{(s + 1) * 0.5:.0f}s' for s in eval_steps]
        avg_l2 = float(np.nanmean(l2_at)) if l2_at else float('nan')
        avg_col = float(np.nanmean(col_at)) if col_at else float('nan')

        # Scalar entries for downstream loggers.
        for s, lbl, l2, cl in zip(eval_steps, labels, l2_at, col_at):
            ret_dict[f'planning/L2_{lbl}'] = l2
            ret_dict[f'planning/Collision_{lbl}'] = cl
        ret_dict['planning/avg.L2'] = avg_l2
        ret_dict['planning/avg.Collision'] = avg_col

        # Pretty-print summary
        lines = ['', 'Planning metrics (lower is better):']
        for s, lbl in zip(eval_steps, labels):
            lines.append(
                f'  L2 @ {lbl:>3}: {l2_sum[s] / max(l2_n[s], 1):.3f} m'
                f'   Collision @ {lbl:>3}: '
                f'{100.0 * col_sum[s] / max(col_n[s], 1):.2f}% '
                f'(N={l2_n[s]})')
        lines.append(f'  avg.L2:        {avg_l2:.3f} m')
        lines.append(f'  avg.Collision: {100.0 * avg_col:.2f}%')

        # Per-command breakdown (only emit lines that have data).
        cmd_name = {0: 'Right', 1: 'Left', 2: 'Straight'}
        per_cmd_lines = ['', 'Planning per command:']
        for cmd_id in (1, 0, 2):
            agg = per_cmd[cmd_id]
            n_total = int(agg['l2_n'].max() if agg['l2_n'].size else 0)
            if n_total == 0:
                continue
            l2_at_c, col_at_c = _format(eval_steps,
                                        agg['l2_sum'], agg['l2_n'],
                                        agg['col_sum'], agg['col_n'])
            avg_l2_c = float(np.nanmean(l2_at_c))
            avg_col_c = float(np.nanmean(col_at_c))
            per_cmd_lines.append(
                f'  {cmd_name[cmd_id]:8s} N={n_total:5d}  '
                f'avg.L2 {avg_l2_c:.3f} m  '
                f'avg.Collision {100.0 * avg_col_c:.2f}%')
            ret_dict[f'planning/{cmd_name[cmd_id]}/avg.L2'] = avg_l2_c
            ret_dict[f'planning/{cmd_name[cmd_id]}/avg.Collision'] = avg_col_c
        if len(per_cmd_lines) > 2:
            lines.extend(per_cmd_lines)

        bucket_title = {
            'static': 'Static',
            'slow': 'Slow',
            'moving_straight': 'MovingStraight',
            'turning': 'Turning',
        }
        per_bucket_lines = ['', 'Planning per GT motion bucket:']
        for bucket_name in bucket_names:
            agg = per_bucket[bucket_name]
            n_total = int(agg['l2_n'].max() if agg['l2_n'].size else 0)
            if n_total == 0:
                continue
            l2_at_b, col_at_b = _format(eval_steps,
                                        agg['l2_sum'], agg['l2_n'],
                                        agg['col_sum'], agg['col_n'])
            avg_l2_b = float(np.nanmean(l2_at_b))
            avg_col_b = float(np.nanmean(col_at_b))
            title = bucket_title[bucket_name]
            per_bucket_lines.append(
                f'  {title:14s} N={n_total:5d}  '
                f'avg.L2 {avg_l2_b:.3f} m  '
                f'avg.Collision {100.0 * avg_col_b:.2f}%')
            ret_dict[f'planning/{title}/avg.L2'] = avg_l2_b
            ret_dict[f'planning/{title}/avg.Collision'] = avg_col_b
            ret_dict[f'planning/{title}/N'] = n_total
            for lbl, l2, cl in zip(labels, l2_at_b, col_at_b):
                ret_dict[f'planning/{title}/L2_{lbl}'] = l2
                ret_dict[f'planning/{title}/Collision_{lbl}'] = cl
        if len(per_bucket_lines) > 2:
            lines.extend(per_bucket_lines)

        for line in lines:
            print_log(line, logger=logger)
        return ret_dict

    @staticmethod
    def _planning_to_numpy(value):
        if isinstance(value, list) and value and hasattr(value[0], 'cpu'):
            value = value[0]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        return np.asarray(value)

    def _planning_pc_range(self):
        """Resolve point_cloud_range used to lay out gt_segmentation.

        Falls back to the dataset attribute, then the standard KL
        ±64/±48 range used by the LiDAR plan config.
        """
        for attr in ('pc_range', 'point_cloud_range'):
            v = getattr(self, attr, None)
            if v is not None:
                return list(v)
        return [-64.0, -48.0, -2.0, 64.0, 48.0, 6.0]

    def _planning_cell_size(self):
        # Match GenerateOccFlowLabels grid_conf['xbound'][2].
        return getattr(self, 'occ_cell_size', 0.8)


    def evaluate(self,
                 results,
                 metric='bbox',
                 iou_thr=(0.25, 0.5),
                 logger=None,
                 jsonfile_prefix=None,
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None,
                 tracking_dist_th=2.0,
                 **kwargs):
        occ_results_computed = None
        if isinstance(results, dict):
            occ_results_computed = results.get('occ_results_computed', None)
            if 'bbox_results' not in results:
                raise KeyError('results dict must contain "bbox_results"')
            results = results['bbox_results']

        assert isinstance(results, list), \
            f'Expect results to be list, got {type(results)}.'
        assert len(results) == len(self), \
            f'The length of results is not equal to the dataset len: {len(results)} != {len(self)}'

        norm_results = []
        for result in results:
            planning = result.get('planning') if isinstance(
                result, dict) else None
            if 'pts_bbox' in result:
                result = result['pts_bbox']
            elif 'img_bbox' in result:
                result = result['img_bbox']
            result = dict(result)
            if planning is not None:
                # Preserve the planning sub-dict the detector emits
                # alongside pts_bbox; the unwrap above would otherwise
                # drop it since we reassign to result['pts_bbox'].
                result['planning'] = planning
            if ('track_ids' in result and 'boxes_3d' in result
                    and 'track_boxes_3d' not in result):
                result['track_boxes_3d'] = result['boxes_3d']
                result['track_labels_3d'] = result.get('labels_3d', None)
            if 'boxes_3d_det' in result:
                result['boxes_3d'] = result['boxes_3d_det']
                result['scores_3d'] = result['scores_3d_det']
                result['labels_3d'] = result['labels_3d_det']
            for box_key in ('boxes_3d', 'track_boxes_3d'):
                if box_key in result and hasattr(result[box_key], 'to'):
                    result[box_key] = result[box_key].to('cpu')
            for tensor_key in ('scores_3d', 'labels_3d', 'track_scores',
                               'track_ids', 'track_labels_3d'):
                if (tensor_key in result
                        and isinstance(result[tensor_key], torch.Tensor)):
                    result[tensor_key] = result[tensor_key].detach().cpu()
            for tensor_key, value in list(result.items()):
                if tensor_key.startswith('traj') and isinstance(
                        value, torch.Tensor):
                    result[tensor_key] = value.detach().cpu()
            norm_results.append(result)

        gt_ann_infos = []
        for idx in range(len(self)):
            ann_info = self.get_ann_info(idx)
            gt_ann_infos.append(ann_info)

        ret_dict = self._evaluate_nusc_style(
            norm_results, gt_ann_infos, logger=logger)
        has_tracking = any(
            'track_ids' in result and 'track_boxes_3d' in result
            for result in norm_results)
        if has_tracking:
            ret_dict.update(
                self._evaluate_tracking(
                    norm_results,
                    gt_ann_infos,
                    logger=logger,
                    dist_th=tracking_dist_th,
                    score_thresholds=kwargs.get(
                        'tracking_score_thresholds', None)))
        has_motion = any('traj' in result for result in norm_results)
        if has_motion:
            ret_dict.update(
                self._evaluate_motion(
                    norm_results,
                    gt_ann_infos,
                    logger=logger,
                    dist_th=kwargs.get('motion_dist_th', tracking_dist_th),
                    miss_threshold=kwargs.get('motion_miss_threshold', 2.0),
                    score_thr=kwargs.get('motion_score_thr', 0.0)))
        if occ_results_computed is not None:
            ret_dict.update(
                self._evaluate_occ_results(
                    occ_results_computed, logger=logger))
        has_map = any('ret_iou' in result for result in norm_results)
        if has_map:
            ret_dict.update(
                self._evaluate_map_results(norm_results, logger=logger))
        has_planning = any('planning' in result for result in norm_results)
        if has_planning:
            ret_dict.update(
                self._evaluate_planning(
                    norm_results,
                    logger=logger,
                    ego_width=kwargs.get('planning_ego_width', 3.0),
                    ego_length=kwargs.get('planning_ego_length', 14.6),
                    eval_steps=kwargs.get('planning_eval_steps',
                                          (1, 3, 5))))

        if show:
            self.show(norm_results, out_dir, pipeline=pipeline)
        return ret_dict


@DATASETS.register_module()
class KlBEVFormerDataset(KlDataset):

    def __init__(self,
                 *args,
                 queue_length=4,
                 max_time_gap=1.0,
                 **kwargs):
        assert queue_length >= 1
        self.queue_length = queue_length
        self.max_time_gap = float(max_time_gap)
        self.token2index = {}
        self.valid_data_indices = None
        super().__init__(*args, **kwargs)
        self.token2index = {
            info.get('token'): idx
            for idx, info in enumerate(self.data_infos)
            if info.get('token')
        }
        if self.test_mode:
            self.valid_data_indices = [
                idx for idx in range(len(self.data_infos))
                if self._collect_queue_indices(idx) is not None
            ]

    def __len__(self):
        if self.valid_data_indices is not None:
            return len(self.valid_data_indices)
        return len(self.data_infos)

    def _to_raw_index(self, index):
        if self.valid_data_indices is None:
            return index
        if index < 0:
            index += len(self.valid_data_indices)
        return self.valid_data_indices[index]

    def get_data_info(self, index):
        return super().get_data_info(self._to_raw_index(index))

    def get_ann_info(self, index):
        return super().get_ann_info(self._to_raw_index(index))

    def prepare_train_data(self, index):
        data = self._prepare_queue_data(self._to_raw_index(index))
        if data is not None:
            return data
        for _ in range(10):
            raw_index = np.random.randint(0, len(self.data_infos))
            data = self._prepare_queue_data(raw_index)
            if data is not None:
                return data
        return None

    def prepare_test_data(self, index):
        return self._prepare_queue_data(self._to_raw_index(index))

    def _prepare_queue_data(self, raw_index):
        indices = self._collect_queue_indices(raw_index)
        if indices is None:
            return None

        queue = []
        raw_meta = []
        for queue_idx, idx in enumerate(indices):
            input_dict = KlDataset.get_data_info(self, idx)
            if input_dict is None:
                return None
            raw_meta.append(self._extract_raw_meta(idx))
            input_dict['_kl_is_current_frame'] = (
                queue_idx == len(indices) - 1)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if example is None:
                return None
            queue.append(example)
        return self._union2one(queue, raw_meta)

    def _collect_queue_indices(self, raw_index):
        if self.queue_length == 1:
            return [raw_index]

        current = self.data_infos[raw_index]
        scene_token = current.get('scene_token')
        indices = [raw_index]
        cursor = current
        prev_token = current.get('prev', '')
        for _ in range(self.queue_length - 1):
            if not prev_token:
                return None
            prev_index = self.token2index.get(prev_token)
            if prev_index is None:
                return None
            prev_info = self.data_infos[prev_index]
            if prev_info.get('scene_token') != scene_token:
                return None
            dt = abs(float(cursor.get('timestamp', 0.0)) -
                     float(prev_info.get('timestamp', 0.0)))
            if dt > self.max_time_gap:
                return None
            indices.append(prev_index)
            cursor = prev_info
            prev_token = prev_info.get('prev', '')

        indices.reverse()
        return indices

    def _extract_raw_meta(self, raw_index):
        info = self.data_infos[raw_index]
        return dict(
            sample_idx=info.get('sample_idx', raw_index),
            scene_token=info.get('scene_token', ''),
            ego2global=np.asarray(info.get('ego2global', np.eye(4)),
                                  dtype=np.float64),
            prev=info.get('prev', None),
            next=info.get('next', None),
            timestamp=float(info.get('timestamp', 0.0)),
            token=info.get('token', ''),
            # VLM scene caption (cross-modal distillation target); None if
            # the frame has no geo_facts/summary. Carried in img_metas so it
            # reaches the detector without touching the shared Collect3D config.
            gt_caption=(info.get('geo_facts') or {}).get('summary'))

    @staticmethod
    def _dc_data(value):
        return value.data if isinstance(value, DC) else value

    def _union2one(self, queue, raw_meta):
        assert len(queue) == len(raw_meta) == self.queue_length
        sample = queue[-1]
        history_points = [
            self._dc_data(frame['points']) for frame in queue[:-1]
        ]
        sample['history_points'] = DC(history_points, stack=False)

        queue_metas = {}
        prev_ego2global = None
        prev_timestamp = None
        for idx, meta in enumerate(raw_meta):
            entry = dict(meta)
            ego2global = meta['ego2global']
            timestamp = meta['timestamp']
            if idx == 0:
                entry['prev_bev_exists'] = False
                entry['ego_motion_delta'] = np.eye(4, dtype=np.float64)
                entry['time_delta'] = 0.0
            else:
                entry['prev_bev_exists'] = True
                entry['ego_motion_delta'] = (
                    np.linalg.inv(ego2global) @ prev_ego2global)
                entry['time_delta'] = float(timestamp - prev_timestamp)
            queue_metas[idx] = entry
            prev_ego2global = ego2global
            prev_timestamp = timestamp

        img_metas = self._dc_data(sample['img_metas'])
        img_metas['queue_metas'] = queue_metas
        # Current-frame VLM caption (cross-modal distillation target).
        img_metas['gt_caption'] = raw_meta[-1].get('gt_caption')
        sample['img_metas'] = DC(img_metas, cpu_only=True)
        return sample


@DATASETS.register_module()
class KlTrackDataset(KlBEVFormerDataset):
    """KL temporal dataset that returns a full queue for tracking training."""

    def __init__(self,
                 *args,
                 occ_receptive_field=None,
                 occ_n_future=None,
                 occ_filter_invalid_sample=False,
                 **kwargs):
        self.occ_receptive_field = occ_receptive_field
        self.occ_n_future = occ_n_future
        self.occ_filter_invalid_sample = occ_filter_invalid_sample
        super().__init__(*args, **kwargs)

    @property
    def with_occ_labels(self):
        return (self.occ_receptive_field is not None
                and self.occ_n_future is not None)

    @staticmethod
    def _as_tensor(value, dtype=None):
        tensor = torch.as_tensor(value)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor

    @staticmethod
    def _ego_pose_parts(meta):
        ego2global = np.asarray(meta['ego2global'], dtype=np.float32)
        return ego2global[:3, :3], ego2global[:3, 3]

    def _prepare_queue_data(self, raw_index):
        indices = self._collect_queue_indices(raw_index)
        if indices is None:
            return None

        queue = []
        raw_meta = []
        for queue_idx, idx in enumerate(indices):
            input_dict = KlDataset.get_data_info(self, idx)
            if input_dict is None:
                return None
            if self.with_occ_labels:
                if 'ann_info' not in input_dict:
                    input_dict['ann_info'] = KlDataset.get_ann_info(self, idx)
                occ_inputs = self._build_occ_inputs(idx)
                if occ_inputs is None:
                    return None
                input_dict.update(occ_inputs)
            raw_meta.append(self._extract_raw_meta(idx))
            input_dict['_kl_is_current_frame'] = (
                queue_idx == len(indices) - 1)
            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            if example is None:
                return None
            queue.append(example)
        return self._union2one(queue, raw_meta)

    def _collect_occ_prev_indices(self, raw_index):
        scene_token = self.data_infos[raw_index].get('scene_token')
        out = []
        prev_token = self.data_infos[raw_index].get('prev', '')
        for _ in range(self.occ_receptive_field - 1):
            prev_idx = self.token2index.get(prev_token, -1) if prev_token else -1
            if (prev_idx < 0
                    or self.data_infos[prev_idx].get('scene_token') != scene_token):
                out.append(-1)
                prev_token = ''
                continue
            out.append(prev_idx)
            prev_token = self.data_infos[prev_idx].get('prev', '')
        out.reverse()
        return out

    def _collect_occ_future_indices(self, raw_index):
        scene_token = self.data_infos[raw_index].get('scene_token')
        out = []
        next_token = self.data_infos[raw_index].get('next', '')
        for _ in range(self.occ_n_future):
            next_idx = self.token2index.get(next_token, -1) if next_token else -1
            if (next_idx < 0
                    or self.data_infos[next_idx].get('scene_token') != scene_token):
                out.append(-1)
                next_token = ''
                continue
            out.append(next_idx)
            next_token = self.data_infos[next_idx].get('next', '')
        return out

    @staticmethod
    def _occ_pose_parts(info):
        ego2global = np.asarray(info.get('ego2global', np.eye(4)),
                                dtype=np.float32)
        l2e_r = np.eye(3, dtype=np.float32)
        l2e_t = np.zeros(3, dtype=np.float32)
        return l2e_r, l2e_t, ego2global[:3, :3], ego2global[:3, 3]

    def _build_occ_inputs(self, raw_index):
        prev_indices = self._collect_occ_prev_indices(raw_index)
        future_indices = self._collect_occ_future_indices(raw_index)
        all_validity_frames = prev_indices + [raw_index] + future_indices
        if self.occ_filter_invalid_sample and -1 in all_validity_frames:
            return None

        future_frames = [raw_index] + future_indices
        future_ann_infos = []
        l2e_r_mats = []
        l2e_t_vecs = []
        e2g_r_mats = []
        e2g_t_vecs = []
        for frame_idx in future_frames:
            if frame_idx < 0:
                future_ann_infos.append(None)
                l2e_r_mats.append(None)
                l2e_t_vecs.append(None)
                e2g_r_mats.append(None)
                e2g_t_vecs.append(None)
                continue
            ann_info = copy.deepcopy(KlDataset.get_ann_info(self, frame_idx))
            ann_info['gt_vis_tokens'] = None
            future_ann_infos.append(ann_info)
            l2e_r, l2e_t, e2g_r, e2g_t = self._occ_pose_parts(
                self.data_infos[frame_idx])
            l2e_r_mats.append(torch.from_numpy(l2e_r))
            l2e_t_vecs.append(torch.from_numpy(l2e_t))
            e2g_r_mats.append(torch.from_numpy(e2g_r))
            e2g_t_vecs.append(torch.from_numpy(e2g_t))

        return dict(
            occ_future_ann_infos=future_ann_infos,
            occ_l2e_r_mats=l2e_r_mats,
            occ_l2e_t_vecs=l2e_t_vecs,
            occ_e2g_r_mats=e2g_r_mats,
            occ_e2g_t_vecs=e2g_t_vecs,
            occ_has_invalid_frame=-1 in all_validity_frames,
            occ_img_is_valid=np.asarray(
                [idx >= 0 for idx in all_validity_frames], dtype=np.bool_))

    def _union2one(self, queue, raw_meta):
        assert len(queue) == len(raw_meta) == self.queue_length
        sample = queue[-1]

        points_list = [self._dc_data(each['points']) for each in queue]
        # Track-level GT (labels/bboxes/inds/past_traj) are absent in
        # test pipelines that only consume drivable-map fields. Mirror
        # the optional handling used below for fut_traj / sdc / planning.
        has_track_gt = all('gt_labels_3d' in each for each in queue)
        if has_track_gt:
            gt_labels_3d_list = [
                self._dc_data(each['gt_labels_3d']) for each in queue
            ]
            gt_bboxes_3d_list = [
                self._dc_data(each['gt_bboxes_3d']) for each in queue
            ]
            gt_inds_list = [
                self._as_tensor(each.get('gt_inds', []), dtype=torch.long)
                for each in queue
            ]
            gt_past_traj_list = [
                self._as_tensor(each.get('gt_past_traj', []),
                                dtype=torch.float32)
                for each in queue
            ]
            gt_past_traj_mask_list = [
                self._as_tensor(each.get('gt_past_traj_mask', []),
                                dtype=torch.float32)
                for each in queue
            ]
        has_fut_traj = all('gt_fut_traj' in each for each in queue)
        if has_fut_traj:
            gt_fut_traj = self._as_tensor(
                queue[-1]['gt_fut_traj'], dtype=torch.float32)
            gt_fut_traj_mask = self._as_tensor(
                queue[-1]['gt_fut_traj_mask'], dtype=torch.float32)

        has_sdc = all('gt_sdc_bbox' in each for each in queue)
        if has_sdc:
            gt_sdc_bbox_list = [
                self._dc_data(each['gt_sdc_bbox']) for each in queue
            ]
            gt_sdc_label_list = [
                self._as_tensor(each['gt_sdc_label'], dtype=torch.long)
                for each in queue
            ]
            # SDC future trajectory only supervises the current frame's
            # motion head, so collapse to a single tensor (mirrors how
            # gt_fut_traj is collected above).
            gt_sdc_fut_traj = self._as_tensor(
                queue[-1]['gt_sdc_fut_traj'], dtype=torch.float32)
            gt_sdc_fut_traj_mask = self._as_tensor(
                queue[-1]['gt_sdc_fut_traj_mask'], dtype=torch.float32)

        # Planning supervision is keyed off sdc_planning being present in
        # the current frame; the field is dormant until the loader op is
        # configured to propagate it.
        has_planning = 'sdc_planning' in queue[-1]
        if has_planning:
            sdc_planning = self._as_tensor(
                queue[-1]['sdc_planning'], dtype=torch.float32)
            sdc_planning_mask = self._as_tensor(
                queue[-1]['sdc_planning_mask'], dtype=torch.float32)
            # PlanningHead's navi_embed indexing expects command to be a
            # scalar per sample (tensor shape [B] after collation, not
            # [B, 1]). Strip the leading dim from the [1] pkl payload.
            command_raw = self._as_tensor(
                queue[-1]['command'], dtype=torch.long)
            command = command_raw.reshape(()) if command_raw.numel() == 1 \
                else command_raw.squeeze(0)

        # gt_future_boxes / gt_future_labels are produced by
        # GenerateOccFlowLabels at the current frame and consumed by
        # PlanningHead's collision loss. They reference the current +
        # occ_n_future frames in the *current* lidar frame.
        has_future_boxes = 'gt_future_boxes' in queue[-1]
        if has_future_boxes:
            gt_future_boxes_list = queue[-1]['gt_future_boxes']
            gt_future_labels_list = [
                self._as_tensor(each, dtype=torch.long)
                for each in queue[-1]['gt_future_labels']
            ]

        l2g_r_mat_list = []
        l2g_t_list = []
        timestamp_list = []
        metas_map = {}
        prev_ego2global = None
        prev_timestamp = None
        for idx, (frame, meta) in enumerate(zip(queue, raw_meta)):
            frame_meta = copy.deepcopy(self._dc_data(frame['img_metas']))
            frame_meta.update(
                sample_idx=meta.get('sample_idx', frame_meta.get('sample_idx')),
                scene_token=meta.get('scene_token', ''),
                token=meta.get('token', ''),
                prev=meta.get('prev', None),
                next=meta.get('next', None),
                timestamp=float(meta.get('timestamp', 0.0)))
            ego2global = np.asarray(meta['ego2global'], dtype=np.float64)
            frame_meta['ego2global'] = ego2global.copy()
            if idx == 0:
                frame_meta['prev_bev'] = False
                frame_meta['prev_bev_exists'] = False
                frame_meta['ego_motion_delta'] = np.eye(4, dtype=np.float64)
                frame_meta['time_delta'] = 0.0
            else:
                frame_meta['prev_bev'] = True
                frame_meta['prev_bev_exists'] = True
                frame_meta['ego_motion_delta'] = (
                    np.linalg.inv(ego2global) @ prev_ego2global)
                frame_meta['time_delta'] = float(meta['timestamp'] -
                                                 prev_timestamp)
            metas_map[idx] = frame_meta

            # Current-frame VLM caption (cross-modal distillation target).
            # Lives on the last queue frame's meta; the detector reads it via
            # _current_caption (which indexes metas_map[max(keys)]).
            if idx == len(queue) - 1:
                frame_meta['gt_caption'] = meta.get('gt_caption')

            l2g_r_mat, l2g_t = self._ego_pose_parts(meta)
            l2g_r_mat_list.append(self._as_tensor(l2g_r_mat))
            l2g_t_list.append(self._as_tensor(l2g_t))
            timestamp_list.append(
                self._as_tensor(float(meta.get('timestamp', 0.0)),
                                dtype=torch.float32))
            prev_ego2global = ego2global
            prev_timestamp = meta['timestamp']

        sample['points'] = DC(points_list, stack=False)
        sample['history_points'] = DC(points_list[:-1], stack=False)
        sample['img_metas'] = DC(metas_map, cpu_only=True)
        if has_track_gt:
            sample['gt_labels_3d'] = DC(gt_labels_3d_list)
            sample['gt_bboxes_3d'] = DC(gt_bboxes_3d_list, cpu_only=True)
            sample['gt_inds'] = DC(gt_inds_list)
            sample['gt_past_traj'] = DC(gt_past_traj_list)
            sample['gt_past_traj_mask'] = DC(gt_past_traj_mask_list)
        if has_fut_traj:
            sample['gt_fut_traj'] = DC(gt_fut_traj)
            sample['gt_fut_traj_mask'] = DC(gt_fut_traj_mask)
        if has_sdc:
            sample['gt_sdc_bbox'] = DC(gt_sdc_bbox_list, cpu_only=True)
            sample['gt_sdc_label'] = DC(gt_sdc_label_list)
            sample['gt_sdc_fut_traj'] = DC(gt_sdc_fut_traj)
            sample['gt_sdc_fut_traj_mask'] = DC(gt_sdc_fut_traj_mask)
        if has_planning:
            # stack=True so DataLoader concatenates along a new batch dim
            # (mirrors how camera nuScenes_e2e returns these as plain
            # numpy that default_collate stacks). pad_dims=None because
            # command is 1-D and the DC default pad_dims=2 would assert.
            sample['sdc_planning'] = DC(
                sdc_planning, stack=True, pad_dims=None)
            sample['sdc_planning_mask'] = DC(
                sdc_planning_mask, stack=True, pad_dims=None)
            sample['command'] = DC(command, stack=True, pad_dims=None)
        if has_future_boxes:
            sample['gt_future_boxes'] = DC(gt_future_boxes_list, cpu_only=True)
            sample['gt_future_labels'] = DC(gt_future_labels_list)
        sample['l2g_r_mat'] = DC(l2g_r_mat_list)
        sample['l2g_t'] = DC(l2g_t_list)
        sample['timestamp'] = DC(timestamp_list)
        return sample
