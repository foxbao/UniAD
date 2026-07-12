import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import DataContainer
from mmcv.runner import get_dist_info

from ..dense_heads.occ_head_plugin import IntersectionOverUnion, PanopticMetric
from ..dense_heads.planning_head_plugin import PlanningMetric

import mmcv
import numpy as np
import pycocotools.mask as mask_util

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]


def _unwrap_model(model):
    return model.module if hasattr(model, 'module') else model


def _scatter_data_for_eval(model, data):
    """Scatter MMCV DataContainer batches before direct DDP eval forward."""
    if not isinstance(data.get('points'), DataContainer):
        return model, data
    if not hasattr(model, 'scatter') or not getattr(model, 'device_ids', None):
        return model, data
    _, scattered_kwargs = model.scatter((), data, model.device_ids)
    return _unwrap_model(model), scattered_kwargs[0]


def _base_dataset(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def _occ_grid_cell_sizes(dataset, height, width):
    dataset = _base_dataset(dataset)
    point_cloud_range = None
    for attr in ('eval_point_cloud_range', 'point_cloud_range', 'pc_range'):
        value = getattr(dataset, attr, None)
        if value is not None:
            point_cloud_range = np.asarray(value, dtype=np.float32)
            break
    if point_cloud_range is not None and point_cloud_range.shape[0] >= 5:
        cell_x = float(point_cloud_range[3] - point_cloud_range[0]) / width
        cell_y = float(point_cloud_range[4] - point_cloud_range[1]) / height
        return cell_y, cell_x
    if hasattr(dataset, '_planning_cell_size'):
        cell = float(dataset._planning_cell_size())
        return cell, cell
    return 0.5, 0.5


def _center_slice(length, size):
    size = max(1, min(int(size), int(length)))
    start = max(0, (int(length) - size) // 2)
    return slice(start, start + size)


def _build_occ_eval_ranges(dataset, occ_tensor):
    height, width = occ_tensor.shape[-2:]
    cell_y, cell_x = _occ_grid_cell_sizes(dataset, height, width)
    center_h = int(round(30.0 / cell_y))
    center_w = int(round(30.0 / cell_x))
    return {
        'center30m': (
            _center_slice(height, center_h),
            _center_slice(width, center_w)),
        'full': (slice(0, height), slice(0, width)),
    }


def _planning_tensor(value):
    if isinstance(value, DataContainer):
        value = value.data
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return _planning_tensor(value[0])
        tensors = [_planning_tensor(v) for v in value]
        if all(torch.is_tensor(v) for v in tensors):
            return torch.stack(tensors, dim=0)
        return tensors[0]
    if torch.is_tensor(value):
        return value
    return torch.as_tensor(value)


def _planning_traj_btc(value, steps=6, channels=2):
    traj = _planning_tensor(value)
    if traj.dim() == 4:
        traj = traj[0] if traj.shape[0] == 1 else traj.flatten(0, 1)
    elif traj.dim() == 2:
        traj = traj.unsqueeze(0)
    if traj.dim() != 3:
        raise ValueError(f'Unexpected planning trajectory shape: {traj.shape}')
    return traj[:, :steps, :channels].contiguous()


def _planning_mask_btc(value, steps=6, channels=2):
    mask = _planning_tensor(value)
    if mask.dim() == 4:
        mask = mask[0] if mask.shape[0] == 1 else mask.flatten(0, 1)
    if mask.dim() == 2:
        if mask.shape[-1] == channels and mask.shape[0] >= steps:
            mask = mask.unsqueeze(0)
        else:
            mask = mask[:, :steps].unsqueeze(-1)
    if mask.dim() != 3:
        raise ValueError(f'Unexpected planning mask shape: {mask.shape}')
    mask = mask[:, :steps]
    if mask.shape[-1] >= channels:
        mask = mask[..., :channels]
    else:
        mask = mask.expand(*mask.shape[:-1], channels)
    return mask.contiguous()


def _planning_seg_bthw(value, batch_size, steps=6):
    seg = _planning_tensor(value)
    if seg.dim() == 5:
        seg = seg[0] if seg.shape[0] == 1 else seg.flatten(0, 1)
    elif seg.dim() == 3:
        seg = seg.unsqueeze(0)
    if seg.dim() != 4:
        raise ValueError(f'Unexpected planning segmentation shape: {seg.shape}')
    if seg.shape[1] >= steps + 1:
        seg = seg[:, 1:steps + 1]
    else:
        seg = seg[:, :steps]
    if seg.shape[0] == 1 and batch_size > 1:
        seg = seg.expand(batch_size, *seg.shape[1:])
    return seg.contiguous()


def _detach_to_cpu(obj):
    if isinstance(obj, DataContainer):
        return _detach_to_cpu(obj.data)
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if hasattr(obj, 'tensor') and hasattr(obj, 'to'):
        return obj.to('cpu')
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


def _planning_eval_tensor(value, dtype=None):
    if value is None:
        return None
    tensor = _planning_tensor(value).detach().cpu()
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def _compact_planning_for_eval(planning):
    """Keep only the fields needed by KlDataset._evaluate_planning."""
    if not isinstance(planning, dict):
        return _detach_to_cpu(planning)

    result_planning = planning.get('result_planning', {})
    planning_gt = planning.get('planning_gt', {})
    compact = dict(result_planning={}, planning_gt={})

    if isinstance(result_planning, dict) and 'sdc_traj' in result_planning:
        compact['result_planning']['sdc_traj'] = _planning_eval_tensor(
            result_planning['sdc_traj'], dtype=torch.float32)
    if isinstance(result_planning, dict):
        selector_field_dtypes = {
            'lane_anchor_selector_target': torch.long,
            'lane_anchor_selector_pred': torch.long,
            'lane_anchor_selector_valid': torch.bool,
            'lane_anchor_selector_num_valid': torch.long,
            'lane_anchor_selector_oracle_l2': torch.float32,
            'lane_anchor_selector_pred_l2': torch.float32,
            'lane_anchor_selector_selected_l2': torch.float32,
            'lane_anchor_selector_entropy': torch.float32,
            'lane_anchor_selector_max_prob': torch.float32,
            'lane_anchor_selector_oracle_anchor': torch.float32,
            'lane_anchor_selector_pred_anchor': torch.float32,
            'lane_anchor_selector_selected_anchor': torch.float32,
            'multimodal_selected_index': torch.long,
            'multimodal_fallback_index': torch.long,
            'multimodal_candidate_count': torch.long,
            'multimodal_map_selected_index': torch.long,
            'multimodal_utility_probability': torch.float32,
            'multimodal_utility_score': torch.float32,
            'multimodal_selected_map_traj': torch.float32,
            'multimodal_fallback_traj': torch.float32,
        }
        for key, dtype in selector_field_dtypes.items():
            if key in result_planning:
                compact['result_planning'][key] = _planning_eval_tensor(
                    result_planning[key], dtype=dtype)

    if isinstance(planning_gt, dict):
        field_dtypes = {
            'sdc_planning': torch.float32,
            'sdc_planning_mask': torch.uint8,
            'segmentation': torch.uint8,
            'command': torch.long,
        }
        for key, dtype in field_dtypes.items():
            if key in planning_gt:
                compact['planning_gt'][key] = _planning_eval_tensor(
                    planning_gt[key], dtype=dtype)

    return compact


def _dataset_wants_planning_payload(dataset):
    return hasattr(_base_dataset(dataset), '_evaluate_planning')


def _strip_eval_intermediates(item, keep_planning=False):
    """Remove per-frame GPU intermediates that are not used by metrics."""
    item.pop('occ', None)
    if keep_planning and 'planning' in item:
        item['planning'] = _compact_planning_for_eval(item['planning'])
    else:
        item.pop('planning', None)
    item.pop('map', None)
    item.pop('args_tuple', None)
    pts_bbox = item.get('pts_bbox', None)
    if isinstance(pts_bbox, dict):
        for key in [
                'bev_embed', 'bev_pos', 'prev_bev',
                'track_query_embeddings', 'track_query_matched_idxes',
                'track_bbox_results', 'sdc_embedding',
                'sdc_track_bbox_results', 'map', 'args_tuple',
                'score_list', 'lane', 'lane_score', 'stuff_score_list',
                'panoptic', 'segm'
        ]:
            pts_bbox.pop(key, None)
        item['pts_bbox'] = _detach_to_cpu(pts_bbox)
    if 'ret_iou' in item:
        item['ret_iou'] = _detach_to_cpu(item['ret_iou'])
    return item


def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()

    # Occ eval init
    model_to_eval = _unwrap_model(model)
    eval_occ = hasattr(model_to_eval, 'with_occ_head') \
                and model_to_eval.with_occ_head
    if eval_occ:
        evaluation_ranges = None
        n_classes = 2
        iou_metrics = {}
        panoptic_metrics = {}
    
    # Plan eval init. KL computes planning metrics from a compact per-sample
    # payload in the dataset, because its ego footprint/grid differ from
    # camera UniAD's PlanningMetric.
    eval_planning =  hasattr(model_to_eval, 'with_planning_head') \
                      and model_to_eval.with_planning_head
    dataset = data_loader.dataset
    keep_planning_payload = eval_planning and _dataset_wants_planning_payload(
        dataset)
    use_streaming_planning_metric = eval_planning and not keep_planning_payload
    if use_streaming_planning_metric:
        planning_metrics = PlanningMetric(conf={
            'xbound': [-12.5, 12.5, 0.5],
            'ybound': [-12.5, 12.5, 0.5],
            'zbound': [-10.0, 10.0, 20.0],
        }).cuda()
        
    bbox_results = []
    mask_results = []
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    plot_mode = os.environ.get('ENABLE_PLOT_MODE', None) is not None
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            forward_model, data = _scatter_data_for_eval(model, data)
            result = forward_model(return_loss=False, rescale=True, **data)

            # EVAL planning
            if eval_planning:
                # TODO: Wrap below into a func
                segmentation = result[0]['planning']['planning_gt']['segmentation']
                sdc_planning = result[0]['planning']['planning_gt']['sdc_planning']
                sdc_planning_mask = result[0]['planning']['planning_gt']['sdc_planning_mask']
                pred_sdc_traj = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj'] = result[0]['planning']['result_planning']['sdc_traj']
                result[0]['planning_traj_gt'] = result[0]['planning']['planning_gt']['sdc_planning']
                result[0]['command'] = result[0]['planning']['planning_gt']['command']
                if use_streaming_planning_metric:
                    pred_metric = _planning_traj_btc(pred_sdc_traj)
                    gt_metric = _planning_traj_btc(sdc_planning)
                    mask_metric = _planning_mask_btc(sdc_planning_mask)
                    seg_metric = _planning_seg_bthw(
                        segmentation, batch_size=pred_metric.shape[0])
                    planning_metrics(pred_metric, gt_metric, mask_metric,
                                     seg_metric)

            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0].keys():
                    if evaluation_ranges is None:
                        evaluation_ranges = _build_occ_eval_ranges(
                            dataset, result[0]['occ']['seg_out'])
                        metric_device = result[0]['occ']['seg_out'].device
                        for key in evaluation_ranges:
                            iou_metrics[key] = IntersectionOverUnion(
                                n_classes).to(metric_device)
                            panoptic_metrics[key] = PanopticMetric(
                                n_classes=n_classes,
                                temporally_consistent=True).to(metric_device)
                    num_occ += 1
                    for key, (y_limits, x_limits) in evaluation_ranges.items():
                        iou_metrics[key](
                            result[0]['occ']['seg_out'][
                                ..., y_limits, x_limits].contiguous(),
                            result[0]['occ']['seg_gt'][
                                ..., y_limits, x_limits].contiguous())
                        panoptic_metrics[key](
                            result[0]['occ']['ins_seg_out'][
                                ..., y_limits, x_limits].contiguous().detach(),
                            result[0]['occ']['ins_seg_gt'][
                                ..., y_limits, x_limits].contiguous())

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            result_items = [result] if isinstance(result, dict) else result
            if not plot_mode:
                for item in result_items:
                    if not isinstance(item, dict):
                        continue
                    _strip_eval_intermediates(
                        item, keep_planning=keep_planning_payload)
            else:
                for item in result_items:
                    if not isinstance(item, dict):
                        continue
                    if 'occ' in item:
                        item['occ'] = _detach_to_cpu(item['occ'])
                    if 'planning' in item:
                        item['planning'] = _detach_to_cpu(item['planning'])
                    if 'map' in item:
                        item['map'] = _detach_to_cpu(item['map'])
                    if 'pts_bbox' in item:
                        item['pts_bbox'] = _detach_to_cpu(item['pts_bbox'])

            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if use_streaming_planning_metric:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    # Pure BEVFormer detection datasets expect a plain list of bbox results.
    # UniAD/e2e datasets need the dict below for mask/occ/planning metrics.
    if not eval_occ and not eval_planning and mask_results is None:
        return bbox_results

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key in iou_metrics:
            panoptic_scores = panoptic_metrics[key].compute()
            for panoptic_key, value in panoptic_scores.items():
                occ_results[f'{panoptic_key}'] = occ_results.get(f'{panoptic_key}', []) + [100 * value[1].item()]
            panoptic_metrics[key].reset()

            iou_scores = iou_metrics[key].compute()
            occ_results['iou'] = occ_results.get('iou', []) + [100 * iou_scores[1].item()]
            iou_metrics[key].reset()

        occ_results['num_occ'] = num_occ  # count on one gpu
        occ_results['ratio_occ'] = num_occ / len(dataset)  # count on one gpu, but reflect the relative ratio
        ret_results['occ_results_computed'] = occ_results
    if use_streaming_planning_metric:
        ret_results['planning_results_computed'] = planning_results

    if mask_results is not None:
        ret_results['mask_results'] = mask_results
    return ret_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)
