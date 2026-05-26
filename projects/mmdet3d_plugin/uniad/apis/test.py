import os
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
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


def _get_occ_eval_slices(key, occ_tensor):
    height, width = occ_tensor.shape[-2:]
    if key == '100x100':
        return slice(0, height), slice(0, width)

    if height == 200 and width == 200:
        # Original UniAD nuScenes occ grid: 0.5m resolution, 30m center crop.
        return slice(70, 130), slice(70, 130)
    if height == 120 and width == 200:
        # KL LiDAR occ grid: 0.8m resolution, approximately 30.4m center crop.
        return slice(41, 79), slice(81, 119)

    crop_h = min(height, 60)
    crop_w = min(width, 60)
    y0 = max((height - crop_h) // 2, 0)
    x0 = max((width - crop_w) // 2, 0)
    return slice(y0, y0 + crop_h), slice(x0, x0 + crop_w)


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
    model_to_eval = model.module if hasattr(model, 'module') else model
    eval_occ = hasattr(model_to_eval, 'with_occ_head') \
                and model_to_eval.with_occ_head
    if eval_occ:
        EVALUATION_RANGE_NAMES = ('30x30', '100x100')
        n_classes = 2
        iou_metrics = {}
        for key in EVALUATION_RANGE_NAMES:
            iou_metrics[key] = IntersectionOverUnion(n_classes).cuda()
        panoptic_metrics = {}
        for key in EVALUATION_RANGE_NAMES:
            panoptic_metrics[key] = PanopticMetric(n_classes=n_classes, temporally_consistent=True).cuda()
    
    # Plan eval init
    eval_planning =  hasattr(model.module, 'with_planning_head') \
                      and model.module.with_planning_head
    if eval_planning:
        planning_metrics = PlanningMetric().cuda()
        
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    num_occ = 0
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)

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
                planning_metrics(pred_sdc_traj[:, :6, :2], sdc_planning[0][0,:, :6, :2], sdc_planning_mask[0][0,:, :6, :2], segmentation[0][:, [1,2,3,4,5,6]])

            # Eval Occ
            if eval_occ:
                occ_has_invalid_frame = data['gt_occ_has_invalid_frame'][0]
                occ_to_eval = not occ_has_invalid_frame.item()
                if occ_to_eval and 'occ' in result[0]:
                    occ_result = result[0]['occ']
                    num_occ += 1
                    for key in EVALUATION_RANGE_NAMES:
                        y_limits, x_limits = _get_occ_eval_slices(
                            key, occ_result['seg_out'])
                        iou_metrics[key](
                            occ_result['seg_out'][..., y_limits,
                                                  x_limits].contiguous(),
                            occ_result['seg_gt'][..., y_limits,
                                                 x_limits].contiguous())
                        panoptic_metrics[key](
                            occ_result['ins_seg_out'][..., y_limits,
                                                      x_limits].contiguous(
                                                      ).detach(),
                            occ_result['ins_seg_gt'][..., y_limits,
                                                     x_limits].contiguous())

            # Pop out unnecessary occ results, avoid appending it to cpu when collect_results_cpu
            if os.environ.get('ENABLE_PLOT_MODE', None) is None:
                result[0].pop('occ', None)
                result[0].pop('planning', None)
            else:
                if 'occ' in result[0]:
                    occ_result = result[0]['occ']
                    for k in ['seg_gt', 'ins_seg_gt', 'pred_ins_sigmoid',
                              'seg_out', 'ins_seg_out']:
                        if k in occ_result:
                            occ_result[k] = occ_result[k].detach().cpu()
                for k in ['bbox', 'segm', 'labels', 'panoptic', 'drivable', 'score_list', 'lane', 'lane_score', 'stuff_score_list']:
                    if k in result[0]['pts_bbox'] and isinstance(result[0]['pts_bbox'][k], torch.Tensor):
                        result[0]['pts_bbox'][k] = result[0]['pts_bbox'][k].detach().cpu()

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

    if eval_planning:
        planning_results = planning_metrics.compute()
        planning_metrics.reset()

    ret_results = dict()
    ret_results['bbox_results'] = bbox_results
    if eval_occ:
        occ_results = {}
        for key in EVALUATION_RANGE_NAMES:
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
    if eval_planning:
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
