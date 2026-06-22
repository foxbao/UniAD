#!/usr/bin/env python
"""C2: Qwen2.5-VL teacher — turn C1 geometric facts + front image into a
natural Chinese scene summary, written back into the pkl.

See documents/llm_integration_plan.md (3.2). Pipeline per frame:
  c1_facts (geometry, authoritative) + CAM_FRONT image
    -> render facts to a Chinese constraint prompt
    -> Qwen2.5-VL: keep the geometry, ADD only image-only semantics
       (loading/unloading activity, crane busy/idle, load state check),
       output ONE fluent Chinese summary sentence
    -> info['c1_facts']['summary']

This is offline data generation: run it in the `qwen_vl` conda env (torch 2.6
+ transformers 4.57.6), NOT in uniad_train. It only reads images + the C1 pkl
and writes the summary field back; the model is never part of inference.

Runs on a single GPU (7B fits in 24G). Use --device cuda:0 to pin.
"""

import argparse
import json
import os.path as osp
import pickle

from tqdm import tqdm

# 13 trained classes (post label_mapping; see base_track_lidar.py), Chinese.
_CLS_ZH = [
    '行人', '小车', '满载IGV', '卡车', '空挂车',
    '满载挂车', '空载IGV', '吊机', '其他车辆', '锥桶',
    '集装箱叉车', '叉车', '轮胎吊',
]
_MOTION_ZH = {'static': '静止', 'moving_slow': '缓行', 'moving': '行驶',
              'unknown': '未知'}
_HEADING_ZH = {'straight': '直行', 'turning_left': '左转',
               'turning_right': '右转', 'static': '静止', 'unknown': ''}
_LOAD_ZH = {'loaded': '满载', 'empty': '空载', 'NA': ''}
_ADVICE_ZH = {'keep': '保持', 'slow': '减速', 'yield': '让行', 'stop': '停车'}
_BEARING_ZH = {
    'front': '正前', 'left-front': '左前', 'left': '左侧', 'left-rear': '左后',
    'rear': '正后', 'right-rear': '右后', 'right': '右侧', 'right-front': '右前',
}
# Classes whose Chinese name already encodes load state -> don't prefix _LOAD_ZH.
_LOAD_IN_NAME = {2, 4, 5, 6}  # 满载IGV/空挂车/满载挂车/空载IGV


def _cls_name(cid):
    return _CLS_ZH[cid] if 0 <= cid < len(_CLS_ZH) else f'类别{cid}'


_SYSTEM = (
    '你是港口自动驾驶场景的标注助手。下面给你一张本车前视相机图像，'
    '以及一份由激光雷达几何计算得到的、准确无误的场景事实。'
    '请严格遵守：\n'
    '1. 事实中的数量、类别、方位、距离、运动、冲突均为准确数据，不得改动或编造；\n'
    '2. 你的任务是结合图像，补充“几何无法判断、但图像能看出”的语义，仅限：'
    '目标是否正在装卸作业、吊机是忙碌还是空闲、满载/空载的目视确认；\n'
    '3. 对图像看不清或不确定的，不要猜测；\n'
    '4. 若事实中标注了“与本车规划路径冲突”的目标，summary 必须明确点出'
    '是哪个目标（方位+类别）、以及预计多少秒后冲突，并给出建议（减速/让行）。'
    '严禁用“障碍物”等笼统说法代替具体目标；\n'
    '5. 最终只输出一句通顺的中文场景描述（不超过60字），面向本车视角，'
    '突出对本车行驶最相关的目标与建议。不要分点，不要复述全部目标。\n'
    '示例（含冲突）：右前方12米处一台满载IGV缓行，预计3秒后与本车路径冲突，建议让行。\n'
    '示例（含作业）：正后方吊机正在装卸作业，右前方空挂车静止，本车可保持。'
)


def _render_facts(facts):
    """C1 facts dict -> Chinese constraint text fed alongside the image."""
    lines = [f'场景共有 {facts.get("n_agents", 0)} 个目标。']
    aoi = set(facts.get('agents_of_interest', []))
    for a in facts.get('agents', []):
        if a['id'] not in aoi:
            continue
        load_pfx = '' if a['cls'] in _LOAD_IN_NAME else _LOAD_ZH.get(a['load'], '')
        bearing = _BEARING_ZH.get(a['bearing'], a['bearing'])
        seg = [f'{bearing}方 {a["range"]}米处一台{load_pfx}'
               f'{_cls_name(a["cls"])}，{_MOTION_ZH.get(a["motion"], "")}']
        h = _HEADING_ZH.get(a['heading'], '')
        if h and a['motion'] != 'static':
            seg.append(h)
        if a['conflict']:
            ttc = a['ttc']
            seg.append(f'约{ttc}秒后与本车规划路径冲突' if ttc else '与本车路径冲突')
        if a['activity_gate']:
            seg.append('（几何上可能在作业，请结合图像确认）')
        lines.append('- ' + '，'.join(seg) + '。')
    advice = _ADVICE_ZH.get(facts.get('ego_advice', 'keep'), '保持')
    lines.append(f'本车建议：{advice}。')
    if facts.get('congestion') == 'queue':
        lines.append('附近有车辆缓行排队。')
    cranes = facts.get('crane_status', [])
    if cranes:
        lines.append(f'场景中有 {len(cranes)} 台吊机（忙闲请结合图像判断）。')
    return '\n'.join(lines)


def _load_model(model_path, device):
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
        device_map={'': device})
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def _infer_one(model, processor, img_path, facts_text, max_new_tokens=96):
    from qwen_vl_utils import process_vision_info
    messages = [
        {'role': 'system', 'content': _SYSTEM},
        {'role': 'user', 'content': [
            {'type': 'image', 'image': img_path},
            {'type': 'text', 'text': '场景事实：\n' + facts_text},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    images, videos = process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos,
                       padding=True, return_tensors='pt').to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=True)[0].strip()


def _resolve_img(info, data_root):
    cams = info.get('sync_info', {}).get('cameras', {})
    front = cams.get('CAM_FRONT')
    if not front or not front.get('valid') or 'path' not in front:
        return None
    p = front['path']
    return p if osp.isabs(p) or osp.exists(p) else osp.join(data_root, p)


def gen_c2_to_pkl(pkl_path, model_path, out_path=None, in_place=False,
                  data_root='', device='cuda:0', limit=0,
                  max_new_tokens=96, dry_run=False,
                  num_shards=1, shard_id=0):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = (data['data_list'] if isinstance(data, dict) and 'data_list' in data
             else data['infos'] if isinstance(data, dict) and 'infos' in data
             else data)
    # Data-parallel sharding: strided split keeps shards balanced and
    # non-overlapping. Each shard writes its own JSON sidecar; merge_summaries
    # can ingest several sidecars (they are disjoint token->summary maps).
    if num_shards > 1:
        infos = infos[shard_id::num_shards]
    model = processor = None
    if not dry_run:
        model, processor = _load_model(model_path, device)

    n_done = n_skip = 0
    summaries = {}  # token -> summary, numpy-version-safe sidecar
    todo = infos if not limit else infos[:limit]
    for info in tqdm(todo, desc=osp.basename(pkl_path)):
        facts = info.get('c1_facts')
        if facts is None:
            n_skip += 1
            continue
        img = _resolve_img(info, data_root)
        facts_text = _render_facts(facts)
        if dry_run:
            print('\n--- IMG:', img, '---')
            print(facts_text)
            n_done += 1
            continue
        token = info.get('token')
        if img is None:
            facts['summary'] = None
            summaries[token] = None
            n_skip += 1
            continue
        facts['summary'] = _infer_one(
            model, processor, img, facts_text, max_new_tokens)
        summaries[token] = facts['summary']
        n_done += 1

    print(f'[{osp.basename(pkl_path)}] summaries={n_done} skipped={n_skip}'
          + (f' (shard {shard_id}/{num_shards})' if num_shards > 1 else ''))
    if not dry_run:
        _write(data, pkl_path, out_path, in_place, summaries,
               num_shards=num_shards, shard_id=shard_id)
    return data


def _write(data, pkl_path, out_path, in_place, summaries,
           num_shards=1, shard_id=0):
    """Write a token->summary JSON sidecar (+ the pkl when not sharding).

    The JSON sidecar is the numpy-version-safe handoff: C2 runs under numpy
    2.x (qwen_vl env) but training reads under numpy 1.x (uniad_train), and a
    full pkl re-pickle would re-serialize numpy arrays into the 2.x format
    that 1.x cannot load. The merge step reads only this JSON.

    In shard mode each process holds only its slice of infos, so dumping the
    pkl would be wrong/wasteful -- we write a shard-suffixed JSON only and let
    merge_summaries ingest all shards.
    """
    if in_place:
        dst = pkl_path
    elif out_path is not None:
        dst = out_path
    else:
        root, ext = osp.splitext(pkl_path)
        dst = f'{root}_c2{ext}'
    json_base = osp.splitext(dst)[0] + '_summaries'
    if num_shards > 1:
        json_dst = f'{json_base}.shard{shard_id}of{num_shards}.json'
        with open(json_dst, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=1)
        print(f'  -> wrote {json_dst} ({len(summaries)} entries)')
        return
    with open(dst, 'wb') as f:
        pickle.dump(data, f)
    json_dst = json_base + '.json'
    with open(json_dst, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    print(f'  -> wrote {dst}')
    print(f'  -> wrote {json_dst} ({len(summaries)} entries)')


def main():
    parser = argparse.ArgumentParser(
        description='C2: Qwen2.5-VL scene summaries for KL pkl files.')
    parser.add_argument('--pkl-path', nargs='+', required=True)
    parser.add_argument('--model-path',
                        default='/mnt/disk1/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--data-root', default='',
                        help='Prefix for relative image paths (usually cwd).')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only process first N frames (0=all).')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--dry-run', action='store_true',
                        help='Render prompts only; do not load the model.')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Data-parallel shards (one process/GPU each).')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='This process shard index in [0, num_shards).')
    args = parser.parse_args()
    if args.out_path is not None and len(args.pkl_path) != 1:
        raise ValueError('--out-path can only be used with one --pkl-path.')
    for pkl_path in args.pkl_path:
        gen_c2_to_pkl(
            pkl_path, args.model_path, out_path=args.out_path,
            in_place=args.in_place, data_root=args.data_root,
            device=args.device, limit=args.limit,
            max_new_tokens=args.max_new_tokens, dry_run=args.dry_run,
            num_shards=args.num_shards, shard_id=args.shard_id)


if __name__ == '__main__':
    main()



