#!/usr/bin/env python
"""VLM caption teacher — turn geometric facts + surround-camera
images into a natural Chinese scene summary, written back into the pkl.

See documents/llm_integration_plan.md (3.2). Pipeline per frame:
  geo_facts (geometry, authoritative) + surround-camera images
    -> _resolve_views routes only the cameras covering this frame's relevant
       agents (AOI/conflict/gate) to the VLM (cost ~2x, not 6x)
    -> render facts to a Chinese constraint prompt
    -> Qwen-VL: keep the geometry, ADD only image-only semantics
       (loading/unloading activity, crane busy/idle, load state check),
       output ONE fluent Chinese summary sentence
    -> info['geo_facts']['summary']

Each image is labeled with its bearing (【前方相机】/【右后方相机】...) so the VLM
aligns activity/load judgements to the correct direction. Multi-camera fixes the
single-front blind spot: rear/side targets that front-only could not see (and so
were never captioned with an activity state — see plan 4.2/4.3).

This is offline data generation: run it in the `qwen_vl` conda env (torch 2.6
+ transformers 4.57.6), NOT in uniad_train. It only reads images + the geo-facts pkl
and writes the summary field back; the model is never part of inference.

Runs on a single GPU by default (Qwen2.5-VL-7B / Qwen3-VL-8B fit in
24G). Larger teachers can be sharded across the visible GPUs with
--device-map auto.
"""

import argparse
import json
import math
import os.path as osp
import pickle
from pathlib import Path

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
_OPERATION_ZH = {
    'working': '正在装卸作业',
    'waiting': '等待作业',
    'idle': '空闲',
    'unknown': '看不清/无法判断',
}
_MOTION_STATE_ZH = {
    'moving': '移动/行驶中',
    'static': '静止',
    'unknown': '未知',
}
# Classes whose Chinese name already encodes load state -> don't prefix _LOAD_ZH.
_LOAD_IN_NAME = {2, 4, 5, 6}  # 满载IGV/空挂车/满载挂车/空载IGV


def _cls_name(cid):
    return _CLS_ZH[cid] if 0 <= cid < len(_CLS_ZH) else f'类别{cid}'


def _canonical_operation_state(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    aliases = {
        'busy': 'working',
        'work': 'working',
        'working': 'working',
        'loading': 'working',
        'unloading': 'working',
        'loading_unloading': 'working',
        'wait': 'waiting',
        'waiting': 'waiting',
        'idle': 'idle',
        'none': 'idle',
        'unclear': 'unknown',
        'unknown': 'unknown',
    }
    return aliases.get(value, value)


def _canonical_motion_state(value):
    if value is None:
        return None
    value = str(value).strip().lower()
    aliases = {
        'moving': 'moving',
        'move': 'moving',
        'driving': 'moving',
        'static': 'static',
        'stopped': 'static',
        'stop': 'static',
        'idle': 'static',
        'unknown': 'unknown',
        'unclear': 'unknown',
    }
    return aliases.get(value, value)


def _normalise_feedback(raw):
    """Normalise human feedback to token -> track_id(str) -> state dict.

    Preferred schema:
      {token: {track_id: {motion_state, operation_state, note}}}
    Backward-compatible shorthand:
      {token: {track_id: {"activity": "busy"}}}
    """
    if not raw:
        return {}
    if isinstance(raw, dict) and 'frames' in raw and isinstance(raw['frames'], dict):
        raw = raw['frames']
    out = {}
    for token, frame_fb in raw.items():
        if not isinstance(frame_fb, dict):
            continue
        dst = {}
        for track_id, entry in frame_fb.items():
            if not isinstance(entry, dict):
                entry = {'operation_state': entry}
            operation = _canonical_operation_state(
                entry.get('operation_state', entry.get('activity')))
            motion = _canonical_motion_state(entry.get('motion_state'))
            norm = {}
            if motion:
                norm['motion_state'] = motion
            if operation:
                norm['operation_state'] = operation
            if entry.get('note'):
                norm['note'] = str(entry['note'])
            if entry.get('source'):
                norm['source'] = str(entry['source'])
            if norm:
                dst[str(track_id)] = norm
        if dst:
            out[str(token)] = dst
    return out


def _load_feedback(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return _normalise_feedback(json.load(f))


def _feedback_for_info(feedback, info):
    return feedback.get(str(info.get('token')), {}) if feedback else {}


def _render_feedback(feedback_for_frame, facts):
    if not feedback_for_frame:
        return ''
    agents = {str(a['id']): a for a in facts.get('agents', [])}
    lines = [
        '人工确认反馈（优先级高于图像模型自行判断；必须服从）：',
        '注意：运动状态和作业状态是两件事，静止的吊机/叉车仍可能正在作业。',
    ]
    for track_id, fb in sorted(feedback_for_frame.items(), key=lambda kv: kv[0]):
        agent = agents.get(str(track_id), {})
        cls = _cls_name(int(agent.get('cls', -1))) if agent else '目标'
        bearing = _BEARING_ZH.get(agent.get('bearing'), agent.get('bearing', ''))
        dist = agent.get('range')
        label = f'#{track_id} {bearing}方{dist}米处{cls}' if dist is not None \
            else f'#{track_id} {cls}'
        states = []
        motion = fb.get('motion_state')
        operation = fb.get('operation_state')
        if motion:
            states.append(f'运动状态={_MOTION_STATE_ZH.get(motion, motion)}')
        if operation:
            states.append(f'作业状态={_OPERATION_ZH.get(operation, operation)}')
        note = f'；备注：{fb["note"]}' if fb.get('note') else ''
        lines.append(f'- {label}：' + '，'.join(states) + note + '。')
    return '\n'.join(lines)


_SYSTEM = (
    '你是港口自动驾驶场景的标注助手。下面给你若干张本车环视相机图像（每张图前'
    '都标注了它的方位，如【前方相机】【右后方相机】），'
    '以及一份由激光雷达几何计算得到的、准确无误的场景事实。'
    '请严格遵守：\n'
    '1. 事实中的数量、类别、方位、距离、运动、冲突均为准确数据，不得改动或编造；'
    '其中 ego_advice 是几何步骤已经算好的本车建议，必须原样保留为“保持/减速/让行/停车”，'
    '不得根据图像改写；\n'
    '2. 你的任务是结合图像，补充“几何无法判断、但图像能看出”的语义，仅限：'
    '目标是否正在装卸作业、是否等待作业、是否空闲、满载/空载的目视确认；'
    '判断某目标时，请看与其方位一致的那张相机图（如右后方的目标看【右后方相机】）；'
    '若本帧没有有效相机图像，只能依据几何事实输出，不得补充作业/载货等视觉判断；\n'
    '3. 必须区分“运动状态”和“作业状态”：设备本体静止不等于空闲。吊机/轮胎吊静止但吊具、吊臂、'
    '下方车辆或箱体处于装卸上下文时，应判为正在装卸作业；叉车/集装箱叉车静止但叉臂/夹具正在取放货，'
    '也应判为正在作业。对图像看不清或不确定的，不要猜测；\n'
    '4. 若事实中标注了“与本车规划路径冲突”的目标，summary 必须明确点出'
    '是哪个目标（方位+类别）、以及预计多少秒后冲突，并给出建议（减速/让行）。'
    '严禁用“障碍物”等笼统说法代替具体目标；\n'
    '5. 若事实中标注了某目标“（请判断作业状态）”，该目标已在某路相机可见范围内，'
    'summary 必须明确说出它的作业状态，四选一：正在装卸作业 / 等待作业 / 空闲 / 看不清，'
    '依据其方位对应相机的图像观察（吊具是否起落、周围是否有箱、是否有车停靠装卸）。不得回避；\n'
    '6. 若有人类确认反馈，则该反馈优先级高于你对图像的自行判断，summary 必须服从；\n'
    '7. 最终只输出一句通顺的中文场景描述（不超过60字），面向本车视角，'
    '突出对本车行驶最相关的目标与建议。不要分点，不要复述全部目标。\n'
    '示例（含冲突）：右前方12米处一台满载IGV缓行，预计3秒后与本车路径冲突，建议让行。\n'
    '示例（含作业）：右后方吊机正在装卸作业，正前方空挂车等待，本车可保持。'
)


def _render_facts(facts):
    """geo_facts dict -> Chinese constraint text fed alongside the image.

    Renders agents that are AOI OR conflict OR activity_gate -- the SAME set
    _resolve_views routes a camera for. Earlier this rendered AOI-only, so a
    gate target outside AOI got a camera routed (the VLM was shown the image)
    but was never named in the prompt -> the VLM was never asked to judge it.
    That mismatch deflated the activity metric. AOI agents are listed first
    (most ego-relevant), then any gate/conflict-only agents.
    """
    lines = [f'场景共有 {facts.get("n_agents", 0)} 个目标。']
    aoi = set(facts.get('agents_of_interest', []))

    def _relevant(a):
        return a['id'] in aoi or a.get('conflict') or a.get('activity_gate')

    agents = [a for a in facts.get('agents', []) if _relevant(a)]
    # AOI first, then conflict/gate-only; stable within each group.
    agents.sort(key=lambda a: a['id'] not in aoi)
    for a in agents:
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
            seg.append('（请判断作业状态）')
        lines.append('- ' + '，'.join(seg) + '。')
    advice = _ADVICE_ZH.get(facts.get('ego_advice', 'keep'), '保持')
    lines.append(f'本车建议（几何事实，必须照抄）：{advice}。')
    if facts.get('congestion') == 'queue':
        lines.append('附近有车辆缓行排队。')
    cranes = facts.get('crane_status', [])
    if cranes:
        lines.append(f'场景中有 {len(cranes)} 台吊机（忙闲请结合图像判断）。')
    return '\n'.join(lines)


def _parse_max_memory(spec):
    if not spec:
        return None
    import torch
    if ':' not in spec:
        return {i: spec for i in range(torch.cuda.device_count())}
    memory = {}
    for item in spec.split(','):
        if not item:
            continue
        key, value = item.split(':', 1)
        key = key.strip()
        value = value.strip()
        memory[int(key) if key.isdigit() else key] = value
    return memory


def _load_model(model_path, device, device_map=None, max_memory=None):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    if device_map in (None, '', 'single'):
        resolved_device_map = {'': device}
    elif device_map == 'auto':
        resolved_device_map = 'auto'
    else:
        raise ValueError(
            f'Unsupported --device-map {device_map!r}; use single or auto.')
    kwargs = dict(
        dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map=resolved_device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True)
    parsed_max_memory = _parse_max_memory(max_memory)
    if parsed_max_memory is not None:
        kwargs['max_memory'] = parsed_max_memory
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, **kwargs)
    model.eval()
    if hasattr(model, 'hf_device_map'):
        print(f'  -> model device_map: {model.hf_device_map}')
    # Keep image preprocessing stable across transformers releases. In 4.57,
    # Qwen2.5-VL defaults to a fast processor with slightly different output.
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False)
    return model, processor


def _infer_one(model, processor, views, facts_text, max_new_tokens=96,
               annotated=False):
    """views: [(cam, zh_label, path), ...]. Feeds each labeled view so the VLM
    aligns activity/load judgements to the correct bearing."""
    content = []
    for _, zh, path in views:
        content.append({'type': 'text', 'text': f'【{zh}相机】'})
        content.append({'type': 'image', 'image': path})
    if not views:
        content.append({
            'type': 'text',
            'text': '【无有效相机图像】本帧没有可用同步相机；请只依据几何事实生成summary，'
                    '涉及作业状态、载货状态等视觉语义时写看不清或不要提及。'
        })
    if annotated:
        facts_text = (
            '注意：图像中的彩色圆点/文字是由激光雷达目标中心投影得到的辅助标注，'
            '格式为“#目标ID 类别 距离”。请优先结合这些标注定位事实中点名的目标。\n'
            + facts_text)
    content.append({'type': 'text', 'text': '场景事实：\n' + facts_text})
    messages = [
        {'role': 'system', 'content': _SYSTEM},
        {'role': 'user', 'content': content},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    processor_kwargs = dict(text=[text], padding=True, return_tensors='pt')
    if views:
        from qwen_vl_utils import process_vision_info
        images, videos = process_vision_info(messages)
        processor_kwargs.update(images=images, videos=videos)
    inputs = processor(**processor_kwargs).to(model.device)
    import torch
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
    trimmed = out[:, inputs.input_ids.shape[1]:]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=True)[0].strip()


# 8-way bearing sector -> surround camera that best sees it (nuScenes layout,
# confirmed from camera_extrinsics). Routes only cameras covering this frame's
# relevant agents to the VLM (cost ~2x, not 6x).
_BEARING_CAM = {
    'front': 'CAM_FRONT', 'left-front': 'CAM_FRONT_LEFT',
    'left': 'CAM_FRONT_LEFT', 'left-rear': 'CAM_BACK_LEFT',
    'rear': 'CAM_BACK', 'right-rear': 'CAM_BACK_RIGHT',
    'right': 'CAM_FRONT_RIGHT', 'right-front': 'CAM_FRONT_RIGHT',
}
_CAM_ZH = {
    'CAM_FRONT': '前方', 'CAM_FRONT_LEFT': '左前方',
    'CAM_FRONT_RIGHT': '右前方', 'CAM_BACK': '后方',
    'CAM_BACK_LEFT': '左后方', 'CAM_BACK_RIGHT': '右后方',
}
_CAM_FALLBACK_ORDER = (
    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
)
_CAM_TO_DISK = {
    'CAM_FRONT': 'front',
    'CAM_FRONT_LEFT': 'left_front',
    'CAM_BACK_LEFT': 'left_rear',
    'CAM_BACK': 'rear',
    'CAM_FRONT_RIGHT': 'right_front',
    'CAM_BACK_RIGHT': 'right_rear',
}
_CALIB_CACHE = {}


def _cam_path(info, cam, data_root):
    cams = info.get('sync_info', {}).get('cameras', {})
    e = cams.get(cam)
    if not e or not e.get('valid') or 'path' not in e:
        return None
    p = e['path']
    return p if osp.isabs(p) or osp.exists(p) else osp.join(data_root, p)


def _valid_views(info, data_root, cams):
    out = []
    seen = set()
    for c in cams:
        if c in seen:
            continue
        seen.add(c)
        p = _cam_path(info, c, data_root)
        if p is not None:
            out.append((c, _CAM_ZH.get(c, c), p))
    return out


def _quat_to_rot(x, y, z, w):
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if norm <= 0:
        raise ValueError('Invalid zero quaternion.')
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    import numpy as np
    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def _transform_from_quat_list(values):
    import numpy as np
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _quat_to_rot(
        values[3], values[4], values[5], values[6])
    transform[:3, 3] = np.asarray(values[:3], dtype=np.float64)
    return transform


def _scene_root_from_path(path):
    p = Path(path)
    if 'camera_undist' not in p.parts:
        return None
    idx = p.parts.index('camera_undist')
    if idx >= 1:
        return Path(*p.parts[:idx - 1])
    return None


def _scene_root_from_info(info, image_path=None):
    if image_path:
        scene_root = _scene_root_from_path(image_path)
        if scene_root is not None:
            return scene_root
    cams = info.get('sync_info', {}).get('cameras', {})
    for entry in cams.values():
        path = entry.get('path') if isinstance(entry, dict) else None
        if not path:
            continue
        scene_root = _scene_root_from_path(path)
        if scene_root is not None:
            return scene_root
    scene_token = info.get('scene_token')
    if scene_token:
        scene = scene_token.split('/')[0]
        return Path('data/kl_8/v1.0-trainval/sample') / scene
    return None


def _load_calib(info, image_path=None):
    scene_root = _scene_root_from_info(info, image_path)
    if scene_root is None:
        return None
    scene_root = scene_root.resolve()
    if scene_root in _CALIB_CACHE:
        return _CALIB_CACHE[scene_root]
    extr_path = scene_root / 'camera_extrinsics.json'
    intr_path = scene_root / 'intrinsics.json'
    if not extr_path.exists() or not intr_path.exists():
        _CALIB_CACHE[scene_root] = None
        return None
    with open(extr_path, 'r') as f:
        extr = json.load(f)
    with open(intr_path, 'r') as f:
        intr = json.load(f)
    calib = dict(extr=extr, intr=intr)
    _CALIB_CACHE[scene_root] = calib
    return calib


def _image_scale_from_intrinsics(image_size, intr):
    width, height = image_size
    sx = width / max(2.0 * float(intr['cx']), 1.0)
    sy = height / max(2.0 * float(intr['cy']), 1.0)
    return 0.5 * (sx + sy)


def _project_point(point_lidar, cam, image_size, calib):
    disk = _CAM_TO_DISK.get(cam)
    if disk is None:
        return None
    extr_key = f'Tx_baselink_camera_{disk}'
    intr_key = f'camera_{disk}'
    if extr_key not in calib['extr'] or intr_key not in calib['intr']:
        return None
    import numpy as np
    t_cam_to_lidar = _transform_from_quat_list(calib['extr'][extr_key])
    t_lidar_to_cam = np.linalg.inv(t_cam_to_lidar)
    intr = calib['intr'][intr_key]
    scale = _image_scale_from_intrinsics(image_size, intr)
    k = np.array([
        [intr['fx'] * scale, 0.0, intr['cx'] * scale],
        [0.0, intr['fy'] * scale, intr['cy'] * scale],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    point_h = np.ones((4,), dtype=np.float64)
    point_h[:3] = point_lidar
    point_cam = t_lidar_to_cam @ point_h
    if point_cam[2] <= 1e-3:
        return None
    uvw = k @ (point_cam[:3] / point_cam[2])
    u, v = float(uvw[0]), float(uvw[1])
    width, height = image_size
    if not (-40 <= u <= width + 40 and -40 <= v <= height + 40):
        return None
    return u, v


def _relevant_agent_ids(facts):
    aoi = set(facts.get('agents_of_interest', []))
    ids = []
    for agent in facts.get('agents', []):
        if (agent['id'] in aoi or agent.get('conflict') or
                agent.get('activity_gate')):
            ids.append(agent['id'])
    return set(ids)


def _annotate_image(path, cam, facts, info, annotated_dir):
    from PIL import Image, ImageDraw, ImageFont
    image = Image.open(path).convert('RGB')
    calib = _load_calib(info, path)
    if calib is None:
        return path
    inst_by_id = {
        inst.get('track_id'): inst
        for inst in info.get('instances', [])
        if 'track_id' in inst and 'bbox_3d' in inst
    }
    agent_by_id = {agent['id']: agent for agent in facts.get('agents', [])}
    relevant_ids = _relevant_agent_ids(facts)
    if not relevant_ids:
        return path
    try:
        font = ImageFont.truetype(
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc', 22)
        small_font = ImageFont.truetype(
            '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 18)
    except OSError:
        font = small_font = ImageFont.load_default()
    colors = [(255, 48, 48), (0, 160, 255), (255, 180, 0), (0, 190, 90),
              (180, 70, 255), (255, 80, 180)]
    draw = ImageDraw.Draw(image)
    n_drawn = 0
    for idx, track_id in enumerate(sorted(relevant_ids)):
        inst = inst_by_id.get(track_id)
        agent = agent_by_id.get(track_id)
        if inst is None or agent is None:
            continue
        uv = _project_point(inst['bbox_3d'][:3], cam, image.size, calib)
        if uv is None:
            continue
        u, v = uv
        color = colors[idx % len(colors)]
        r = 8
        draw.ellipse((u - r, v - r, u + r, v + r), fill=color,
                     outline=(255, 255, 255), width=2)
        label = f'#{track_id} {_cls_name(agent["cls"])} {agent["range"]}m'
        text_box0 = draw.textbbox((0, 0), label, font=font)
        text_w = text_box0[2] - text_box0[0]
        text_h = text_box0[3] - text_box0[1]
        text_xy = (max(0, min(u + 10, image.width - text_w - 8)),
                   max(0, min(v - text_h - 10, image.height - text_h - 4)))
        box = draw.textbbox(text_xy, label, font=font)
        draw.rectangle((box[0] - 4, box[1] - 2, box[2] + 4, box[3] + 2),
                       fill=(0, 0, 0))
        draw.text(text_xy, label, font=font, fill=color)
        n_drawn += 1
    if n_drawn == 0:
        return path
    draw.rectangle((0, image.height - 30, image.width, image.height),
                   fill=(0, 0, 0))
    draw.text((8, image.height - 27), 'LiDAR投影辅助标注：彩色点为目标中心',
              font=small_font, fill=(255, 255, 255))
    annotated_dir = Path(annotated_dir)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    out_path = annotated_dir / f'{info.get("token", "no_token")}_{cam}.jpg'
    image.save(out_path, quality=92)
    return str(out_path)


def _resolve_views(facts, info, data_root):
    """Cameras covering this frame's relevant agents (AOI/conflict/gate),
    deduped; returns [(cam, zh_label, path), ...] for valid views.

    Some KL frames only have a subset of synced cameras. If the bearing-routed
    cameras are unavailable, fall back to any valid camera instead of dropping
    the frame to a None summary.
    """
    aoi = set(facts.get('agents_of_interest', []))
    cams = []
    for a in facts.get('agents', []):
        if a['id'] in aoi or a.get('conflict') or a.get('activity_gate'):
            c = _BEARING_CAM.get(a['bearing'])
            if c and c not in cams:
                cams.append(c)
    if not cams:
        cams = ['CAM_FRONT']
    out = _valid_views(info, data_root, cams)
    if out:
        return out
    return _valid_views(info, data_root, _CAM_FALLBACK_ORDER)


def _maybe_annotate_views(views, facts, info, annotated_dir=None):
    if not annotated_dir:
        return views
    return [
        (cam, zh, _annotate_image(path, cam, facts, info, annotated_dir))
        for cam, zh, path in views
    ]


def gen_vlm_to_pkl(pkl_path, model_path, out_path=None, in_place=False,
                  data_root='', device='cuda:0', limit=0,
                  max_new_tokens=96, dry_run=False,
                  num_shards=1, shard_id=0, annotated_dir=None,
                  feedback=None, device_map='single', max_memory=None,
                  load_only=False):
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
        model, processor = _load_model(
            model_path, device, device_map=device_map,
            max_memory=max_memory)
        if load_only:
            print('load-only smoke passed.')
            return

    n_done = n_skip = 0
    summaries = {}  # token -> summary, numpy-version-safe sidecar
    summary_meta = {}
    todo = infos if not limit else infos[:limit]
    for info in tqdm(todo, desc=osp.basename(pkl_path)):
        facts = info.get('geo_facts')
        if facts is None:
            n_skip += 1
            continue
        views = _resolve_views(facts, info, data_root)
        views = _maybe_annotate_views(views, facts, info, annotated_dir)
        facts_text = _render_facts(facts)
        token = info.get('token')
        frame_feedback = _feedback_for_info(feedback, info)
        feedback_text = _render_feedback(frame_feedback, facts)
        if feedback_text:
            facts_text = facts_text + '\n\n' + feedback_text
        if dry_run:
            print('\n--- VIEWS:', [v[0] for v in views], '---')
            print(facts_text)
            n_done += 1
            continue
        facts['summary'] = _infer_one(
            model, processor, views, facts_text, max_new_tokens,
            annotated=bool(annotated_dir and views))
        summaries[token] = facts['summary']
        meta = {}
        if frame_feedback:
            meta['human_feedback'] = frame_feedback
        if not views:
            meta['no_valid_camera'] = True
        if meta:
            summary_meta[token] = meta
        n_done += 1

    print(f'[{osp.basename(pkl_path)}] summaries={n_done} skipped={n_skip}'
          + (f' (shard {shard_id}/{num_shards})' if num_shards > 1 else ''))
    if not dry_run:
        _write(data, pkl_path, out_path, in_place, summaries, summary_meta,
               num_shards=num_shards, shard_id=shard_id)
    return data


def _write(data, pkl_path, out_path, in_place, summaries, summary_meta=None,
           num_shards=1, shard_id=0):
    """Write a token->summary JSON sidecar (+ the pkl when not sharding).

    The JSON sidecar is the numpy-version-safe handoff: VLM caption gen runs under numpy
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
        dst = f'{root}_vlmcap{ext}'
    json_base = osp.splitext(dst)[0] + '_summaries'
    meta_base = osp.splitext(dst)[0] + '_summary_meta'
    if num_shards > 1:
        json_dst = f'{json_base}.shard{shard_id}of{num_shards}.json'
        with open(json_dst, 'w', encoding='utf-8') as f:
            json.dump(summaries, f, ensure_ascii=False, indent=1)
        print(f'  -> wrote {json_dst} ({len(summaries)} entries)')
        if summary_meta:
            meta_dst = f'{meta_base}.shard{shard_id}of{num_shards}.json'
            with open(meta_dst, 'w', encoding='utf-8') as f:
                json.dump(summary_meta, f, ensure_ascii=False, indent=1)
            print(f'  -> wrote {meta_dst} ({len(summary_meta)} entries)')
        return
    with open(dst, 'wb') as f:
        pickle.dump(data, f)
    json_dst = json_base + '.json'
    with open(json_dst, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=1)
    print(f'  -> wrote {dst}')
    print(f'  -> wrote {json_dst} ({len(summaries)} entries)')
    if summary_meta:
        meta_dst = meta_base + '.json'
        with open(meta_dst, 'w', encoding='utf-8') as f:
            json.dump(summary_meta, f, ensure_ascii=False, indent=1)
        print(f'  -> wrote {meta_dst} ({len(summary_meta)} entries)')


def main():
    parser = argparse.ArgumentParser(
        description='VLM caption scene summaries for KL pkl files.')
    parser.add_argument('--pkl-path', nargs='+', required=True)
    parser.add_argument('--model-path',
                        default='/mnt/disk1/models/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--data-root', default='',
                        help='Prefix for relative image paths (usually cwd).')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--device-map', default='single',
                        choices=['single', 'auto'],
                        help='single: put the whole model on --device. auto: '
                        'let Transformers/Accelerate shard the model across '
                        'visible GPUs, e.g. CUDA_VISIBLE_DEVICES=0,1,2,3.')
    parser.add_argument('--max-memory', default=None,
                        help='Optional max_memory for --device-map auto. '
                        'Use one value for every visible GPU, e.g. 22GiB, or '
                        'comma items such as 0:22GiB,1:22GiB,cpu:64GiB.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only process first N frames (0=all).')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--dry-run', action='store_true',
                        help='Render prompts only; do not load the model.')
    parser.add_argument('--load-only', action='store_true',
                        help='Load model/processor and exit. Useful for '
                        'multi-GPU memory smoke tests.')
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Data-parallel shards (one process/GPU each).')
    parser.add_argument('--shard-id', type=int, default=0,
                        help='This process shard index in [0, num_shards).')
    parser.add_argument('--annotate-targets', action='store_true',
                        help='Draw LiDAR-projected relevant target points/labels '
                        'on camera images before sending them to the VLM.')
    parser.add_argument('--annotated-dir', default='/tmp/vlmcap_annotated',
                        help='Where --annotate-targets writes temporary images.')
    parser.add_argument('--feedback-json', default=None,
                        help='Optional human feedback JSON: token -> track_id -> '
                        '{motion_state, operation_state, note}. This is injected '
                        'into the teacher prompt with higher priority than the '
                        'VLM image judgement.')
    args = parser.parse_args()
    if args.out_path is not None and len(args.pkl_path) != 1:
        raise ValueError('--out-path can only be used with one --pkl-path.')
    feedback = _load_feedback(args.feedback_json)
    for pkl_path in args.pkl_path:
        gen_vlm_to_pkl(
            pkl_path, args.model_path, out_path=args.out_path,
            in_place=args.in_place, data_root=args.data_root,
            device=args.device, limit=args.limit,
            max_new_tokens=args.max_new_tokens, dry_run=args.dry_run,
            num_shards=args.num_shards, shard_id=args.shard_id,
            annotated_dir=(args.annotated_dir if args.annotate_targets else None),
            feedback=feedback, device_map=args.device_map,
            max_memory=args.max_memory, load_only=args.load_only)


if __name__ == '__main__':
    main()
