#!/usr/bin/env python3
"""Generate hard-GT VLM/LLM QA pairs for KL frames.

This is the first, deterministic QA layer: it uses only structured KL ground
truth already present in the pkl, mainly info['geo_facts']. No images or VLM
teacher are involved, so every generated answer should be treated as hard
geometry/trajectory truth.

Output fields per frame:
  info['vlm_qa'] = [...]
  info['geo_facts']['qa'] = same list

The schema is intentionally simple and close to DriveLM/NuScenes-QA style:
  {
    "id": "...",
    "question": "...",
    "answer": "...",
    "answer_type": "count|spatial|motion|risk|advice|activity_gate|summary",
    "gt_source": "geo_facts|3d_box|future_traj|rule",
    "target_ids": [track_id, ...],
  }
"""

import argparse
import os.path as osp
import pickle
from collections import Counter


CLS_ZH = [
    '行人', '小车', '满载IGV', '卡车', '空挂车',
    '满载挂车', '空载IGV', '正面吊', '其他车辆', '锥桶',
    '集装箱叉车', '叉车', '轮胎吊',
]

BEARING_ZH = {
    'front': '正前方',
    'left-front': '左前方',
    'left': '左侧',
    'left-rear': '左后方',
    'rear': '正后方',
    'right-rear': '右后方',
    'right': '右侧',
    'right-front': '右前方',
}

MOTION_ZH = {
    'static': '静止',
    'moving_slow': '缓行',
    'moving': '行驶',
    'unknown': '未知',
}

HEADING_ZH = {
    'straight': '直行',
    'turning_left': '左转',
    'turning_right': '右转',
    'static': '静止',
    'unknown': '',
}

ADVICE_ZH = {
    'keep': '保持',
    'slow': '减速',
    'yield': '让行',
    'stop': '停车',
}

LOAD_ZH = {
    'loaded': '满载',
    'empty': '空载',
    'NA': '',
}

LOAD_IN_NAME = {2, 4, 5, 6}
HANDLER_IDS = {7, 10, 11, 12}
COUNT_CLASS_IDS = [2, 3, 4, 5, 6, 7, 10, 11, 12]
BEARING_ORDER = [
    'front', 'left-front', 'right-front', 'left', 'right',
    'rear', 'left-rear', 'right-rear',
]


def _infos(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    return data


def _cls_name(cls_id):
    return CLS_ZH[cls_id] if 0 <= int(cls_id) < len(CLS_ZH) else f'类别{cls_id}'


def _bearing_name(bearing):
    return BEARING_ZH.get(bearing, str(bearing))


def _unit(cls_id):
    cls_id = int(cls_id)
    if cls_id == 0:
        return '一名'
    if cls_id == 9:
        return '一个'
    return '一台'


def _motion_name(motion):
    return MOTION_ZH.get(motion, str(motion))


def _advice_name(advice):
    return ADVICE_ZH.get(advice, str(advice))


def _load_prefix(agent):
    cls_id = int(agent.get('cls', -1))
    if cls_id in LOAD_IN_NAME:
        return ''
    return LOAD_ZH.get(agent.get('load', 'NA'), '')


def _motion_clause(agent):
    return f"当前{_motion_name(agent.get('motion', 'unknown'))}"


def _agent_phrase(agent, with_motion=False, with_heading=False):
    cls_id = int(agent.get('cls', -1))
    phrase = (f"{_bearing_name(agent.get('bearing'))}"
              f"{float(agent.get('range', 0.0)):.1f}米处"
              f"{_unit(cls_id)}{_load_prefix(agent)}{_cls_name(cls_id)}")
    if with_motion:
        phrase += f"，{_motion_clause(agent)}"
        if with_heading and agent.get('motion') != 'static':
            heading = HEADING_ZH.get(agent.get('heading'), '')
            if heading and heading != '静止':
                phrase += f"、{heading}"
    return phrase


def _qid(token, idx):
    token = str(token or 'frame')
    return f'{token}:{idx:03d}'


def _qa(token, idx, question, answer, answer_type, gt_source,
        target_ids=None):
    return {
        'id': _qid(token, idx),
        'question': question,
        'answer': answer,
        'answer_type': answer_type,
        'gt_source': gt_source,
        'target_ids': [int(x) for x in (target_ids or [])],
    }


def _sorted_agents(facts):
    agents = list(facts.get('agents') or [])
    return sorted(agents, key=lambda a: float(a.get('range', 1e9)))


def _nearest_agent(agents, pred=lambda a: True):
    for agent in agents:
        if pred(agent):
            return agent
    return None


def _agents_by_bearing(agents, bearing):
    return [a for a in agents if a.get('bearing') == bearing]


def _count_answer(n, cls_name, bearing=None):
    prefix = _bearing_name(bearing) if bearing else '本车周围'
    return f'{prefix}共有{n}个{cls_name}。'


def _make_count_qas(token, start_idx, agents):
    qas = []
    counts = Counter(int(a.get('cls', -1)) for a in agents)
    idx = start_idx
    for cls_id in COUNT_CLASS_IDS:
        n = counts.get(cls_id, 0)
        if n <= 0:
            continue
        name = _cls_name(cls_id)
        qas.append(_qa(
            token, idx,
            f'本车周围有几个{name}？',
            _count_answer(n, name),
            'count', '3d_box',
            [a.get('id') for a in agents if int(a.get('cls', -1)) == cls_id]))
        idx += 1
    return qas, idx


def _make_spatial_qas(token, start_idx, agents):
    qas = []
    idx = start_idx
    nearest = _nearest_agent(agents)
    if nearest is not None:
        qas.append(_qa(
            token, idx,
            '距离本车最近的目标是什么？',
            _agent_phrase(nearest, with_motion=True),
            'spatial', '3d_box', [nearest.get('id')]))
        idx += 1

    for bearing in BEARING_ORDER:
        cand = _agents_by_bearing(agents, bearing)
        if not cand:
            continue
        nearest_b = cand[0]
        qas.append(_qa(
            token, idx,
            f'本车{_bearing_name(bearing)}最近的目标是什么？',
            _agent_phrase(nearest_b, with_motion=True),
            'spatial', '3d_box', [nearest_b.get('id')]))
        idx += 1
    return qas, idx


def _make_motion_qas(token, start_idx, facts, agents, max_items=3):
    qas = []
    idx = start_idx
    aoi = set(facts.get('agents_of_interest') or [])
    chosen = [a for a in agents if a.get('id') in aoi]
    chosen = chosen[:max_items]
    for agent in chosen:
        qas.append(_qa(
            token, idx,
            f"{_agent_phrase(agent)}的运动状态是什么？",
            f"{_agent_phrase(agent)}{_motion_clause(agent)}。",
            'motion', 'geo_facts', [agent.get('id')]))
        idx += 1
    return qas, idx


def _make_risk_qas(token, start_idx, facts, agents):
    qas = []
    idx = start_idx
    conflicts = [a for a in agents if a.get('conflict')]
    conflicts.sort(key=lambda a: (
        a.get('ttc') is None,
        float(a.get('ttc') or 1e9),
        float(a.get('range', 1e9))))
    if conflicts:
        first = conflicts[0]
        ttc = first.get('ttc')
        ttc_text = f'，预计{float(ttc):.1f}秒后冲突' if ttc is not None else ''
        qas.append(_qa(
            token, idx,
            '是否有目标会与本车未来路径冲突？',
            f"有，{_agent_phrase(first)}{_motion_clause(first)}，"
            f"会与本车未来路径冲突{ttc_text}。",
            'risk', 'future_traj', [first.get('id')]))
        idx += 1
        if len(conflicts) > 1:
            ids = [a.get('id') for a in conflicts]
            desc = '；'.join(
                f"{_agent_phrase(a)}{_motion_clause(a)}"
                + (f"，约{float(a.get('ttc')):.1f}秒后冲突"
                   if a.get('ttc') is not None else '')
                for a in conflicts[:3])
            qas.append(_qa(
                token, idx,
                '哪些目标与本车未来路径存在冲突？',
                desc + '。',
                'risk', 'future_traj', ids[:3]))
            idx += 1
    else:
        qas.append(_qa(
            token, idx,
            '是否有目标会与本车未来路径冲突？',
            '没有发现会与本车未来路径冲突的目标。',
            'risk', 'future_traj', []))
        idx += 1

    advice = facts.get('ego_advice', 'keep')
    qas.append(_qa(
        token, idx,
        '根据当前几何关系，本车应该怎么做？',
        f"本车建议{_advice_name(advice)}。",
        'advice', 'rule', []))
    idx += 1
    return qas, idx


def _make_activity_gate_qas(token, start_idx, agents):
    qas = []
    idx = start_idx
    gated = [a for a in agents if a.get('activity_gate')]
    if not gated:
        qas.append(_qa(
            token, idx,
            '当前是否有需要视觉确认作业状态的设备？',
            '没有需要视觉确认作业状态的设备。',
            'activity_gate', 'geo_facts', []))
        return qas, idx + 1

    ids = [a.get('id') for a in gated]
    desc = '；'.join(_agent_phrase(a) for a in gated[:5])
    qas.append(_qa(
        token, idx,
        '当前哪些设备需要结合图像确认作业状态？',
        f'{desc}需要结合图像确认是否正在装卸、等待或空闲。',
        'activity_gate', 'geo_facts', ids[:5]))
    idx += 1

    for agent in gated[:3]:
        cls_id = int(agent.get('cls', -1))
        if cls_id in HANDLER_IDS:
            answer = (f"{_agent_phrase(agent)}满足几何门控条件，"
                      "需要结合对应相机图像判断作业状态。")
        else:
            answer = (f"{_agent_phrase(agent)}满足几何门控条件，"
                      "但是否作业仍需要图像确认。")
        qas.append(_qa(
            token, idx,
            f"{_agent_phrase(agent)}是否正在作业？",
            answer,
            'activity_gate', 'geo_facts', [agent.get('id')]))
        idx += 1
    return qas, idx


def _make_summary_qa(token, start_idx, facts, agents):
    conflicts = [a for a in agents if a.get('conflict')]
    target_ids = []
    if conflicts:
        conflicts.sort(key=lambda a: (
            a.get('ttc') is None,
            float(a.get('ttc') or 1e9)))
        focus = conflicts[0]
        target_ids = [focus.get('id')]
        ttc = focus.get('ttc')
        ttc_text = f'，预计{float(ttc):.1f}秒后冲突' if ttc is not None else ''
        answer = (f"{_agent_phrase(focus)}{_motion_clause(focus)}，"
                  f"会与本车未来路径冲突{ttc_text}，"
                  f"建议{_advice_name(facts.get('ego_advice', 'slow'))}。")
    elif facts.get('congestion') == 'queue':
        answer = '附近有车辆缓行排队，本车建议减速观察。'
    else:
        nearest = _nearest_agent(agents)
        if nearest is None:
            answer = f"当前未发现有效目标，本车建议{_advice_name(facts.get('ego_advice', 'keep'))}。"
        else:
            target_ids = [nearest.get('id')]
            answer = (f"{_agent_phrase(nearest, with_motion=True)}，"
                      f"本车建议{_advice_name(facts.get('ego_advice', 'keep'))}。")
    return [_qa(
        token, start_idx,
        '请用一句话概括当前本车周围最重要的几何风险。',
        answer,
        'summary', 'geo_facts', target_ids)]


def make_frame_qa(info, max_qas=24):
    facts = info.get('geo_facts') or {}
    if not facts:
        return []

    token = info.get('token') or info.get('sample_idx') or ''
    agents = _sorted_agents(facts)
    idx = 0
    qas = []

    summary_qas = _make_summary_qa(token, idx, facts, agents)
    idx += len(summary_qas)

    builders = [
        _make_risk_qas,
        _make_activity_gate_qas,
        _make_spatial_qas,
        _make_motion_qas,
        _make_count_qas,
    ]
    for builder in builders:
        new_qas, idx = builder(token, idx, facts, agents) \
            if builder is _make_motion_qas or builder is _make_risk_qas \
            else builder(token, idx, agents)
        qas.extend(new_qas)
        if len(qas) >= max_qas - len(summary_qas):
            return (summary_qas + qas)[:max_qas]

    return (summary_qas + qas)[:max_qas]


def generate(pkl_path, out_path=None, in_place=False, limit=None, max_qas=24,
             clear_existing=False):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos = _infos(data)

    type_counts = Counter()
    n_frames = 0
    n_with_qa = 0
    n_qas = 0
    for info in infos:
        if limit is not None and n_frames >= limit:
            break
        n_frames += 1
        if clear_existing:
            info.pop('vlm_qa', None)
            facts = info.get('geo_facts')
            if isinstance(facts, dict):
                facts.pop('qa', None)
        qas = make_frame_qa(info, max_qas=max_qas)
        if not qas:
            continue
        info['vlm_qa'] = qas
        info.setdefault('geo_facts', {})['qa'] = qas
        n_with_qa += 1
        n_qas += len(qas)
        type_counts.update(q['answer_type'] for q in qas)

    if limit is not None and not in_place:
        # Keep quick trial outputs small and honest: limit means output only the
        # processed prefix, not a full pkl where the tail has no QA.
        if isinstance(data, dict) and 'data_list' in data:
            data = dict(data)
            data['data_list'] = infos[:limit]
        elif isinstance(data, dict) and 'infos' in data:
            data = dict(data)
            data['infos'] = infos[:limit]
        else:
            data = infos[:limit]

    if in_place:
        dst = pkl_path
    elif out_path is not None:
        dst = out_path
    else:
        root, ext = osp.splitext(pkl_path)
        dst = f'{root}_vlmqa{ext}'

    with open(dst, 'wb') as f:
        pickle.dump(data, f)

    print(f'[{osp.basename(pkl_path)}] frames_seen={n_frames} '
          f'frames_with_qa={n_with_qa} qas={n_qas}')
    print('answer_type:', dict(sorted(type_counts.items())))
    print(f'  -> wrote {dst}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate deterministic hard-GT QA pairs for KL pkl files.')
    parser.add_argument('--pkl-path', required=True)
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only process/write the first N frames for quick checks.')
    parser.add_argument('--max-qas', type=int, default=24,
                        help='Maximum QA pairs per frame.')
    parser.add_argument('--clear-existing', action='store_true',
                        help='Remove existing vlm_qa/geo_facts.qa before writing.')
    args = parser.parse_args()
    generate(args.pkl_path, args.out_path, args.in_place, args.limit,
             args.max_qas, args.clear_existing)


if __name__ == '__main__':
    main()
