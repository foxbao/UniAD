#!/usr/bin/env python
"""Evaluate LLMBridgeHead captions against geometric GT (key-semantic hit
rates + hallucination), with baselines that prove the LiDAR query matters.

Why not BLEU: port captions are highly templated, so BLEU is inflated and
says nothing about whether the model recovered the *scene semantics*. Instead
we parse the generated Chinese caption into structured fields with regex and
score them against geo_facts (objective, deterministic).

Metrics (per frame, then averaged):
  - bearing/class hit of the conflict agent
  - TTC bucket accuracy (<2s / 2-4s / >4s)
  - ego_advice accuracy (keep/slow/yield)
  - hallucination rate: object classes mentioned that are NOT in geo_facts

Baselines (run on the SAME val frames; --mode):
  - model    : the trained LLMBridgeHead (needs a checkpoint)
  - template : deterministic geo_facts -> sentence (rule, the floor)
  - shuffle  : feed frame i's query but score against frame j's GT (the key
               control — if hits don't drop, the LLM ignores the query)
  - noquery  : caption from prompt only (LLM language prior)

This script only needs uniad_train + transformers (for model/shuffle/noquery
modes); template mode needs neither GPU nor transformers.
See documents/llm_integration_plan.md sec 4.3-B.
"""
import argparse
import pickle
import re

# Chinese surface forms -> canonical labels (mirrors gen_vlm_caption._*_ZH).
_BEARING_ZH = {
    '正前': 'front', '左前': 'left-front', '左侧': 'left', '左后': 'left-rear',
    '正后': 'rear', '右后': 'right-rear', '右侧': 'right', '右前': 'right-front',
}
_CLS_ZH = {
    '行人': 0, '小车': 1, '满载IGV': 2, '卡车': 3, '空挂车': 4, '满载挂车': 5,
    '空载IGV': 6, '吊机': 7, '其他车辆': 8, '锥桶': 9, '集装箱叉车': 10,
    '叉车': 11, '轮胎吊': 12,
}
_ADVICE_ZH = [('让行', 'yield'), ('停车', 'stop'), ('减速', 'slow'),
              ('保持', 'keep')]


def parse_caption(text):
    """Regex-parse a Chinese caption into structured fields."""
    if not text:
        return dict(classes=set(), bearings=set(), ttc=None, advice=None,
                    activity=False)
    classes = {cid for name, cid in _CLS_ZH.items() if name in text}
    bearings = {canon for zh, canon in _BEARING_ZH.items() if zh in text}
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*秒', text)
    ttc = float(m.group(1)) if m else None
    advice = next((c for zh, c in _ADVICE_ZH if zh in text), None)
    activity = ('作业' in text) or ('装卸' in text)
    return dict(classes=classes, bearings=bearings, ttc=ttc,
                advice=advice, activity=activity)


def _ttc_bucket(t):
    if t is None:
        return 'none'
    return 'near' if t < 2 else ('mid' if t < 4 else 'far')


def gt_fields(facts):
    """Extract scorable GT from a frame's geo_facts."""
    agents = facts.get('agents', [])
    present = {int(a['cls']) for a in agents}
    conflicts = [a for a in agents if a.get('conflict')]
    conflicts.sort(key=lambda a: (a['ttc'] is None, a['ttc']))
    top = conflicts[0] if conflicts else None
    return dict(
        present_classes=present,
        conflict_cls=int(top['cls']) if top else None,
        conflict_bearing=top['bearing'] if top else None,
        conflict_ttc=top['ttc'] if top else None,
        advice=facts.get('ego_advice'),
        has_activity=any(a.get('activity_gate') for a in agents))


def score_frame(pred, gt):
    """One frame -> dict of per-metric (hit, applicable) tuples.

    applicable=False means the metric doesn't apply to this frame (e.g. no
    conflict), so it's excluded from that metric's denominator.
    """
    s = {}
    # conflict class / bearing — only on frames that actually have a conflict
    if gt['conflict_cls'] is not None:
        s['conflict_cls'] = (gt['conflict_cls'] in pred['classes'], True)
        s['conflict_bearing'] = (gt['conflict_bearing'] in pred['bearings'], True)
        s['ttc_bucket'] = (
            _ttc_bucket(pred['ttc']) == _ttc_bucket(gt['conflict_ttc']), True)
    else:
        s['conflict_cls'] = (False, False)
        s['conflict_bearing'] = (False, False)
        s['ttc_bucket'] = (False, False)
    # advice — always applicable
    s['advice'] = (pred['advice'] == gt['advice'], True)
    # activity — only where geometry gated it on
    s['activity'] = (pred['activity'] == gt['has_activity'],
                     gt['has_activity'])
    # hallucination: predicted classes not present in GT (lower better)
    halluc = pred['classes'] - gt['present_classes']
    s['halluc'] = (len(halluc) == 0, len(pred['classes']) > 0)
    return s


_CLS_NAME = {v: k for k, v in _CLS_ZH.items()}
_BEARING_NAME = {v: k for k, v in _BEARING_ZH.items()}
_ADVICE_NAME = {c: zh for zh, c in _ADVICE_ZH}


def template_caption(facts):
    """Deterministic geo_facts -> Chinese sentence (the 'template' baseline).

    This is the rule-only floor: it nails every geometric field by construction
    but can never add image-only semantics (activity/load). If the trained
    model can't beat this on image-only fields, the VLM distillation added
    nothing.
    """
    agents = facts.get('agents', [])
    aoi = set(facts.get('agents_of_interest', []))
    parts = []
    for a in agents:
        if a['id'] not in aoi:
            continue
        b = _BEARING_NAME.get(a['bearing'], '')
        c = _CLS_NAME.get(int(a['cls']), '目标')
        seg = f'{b}方{a["range"]}米处{c}'
        if a.get('conflict') and a.get('ttc') is not None:
            seg += f'，约{a["ttc"]}秒后冲突'
        parts.append(seg)
    advice = _ADVICE_NAME.get(facts.get('ego_advice', 'keep'), '保持')
    return '；'.join(parts) + f'，本车建议{advice}。'


def aggregate(rows):
    """rows: list of per-frame score dicts -> per-metric rate."""
    keys = ['conflict_cls', 'conflict_bearing', 'ttc_bucket', 'advice',
            'activity', 'halluc']
    out = {}
    for k in keys:
        hits = sum(1 for r in rows if r[k][1] and r[k][0])
        appl = sum(1 for r in rows if r[k][1])
        out[k] = (hits / appl, appl) if appl else (float('nan'), 0)
    return out


def _load_infos(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    return d['data_list'] if isinstance(d, dict) and 'data_list' in d \
        else d['infos'] if isinstance(d, dict) and 'infos' in d else d


def main():
    p = argparse.ArgumentParser(
        description='Eval LLM captions vs geometric GT, with baselines.')
    p.add_argument('--pkl-path', required=True,
                   help='val pkl with geo_facts (+summary for the teacher).')
    p.add_argument('--mode', default='template',
                   choices=['template', 'teacher', 'model', 'shuffle',
                            'noquery'],
                   help='caption source. template/teacher need no GPU; '
                        'model/shuffle/noquery need a trained checkpoint '
                        '(via --config/--checkpoint, TODO once trained).')
    p.add_argument('--config', default=None)
    p.add_argument('--checkpoint', default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--limit', type=int, default=0)
    args = p.parse_args()

    infos = _load_infos(args.pkl_path)
    infos = [i for i in infos if i.get('geo_facts')]
    if args.limit:
        infos = infos[:args.limit]

    # caption source per mode
    if args.mode == 'template':
        captions = [template_caption(i['geo_facts']) for i in infos]
    elif args.mode == 'teacher':
        # the VLM teacher's own summary — upper bound for what we distil
        captions = [i['geo_facts'].get('summary') or '' for i in infos]
    else:
        raise NotImplementedError(
            f'mode={args.mode} needs a trained checkpoint; run after training '
            '(load config+ckpt, build detector, forward_test per frame; for '
            'shuffle, pair frame i query with frame j GT). See sec 4.3-B.')

    # GT pairing: shuffle deranges the GT index to test query-dependence
    gts = [gt_fields(i['geo_facts']) for i in infos]
    if args.mode == 'shuffle':
        import random
        idx = list(range(len(gts)))
        random.Random(args.seed).shuffle(idx)
        gts = [gts[j] for j in idx]

    rows = [score_frame(parse_caption(c), g) for c, g in zip(captions, gts)]
    res = aggregate(rows)
    print(f'mode={args.mode}  frames={len(rows)}')
    for k, (rate, n) in res.items():
        print(f'  {k:16s}: {rate:.3f}  (n={n})')


if __name__ == '__main__':
    main()
