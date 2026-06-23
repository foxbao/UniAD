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
  - activity_addressed: on gated frames, did the VLM give ANY three-way
    activity judgement (busy/waiting/idle)? This is the real test of whether
    it looked at the routed camera and responded to the gate.
  - activity_busy: of addressed gated frames, the fraction judged 'busy'
    (active loading/unloading) — a rate, not a hit/miss score.
  - hallucination rate: object classes mentioned that are NOT in geo_facts

Baselines (run on the SAME val frames; --mode):
  - model    : the trained LLMBridgeHead (needs --config + --checkpoint)
  - template : deterministic geo_facts -> sentence (rule, the floor)
  - shuffle  : generate frame i's caption from a DERANGED frame's query, score
               against frame i's GT (the key control -- if hits don't drop, the
               LLM ignores the query and rides the language prior)
  - noquery  : caption from prompt only, no object tokens (LLM language prior)

The decisive read is model > template AND model >> shuffle/noquery: that means
the head genuinely reads the LiDAR query rather than parroting templates or the
language prior. If shuffle ~ model, the distillation didn't transfer.

Run from the repo root WITH PYTHONPATH set (model/shuffle/noquery build the
detector via third_party.uniad_mmdet3d):
  PYTHONPATH=$(pwd) python tools/analysis_tools/eval_llm_caption.py \
    --pkl-path data/kl_8/kl_infos_val_sub6cam_v3_vlmcap.pkl --mode model \
    --config projects/configs/stage2_e2e_lidar/base_e2e_lidar_occ_llm_train.py \
    --checkpoint <work_dir>/latest.pth
template/teacher modes need neither GPU, transformers, nor PYTHONPATH.
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
    """Regex-parse a Chinese caption into structured fields.

    activity_state is the three-way gate judgement the VLM is asked to make:
    'busy' (装卸/作业 in progress), 'waiting' (等待), 'idle' (空闲), or None if
    the caption gave no activity judgement at all. "addressed" = state is not
    None. Note: idle/waiting are valid judgements (a rear crane 40m away is
    usually genuinely idle), so they must NOT be scored as misses.
    """
    if not text:
        return dict(classes=set(), bearings=set(), ttc=None, advice=None,
                    activity_state=None)
    classes = {cid for name, cid in _CLS_ZH.items() if name in text}
    bearings = {canon for zh, canon in _BEARING_ZH.items() if zh in text}
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*秒', text)
    ttc = float(m.group(1)) if m else None
    advice = next((c for zh, c in _ADVICE_ZH if zh in text), None)
    if ('作业' in text) or ('装卸' in text):
        activity_state = 'busy'
    elif '等待' in text:
        activity_state = 'waiting'
    elif '空闲' in text:
        activity_state = 'idle'
    else:
        activity_state = None
    return dict(classes=classes, bearings=bearings, ttc=ttc,
                advice=advice, activity_state=activity_state)


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
    # activity — only on gated frames. Two metrics, because "idle"/"waiting"
    # are valid judgements, not misses:
    #   activity_addressed: did the VLM give ANY three-way judgement (the real
    #     test of whether it looked at the image and responded to the gate)?
    #   activity_busy: of addressed gated frames, how many were 'busy' (the
    #     active-operation rate — informative, not a hit/miss score).
    addressed = pred['activity_state'] is not None
    s['activity_addressed'] = (addressed, gt['has_activity'])
    s['activity_busy'] = (pred['activity_state'] == 'busy',
                          gt['has_activity'] and addressed)
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
            'activity_addressed', 'activity_busy', 'halluc']
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


def _run_model_captions(args, n_frames):
    """Build detector+ckpt, run inference, return captions for model/shuffle/
    noquery modes (aligned to the same frame order as _load_infos).

    Implementation: drive the dataloader once; for each frame capture the head's
    (agent_query, centres) into a bank, then generate captions per mode --
      model   : frame i's own query
      noquery : no query (prompt only) -> exposes the LLM's language prior
      shuffle : frame i's query paired with a DERANGED frame's query, so if the
                score doesn't drop, the LLM isn't really reading the query.
    The query bank is captured by monkeypatching the head's generate_from_query
    so we get exactly the (query, centres) the detector would have used, without
    editing the detector/head inference path. Needs --config + --checkpoint.
    """
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from third_party.uniad_mmdet3d.datasets.builder import (
        build_dataloader, build_dataset)
    from third_party.uniad_mmdet3d.models.builder import build_model

    assert args.config and args.checkpoint, \
        'model/shuffle/noquery modes need --config and --checkpoint'
    cfg = Config.fromfile(args.config)
    if cfg.get('plugin_dir'):
        import importlib, os.path as _osp
        _m = cfg.plugin_dir.rstrip('/').replace('/', '.')
        importlib.import_module(_m)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=1,
                              dist=False, shuffle=False)
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    head = model.module.llm_head
    assert head is not None, 'config has no llm_head'

    # Capture the (query, centres) the head would use, per frame, by wrapping
    # generate_from_query. Store CPU clones (the bank must outlive each frame).
    bank = []          # [(query_or_None, centres_or_None)]
    orig = head.generate_from_query

    def capture(agent_query, centres, max_new_tokens=64):
        bank.append((
            None if agent_query is None else agent_query.detach().cpu(),
            None if centres is None else centres.detach().cpu()))
        return orig(agent_query, centres, max_new_tokens=max_new_tokens)

    head.generate_from_query = capture
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            if n_frames and i >= n_frames:
                break
            model(return_loss=False, rescale=True, **data)
    head.generate_from_query = orig

    dev = next(head.projector.parameters()).device

    def gen(query, centres):
        q = None if query is None else query.to(dev)
        c = None if centres is None else centres.to(dev)
        with torch.no_grad():
            return orig(q, c)

    if args.mode == 'model':
        return [gen(q, c) for q, c in bank]
    if args.mode == 'noquery':
        return [gen(None, None) for _ in bank]
    # shuffle: derange the query bank index, keep frame order for GT pairing
    import random
    idx = list(range(len(bank)))
    random.Random(args.seed).shuffle(idx)
    return [gen(bank[j][0], bank[j][1]) for j in idx]


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
        # model / shuffle / noquery: generate from the trained head.
        captions = _run_model_captions(args, len(infos))

    # GT pairing. shuffle is realised inside _run_model_captions by deranging
    # the QUERY (frame i's GT vs another frame's query), so GT here stays in
    # frame order for ALL modes -- do NOT also derange GT (that would be a
    # double shuffle and break the control).
    gts = [gt_fields(i['geo_facts']) for i in infos]

    rows = [score_frame(parse_caption(c), g) for c, g in zip(captions, gts)]
    res = aggregate(rows)
    print(f'mode={args.mode}  frames={len(rows)}')
    for k, (rate, n) in res.items():
        print(f'  {k:16s}: {rate:.3f}  (n={n})')


if __name__ == '__main__':
    main()
