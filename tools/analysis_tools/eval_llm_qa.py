#!/usr/bin/env python3
"""Evaluate LLMBridgeHead hard-GT QA answers.

This mirrors eval_llm_caption.py's decisive controls:

  model   : answer frame i's question with frame i's LiDAR query
  shuffle : answer frame i's question with a different frame's LiDAR query
  noquery : answer from the question prompt only, no object tokens

If model is not clearly better than shuffle/noquery, the head may be relying on
language priors or templates instead of the same-frame LiDAR query.
"""

import argparse
import json
import os.path as osp
import pickle
import random
import re
from collections import defaultdict


TYPE_ORDER = [
    'summary', 'risk', 'advice', 'activity_gate',
    'spatial', 'motion', 'count',
]


def _load_infos(pkl_path):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    return data


def _norm(text):
    text = '' if text is None else str(text)
    text = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    text = re.split(r'[\r\n]', text.strip())[0]
    return re.sub(r'[\s，。！？、；：,.!?;:（）()“”"\'`]+', '', text)


def _char_f1(pred, gt):
    pred = _norm(pred)
    gt = _norm(gt)
    if not pred and not gt:
        return 1.0
    if not pred or not gt:
        return 0.0
    pc = defaultdict(int)
    for ch in pred:
        pc[ch] += 1
    overlap = 0
    for ch in gt:
        if pc[ch] > 0:
            overlap += 1
            pc[ch] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / max(len(pred), 1)
    recall = overlap / max(len(gt), 1)
    return 2 * precision * recall / (precision + recall)


def _score(pred, gt):
    pn = _norm(pred)
    gn = _norm(gt)
    return dict(
        exact=pn == gn,
        contains=(gn in pn) if gn else False,
        char_f1=_char_f1(pred, gt),
    )


def _qa_list(info):
    return info.get('vlm_qa') or (info.get('geo_facts') or {}).get('qa') or []


def _select_qas(info, answer_types, strategy, max_qas):
    qas = [
        q for q in _qa_list(info)
        if isinstance(q, dict)
        and (not answer_types or q.get('answer_type') in answer_types)
        and q.get('question') and q.get('answer')
    ]
    if not qas:
        return []
    if strategy == 'hash':
        key = str(qas[0].get('id', ''))
        return [qas[sum(ord(ch) for ch in key) % len(qas)]]
    if strategy == 'all':
        return qas[:max_qas] if max_qas else qas

    # per_type: first QA of each answer_type, then fill remaining slots in
    # original order. This gives a compact, balanced smoke eval.
    selected = []
    used_ids = set()
    for answer_type in TYPE_ORDER:
        for q in qas:
            if q.get('answer_type') == answer_type:
                selected.append(q)
                used_ids.add(q.get('id'))
                break
        if max_qas and len(selected) >= max_qas:
            return selected
    for q in qas:
        if q.get('id') in used_ids:
            continue
        selected.append(q)
        if max_qas and len(selected) >= max_qas:
            break
    return selected


def _build_model_and_bank(args):
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from third_party.uniad_mmdet3d.datasets.builder import (
        build_dataloader, build_dataset)
    from third_party.uniad_mmdet3d.models.builder import build_model

    cfg = Config.fromfile(args.config)
    if cfg.get('plugin_dir'):
        import importlib
        importlib.import_module(cfg.plugin_dir.rstrip('/').replace('/', '.'))

    cfg.data.test.ann_file = osp.abspath(args.pkl_path)
    cfg.data.test.test_mode = True
    cfg.data.workers_per_gpu = args.workers
    dataset = build_dataset(cfg.data.test)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=args.workers,
        dist=False, shuffle=False)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model.cuda(), device_ids=[0])
    model.eval()
    head = model.module.llm_head
    assert head is not None, 'config has no llm_head'

    bank = []
    orig_generate = head.generate_from_query

    def capture(agent_query, centres, max_new_tokens=64):
        bank.append((
            None if agent_query is None else agent_query.detach().cpu(),
            None if centres is None else centres.detach().cpu()))
        return ''

    head.generate_from_query = capture
    eval_infos = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            if args.limit and i >= args.limit:
                break
            eval_infos.append(dataset.data_infos[dataset._to_raw_index(i)])
            model(return_loss=False, rescale=True, **data)
    head.generate_from_query = orig_generate
    assert len(bank) == len(eval_infos), (
        f'query bank ({len(bank)}) != eval infos ({len(eval_infos)})')
    return model, head, orig_generate, bank, eval_infos


def _derangement(n, seed):
    idx = list(range(n))
    if n <= 1:
        return idx
    random.Random(seed).shuffle(idx)
    for i in range(n):
        if idx[i] == i:
            idx[i], idx[(i + 1) % n] = idx[(i + 1) % n], idx[i]
    return idx


def _qa_prompt(head, qa):
    template = getattr(head, 'qa_prompt_template', '请回答问题：{question}')
    return template.format(
        question=qa.get('question', ''),
        answer_type=qa.get('answer_type', ''),
        gt_source=qa.get('gt_source', ''))


def _generate_mode(args, head, orig_generate, bank, infos, mode):
    dev = next(head.projector.parameters()).device
    idx = _derangement(len(bank), args.seed) if mode == 'shuffle' else None
    old_prompt = head.prompt
    rows = []
    try:
        for i, info in enumerate(infos):
            qas = _select_qas(
                info, args.answer_types, args.qa_strategy,
                args.max_qas_per_frame)
            if not qas:
                continue
            if mode == 'model':
                src_q, src_c = bank[i]
            elif mode == 'shuffle':
                src_q, src_c = bank[idx[i]]
            elif mode == 'noquery':
                src_q, src_c = None, None
            else:
                raise ValueError(mode)
            q = None if src_q is None else src_q.to(dev)
            c = None if src_c is None else src_c.to(dev)
            for qa in qas:
                head.prompt = _qa_prompt(head, qa)
                pred = orig_generate(q, c, max_new_tokens=args.max_new_tokens)
                score = _score(pred, qa.get('answer', ''))
                rows.append(dict(
                    mode=mode,
                    token=info.get('token', ''),
                    qa_id=qa.get('id', ''),
                    answer_type=qa.get('answer_type', ''),
                    question=qa.get('question', ''),
                    gt=qa.get('answer', ''),
                    pred=pred,
                    **score))
    finally:
        head.prompt = old_prompt
    return rows


def _summarise(rows):
    by_type = defaultdict(list)
    by_type['ALL'] = rows
    for row in rows:
        by_type[row['answer_type']].append(row)

    out = {}
    for key in ['ALL'] + [t for t in TYPE_ORDER if t in by_type]:
        items = by_type.get(key, [])
        if not items:
            continue
        out[key] = dict(
            n=len(items),
            exact=sum(r['exact'] for r in items) / len(items),
            contains=sum(r['contains'] for r in items) / len(items),
            char_f1=sum(r['char_f1'] for r in items) / len(items),
        )
    return out


def _print_summary(mode, summary):
    print(f'\n[{mode}]')
    print(f'{"type":<15} {"n":>6} {"exact":>8} {"contains":>9} {"char_f1":>8}')
    for key, m in summary.items():
        print(f'{key:<15} {m["n"]:>6d} {m["exact"]:>8.3f} '
              f'{m["contains"]:>9.3f} {m["char_f1"]:>8.3f}')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate hard-GT QA with model/shuffle/noquery controls.')
    parser.add_argument('--pkl-path', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--modes', nargs='+',
                        default=['model', 'shuffle', 'noquery'],
                        choices=['model', 'shuffle', 'noquery'])
    parser.add_argument('--limit', type=int, default=50,
                        help='Number of queue-able eval frames to score.')
    parser.add_argument('--qa-strategy', default='per_type',
                        choices=['per_type', 'hash', 'all'])
    parser.add_argument('--max-qas-per-frame', type=int, default=7)
    parser.add_argument('--answer-types', nargs='*', default=None)
    parser.add_argument('--max-new-tokens', type=int, default=80)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--out-json', default=None)
    args = parser.parse_args()

    _, head, orig_generate, bank, infos = _build_model_and_bank(args)
    all_rows = []
    summaries = {}
    print(f'eval_frames={len(infos)} modes={args.modes} '
          f'qa_strategy={args.qa_strategy} max_qas/frame={args.max_qas_per_frame}')
    for mode in args.modes:
        rows = _generate_mode(args, head, orig_generate, bank, infos, mode)
        all_rows.extend(rows)
        summaries[mode] = _summarise(rows)
        _print_summary(mode, summaries[mode])

    if args.out_json:
        payload = dict(args=vars(args), summaries=summaries, rows=all_rows)
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f'\nwrote {args.out_json}')


if __name__ == '__main__':
    main()
