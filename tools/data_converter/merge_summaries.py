#!/usr/bin/env python
"""Step 5: merge VLM-caption summaries (JSON sidecar) back into a KL geo-facts pkl.

This runs in the TRAINING env (uniad_train, numpy 1.x). VLM caption (gen_vlm_caption.py)
runs under qwen_vl (numpy 2.x) and emits a token->summary JSON sidecar instead
of rewriting the pkl, because a numpy-2.x pickle of the pkl cannot be loaded by
numpy 1.x. Here we read that JSON and write summary into
info['geo_facts']['summary'], producing a pkl the training stack can load.

See documents/llm_integration_plan.md (0.4 Step 5).
"""

import argparse
import json
import os.path as osp
import pickle


def _infos(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    return data


def merge(pkl_path, json_path, out_path=None, in_place=False):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    # json_path may be one path or several (e.g. one per VLM-caption shard); shards are
    # disjoint token->summary maps, so a plain update merges them.
    if isinstance(json_path, str):
        json_path = [json_path]
    summaries = {}
    for jp in json_path:
        with open(jp, 'r', encoding='utf-8') as f:
            summaries.update(json.load(f))
    print(f'loaded {len(summaries)} summaries from {len(json_path)} sidecar(s)')
    infos = _infos(data)

    n_set = n_miss = 0
    for info in infos:
        facts = info.get('geo_facts')
        if facts is None:
            continue
        token = info.get('token')
        if token in summaries:
            facts['summary'] = summaries[token]
            n_set += 1
        else:
            facts.setdefault('summary', None)
            n_miss += 1

    print(f'[{osp.basename(pkl_path)}] merged summaries: set={n_set} '
          f'not_in_json={n_miss} (json has {len(summaries)} entries)')

    if in_place:
        dst = pkl_path
    elif out_path is not None:
        dst = out_path
    else:
        root, ext = osp.splitext(pkl_path)
        dst = f'{root}_vlmcap{ext}'
    with open(dst, 'wb') as f:
        pickle.dump(data, f)
    print(f'  -> wrote {dst}')


def main():
    p = argparse.ArgumentParser(
        description='Merge VLM-caption summary JSON sidecar into a KL c1 pkl.')
    p.add_argument('--pkl-path', required=True,
                   help='The geo-facts pkl (numpy-1.x written, e.g. *_with_cam_geo.pkl '
                        'or a subset).')
    p.add_argument('--json-path', required=True, nargs='+',
                   help='VLM-caption sidecar(s), *_summaries.json. Pass several '
                        '(or a shell glob) to merge all VLM-caption shards at once.')
    p.add_argument('--out-path', default=None)
    p.add_argument('--in-place', action='store_true')
    args = p.parse_args()
    merge(args.pkl_path, args.json_path, args.out_path, args.in_place)


if __name__ == '__main__':
    main()
