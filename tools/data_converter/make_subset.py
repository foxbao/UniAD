#!/usr/bin/env python
"""Build a stratified subset of a KL geo-facts pkl for fast VLM-caption/LLM iteration.

Rare-but-valuable scenes (queue / conflict / activity-gate) are kept in full;
ordinary frames are randomly sampled to fill up to --size. This keeps the
subset small (fast VLM generation + quick LLMBridgeHead smoke tests) while
guaranteeing the interesting cases are not drowned out.

prev/next tokens are left untouched (VLM caption generation is per-frame and ignores
time order); the subset is therefore NOT meant for temporal track training as
its prev/next may point outside the subset. See documents/llm_integration_plan.md.
"""

import argparse
import os.path as osp
import pickle
import random


def _infos(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list'], 'data_list'
    if isinstance(data, dict) and 'infos' in data:
        return data['infos'], 'infos'
    return data, None


def _is_rare(facts):
    if facts is None:
        return False
    if facts.get('congestion') == 'queue':
        return True
    agents = facts.get('agents', [])
    return (any(a.get('conflict') for a in agents)
            or any(a.get('activity_gate') for a in agents))


def make_subset(pkl_path, size, out_path, seed=0):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    infos, key = _infos(data)

    rare = [i for i in infos if _is_rare(i.get('geo_facts'))]
    ordinary = [i for i in infos if not _is_rare(i.get('geo_facts'))]
    rng = random.Random(seed)
    rng.shuffle(ordinary)
    n_fill = max(0, size - len(rare))
    chosen = rare + ordinary[:n_fill]
    # Preserve original order for readability/debugging.
    order = {id(i): k for k, i in enumerate(infos)}
    chosen.sort(key=lambda i: order[id(i)])

    print(f'[{osp.basename(pkl_path)}] total={len(infos)} '
          f'rare={len(rare)} ordinary_added={n_fill} -> subset={len(chosen)}')
    if isinstance(data, dict):
        out = dict(data)
        out[key] = chosen
    else:
        out = chosen
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f'  -> wrote {out_path}')


def main():
    p = argparse.ArgumentParser(description='Stratified subset of a KL geo-facts pkl.')
    p.add_argument('--pkl-path', required=True)
    p.add_argument('--out-path', required=True)
    p.add_argument('--size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    make_subset(args.pkl_path, args.size, args.out_path, args.seed)


if __name__ == '__main__':
    main()
