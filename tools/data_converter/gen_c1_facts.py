#!/usr/bin/env python
"""C1: derive per-frame geometric scene facts from KL pkl (for the LLM teacher).

See documents/llm_integration_plan.md (3.2). This produces, per LiDAR frame,
a structured fact dict that (a) seeds the scene-level summary and (b) is fed
to the C2 Qwen2.5-VL teacher as a grounding/constraint prompt so it does not
hallucinate geometry. It also dumps two distributions (per-agent static
duration and nearest-Crane distance) so the 'activity' geometric-gate
thresholds can be set from data rather than guessed.

All facts are computed from existing GT only (boxes, velocity, past/future
trajectories). fut_traj convention (verified): cumulative displacement from
each agent's own current position, in the LiDAR frame; agent future absolute
xy = bbox_xy + fut_traj[t], ego future absolute xy = sdc_fut_traj[t].

Output: writes facts into info['c1_facts'] (incremental, like add_cam_sync),
and prints distribution summaries.
"""

import argparse
import os.path as osp

import mmcv
import numpy as np
from tqdm import tqdm

# 8-way bearing sectors (LiDAR frame: +x forward, +y left), degrees of yaw
# measured from +x CCW. Names are ego-centric.
_BEARING_NAMES = [
    'front', 'left-front', 'left', 'left-rear',
    'rear', 'right-rear', 'right', 'right-front',
]


def _get_infos(data):
    if isinstance(data, dict) and 'data_list' in data:
        return data['data_list']
    if isinstance(data, dict) and 'infos' in data:
        return data['infos']
    if isinstance(data, list):
        return data
    raise KeyError('Expected pkl to contain "data_list" or "infos".')


# Class-id sets (post label_mapping, 13 classes; see base_track_lidar.py).
_FULL_IDS = {2, 5}     # IGV-Full, Trailer-Full
_EMPTY_IDS = {4, 6}    # Trailer-Empty, IGV-Empty
_CRANE_IDS = {7, 12}   # Crane, WheelCrane
# Static roadside markers that should NOT count as ego path conflicts. In this
# PORT scene, cones delineate keep-out zones at the roadside (not in-lane
# emergency markers), so the ego routinely drives past them; treating "path
# near a cone" as a conflict produced spurious "yield to cone" summaries.
_NO_CONFLICT_IDS = {9}  # Cone


def _bearing(x, y):
    """Ego-centric sector name for a point at (x, y) in the LiDAR frame."""
    ang = np.degrees(np.arctan2(y, x)) % 360.0
    idx = int(((ang + 22.5) % 360.0) // 45.0)
    return _BEARING_NAMES[idx]


def _motion_bin(speed, static_th=0.3, slow_th=1.5):
    if speed < static_th:
        return 'static'
    if speed < slow_th:
        return 'moving_slow'
    return 'moving'


def _heading(fut_xy, turn_th=2.0):
    """straight / turning_left / turning_right from future-traj curvature.

    Uses the signed lateral offset of the trajectory endpoint relative to the
    initial heading direction. fut_xy is [T, 2] displacement from current pos.
    """
    if fut_xy.shape[0] < 3:
        return 'unknown'
    v0 = fut_xy[min(1, fut_xy.shape[0] - 1)]
    n = np.linalg.norm(v0)
    if n < 1e-3:
        return 'static'
    fwd = v0 / n
    left = np.array([-fwd[1], fwd[0]])
    lateral = float(fut_xy[-1] @ left)
    if lateral > turn_th:
        return 'turning_left'
    if lateral < -turn_th:
        return 'turning_right'
    return 'straight'


def _load_state(label_id):
    if label_id in _FULL_IDS:
        return 'loaded'
    if label_id in _EMPTY_IDS:
        return 'empty'
    return 'NA'


def _conflict(agent_xy, agent_fut, ego_fut, dist_th=4.0, dt=0.5):
    """Closest approach between agent and ego future paths.

    agent_fut/ego_fut are [T, 2] cumulative displacements from each one's
    current position; agent_xy is the agent's current pos. Returns
    (flag, ttc_seconds, min_dist). ttc = first step entering dist_th.
    """
    t = min(agent_fut.shape[0], ego_fut.shape[0])
    if t == 0:
        return False, None, None
    agent_abs = agent_xy[None, :] + agent_fut[:t]
    ego_abs = ego_fut[:t]
    d = np.linalg.norm(agent_abs - ego_abs, axis=1)
    j = int(np.argmin(d))
    hit = np.where(d < dist_th)[0]
    if hit.size == 0:
        return False, None, float(d[j])
    return True, float((hit[0] + 1) * dt), float(d[j])


def _static_duration(past_xy, step_th=0.15, dt=0.5):
    """Trailing near-static duration (s) from past-traj displacements.

    past_xy is [P, 2] cumulative displacement over the past P steps. We count
    how many of the most recent steps moved < step_th, i.e. how long the agent
    has been (near) stationary up to now.
    """
    if past_xy.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(past_xy, axis=0), axis=1)
    cnt = 0
    for s in steps[::-1]:
        if s < step_th:
            cnt += 1
        else:
            break
    return float(cnt * dt)


def _nearest_crane_dist(agent_xy, crane_xys):
    if not crane_xys:
        return None
    arr = np.asarray(crane_xys)
    return float(np.min(np.linalg.norm(arr - agent_xy[None, :], axis=1)))


def _frame_facts(info, cfg, dist_stats):
    """Build the c1_facts dict for one frame; append to dist_stats lists."""
    instances = info.get('instances', [])
    ego_fut = np.asarray(info.get('gt_sdc_fut_traj', [[]]))
    ego_fut = ego_fut[0] if ego_fut.ndim == 3 else ego_fut
    crane_xys = [inst['bbox_3d'][:2] for inst in instances
                 if int(inst.get('bbox_label_3d', -1)) in _CRANE_IDS]

    agents = []
    for inst in instances:
        label = int(inst.get('bbox_label_3d', -1))
        if label < 0:
            continue
        xy = np.asarray(inst['bbox_3d'][:2], dtype=np.float64)
        speed = float(np.linalg.norm(inst.get('velocity', [0.0, 0.0])))
        fut = np.asarray(inst.get('gt_fut_traj_locs', []), dtype=np.float64)
        past = np.asarray(inst.get('gt_track_traj_locs', []), dtype=np.float64)
        flag, ttc, mind = _conflict(
            xy, fut, ego_fut, dist_th=cfg['conflict_dist']) \
            if fut.size and ego_fut.size else (False, None, None)
        # Static roadside markers (cones) delineate keep-out zones, not in-lane
        # hazards; never flag them as ego conflicts (see _NO_CONFLICT_IDS).
        if label in _NO_CONFLICT_IDS:
            flag, ttc = False, None
        sdur = _static_duration(past) if past.size else 0.0
        cdist = _nearest_crane_dist(xy, crane_xys)
        gate = (sdur >= cfg['gate_static_s']
                and cdist is not None and cdist <= cfg['gate_crane_m'])
        agents.append({
            'id': int(inst.get('track_id', -1)),
            'cls': label,
            'range': round(float(np.linalg.norm(xy)), 1),
            'bearing': _bearing(xy[0], xy[1]),
            'motion': _motion_bin(speed),
            'heading': _heading(fut) if fut.size else 'unknown',
            'load': _load_state(label),
            'conflict': bool(flag),
            'ttc': round(ttc, 1) if ttc is not None else None,
            'activity_gate': bool(gate),
        })
        dist_stats['static_dur'].append(sdur)
        if cdist is not None:
            dist_stats['crane_dist'].append(cdist)
    return _scene_summary(agents, cfg)


def _scene_summary(agents, cfg):
    """Aggregate per-agent facts into the scene-level fact dict."""
    conflicting = [a for a in agents if a['conflict']]
    # agents_of_interest: conflicts first (by ttc), then nearest others.
    conflicting.sort(key=lambda a: (a['ttc'] is None, a['ttc']))
    others = sorted([a for a in agents if not a['conflict']],
                    key=lambda a: a['range'])
    aoi = [a['id'] for a in conflicting + others][:cfg['top_k']]

    if conflicting:
        ttc0 = conflicting[0]['ttc']
        advice = 'yield' if (ttc0 is not None and ttc0 < cfg['yield_ttc']) \
            else 'slow'
    else:
        advice = 'keep'

    # congestion: a moving queue, not a parking lot -> count only slow-MOVING
    # agents near ego (exclude fully static parked vehicles).
    slow_near = sum(1 for a in agents
                    if a['motion'] == 'moving_slow' and a['range'] < cfg['near_m'])
    congestion = 'queue' if slow_near >= cfg['queue_n'] else 'none'

    cranes = [{'id': a['id'], 'state': 'working' if a['activity_gate']
               else 'idle'} for a in agents if a['cls'] in _CRANE_IDS]

    return {
        'n_agents': len(agents),
        'agents': agents,
        'agents_of_interest': aoi,
        'ego_advice': advice,
        'congestion': congestion,
        'crane_status': cranes,
    }


def gen_c1_to_pkl(pkl_path, out_path=None, in_place=False, cfg=None):
    cfg = cfg or _default_cfg()
    data = mmcv.load(pkl_path)
    infos = _get_infos(data)
    dist_stats = {'static_dur': [], 'crane_dist': []}
    for info in tqdm(infos, desc=osp.basename(pkl_path)):
        info['c1_facts'] = _frame_facts(info, cfg, dist_stats)
    _report_dist(pkl_path, dist_stats, cfg)
    _write(data, pkl_path, out_path, in_place)
    return data


def _default_cfg():
    return dict(
        conflict_dist=4.0,   # m, closest-approach threshold for ego conflict
        yield_ttc=4.0,       # s, ttc below which advice=yield (else slow)
        near_m=30.0,         # m, "near ego" radius for congestion
        queue_n=4,           # >=N slow-MOVING agents near ego -> queue
        top_k=5,             # agents_of_interest cap
        # activity geometric gate (LOOSE pre-filter; C2 image decides final):
        gate_static_s=2.0,   # >= this static duration AND
        gate_crane_m=30.0,   # <= this distance to nearest Crane -> gate True
    )


def _report_dist(pkl_path, dist_stats, cfg):
    print(f'[{osp.basename(pkl_path)}] activity-gate distributions '
          f'(static>={cfg["gate_static_s"]}s & crane<={cfg["gate_crane_m"]}m):')
    for key, unit in (('static_dur', 's'), ('crane_dist', 'm')):
        v = np.array(dist_stats[key]) if dist_stats[key] else np.empty((0,))
        if v.size == 0:
            print(f'  {key}: (empty)')
            continue
        pct = [np.percentile(v, p) for p in (50, 75, 90, 95, 99)]
        print(f'  {key} ({unit}): n={v.size} '
              f'p50={pct[0]:.1f} p75={pct[1]:.1f} p90={pct[2]:.1f} '
              f'p95={pct[3]:.1f} p99={pct[4]:.1f} max={v.max():.1f}')


def _write(data, pkl_path, out_path, in_place):
    if in_place:
        dst = pkl_path
    elif out_path is not None:
        dst = out_path
    else:
        root, ext = osp.splitext(pkl_path)
        dst = f'{root}_c1{ext}'
    mmcv.dump(data, dst)
    print(f'  -> wrote {dst}')


def main():
    parser = argparse.ArgumentParser(
        description='Generate C1 geometric scene facts for KL pkl files.')
    parser.add_argument('--pkl-path', nargs='+', required=True)
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--in-place', action='store_true')
    parser.add_argument('--conflict-dist', type=float, default=4.0)
    parser.add_argument('--gate-static-s', type=float, default=2.0)
    parser.add_argument('--gate-crane-m', type=float, default=30.0)
    args = parser.parse_args()
    if args.out_path is not None and len(args.pkl_path) != 1:
        raise ValueError('--out-path can only be used with one --pkl-path.')
    cfg = _default_cfg()
    cfg.update(conflict_dist=args.conflict_dist,
               gate_static_s=args.gate_static_s,
               gate_crane_m=args.gate_crane_m)
    for pkl_path in args.pkl_path:
        gen_c1_to_pkl(pkl_path, out_path=args.out_path,
                      in_place=args.in_place, cfg=cfg)


if __name__ == '__main__':
    main()





