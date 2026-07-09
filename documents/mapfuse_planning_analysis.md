# HD-Map Fusion for LiDAR Planning — Analysis & Findings

**Status:** In progress (v2 retrain at epoch 2/6 as of 2026-07-08). Early evidence
strongly suggests the HD-map branch adds no measurable value to planning in this
port ODD, but the final verdict awaits the epoch-6 ablation.

**Audience:** A collaborator/agent analyzing whether the HD-map planning fusion is
worth keeping. This doc is self-contained — no prior context needed.

---

## 1. What is being tested

UniAD-tiny LiDAR-only stage-2 model (track + motion + occ + planning) on a private
**port/terminal dataset** (`kl_8`), deployed target NVIDIA DRIVE Orin. We added an
**HD-map lane prior** that is fused *only inside the planning head* (not motion):

- A `MapLaneEncoder` parses a static surveyed map (`data/kl_8/map/base_map.txt`,
  354 lanes), transforms lanes to the ego frame per-frame via `ego2global`, crops
  to the nearest 64 lanes, encodes each as a 7-dim feature (centerline xy, tangent,
  arc-length, L/R boundary offsets) → `lane_query`.
- In `PlanningHeadSingleMode._apply_map_lane_attention`, the single ego plan query
  cross-attends to the K=16 nearest lanes, and the result is fused as a **gated
  residual**:
  ```
  map_delta = map_delta_proj(map_context - plan_query)
  map_gate  = sigmoid(map_gate([plan_query, map_context]))   # data-adaptive gate
  plan_query = plan_query + map_gate * map_delta
  ```

**Question:** does this map prior actually improve the planned trajectory, beyond
what the LiDAR BEV already provides?

---

## 2. Key metric & how we measure "map value"

Planning is scored by **L2** (m, lower better) and **collision rate** at 1s/2s/3s
horizons (dt=0.5s, 6 steps), plus a breakdown by GT-motion bucket (static / slow /
moving_straight / turning). `avg.L2` averages the 1/2/3s L2.

The **decisive test is an ablation**: take one trained checkpoint and evaluate it
twice —
- **map-ON**: config `base_e2e_lidar_plan_mapfuse_v2.py` (`use_map_lane=True`)
- **map-OFF**: config `base_e2e_lidar_plan_mapfuse_v2_nomap_eval.py`
  (`use_map_lane=False` → `_apply_map_lane_attention` becomes a pass-through;
  everything else bit-identical)

The **map-ON vs map-OFF delta on the SAME checkpoint** = the net causal contribution
of the map branch. Everything except the map forward path is identical, so any L2
delta is attributable to the map alone.

> Note: a **planning-eval mask bug** was found and fixed earlier this session
> (`kl_dataset._evaluate_planning` mis-parsed the 4D `sdc_planning_mask`, inflating
> Static bucket counts and metrics). All numbers below use the **fixed** eval code.

---

## 3. History: v1 (the dead branch)

The original `base_e2e_lidar_plan_mapfuse` (call it **v1**) trained the map branch to
a **no-op**. Ablation on v1's epoch_6:

- map-ON vs map-OFF differed by **< 1e-5** on every planning metric (i.e. the logged
  difference was just float rounding — no real effect).
- Weight evidence: `map_gate` sigmoid ≈ 0.135 (nearly closed, not adaptive),
  `map_delta_proj` was zero-initialized and barely grew.

**Root cause = training recipe, not the mechanism:**
1. **Start point.** v1 `load_from` a *fully-trained* planner (avg.L2 ≈ 0.58). With
   the planner already good, the map's marginal gradient is tiny → optimizer keeps
   the branch shut.
2. **Initialization.** `map_delta_proj` zero-init (delta ≡ 0 at step 0) +
   `map_gate_init = -2.0` (sigmoid ≈ 0.12, nearly closed). In `gate * delta` both
   factors start ≈ 0 → the branch receives ≈ 0 gradient → stays a no-op.

v1's clean baseline (fixed eval): **avg.L2 0.580, avg.Collision 0.010**. Bucket order
(best→worst): Static 0.213 < Slow 0.452 < Straight 0.583 < Right 0.664 <
MovingStraight 0.717 < **Turning 0.876**.

---

## 4. v2: giving the map branch a fair chance

**v2 = `base_e2e_lidar_plan_mapfuse_v2.py`** changes exactly two coupled levers to
un-starve the branch (both made config-controllable; defaults preserve v1):

1. **Start point** → `load_from = base_e2e_lidar_occ/latest.pth` (planner *not yet
   formed*), so planner and map co-adapt.
2. **Initialization** → `map_gate_init = 0.0` (sigmoid 0.5, half-open) +
   `map_delta_init = 'small'` (xavier×0.1 random, non-degenerate delta from step 0).

Everything else (map_lane_encoder, motion `map_agent_scope='none'`, losses, data) is
inherited unchanged. Training: 6 epochs, AdamW lr 2e-4, cosine.

---

## 5. Results so far

### 5a. Map-branch weights are learning (mechanism activated)

| checkpoint | gate σ mean | gate σ std | delta_proj absmax | delta_proj std |
|---|---|---|---|---|
| init       | 0.5000 | 0.0000 | 0.0108 | 0.0030 |
| ep1        | 0.4978 | 0.0020 | 0.0532 | 0.0084 |
| ep2        | 0.4974 | 0.0024 | 0.0662 | 0.0094 |

- **delta_proj grows** (0.011→0.053→0.066): unlike v1, the branch is not a zero
  dead-water; it receives gradient and learns. BUT growth is **decelerating**
  (+0.042 then +0.013).
- **gate stays ≈ 0.5** with tiny std: the gate has **not** become selective/adaptive
  — it passes a near-constant half of the delta for all frames.

### 5b. Planner is converging well (training healthy)

| ep | total loss | plan.ade (train) | plan.col0 | motion.min_ade |
|---|---|---|---|---|
| 1 | 21.91 | 0.983 | 0.483 | 0.408 |
| 2 | 19.53 | 0.693 | 0.398 | 0.410 |
| 3 (partial) | 18.48 | 0.619 | 0.288 | 0.396 |

`plan.ade` 0.98→0.69→0.62, approaching v1's mature 0.58. Planner forming normally
from the occ start (higher initial than v1's fine-tune 0.706, as expected).

### 5c. **The decisive ablation: map-ON vs map-OFF (same checkpoint)**

**epoch_1** (planner immature, avg.L2 ≈ 1.39):

| metric | map-ON | map-OFF | Δ (on−off) |
|---|---|---|---|
| avg.L2 | 1.38994 | 1.38727 | **+0.00267** |
| avg.Collision | 0.00956 | 0.00972 | −0.00016 |
| Static/L2 | 0.4577 | 0.4590 | −0.0013 |
| Slow/L2 | 0.6947 | 0.6988 | −0.0041 |
| MovingStraight/L2 | 1.9461 | 1.9389 | +0.0072 |
| Turning/L2 | 1.1428 | 1.1505 | −0.0077 |

**epoch_2** (planner maturing, avg.L2 ≈ 0.95):

| metric | map-ON | map-OFF | Δ (on−off) |
|---|---|---|---|
| avg.L2 | 0.9511435 | 0.9511405 | **+0.0000030** |
| avg.Collision | 0.00994866 | 0.00994866 | 0 (identical) |
| Static/L2 | 0.27095841 | 0.27095629 | +0.0000021 |
| Slow/L2 | 0.58978556 | 0.58977721 | +0.0000084 |
| MovingStraight/L2 | 1.28433905 | 1.28433800 | +0.0000011 |
| Turning/L2 | 0.99436300 | 0.99436034 | +0.0000027 |

### 5d. The trend that matters

| checkpoint | planner avg.L2 | map-ON/OFF Δ (avg.L2) |
|---|---|---|
| v1 ep6 (dead) | 0.580 | < 1e-5 |
| v2 ep1 | 1.390 | ±0.003 (mixed sign) |
| **v2 ep2** | 0.951 | **~3e-6 (collapsed)** |

**As the planner matures, the map's net effect collapses back toward zero.** At ep1
the immature planner let the map's perturbation register as ±0.003 (mixed sign =
noise, not a consistent improvement). By ep2 the planner is much better and the map
delta has shrunk to ~1e-6 — the same order as v1's dead branch.

---

## 6. Interpretation (current, pre-ep6)

- **Mechanism: works.** v2's init fix genuinely activated the branch (weights learn,
  ep1 Δ >> v1's <1e-5). The v1 no-op was a recipe artifact, now fixed.
- **Value: appears near-zero for this ODD.** Once the planner is competent, turning
  the map on vs off changes the plan by ~1e-6. The leading hypothesis:
  1. **BEV already encodes drivable space** (the LiDAR drivable head reaches IoU
     0.907). The map's "where can I drive" is largely redundant with BEV.
  2. **Port ODD weakens map value**: ego routes are quasi-fixed (container haul
     loops), no complex multi-path intersections, high static/slow fraction — exactly
     where a lane prior contributes least.
  3. **Short horizon**: 3s / 6-step plans rarely need long-range lane topology.
- The **gate never became adaptive** (stuck at 0.5), and delta growth is decelerating
  — both consistent with "the branch has little useful signal to lock onto."

**This is early (ep2/6).** The planner isn't fully converged (0.95 vs target ~0.58).
A reversal is unlikely given the trend but not impossible (e.g. gate could still
learn to amplify map in specific scenes in later epochs).

---

## 7. What would change the conclusion

The **epoch-6 ablation** is the final verdict. Re-run both configs on `epoch_6.pth`
(4-GPU eval, ~15 min each):
- If **Δ avg.L2 > ~1%** (e.g. 0.58 → 0.57) → the map adds real value beyond BEV; keep
  and optimize it (e.g. auxiliary map supervision, tune gate).
- If **Δ avg.L2 < ~0.1%** (≈ current ~1e-6) → the map is confirmed to add nothing in
  this ODD; **drop it** and redirect effort to the true weak buckets (Turning 0.876,
  MovingStraight 0.717 in v1's clean numbers).

---

## 8. Open questions for the collaborator

1. Is the ~1e-6 ep2 delta truly "no value", or is the **gated-residual design itself**
   too conservative (single ego query, K=16 lanes, one attn layer)? Would a stronger
   coupling (e.g. map-conditioned anchor selection, or auxiliary loss forcing the
   plan query to predict lane alignment) extract value the current design can't?
2. Does the **static surveyed map + per-frame ego2global transform** have enough
   localization accuracy to be trustworthy? (Deployment note: this path had an FP16
   catastrophic-cancellation NaN issue from ~3000m global coords, fixed separately.)
3. Given BEV drivable IoU 0.907, is there *any* planning-relevant information the map
   holds that BEV structurally cannot (occluded/far lane topology)? If yes, the
   experiment design (short-horizon L2) may not be sensitive to it — a topology/
   route-adherence metric might be needed instead.

---

## 9. Literature survey: how others fuse map into planning (2026-07-08)

A multi-source, adversarially-verified survey (23 primary sources, 25 claims
verified 3-vote, 22 confirmed / 3 refuted) answering "is my fusion under-powered, or
does map genuinely add little here?". **Answer: both — but the dominant cause is
structural (ego-status dominance + BEV redundancy), not fusion-mechanism weakness.**

### 9a. Fusion mechanisms, strong → weak

**Strong / standard:**
- **Bidirectional per-agent agent↔lane cross-attention.** LaneGCN: four directional
  modules A2L→L2L→L2A→A2A, per-actor local selection (L2 thresholds 7m/6m), **plain
  additive residual (no gate)**. VectorNet/TNT/DenseTNT: fully-connected bidirectional
  self-attention over all lane+agent polylines. PGP: agent→lane-node attention + GNN
  message passing over the lane graph, per-prediction path traversal.
  [arXiv 2007.13732; ar5iv 2005.04259; arXiv 2106.15004]
- **Trajectory-aligned local map, iteratively refined.** MTR "dynamic map collection":
  per motion-query **and per decoder layer** re-gathers the L polylines nearest the
  *currently predicted* trajectory (6 layers, 128 polylines). Contrast our **static
  one-shot K=16 + single layer**. [NeurIPS 2022 MTR]
- **Map-as-constraint (bypasses attention).** TNT samples trajectory target candidates
  directly on HD-map centerlines (map = output space). VAD adds loss-side
  boundary/directional/collision constraints on the closest lane per waypoint.
  [PMLR v155 TNT; ICCV 2023 VAD]

**Weak (where our design sits):** single ego query, one attn layer, conservative
sigmoid-gated residual. A gate is the weakest coupling — it structurally *permits*
the model to learn to shut the branch off, which is exactly what v1 did.

### 9b. Closest published analog = VAD — and it is stronger than ours

VAD is the nearest match (single ego query cross-attends to vectorized map queries,
ego=Q / map=K,V), BUT it fuses by **concatenation** into the plan-head MLP
(`f_ego=[Q'_ego, Q''_ego, s_ego]`) with **no sigmoid gate**, plus loss-side geometric
constraints. So even the closest design injects map more forcefully than our gate.
[ICCV 2023 VAD]

### 9c. The likely real cause: ego-status dominance

CVPR 2024 "Is Ego Status All You Need?" (verified 3-0): in UniAD/VAD-style planners,
predicted trajectories are **dominated by ego status** — blanking camera collapses
perception (NDS→0) yet planning barely changes; perturbing ego velocity changes
planning drastically. In a **fixed-route port ODD** (low curvature, strong kinematic
prior) this dominance is maximal and structurally subsumes any map delta. This, more
than gate weakness, explains our <0.1% ablation. Also note VectorNet's Argoverse
ablation: adding map dropped ADE 2.36→1.75 (large) — map DOES help prediction in open
ODDs, reinforcing that our null result is ODD-specific, not universal.
[arXiv 2312.03031; PMLR v155]

**Honest correction (refuted claims):** "adding map *degrades* planning" was refuted
0-3; "Ego-MLP matches SOTA" refuted 1-2. Accurate statement: map contribution is
*diminished/subsumed*, not negative or literally zero-value.

### 9d. If we want one more attempt to make map matter (strong→weak effort)

1. **Concat instead of gated residual** (VAD-style): force map into the plan head.
2. **Loss-side geometric constraints** (VAD-style): lane-direction / boundary terms —
   plausibly most useful on our weak buckets (Turning 0.876), where lane geometry
   binds hardest.
3. **Cleanest diagnostic:** remove ego-status injection from the plan query and re-run
   the ablation. This disambiguates "fusion too weak" vs "ego-status dominates". If
   map delta stays ~0 even without ego-status, the map is truly redundant with BEV
   here; if it grows, the kinematic prior was capping it.

### 9e. Second independent survey — corroboration + extra methods

A separate agent ran an independent survey and **converged on the same core verdict**
(our gated-residual is too soft; port LiDAR-BEV already learns drivable well; map is
easily treated as an ignorable perturbation). Two independent surveys agreeing is a
strong signal. It surfaced two methods worth adding:

- **MAP — Map-Assisted Planning (ICCV 2025 Workshop, DriveX).** The closest analog to
  our design that WORKS: instead of a residual on one ego query, the map branch
  **produces its own planning query** `Q_map` (cross-attn: online map-seg memory +
  ego status), a parallel `Q_plan` comes from BEV + ego status, and they fuse
  **peer-to-peer via an ego-status-driven weight adapter**: `Q_final = α·Q_map +
  (1-α)·Q_plan`. Reports L2 −16.6%, off-road −56.2% vs UniV2X. Note the adapter is
  **gated on ego status** — dovetails with the ego-dominance finding (§9c).
  [ICCV 2025W MAP]
- **PDM / PDM-Closed (nuPlan, arXiv 2306.07962).** Map as **reference-path / proposal
  skeleton**: pick route centerline from the lane graph → generate Frenet
  lateral/velocity proposals → rollout → rule-score → select. Map becomes the planner's
  *search space*, not a soft feature. A simple centerline prior is very strong in
  closed loop.

**Suggested stronger re-attachments (from both surveys, strong→weak effort):**
- **Version A — map-conditioned anchor/proposal:** sample a centerline from the
  nearest/route lane and predict an **offset/residual to it**, instead of predicting
  global xy from scratch. (PDM / TNT lineage.)
- **Version B — map consistency loss:** lane-direction alignment + boundary-overstep +
  off-drivable penalties on the plan waypoints; **inspect the Turning bucket** (our
  weak spot 0.876) where lane geometry binds hardest. (VAD lineage.)
- **Version C — independent Q_map fusion:** let the map branch emit its own `Q_map`
  and fuse peer-to-peer with a learned/ego-gated weight, not a residual on one query.
  (MAP / VAD lineage.)

**⚠️ Domain-transfer caveat (cross-check between the two surveys):** MAP's headline
gains (−16.6% L2 / −56.2% off-road) are on **UniV2X + camera + open urban nuScenes**,
where BEV learns drivable *worse* than our LiDAR (IoU 0.907) and map has more headroom.
Combined with the ego-status-dominance finding, expect **any of A/B/C to yield far less
here** than the MAP paper's numbers. Do the cheap diagnostic (§9d.3 / §11) FIRST to
learn whether there is *any* headroom before investing in A/B/C.

Caveat: no source measures map fusion in a port/fixed-route ODD specifically; the
ODD-specific conclusion is inferred from the ego-dominance mechanism + low-curvature
routes. MTR/LaneGCN/VectorNet/PGP are *prediction* models (ADE/FDE), not closed-loop
planners — mechanisms transfer conceptually.

---

## 10. ego-status diagnostic — RESULT (v2 ep2): the literature hypothesis was wrong here

We ran the §9d.3 diagnostic to disambiguate "map fusion too weak" vs "ego-status
dominates". Added an eval-only `ablate_ego_status` switch to `PlanningHeadSingleMode`
(default `'none'`, no training impact). `plan_query = concat(sdc_traj_query,
sdc_track_query, navi_embed)` → mlp_fuser. The switch zeros ego pathways at inference:
`zero`=zero sdc_track_query (detected ego box pos+vel), `traj`=zero sdc_traj_query
(motion-predicted ego kinematic trend), `both`=zero both.

**Results on v2 epoch_2 (avg.L2, baseline none = 0.9511435):**

| ablation (map ON) | avg.L2 | Δ vs baseline |
|---|---|---|
| none (baseline) | 0.9511435 | — |
| track-off (`zero`) | 0.9511449 | +1.4e-6 |
| traj-off (`traj`) | 0.9511629 | +1.9e-5 |
| **both-off (`both`)** | 0.9511456 | +2.1e-6 |

And the ego×map 2×2 (avg.L2): with ego ON, map Δ = +3.0e-6; with ego OFF (`zero`),
map Δ = +2.1e-6 — **map delta does NOT grow when ego-status is removed.**

**Interpretation — this REFUTES the ego-status-dominance hypothesis (§9c) for THIS
planner:**
- Zeroing *all* ego inputs (both-off, which rules out one-pathway-compensates-the-other
  false negatives) barely moves the plan (+2e-6). So ego-status is **not** dominating
  and capping the map.
- The real picture: **every prior-type input in plan_query (ego traj/track AND map) is
  only a weak modulation** — the planner's trajectory is driven almost entirely by
  **BEV cross-attention + the navi_embed command**. Map redundancy is a symptom of the
  whole plan_query branch being weak relative to BEV, not of the map fusion mechanism
  or ego dominance specifically.
- **Implication:** stronger map re-attachments (Versions A/B/C) may ALSO underperform,
  because the bottleneck is the plan_query↔BEV structure, not just the map sub-path.
  To make *any* prior matter, the plan_query/BEV interaction likely needs restructuring.

**ep3 re-check (planner matured 0.951 → 0.745, approaching target 0.58) — conclusions
hold.** Full matrix re-run on epoch_3; every ablation Δ is unchanged in magnitude
(1e-6~1e-8): map Δ(ego-on) −1.4e-6, track-off +1.6e-6, traj-off +2e-8,
**both-off −3.6e-6**, map Δ(ego-off) +0.7e-6.

| ablation Δ vs baseline | ep2 (0.951) | ep3 (0.745) |
|---|---|---|
| map Δ (ego-on) | +3.0e-6 | −1.4e-6 |
| track-off | +1.4e-6 | +1.6e-6 |
| traj-off | +1.9e-5 | +2e-8 |
| both-off | +2.1e-6 | −3.6e-6 |
| map Δ (ego-off) | +2.1e-6 | +0.7e-6 |

The planner clearly strengthened but the ablation deltas did NOT grow — this **rules out
the "ep2 too early" caveat**: both checkpoints agree. So both conclusions are stable
under maturation, not early noise: (a) **map adds no measurable value** (map Δ ~1e-6
everywhere), and (b) **ego pathways are weak, not dominant** (both-off ≈ 0). The earlier
runaway-static observation was on a *different* recipe (balanced_norm with
planning_motion_loss_weights); the v2 line shows no such ego dominance. Weight evidence
corroborates without any eval: map gate σ pinned at 0.497 (ep1→ep3), delta_absmax
0.0662→0.0653 (stopped growing) while plan.ade keeps dropping (0.69→0.61→0.56) — the map
branch learned all it could and plateaued. An epoch_6 re-run would be confirmatory only.

---

## 11. v3 lane-anchor re-attachment — RESULT (epoch_4): map helps only as a weak prior

After confirming v2 query-space fusion was effectively dead, we tested the stronger
"Version A" path from §9d: attach map geometry directly at the planning output surface.
The implementation exposes ego-frame HD-map lane centerlines from `map_lane_encoder` as
`outs_map['lane_points']`, then lets `PlanningHeadSingleMode` build a lane-anchor
trajectory from those points.

**Implementation knobs added to `PlanningHeadSingleMode`:**

- `lane_anchor_mode`: `none` / `replace` / `residual` / `blend`
- `lane_anchor_sample_mode`: `whole_lane` / `local_forward`
- `lane_anchor_reference`: `absolute` / `relative_start`
- `lane_anchor_select_mode`: `first_point` / `closest_point` / `best_endpoint`
- `lane_anchor_direction_mode`: `forward` / `bidirectional`
- `lane_anchor_candidate_k`: top-K nearby lanes considered for best-endpoint selection

**Key design correction:** naive lane anchoring used the nearest lane (originally even
nearest by the first lane point) and treated map coordinates as the trajectory itself.
That is wrong for this port ODD: the lane centerline's lateral offset from ego is not
future ego motion. The useful variant samples a local forward segment from the lane
point nearest ego and subtracts that start point (`relative_start`), then blends the
result with the learned planner trajectory:

```python
lane_anchor = local_relative_centerline(best_lane)
sdc_traj = sdc_traj + alpha * (lane_anchor - sdc_traj)
```

For `best_endpoint`, we consider 16 nearby lanes, test both lane point directions, and
pick the anchor whose endpoint/shape best matches the base planner output. This is an
eval-only diagnostic: it asks "does the HD map contain a locally compatible lane prior?"
rather than claiming a fully learned route-selection module.

**epoch_4 comparison (checkpoint `base_e2e_lidar_plan_mapfuse_v2/epoch_4.pth`):**

| variant | avg.L2 | L2_1s | L2_2s | L2_3s | avg.Collision | pred final disp | conclusion |
|---|---:|---:|---:|---:|---:|---:|---|
| v2 map-OFF baseline | 0.617239 | 0.255437 | 0.590340 | 1.005941 | 0.009717 | 5.658 | baseline |
| v3 force query x5 | 5.019780 | — | — | — | 0.006421 | 8.894 | map query can move output, but destructively |
| whole-lane replace | 80.709780 | — | — | — | 0.084452 | 122.195 | absolute map coords are unusable as plan |
| local-relative replace | 4.302368 | 2.120846 | 4.292323 | 6.493934 | 0.048881 | 5.650 | scale fixed, lane choice still wrong |
| best-endpoint replace | 0.658930 | 0.276255 | 0.633247 | 1.067288 | 0.009483 | 5.601 | no longer catastrophic, but hard replacement hurts |
| best-endpoint blend α=0.1 | 0.611897 | 0.252660 | 0.585053 | 0.997976 | 0.009717 | 5.650 | weak map prior helps |
| best-endpoint blend α=0.2 | 0.609052 | 0.251073 | 0.582225 | 0.993858 | 0.009717 | 5.642 | better |
| **best-endpoint blend α=0.3** | **0.608917** | **0.250921** | **0.582260** | **0.993571** | **0.009717** | **5.635** | best tested |

**Bucket deltas for the best run (α=0.3 vs v2 map-OFF baseline where available):**

- `avg.L2`: 0.617239 → **0.608917** (−0.008322 / −1.35%)
- `L2_1s`: 0.255437 → **0.250921**
- `L2_2s`: 0.590340 → **0.582260**
- `L2_3s`: 1.005941 → **0.993571**
- `avg.Collision`: unchanged at **0.009717**
- `pred_final_disp`: 5.658 → **5.635**, closer to GT 5.326 but still long
- Improvements are concentrated in local geometry / obstacle-aware slices:
  `FrontObstacle/avg.L2=0.573544`, `Slow/avg.L2=0.455388`,
  `Turning/avg.L2=0.938360`.

**Interpretation:** the map is not useless; the previous connection was. Query residual
fusion was too weak to matter, and hard lane replacement is too strong. The viable
shape is a **small geometric prior blended into the learned planner**, after robust
multi-candidate lane selection and local-relative sampling. The gain is modest but real
on epoch_4: about **1.3% avg.L2** without worsening collision or detection/NDS.

**Current best eval config:**
`projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_bestendpoint_blend03_eval.py`

**Logs:**

- α=0.1:
  `projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_bestendpoint_blend01_eval/logs/eval.07091221`
- α=0.2:
  `projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_bestendpoint_blend02_eval/logs/eval.07091205`
- α=0.3:
  `projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_bestendpoint_blend03_eval/logs/eval.07091237`

**Next engineering step:** train this path instead of only eval-time blending. Use
α≈0.2–0.3 as an initialization/regularizer, but learn either (a) a per-frame confidence
gate or (b) a residual-to-anchor head. Static scenes still predict ~0.42m final
movement, so a speed/stop gate should be added before letting lane priors pull the
planner in low-motion frames.

---

## 12. A-route learned-blend training and eval plan

The current A-route training config is:

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train.py
```

It starts from `base_e2e_lidar_plan_mapfuse_v2/epoch_4.pth`, freezes the
pre-existing UniAD stack, and trains only:

- `planning_head.lane_anchor_gate_head`
- `planning_head.lane_anchor_residual_head`

Use the resulting checkpoints to close the A-route loop:

```bash
cd UniAD_train/UniAD
export TORCHRUN=~/anaconda3/envs/uniad_train/bin/torchrun
export PYTHONPATH=$(pwd):${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

CFG=projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train.py
WORK=projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_laneanchor_learnedblend_train

MASTER_PORT=28621 ./tools/uniad_dist_eval.sh $CFG $WORK/epoch_1.pth 5
MASTER_PORT=28622 ./tools/uniad_dist_eval.sh $CFG $WORK/epoch_2.pth 5
```

Compare against:

| baseline | avg.L2 | avg.Collision | note |
|---|---:|---:|---|
| v2 map-OFF baseline | 0.617239 | 0.009717 | no map benefit |
| fixed best-endpoint blend α=0.3 | 0.608917 | 0.009717 | current best non-learned A route |
| learned blend epoch_1 | TBD | TBD | run after checkpoint appears |
| learned blend epoch_2 | TBD | TBD | run after checkpoint appears |

After eval, inspect whether the learned map path is actually being used:

```bash
python tools/analysis_tools/diagnose_lane_anchor_gate.py \
  $CFG $WORK/epoch_2.pth \
  --out-dir $WORK/lane_anchor_gate_diag_epoch2 \
  --device cuda:0 \
  --max-samples 512
```

Read:

- `lane_anchor_gate_summary.csv`: gate/residual statistics by motion bucket.
- `lane_anchor_gate_samples.csv`: per-sample gate, residual, and anchor-vs-base
  displacement.

Success criteria for A are stricter than "training loss went down":

- `planning/avg.L2` should beat fixed α=0.3 (`0.608917`) or at least beat v2
  map-OFF (`0.617239`) without collision regression.
- `lane_anchor_gate` should not remain pinned at the initialization value
  (`~0.3`) for every bucket.
- Static/low-motion samples should have lower gate or small final delta from
  base plan.
- Residual magnitude should remain small enough to act as correction rather
  than a second uncontrolled planner.

## 13. B-route preparation: lightly unfreeze map fusion

B1 is prepared as:

```text
projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v3_b1_mapfusion_train.py
```

It keeps the A-route lane-anchor learned blend, but additionally unfreezes the
latent map-fusion modules:

- `planning_head.map_attn_module`
- `planning_head.map_delta_proj`
- `planning_head.map_gate`

The intent is to test whether map features can improve the planning latent
state once the output-space geometric prior has been stabilized. B1 should be
started after the A checkpoint is evaluated. If A underperforms fixed α=0.3,
B1 is the next least-invasive step before a full lane-candidate planner.

---

## 14. Reproduce

```bash
cd UniAD_train/UniAD
export PATH=~/anaconda3/envs/uniad_train/bin:$PATH
export TORCHRUN=~/anaconda3/envs/uniad_train/bin/torchrun
export CUDA_VISIBLE_DEVICES=0,2,3,4   # pick free/low-load cards
CKPT=projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2/epoch_6.pth

# map-ON
MASTER_PORT=28693 ./tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2.py $CKPT 4
# map-OFF (ablation)
MASTER_PORT=28694 ./tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v2_nomap_eval.py $CKPT 4
```

Planning metrics print near the end of each eval log
(`.../logs/eval.<timestamp>`), keys `planning/avg.L2`, `planning/avg.Collision`,
`planning/<Bucket>/avg.L2`. Weight inspection: load the `.pth` state_dict, check
`planning_head.map_gate.2.bias` (sigmoid) and `planning_head.map_delta_proj.weight`
(absmax).
