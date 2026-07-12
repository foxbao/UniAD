# HD-Map Fusion for LiDAR Planning — Analysis & Findings

**Status:** Updated 2026-07-12. Latent map-feature fusion remains ineffective, but
the C2.1 straight-through discrete lane-anchor path has now shown a measurable
causal map contribution and is the current planning champion on full validation.

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

For the later explicit lane-anchor routes, `use_map_lane=False` is not sufficient:
that switch only disables latent map attention, while the selector still reads
`outs_map['lane_points']`. Their strict map-OFF evaluation instead uses the
eval-only `ablate_lane_anchor=True` switch, which preserves all checkpoint modules
but skips lane-anchor construction and fusion in `forward`.

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
`projects/configs/stage2_e2e_lidar/eval/`

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

## 14. C0 oracle: explicit map-anchor upper bound

After A/B1/B2, the evidence is now:

- A learned lane-anchor blend is the current champion (`avg.L2 0.60495`).
- B1 pre-BEV map fusion has no measurable causal effect.
- B2 post-attention map fusion receives gradient, but its learned residual hurts
  the final trajectory (`avg.L2 0.6082`), and map-off returns exactly to A
  (`avg.L2 0.60495`).

The next route should therefore stop treating map as a weak hidden feature and
instead test explicit map geometry. `tools/analyze_map_anchor_oracle.py` is the
C0 diagnostic: it does not run the detector or train anything. It reads
`kl_infos_val.pkl`, transforms the surveyed map lanes into each ego frame using
`ego2global`, and asks how close a local lane-centerline anchor can get to GT
planning if the best lane/direction were selected by oracle.

Run:

```bash
cd UniAD_train/UniAD
/home/baojiali/anaconda3/envs/uniad_train/bin/python \
  tools/analyze_map_anchor_oracle.py
```

Full validation result:

| split | N | L2@1s | L2@2s | L2@3s | avg.L2 | anchor/GT final disp |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 4963 | 0.1637 | 0.2479 | 0.2536 | **0.2653** | 5.100 / 5.298 |
| Static | 811 | 0.0166 | 0.0220 | 0.0266 | 0.0318 | 0.062 / 0.072 |
| Slow | 1195 | 0.1905 | 0.3157 | 0.4342 | 0.3419 | 0.895 / 1.156 |
| MovingStraight | 2816 | 0.1915 | 0.2752 | 0.2284 | 0.2960 | 8.360 / 8.596 |
| Turning | 141 | 0.1947 | 0.3510 | 0.4745 | 0.3464 | 4.605 / 4.576 |

Compared with A epoch1 (`avg.L2 0.60495`), the oracle lane-anchor upper bound is
`-0.3397` avg.L2. This is a strong positive signal: the surveyed map geometry is
not useless. The failure is that the current planner cannot select/use the right
map anchor from BEV/query features. C1 should therefore make map-anchor selection
and conditioning explicit instead of adding another latent feature-fusion block.

### C1 learned hard selector result and diagnosis (2026-07-11)

C1 added a 16-candidate lane-anchor classifier, trained with the GT-best
candidate and teacher-forced anchor. Its epoch-1 validation result was:

| model | L2@1s | L2@2s | L2@3s | avg.L2 | avg.Collision |
|---|---:|---:|---:|---:|---:|
| A epoch1 | 0.2536 | 0.5735 | 0.9878 | **0.60495** | 0.9637% |
| C1 hard selector | 0.2611 | 0.5974 | 1.0304 | **0.62966** | 0.9483% |

C1 regressed avg.L2 by `+0.02471` (`+4.1%`) while collision stayed flat. The
new validation diagnostics pass GT only to the scoring path; prediction remains
GT-independent. Full-val selector results:

| split | N | top-1 acc | oracle score | predicted score | gap |
|---|---:|---:|---:|---:|---:|
| Overall | 4585 | 40.9% | 1.013 | 1.162 | 0.148 |
| Static | 772 | 15.0% | 0.203 | 0.492 | 0.289 |
| Slow | 1093 | 49.1% | 0.858 | 0.990 | 0.131 |
| MovingStraight | 2578 | 45.9% | 1.292 | 1.391 | 0.100 |
| Turning | 142 | 25.4% | 1.558 | 1.956 | 0.398 |

The score is the selector objective (`endpoint L2 + 0.25 * trajectory mean
L2`), not planning avg.L2. Static and turning have the largest selection gaps.
Two train/eval mismatches explain why training ADE looked good but validation
regressed:

1. Hard `argmax` blocks planning ADE gradients from reaching the selector; only
   cross-entropy trains it.
2. C1 training generated anchor distance samples from GT trajectory speed and
   selected lane direction against GT, while inference used the base planner.
   The selector therefore saw different candidate geometry at train and test.

C2 is implemented in
`base_e2e_lidar_plan_mapfuse_v4_c2_soft_selector_train.py`. It always generates
candidates from the frozen base trajectory, uses GT only to score supervision,
and replaces hard selection with a temperature-0.5 softmax mixture. This makes
the final planning loss differentiable with respect to selector logits and
removes teacher forcing. It starts from C1 epoch1 and trains selector, anchor
gate, and anchor residual heads for one epoch.

### C2 pre-training audit: current route is not yet effective (2026-07-11)

The first 5-GPU C2 launch did not train: a batch with no valid planning GT on
one rank omitted selector log keys, so MMDistributedDataParallel stopped at the
first iteration with `loss log variables are different across GPUs`. Selector
training now emits the same loss/stat keys on every rank, including zero-valued
keys for ranks with no valid target. A 20-sample, 5-GPU smoke run completed all
four iterations after the fix.

More importantly, the original C0 `avg.L2=0.2653` is not a realizable C2 upper
bound. C0 used GT future speed/distance and GT direction to construct its lane
anchor. C2 must construct candidates from the frozen base prediction. The
evaluator now reports true planning L2 for three deployable candidate outputs:

- `oracle anchor`: best candidate chosen with GT, but candidate geometry is
  generated only from the frozen base trajectory and map.
- `pred anchor`: hard candidate selected by the C1 selector.
- `selected anchor`: temperature-0.5 soft mixture used by C2.

Full-validation zero-shot evaluation loaded C1 epoch1 into the C2 structure,
without any C2 optimization:

| output | L2@1s | L2@2s | L2@3s | avg.L2 |
|---|---:|---:|---:|---:|
| realizable oracle anchor | 0.2555 | 0.5714 | 0.9396 | **0.58884** |
| selector hard anchor | 0.2814 | 0.6428 | 1.0809 | **0.66837** |
| selector soft anchor | 0.2766 | 0.6341 | 1.0639 | **0.65819** |
| C2 zero-shot final plan | 0.2595 | 0.5939 | 1.0234 | **0.62562** |

The C2 zero-shot final collision rate is `0.9477%`. For comparison, A is
`0.60495 / 0.9637%` and C1 is `0.62966 / 0.9483%` for avg.L2 / collision.
Soft selection recovers only `0.00404` avg.L2 from C1 and remains `0.02067`
(`3.4%`) worse than A. The realizable oracle is only `0.01611` (`2.7%`) better
than A, far smaller than the optimistic C0 gap.

The selector is also too uncertain for geometric averaging: overall entropy is
`2.342`, max probability is `0.148`, and the soft anchor is `0.06935` avg.L2
worse than the realizable oracle. Static and turning remain the weakest groups:

| split | oracle anchor | hard anchor | soft anchor |
|---|---:|---:|---:|
| Static | 0.125 | 0.309 | 0.269 |
| Slow | 0.520 | 0.591 | 0.580 |
| MovingStraight | 0.730 | 0.778 | 0.778 |
| Turning | 0.857 | 1.038 | 0.995 |

Decision: do not spend a full epoch on the current C2 unchanged. It has a
small candidate-level upper bound, but neither the learned selector nor the
soft mixture realizes it, and averaging multiple lanes can create a trajectory
that is not itself a lane. The next experiment should first improve candidate
construction/representation and selector supervision, then require a short
subset run to close a meaningful part of the `0.06935` oracle-selection gap.
Only after that should C2 be trained on the full set. A map-off causal ablation
remains mandatory for any trained successor.

### C2.1: straight-through discrete selector

C2.1 is implemented in
`base_e2e_lidar_plan_mapfuse_v4_c21_st_selector_train.py`. It changes only the
candidate selection relaxation:

- Forward train/test uses the selector's hard argmax, so the anchor is always
  one surveyed lane candidate rather than a coordinate average of lanes.
- During training, `hard - soft.detach() + soft` supplies a straight-through
  softmax gradient, allowing planning ADE to update selector logits.
- Candidate sampling still uses the frozen base trajectory and no GT geometry;
  GT is used only by the selector supervision score.
- C1 and C2 modes remain unchanged and their checkpoints remain reproducible.

A deterministic tensor test verified that train and eval forward outputs are
exactly the argmax candidate, while backpropagation produces nonzero selector
logit gradients. A 100-sample, 5-GPU smoke train then completed 20/20 iterations
without DDP mismatch, NaN, or process residue. At iteration 20, `loss_ade` was
`0.2416`, selector loss `1.4383`, and gradient norm `1.8742`. This validates the
implementation only; the tiny run is not evidence of planning improvement.

The promotion gate is a paired pilot: train C2.1 on a fixed 2000-sample train
subset, then compare C1 and C2.1 on the same 1500-sample validation subset. Do
not start full training unless C2.1 improves hard-anchor selection and final
planning L2 without a collision regression.

#### C2.1 pilot result

The deterministic pilot used the first 24 complete training scenes (`2091`
frames, `4.8%` of the full train set) and 419 optimizer iterations on five
GPUs. `make_scene_subset.py` creates the same temporal subset without breaking
scene queues. The first paired check used `kl_infos_val_sub1k_geo.pkl`: despite
its filename it contains 1500 raw frames and produced 316 valid planning
samples, heavily biased toward Static (`188/316`).

| 1500-frame subset | C1 | C2.1 pilot | delta |
|---|---:|---:|---:|
| planning avg.L2 | 0.36116 | **0.33361** | -7.63% |
| avg.Collision | 14.2577% | 14.2577% | 0 |
| selector top-1 | 21.8% | **59.2%** | +37.4 pp |
| hard-anchor avg.L2 | 0.34459 | **0.28604** | -17.0% |
| selector score gap | 0.190 | **0.096** | -49.4% |

Because that subset is biased, the pilot checkpoint was then evaluated on the
full validation set (`4585` valid selector samples):

| full validation | A | C1 | C2.1 pilot |
|---|---:|---:|---:|
| planning avg.L2 | **0.60495** | 0.62966 | 0.60656 |
| avg.Collision | 0.9637% | 0.9483% | 0.9557% |
| selector top-1 | - | 40.9% | **48.1%** |
| hard-anchor avg.L2 | - | 0.66837 | **0.65512** |
| selector score gap | - | 0.148 | **0.132** |

C2.1 improves C1 avg.L2 by `0.02310` (`3.67%`) and comes within `0.00161`
(`0.27%`) of A after seeing only 4.8% of the training set. Collision remains
flat. Static, MovingStraight, and Turning final L2 improve relative to C1;
Slow regresses (`0.49252 -> 0.51793`). The full-validation selector changes are:

| split | C1 acc | C2.1 acc | C1 gap | C2.1 gap |
|---|---:|---:|---:|---:|
| Overall | 40.9% | **48.1%** | 0.148 | **0.132** |
| Static | 15.0% | **62.2%** | 0.289 | **0.148** |
| Slow | **49.1%** | 42.7% | **0.131** | 0.209 |
| MovingStraight | 45.9% | **47.4%** | 0.100 | **0.084** |
| Turning | 25.4% | **26.1%** | 0.398 | **0.330** |

Decision: C2.1 passes the pilot gate and is worth one full training epoch. It
has not yet beaten A, so the full run is still an experiment, not the new
champion. Full-run acceptance requires avg.L2 below `0.60495`, no collision
regression, and no material Slow-bucket regression. If Slow remains worse, add
bucket-balanced selector supervision before changing candidate geometry again.

#### C2.1 full-training result and causal ablation (2026-07-12)

The promoted C2.1 model trained for one full epoch and completed its built-in
full validation normally. It passes all three acceptance criteria:

| model | L2@1s | L2@2s | L2@3s | avg.L2 | avg.Collision |
|---|---:|---:|---:|---:|---:|
| A epoch1 | 0.2536 | 0.5735 | 0.9878 | 0.60495 | 0.9637% |
| C1 hard selector | 0.2611 | 0.5974 | 1.0304 | 0.62966 | 0.9483% |
| C2.1 pilot | - | - | - | 0.60656 | 0.9557% |
| **C2.1 full** | **0.25124** | **0.56975** | **0.98261** | **0.60120** | **0.9637%** |

C2.1 full improves A by `0.00375` avg.L2 (`0.62%`) with effectively identical
collision, and improves C1 by `0.02846` (`4.52%`). The pilot's Slow regression
did not persist: Slow improved from `0.51793` in the pilot to `0.48401` in the
full run. Full-run motion buckets are Static `0.28121`, Slow `0.48401`,
MovingStraight `0.72017`, and Turning `0.89723`.

Selector top-1 accuracy is `48.1%`, the selector score gap is `0.11915`, and the
selected hard-anchor avg.L2 is `0.64982`. Accuracy did not rise beyond the pilot,
but the score gap and hard-anchor quality improved, and the learned gate/residual
converted that anchor into the best final planning result so far. Turning remains
the weakest selector bucket (`26.8%` accuracy, `0.348` score gap), so it is the
main remaining target for selector/candidate work.

The strict causal ablation evaluates the same C2.1 checkpoint with
`ablate_lane_anchor=True`; latent map fusion is already disabled by
`map_force_scale=0`. Model structure and loaded weights are otherwise identical:

| metric | map-ON | strict map-OFF | map benefit (OFF-ON) |
|---|---:|---:|---:|
| avg.L2 | **0.60120** | 0.61519 | **0.01399 (2.27%)** |
| avg.Collision | **0.9637%** | 0.9797% | 0.0160 pp |
| Static avg.L2 | **0.28121** | 0.29269 | 0.01148 |
| Slow avg.L2 | 0.48401 | **0.46086** | -0.02315 |
| MovingStraight avg.L2 | **0.72017** | 0.74560 | 0.02543 |
| Turning avg.L2 | **0.89723** | 0.98744 | 0.09021 |
| FrontClear avg.L2 | **0.62800** | 0.69137 | 0.06337 |
| FrontObstacle avg.L2 | 0.58970 | **0.58305** | -0.00665 |

This is the first route in this project with a clear same-checkpoint map
contribution. The gain is concentrated in MovingStraight, Turning, and
FrontClear scenes; Slow and FrontObstacle still prefer the base trajectory.
The next iteration should therefore keep the C2.1 discrete anchor path and make
its application more selective by motion/occupancy context, rather than replacing
it with another latent fusion block.

#### C2.1 gate diagnostic and C2.2 proposal (2026-07-12)

The first version of `tools/analysis_tools/diagnose_lane_anchor_gate.py` did not
apply `sdc_planning_mask`, so its Static statistics included invalid padded
samples. The tool was corrected before interpreting the diagnostic below. On
the first 512 validation frames, it retained 503 samples with valid planning GT
and additionally measured base/final/anchor L2 against the valid GT steps.

The final C2.1 trajectory improved over the base trajectory on only `41.7%` of
these samples. The current learned gate is not selecting those samples well:

| group | N | gate mean | final gain vs base | selector accuracy |
|---|---:|---:|---:|---:|
| map helped | 210 | 0.2179 | +0.1266 m | 53.8% |
| map hurt | 293 | 0.2330 | -0.0784 m | 29.7% |

Gate-to-gain correlation is `-0.164`, while selector max-probability-to-gain
correlation is only `-0.078`. The selector's confidence is therefore not yet a
safe inference threshold. Selector correctness, however, is strongly associated
with map utility, which supports improving selection and gate supervision
together rather than deleting the map path.

The next experiment is **C2.2 utility-supervised gate**:

1. Keep C2.1's discrete selector and candidate construction unchanged.
2. For each valid training sample, form the candidate endpoint after the current
   residual head, `map_traj`, and the base trajectory `base_traj`.
3. Compute the continuous gate target that minimizes squared trajectory error on
   the line `base_traj + alpha * (map_traj - base_traj)`, with `alpha` clamped to
   `[0, 1]`. Stop-gradient through this target.
4. Add an auxiliary MSE loss from the predicted `lane_anchor_gate` to this
   target. Initially train only the gate head from the C2.1 checkpoint, keeping
   selector and residual weights frozen; this isolates whether selective map use
   fixes the Slow/FrontObstacle regressions.

The first C2.2 pilot should use a small scene subset and compare the same full
validation protocol. Promotion requires `avg.L2 <= 0.60120`, no collision
regression, and no Slow regression. If gate-only C2.2 fails, retain C2.1 as the
baseline and then investigate selector ranking/Turning candidates separately.

C2.2 is implemented in
`base_e2e_lidar_plan_mapfuse_v4_c22_utility_gate_train.py`. The new utility loss
is disabled by default, so older configs and checkpoints are unchanged. The
training config loads C2.1 epoch1, disables the older static-gate and frozen
selector losses, and exposes only the four `lane_anchor_gate_head` parameter
tensors (`37,633` parameters) to the optimizer.

Unit tests recovered exact targets `0.0`, `0.5`, and `1.0` for synthetic base,
midpoint, and map GT trajectories; invalid masks were excluded. Backpropagation
updated the predicted gate while the detached target produced no residual/map
gradient. A 148-frame scene-complete, five-GPU smoke train completed 30/30
iterations without DDP mismatch, NaN, OOM, or process residue. Across iterations
10/20/30, utility-target valid rate was `0.90/0.94/0.90`, target mean
`0.469/0.344/0.369`, gate MAE `0.408/0.318/0.356`, and gradient norm
`2.64/1.96/1.82`. This validates the implementation and loss scale; it is not
an accuracy result. A strict smoke rerun also verified that only the four gate
parameter tensors changed from the C2.1 checkpoint; all other parameters and
buffers were bit-identical. The C2.2 config therefore enables
`freeze_except_eval=True`, which freezes BatchNorm/Dropout state without changing
the task heads' `self.training` control flow.

The first full pilot attempt produced `avg.L2=0.6122`, but it used the earlier
parameter-only freeze and allowed frozen BatchNorm state to update. Because its
selector diagnostics changed despite selector weights being frozen, that result
is invalid and is discarded. The strict-frozen-eval pilot is the authoritative
test.

#### C2.2 strict pilot result

The strict pilot used the same first 24 complete scenes and 2091 frames as the
C2.1 pilot. Checkpoint comparison confirmed that exactly four tensors changed,
all under `planning_head.lane_anchor_gate_head`; every other parameter and buffer
was bit-identical to C2.1 epoch1.

| full validation | C2.1 full | C2.2 strict pilot | delta |
|---|---:|---:|---:|
| L2@1s | 0.2512 | **0.2500** | -0.0012 |
| L2@2s | 0.5697 | **0.5680** | -0.0017 |
| L2@3s | 0.9826 | **0.9805** | -0.0021 |
| avg.L2 | 0.6012 | **0.5995** | -0.0017 (-0.28%) |
| avg.Collision | 0.9637% | 0.96% | effectively flat |
| Static avg.L2 | **0.2812** | 0.2814 | +0.0002 |
| Slow avg.L2 | **0.4840** | 0.4914 | +0.0074 |
| MovingStraight avg.L2 | 0.7202 | **0.7148** | -0.0054 |
| Turning avg.L2 | 0.8972 | **0.8854** | -0.0118 |
| FrontClear avg.L2 | 0.6280 | **0.6144** | -0.0136 |
| FrontObstacle avg.L2 | **0.5897** | 0.5930 | +0.0033 |

C2.2 improves every reported horizon and the global avg.L2, but fails the
pre-declared promotion gate because Slow and FrontObstacle regress. It should
not yet be trained on the full dataset.

On the same 503 valid diagnostic samples, C2.2 increased mean gate from `0.227`
to `0.277` and changed gate-to-final-gain correlation from `-0.164` to `+0.261`.
This confirms that direct utility supervision is useful. However, it raises map
use broadly: Slow gate `0.219 -> 0.277`, Moving `0.231 -> 0.285`, and Turning
`0.275 -> 0.340`. The small diagnostic subset shows local improvements, while
the 1093-sample full-validation Slow bucket regresses, exposing a distribution
and objective mismatch.

Recommended successor: keep strict gate-only isolation, but replace the current
six-step least-squares target with an evaluation-aligned target. Search alpha on
the base-to-map line using Euclidean errors at the actual 1s/2s/3s evaluation
steps, and default to alpha=0 when the best map blend does not beat base by a
minimum margin. Use a scene subset with adequate Slow/Turning representation
before considering full training.

#### C2.3 evaluation-aligned gate setup and smoke test (2026-07-12)

C2.3 implements that successor while retaining C2.1's selector, candidate,
residual, and inference path. For each valid sample it evaluates 21 uniformly
spaced blend coefficients on `[0, 1]` using mean Euclidean L2 at trajectory
indices 1/3/5 (the evaluator's 1s/2s/3s horizons). The supervised target is the
best coefficient only when it improves over the base trajectory by at least
`0.01 m`; otherwise the target is zero. The old least-squares target remains the
default, so C2.1/C2.2 and older configs are behaviorally unchanged.

The strict C2.3 config inherits C2.2's state-safe gate-only freeze and loads the
C2.1 epoch1 checkpoint. Synthetic tests recovered exact grid targets `0.0`,
`0.5`, and `1.0`, verified invalid-horizon masking, and confirmed that a
`0.005 m` gain is rejected by the `0.01 m` margin. The C2.2 least-squares compatibility
test also remains unchanged.

A 148-frame, two-scene, five-GPU smoke run completed 30/30 iterations without
DDP mismatch, NaN, OOM, or residual UniAD processes. At iterations 10/20/30,
target use rate was `0.54/0.46/0.42`, target mean `0.394/0.310/0.310`, gate MAE
was `0.329/0.291/0.303`, and gradient norm was `2.33/1.69/1.82`. Checkpoint
comparison found exactly four changed tensors, all in
`planning_head.lane_anchor_gate_head`; the other 1697 parameters and buffers
were bit-identical to C2.1 epoch1.

The full training set contains 43,981 samples across 584 scenes, including
6,567 Slow and 1,458 Turning samples. To avoid repeating C2.2's scene-distribution
mismatch, `make_planning_bucket_scene_subset.py` builds deterministic,
scene-complete subsets with planning-bucket quotas. The C2.3 pilot subset has 41
scenes and 2,010 samples: 258 Static, 428 Slow, 1,105 MovingStraight, 156
Turning, and 63 without valid planning GT. The next decision is based on this
pilot followed by the unchanged full validation set; the smoke run is only an
implementation check.

#### C2.3 balanced pilot result

The 2,010-frame balanced pilot completed 402/402 iterations and the unchanged
4,676-frame full validation. Checkpoint comparison again found exactly four
changed gate-head tensors; all other 1,697 parameters and buffers were
bit-identical to C2.1 epoch1.

| full validation | C2.1 full | C2.2 strict | C2.3 balanced | C2.3 vs C2.1 |
|---|---:|---:|---:|---:|
| L2@1s | 0.2512 | 0.2500 | **0.2443** | -0.0069 |
| L2@2s | 0.5697 | 0.5680 | **0.5588** | -0.0109 |
| L2@3s | 0.9826 | 0.9805 | **0.9673** | -0.0153 |
| avg.L2 | 0.6012 | 0.5995 | **0.5901** | -0.0111 (-1.85%) |
| avg.Collision | 0.9637% | 0.96% | **0.96%** | effectively flat |
| Static avg.L2 | 0.2812 | 0.2814 | **0.2803** | -0.0009 |
| Slow avg.L2 | 0.4840 | 0.4914 | **0.4838** | -0.0002 |
| MovingStraight avg.L2 | 0.7202 | 0.7148 | **0.7023** | -0.0179 |
| Turning avg.L2 | 0.8972 | 0.8854 | **0.8782** | -0.0190 |
| FrontClear avg.L2 | 0.6280 | 0.6144 | **0.6006** | -0.0274 |
| FrontObstacle avg.L2 | 0.5897 | 0.5930 | **0.5854** | -0.0043 |

C2.3 passes every pre-declared promotion condition: global L2 improves,
collision remains flat, and neither Slow nor FrontObstacle regresses. It also
retains and strengthens the MovingStraight, Turning, and FrontClear gains. This
is a clean objective-alignment result because the architecture and inference
path are unchanged from C2.1 and only the gate supervision and pilot sampling
changed.

On the same first 512 validation frames used for the earlier diagnostics, 503
samples had valid planning GT. C2.3's mean gate is `0.238`, lower than C2.2's
`0.277`. Samples where the final map path helps have mean gate `0.251`, compared
with `0.229` where it hurts; gate-to-gain correlation is positive at `+0.148`.
Bucket gate means are Static `0.196`, Slow `0.244`, Moving `0.237`, and Turning
`0.337`. C2.3 therefore fixes C2.2's broad gate inflation while preserving
stronger map use for turns.

Decision: promote C2.3 as the current map-planning baseline. The next controlled
experiment is a full-dataset gate-only run from the same C2.1 checkpoint, using
the evaluation-aligned target. Evaluate each epoch and retain the best full
validation checkpoint; do not unfreeze the selector or residual path until that
run confirms the balanced-pilot gain at full training scale.

An earlier attempted ablation using only `use_map_lane=False` produced values
identical to map-ON and is invalid: the explicit selector continued consuming
`outs_map['lane_points']`. Those numbers must not be used as evidence that C2.1
ignores the map.

Reproduce the zero-shot audit with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 MASTER_PORT=28763 \
  ./tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v4_c2_soft_selector_zeroshot_eval.py \
  projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v4_c1_lane_selector_train/epoch_1.pth \
  5
```

Recommended C1 direction:

- Generate or supervise a discrete lane-anchor target from GT future endpoint /
  short-horizon trajectory.
- Let the planner predict/select among nearby candidate lane anchors.
- Condition trajectory regression on the selected anchor, with A as the fallback
  base trajectory.
- Keep C0 as the gate: if future C1 cannot beat A, inspect anchor selection
  accuracy before tuning trajectory regression.

---

## 15. Reproduce

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
  projects/configs/stage2_e2e_lidar/eval/ $CKPT 4
```

Planning metrics print near the end of each eval log
(`.../logs/eval.<timestamp>`), keys `planning/avg.L2`, `planning/avg.Collision`,
`planning/<Bucket>/avg.L2`. Weight inspection: load the `.pth` state_dict, check
`planning_head.map_gate.2.bias` (sigmoid) and `planning_head.map_delta_proj.weight`
(absmax).
