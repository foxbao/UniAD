# HD-map planning: current status and roadmap

> **Authoritative current document. Updated 2026-07-13.** Read this file first.
> Detailed chronological evidence is preserved in
> [`archive/mapfuse_planning_experiment_history.md`](archive/mapfuse_planning_experiment_history.md).
> LLM/VLA research is in
> [`llm_assisted_map_planning_research.md`](llm_assisted_map_planning_research.md).

## 1. Current verdict

The surveyed HD map is useful, but the original latent query fusion was almost
a no-op. The successful direction is to use map topology to construct explicit
multimodal trajectory candidates and learn when each candidate is better than
the exact C2.3 fallback.

The current promoted checkpoint is **D2 medium**:

```text
projects/work_dirs/stage2_e2e_lidar/
  base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_medium_train/epoch_1.pth
```

It improves `avg.L2` from the C2.3 fallback's `0.59011` to `0.57274` with the
same `0.9637%` average collision rate. The independent 43,981-frame D2 full run
is in progress. D2 medium remains the reference until full-data evaluation
proves at least comparable calibration.

The next core architecture is **D3**: D2 top-K shortlist, exact fallback,
agent/occupancy/map-risk features, and a set-aware joint reranker. An LLM
Planning-IR audit may run in parallel, but it does not replace D3. Diffusion or
a full VLA starts only after we prove proposal coverage, rather than ranking,
is the dominant bottleneck.

## 2. Evaluation contract

All promotion decisions use the unchanged full validation evaluator:

- planner horizon: 6 steps / 3 seconds;
- primary metric: mean L2 at evaluator horizons 1s, 2s, and 3s;
- safety metric: average collision rate;
- required buckets: Static, Slow, MovingStraight, Turning, FrontClear, and
  FrontObstacle;
- runtime evaluation queue: 4,676 frames with valid temporal context;
- oracle-only map analysis may use 4,963 frames with valid planning GT and is
  not directly comparable as a deployable model result.

A map route is promoted only when it has:

1. lower global `avg.L2` than its declared fallback/reference;
2. no collision regression;
3. no motion-bucket regression above `0.01 m`;
4. non-zero, useful map selection;
5. a causal candidate-off or map-off degradation;
6. no GT-dependent input in the inference path.

Training loss and map-selection rate alone are not evidence of map value.

## 3. Current architecture

```text
surveyed HD-map topology
        |
        v
lane chains x 8 speed profiles x 9 lateral offsets
        |
        v
roughly 1,000 deployable map candidates + exact C2.3 fallback
        |
        v
D1.1 candidate representation and bounded map residual
        |
        v
D2 shared 1s/2s/3s candidate-cost head
        |
        v
minimum predicted mean cost -> selected trajectory
```

Important invariants:

- The fallback residual is hard-zero, so selecting fallback is byte-for-byte
  C2.3 behavior.
- D2 has no separate utility threshold. Map candidates and fallback are scored
  on one calibrated cost scale.
- D2 full freezes all inherited modules and trains only the 69,635-parameter
  `candidate_cost_head`.
- Runtime candidates use only surveyed map topology, ego pose, and
  training-derived speed profiles. They do not use per-sample GT speed.

## 4. Best evidence

### 4.1 Main comparison

| system | avg.L2 | avg.Collision | status |
|---|---:|---:|---|
| v2 latent map fusion | map-ON/OFF delta about `1e-6` | unchanged | rejected |
| A learned lane blend | `0.60495` | `0.9637%` | real but limited gain |
| C2.3 exact fallback | `0.59011` | `0.9637%` | safety/reference baseline |
| D0 deployable candidate oracle | `0.3282` | oracle only | strong coverage upper bound |
| D2 2,010-frame pilot | `0.5952` | `0.9637%` | rejected: miscalibrated |
| **D2 10,238-frame medium** | **`0.57274`** | **`0.9637%`** | **promoted** |
| D2 43,981-frame full | pending | pending | training/eval required |

### 4.2 D2 medium buckets

| split | C2.3 fallback | D2 medium | delta |
|---|---:|---:|---:|
| Static | 0.28027 | **0.26078** | -0.01949 |
| Slow | 0.48379 | **0.48338** | -0.00041 |
| MovingStraight | 0.70230 | **0.67745** | -0.02485 |
| Turning | **0.87815** | 0.88123 | +0.00308 |
| FrontClear | 0.6006 | **0.5990** | about -0.0016 |
| FrontObstacle | 0.5854 | **0.5615** | about -0.0239 |

D2 medium selects map on `17.88%` of validation frames. Among 822 valid map
selections, 566 (`68.9%`) beat fallback. The selected-map actual mean gain is
`0.09546 m`; the predicted mean gain is `0.09816 m`, leaving only `0.00270 m`
calibration error. Its selected-trajectory-versus-fallback oracle is `0.56385`,
so only about `0.00889 m` remains on that exact selected set.

### 4.3 Candidate coverage diagnosis

D0 combines up to 16 topology chains, 8 training-derived speed profiles, and 9
lateral offsets. It covers all 4,963 valid-GT validation frames:

| split | deployable oracle avg.L2 |
|---|---:|
| ALL | **0.3282** |
| Static | 0.0455 |
| Slow | 0.3636 |
| MovingStraight | 0.3900 |
| Turning | 0.3186 |

The exact-speed geometry diagnostic is `0.1105`. Therefore the current main
bottleneck is candidate ranking, speed-profile choice, and context-aware
selection, not missing map geometry.

## 5. What has been established

| route | finding | decision |
|---|---|---|
| v1/v2 latent query fusion | Branch weights can learn, but mature map-ON/OFF output is effectively identical. Ego-status ablation does not revive map value. | Do not return to single-query residual fusion. |
| A output-space lane blend | A carefully selected local-relative lane prior gives a modest real gain; hard replacement is destructive. | Keep only as fallback ancestry, not final architecture. |
| B pre/post-BEV map fusion | Pre-BEV has no causal effect; post-BEV receives gradients but hurts the final plan. | Rejected. |
| C explicit lane selector/gates | Oracle geometry is strong, but hard/soft selection and post-hoc binary/continuous utility gates are unstable or misallocated. | Stop adding scalar gates. |
| C2.3 balanced gate | Provides a stable exact fallback and `0.59011` reference. | Retain as safety baseline. |
| D0 candidate oracle | Explicit topology/speed/lateral candidates have large headroom. | Candidate route justified. |
| D1/D1.1/D1.2 | Ranking signal exists, but fallback shortcuts and separate utility calibration prevent useful deployment. | Unified calibration required. |
| D2 | Shared horizon-cost prediction works after enough scene-complete data. | Current promoted route. |

The compact lesson is:

> The map was not intrinsically useless. Weak hidden fusion could not expose its
> value; explicit candidates plus calibrated selection can.

## 6. Active experiment: D2 full

Configuration:

```text
projects/configs/stage2_e2e_lidar/
  base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train.py
```

Training data and schedule:

- 43,981 training frames;
- one epoch / 8,797 five-GPU iterations;
- starts from the D1.1 pilot checkpoint;
- trains only `planning_head.map_multimodal_planner.candidate_cost_head`;
- `AdamW`, base learning rate `5e-4`.

Launch command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 MASTER_PORT=28598 \
  ./tools/uniad_dist_train.sh \
  projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train.py \
  5
```

Evaluation after `epoch_1.pth` appears:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 MASTER_PORT=28820 \
  ./tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_eval.py \
  projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train/epoch_1.pth \
  5
```

Decision after evaluation:

- Promote full if it is at least comparable to D2 medium globally and by
  bucket, while preserving collision and causal map use.
- Otherwise retain D2 medium. Full-data degradation would indicate distribution
  imbalance or overfitting of independent candidate costs, not failure of the
  entire D route.

## 7. Next architecture: D3

D3 is the default next implementation after D2 full evaluation:

1. Use D2 to shortlist approximately 8-16 map candidates.
2. Append the exact C2.3 fallback unconditionally.
3. Encode explicit candidate-to-agent, candidate-to-occupancy, route,
   boundary, speed-limit, and map-risk features.
4. Jointly compare the set with a set-aware Transformer.
5. Predict final cost and optionally a bounded residual.

D3 addresses two measured D2 limitations:

- independent minimum selection over roughly 1,000 noisy costs;
- no explicit relative reasoning between candidates, agents, occupancy, and
  route constraints.

Required D3 controls are: no-set-reranker, candidate-off, map-off, exact
fallback-only, top-K recall, and selected-vs-fallback oracle.

## 8. Parallel research routes

### 8.1 LLM Planning IR

The recommended LLM route is not direct XY generation. A teacher or occasional
slow reasoner emits a validated Planning IR such as maneuver, route branch,
yield target, speed envelope, forbidden zone, and confidence. A compact
frame-rate student conditions D3.

Only the no-training semantic-value audit (`P0`) should run before D3 proves its
baseline. Promotion requires scene-specific gain over both D2 and an equal-input
non-LLM graph/set model, especially on Turning and port long-tail cases. See
[`llm_assisted_map_planning_research.md`](llm_assisted_map_planning_research.md).

The P0 code path is now implemented: a default-off D2 audit payload, exact
candidate-factor serializer, strict Planning IR validator, local Qwen teacher,
and offline reselection evaluator. Real audit inference is pending an available
GPU/checkpoint window. The exact protocol and commands are in
[`planning_ir_schema.md`](planning_ir_schema.md).

### 8.2 Diffusion/VLA/world model

The preferred radical variant is hybrid:

```text
topology path shortlist
  -> flow/diffusion speed and lateral residual proposals
  -> D3 set-aware reranker
```

Start this only if D3 shows that top-K proposal coverage, rather than ranking,
is the remaining bottleneck, or if research novelty is prioritized over the
shortest deployment path.

## 9. Update rules

Keep this file short and current:

- update only promoted results, active experiment, current diagnosis, and next
  decision;
- move per-iteration logs, abandoned hypotheses, and complete command history
  to the archive;
- never replace a reference checkpoint until full validation and causal
  ablations pass;
- label oracle, pilot, medium, full, and deployable results explicitly;
- after D2 full evaluation, update Sections 1, 4, 6, and 7 in one change.
