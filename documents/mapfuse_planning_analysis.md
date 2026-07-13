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

The formal promoted reference remains **D2 medium** because it satisfies the
motion-bucket gate:

```text
projects/work_dirs/stage2_e2e_lidar/
  base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_medium_train/epoch_1.pth
```

The completed 43,981-frame **D2 full** checkpoint remains the strict safety
reference:

```text
projects/work_dirs/stage2_e2e_lidar/
  base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train/epoch_1.pth
```

It reaches `avg.L2=0.56905` and `avg.Collision=0.9563%`. The D2.1 10k control
improves global L2 further to `0.54019` and fixes the Turning regression, but
collision rises to `1.003%`. Candidate-off exactly reproduces C2.3, proving the
accuracy gain is causal map value rather than fallback drift.

Do not run D2.1 full as-is. **D3-A is now implemented**: it uses D2.1 for a
Top-16 shortlist, appends the exact fallback, adds explicit online-agent safety
features, and trains a set-aware collision-aware reranker. The next experiment
is the scene-complete 10k D3-A medium run. Occupancy features remain D3-B work,
not part of the current implementation.

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
Top-16 map shortlist + exact fallback
        |
        +---- online MotionFormer vehicle futures
        |       -> center distance / AABB clearance / risk
        v
D3-A set Transformer
        -> bounded horizon-cost delta + collision logits
        -> selected raw candidate
```

Important invariants:

- The fallback residual is hard-zero, so selecting fallback is byte-for-byte
  C2.3 behavior.
- D2 has no separate utility threshold. Map candidates and fallback are scored
  on one calibrated cost scale.
- D2 full freezes all inherited modules and trains only the 69,635-parameter
  `candidate_cost_head`.
- D3-A freezes D2.1 and trains about 1.19M new set-reranker parameters. It uses
  raw candidates because D2.1 residual refinement added two collision events.
- D3-A inference safety features use only online MotionFormer predictions;
  future GT boxes are supervision only.
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
| D2 43,981-frame full | **`0.56905`** | **`0.9563%`** | safety best; Turning gate fails |
| D2.1 10,238-frame joint representation | **`0.54019`** | `1.003%` | accuracy best; collision gate fails |

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

### 4.3 D2 full buckets

| split | D2 medium | D2 full | full - medium |
|---|---:|---:|---:|
| Static | **0.26078** | 0.26848 | +0.00770 |
| Slow | 0.48338 | **0.46428** | -0.01910 |
| MovingStraight | 0.67745 | **0.67588** | -0.00157 |
| Turning | **0.88123** | 0.89471 | +0.01348 |
| FrontClear | about 0.5990 | **0.59772** | about -0.0013 |
| FrontObstacle | about 0.5615 | **0.55674** | about -0.0048 |

D2 full selects map on `14.86%` of validation frames, down from medium's
`17.88%`. The global gain is real, but the Turning regression and lower map
use motivate jointly adapting the frozen candidate representation rather than
training another scalar gate.

### 4.4 D2.1 controlled result

| split | D2 full | D2.1 | D2.1 - D2 full |
|---|---:|---:|---:|
| Static | 0.26848 | **0.19888** | -0.06960 |
| Slow | **0.46428** | 0.46687 | +0.00259 |
| MovingStraight | 0.67588 | **0.64487** | -0.03101 |
| Turning | 0.89471 | **0.87864** | -0.01607 |
| FrontClear | 0.59772 | **0.56650** | -0.03122 |
| FrontObstacle | 0.55674 | **0.52894** | -0.02780 |

D2.1 selects map on `58.23%` of validation frames. Its candidate-off control
returns exactly to `avg.L2=0.59011`, `avg.Collision=0.9637%`, and zero map
selection. The normal-vs-candidate-off gap therefore proves substantial causal
map value.

The paired collision audit covers `12,972` valid frame-horizon events:

| trajectory | avg.L2 | avg.Collision | collision events |
|---|---:|---:|---:|
| D2 full | 0.56905 | 0.9563% | 124 |
| exact fallback | 0.59011 | 0.9637% | 125 |
| D2.1 selected raw candidate | 0.55852 | 0.9878% | 128 |
| D2.1 selected refined candidate | **0.54022** | 1.0032% | 130 |

Relative to D2 full, D2.1 refined adds six collision events and removes none.
Five events are in four consecutive Static frames from scene
`20251106_007/202511061345_record`; one is a Slow-frame 3-second event from
`20260311_007/20260311165726_record`. Residual refinement accounts for two net
events, while raw candidate selection still accounts for four net events.

Neither refined nor raw cost-margin fallback passes the promotion contract.
For example, raw with a `0.03 m` margin reaches `avg.L2=0.55358` but collision
remains `0.9797%`. At `0.125 m`, collision only returns to fallback's
`0.9637%`, while L2 regresses to `0.57458`. A scalar threshold cannot reproduce
D2 full's candidate-specific collision avoidance.

### 4.5 Candidate coverage diagnosis

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
| D2 | Shared horizon-cost prediction works after enough scene-complete data. Full data improves global L2 but exposes Turning/representation limits. | Keep medium as formal reference; use full as D2.1 initialization. |

The compact lesson is:

> The map was not intrinsically useless. Weak hidden fusion could not expose its
> value; explicit candidates plus calibrated selection can.

## 6. Completed experiment: D2 full

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

The built-in full validation completed after training. A dedicated audit eval
also completed and produced the 4,676-frame Top-16-plus-fallback result file.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 MASTER_PORT=28820 \
  ./tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_eval.py \
  projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train/epoch_1.pth \
  5
```

Decision at the end of D2: retain D2 medium as the formal promoted reference
and use D2 full as the D2.1 initialization. After D2.1, D2 full remains the
strict collision reference while D2.1 is the better representation and global
L2 initialization.

## 7. Completed D2.1 and implemented D3-A

D2.1 unfreezes only:

- `candidate_encoder`, `source_embed`;
- `map_attention`, `attention_norm`;
- `ffn`, `ffn_norm`;
- `residual_head`;
- `candidate_cost_head`.

This raises trainable parameters from `69,635` to `735,759` (roughly `1.10%`
of the model). The LiDAR/tracking/motion/occupancy stack, base planning
trajectory, and map candidate generator remain frozen. Candidate-cost mode
still hard-zeros fallback residuals, so selecting fallback remains byte-exact
D2/C2.3 behavior.

The completed implementation and audit use:

- full config:
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_train.py`;
- 10k scene-complete control:
  `projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_medium_train.py`;
- normal and fallback-only evaluation:
  `projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d21_joint_repr_eval.py`
  and
  `projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d21_candidateoff_eval.py`;
- Top-16 collision audit:
  `projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d21_collision_audit_eval.py`
  and `tools/analysis_tools/analyze_d21_collision_audit.py`.

Decision: retain the D2.1 checkpoint as the best representation/accuracy
initialization, but do not promote it and do not run the 43,981-frame D2.1
schedule unchanged. The strict collision gate fails, and both residual-off and
scalar-margin controls show that further D2.1 tuning is unlikely to address the
missing safety relation.

D3 is now the default architecture replacement. The implemented D3-A stage:

1. Use D2.1 to shortlist approximately 8-16 map candidates.
2. Append the exact C2.3 fallback unconditionally.
3. Encodes candidate-to-online-agent center distance, heading-aware AABB
   clearance, and confidence-weighted risk at 1s/2s/3s plus aggregate values.
4. Jointly compares the 17-member set with a two-layer Transformer.
5. Predicts bounded horizon-cost corrections and collision logits, supervised
   by GT future boxes while keeping inference GT-free.
6. Selects raw candidates and keeps fallback byte-exact.

Implementation/configs:

- `projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_train.py`;
- `projects/configs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_medium_train.py`;
- `projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v7_d3a_set_reranker_eval.py`;
- `projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v7_d3a_candidateoff_eval.py`.

Completed engineering gates on 2026-07-13:

- D2.1 checkpoint load: 42 missing keys, all from the new D3-A modules; zero
  unexpected keys;
- trainable parameters: `1,193,222`;
- five-GPU scene-complete smoke: 30/30 iterations, no cross-rank log-key
  mismatch, finite losses and gradients;
- smoke inference: 93 evaluated planning frames, set size `17.0`, online actor
  signal present in `98.9%` of frames;
- smoke `avg.L2=0.7607`, collision `0%`; this is only an interface check after
  30 updates and is not model-quality evidence.

The next command is the one-epoch 10k medium run. Promotion is decided only
after full-validation evaluation of its checkpoint against D2 full, D2.1,
fallback-only, and candidate-off controls.

D3 addresses two measured D2 limitations:

- independent minimum selection over roughly 1,000 noisy costs;
- no explicit relative reasoning between candidates, agents, occupancy, and
  route constraints.

Required D3 controls are: no-set-reranker, candidate-off, map-off, exact
fallback-only, top-K recall, selected-vs-fallback oracle, selected collision
probability calibration, and actor-signal coverage. Occupancy-conditioned
features are deferred to D3-B after D3-A establishes a measurable gain.

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

The P0 code path and real 200-frame balanced pilot are complete. Raw Qwen
selection improves balanced-set L2 from D2 `0.5971` and fallback `0.5927` to
`0.5824` with unchanged 3s collision, but it over-selects map (`84%`) and hurts
Slow. An online D2 predicted-cost-delta gate at an exploratory `+0.125 m`
reduces map use to `49.5%`, retains `0.5826` L2, and limits every motion-bucket
regression to below `0.01 m` on this same pilot. This threshold is diagnostic,
not promoted; it needs a separate holdout and shuffled/equal-input controls.
The exact protocol and commands are in
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
- keep exploratory P0 thresholds labeled as same-set diagnostics until a
  separate holdout confirms them.
