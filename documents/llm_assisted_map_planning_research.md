# LLM-assisted HD-map planning research

> Research date: 2026-07-13. Scope: the current LiDAR UniAD + surveyed HD-map
> planning branch, especially the post-D2/D3 roadmap. This document separates
> published evidence, very recent preprint signals, and project-specific
> engineering hypotheses.

## 1. Executive conclusion

An LLM can add value, but **replacing the trajectory head with an LLM that
prints XY coordinates is not the recommended route**. The stronger and more
deployable pattern in the literature is hierarchical:

1. A language or vision-language model handles semantic scene understanding,
   high-level maneuver decisions, rule interpretation, and rare-event
   reasoning.
2. A conventional differentiable planner produces precise continuous
   trajectories.
3. A verifier or fallback path enforces geometric and safety constraints.

For this project, the most promising paradigm change is therefore:

> Turn the HD map from a geometric feature tensor into a **semantic task graph**,
> let an LLM compile map topology, traffic rules, port task context, and actor
> interactions into a machine-checkable **Planning IR**, then let a fast D3
> candidate reranker and continuous refiner execute that IR.

The LLM should initially be an **offline teacher and occasional slow planner**.
The frame-rate path should remain a small deterministic student plus D3. This
preserves the exact fallback and avoids making LLM latency or hallucination part
of every control cycle.

This does not replace the current D2 evaluation or D3 plan. It adds a semantic
reasoning channel to D3. A more radical AnchorVLA/flow/diffusion planner is a
later route, after we prove that semantic intent changes candidate selection.

## 2. Why this fits the current project

The current measurements already tell us where extra intelligence is needed:

| Fact | Current evidence | Consequence |
|---|---:|---|
| Deployable map-candidate oracle | `avg.L2=0.3282` | Candidate coverage is already strong. |
| Exact-speed geometry oracle | `avg.L2=0.1105` | Geometry is not the main bottleneck. |
| D2 medium deployed result | `avg.L2=0.5727` | Learned calibration helps, but leaves a large ranking gap. |
| D2 medium map selection | `17.88%` | The map path is useful but selective. |
| D2 medium Turning | `0.8812` vs fallback `0.8782` | High-level branch/intent reasoning remains weak. |
| D2 selected-vs-fallback oracle | `0.56385` | Better calibration alone has only about `0.009 m` left on the same selected set. |

D0 already factorizes each candidate into lane chain, speed profile, and
lateral offset. That is almost an action vocabulary. The missing parts are not
more raw trajectories, but stronger answers to questions such as:

- Which topological branch matches the route and current task?
- Should the vehicle stop, yield, reverse, overtake, or enter a work zone?
- Which actor has priority, and for how long should the decision persist?
- Which candidates violate a temporary rule or semantic zone that geometry
  alone does not encode?

The repository also already contains useful building blocks:

- `HDMapParser` preserves lane IDs, predecessors/successors, neighbors,
  direction, turn type, speed limit, and boundaries.
- D0/D1/D2 provide candidate generation, fallback preservation, candidate
  representations, cost supervision, and diagnostic metrics.
- The earlier `LLMBridgeHead` work projects LiDAR object queries into a small
  Qwen model and has a VLM-teacher/caption pipeline. Its spatial QA result shows
  that same-frame queries contain some usable semantics, although exact count
  and operation-state reasoning remain weak.
- Qwen3-VL teacher models and Qwen3 student environments are already available
  locally, reducing the cost of an offline pilot.

One hard limitation must remain explicit: if the input contains only the same
map geometry, actor boxes, and three-way command already available to D3, an
LLM cannot manufacture missing task intent. The larger gain requires at least
one genuinely new semantic source: port job instructions, map-zone/rule
metadata, camera-derived operation state, traffic signals/gestures, or episodic
memory from rare scenarios.

## 3. What existing systems actually do

### 3.1 Mature and relatively established patterns

| Pattern | Representative work | How language is connected | Evidence and lesson |
|---|---|---|---|
| Direct trajectory text | [GPT-Driver](https://arxiv.org/abs/2310.01415), [EMMA](https://arxiv.org/abs/2410.23262) | Serialize scene and coordinates as tokens; autoregressively generate waypoints. | Demonstrates feasibility and unified multitask training, mainly on open-loop benchmarks. Numerical precision, token length, latency, and closed-loop validation remain concerns. |
| High-level action + precise E2E planner | [Senna](https://arxiv.org/abs/2410.22313) | VLM predicts discrete speed/path meta-actions; a separate E2E model predicts trajectory. | The paper explicitly rejects LVLMs as precise numerical planners. With large-scale pretraining, it reports `27.12%` lower planning error and `33.33%` lower collision than its no-pretraining counterpart. This is the closest mature design principle to our proposed IR. |
| Hybrid VLM + traditional stack | [DriveVLM](https://arxiv.org/abs/2402.12289) | VLM performs description, analysis, and hierarchical planning; DriveVLM-Dual fuses 3D perception and a conventional planner. | The authors use the dual system specifically because VLM spatial reasoning and runtime are weak. It was also deployed on a production vehicle. |
| Agent with tools and reflection | [Agent-Driver](https://arxiv.org/abs/2311.10813), [PlanAgent](https://arxiv.org/abs/2406.01587) | The LLM calls perception/map/collision tools, retrieves memory, reasons, generates a planner configuration or code, then reflects on simulated results. | Tool grounding and verification are more credible than free-form trajectory text. PlanAgent consumes a BEV map plus lane-graph text, but reports about `5 s` for a GPT-4V call, so it is a slow-path design rather than a frame-rate controller. |
| Asynchronous guidance | [AsyncDriver](https://arxiv.org/abs/2406.14556) | Vectorized scene and route instructions enter an LLM; latent instruction features are injected into a real-time planner at a lower rate. | Calling the LLM every three frames reduces inference cost by about `40%` with about `1%` performance loss in the paper. This directly motivates cached slow guidance. |
| Offline LLM distillation | [DiMA](https://arxiv.org/abs/2501.09757) | An MLLM and vision planner share structured representations and surrogate tasks; the LLM is optional at inference. | Reports `37%` lower trajectory L2, `80%` lower collision, and `44%` lower long-tail trajectory error for the distilled vision planner. This is strong evidence for teacher-only use, while remembering that the numbers are relative to its own baseline and benchmark protocol. |
| BEV features into an LLM | [BEVDriver](https://arxiv.org/abs/2503.03074) | Camera/LiDAR BEV features pass through a Q-Former into an LLM that predicts trajectories under language navigation. | Reports up to `18.9%` higher CARLA LangAuto Driving Score than prior methods. This proves a practical BEV-to-LLM interface, but still relies on simulator closed-loop evidence. |

Other directly relevant interfaces are
[Driving with LLMs](https://arxiv.org/abs/2310.01957), which projects object-level
numeric vectors into an LLM, and
[Talk2BEV](https://arxiv.org/abs/2310.02251), which turns a structured BEV map
into a language-queryable scene representation. They support our existing
query-projector approach, but neither by itself proves a planning gain on this
port dataset.

### 3.2 2026 frontier signals

The following papers are useful architectural signals, but most were submitted
only weeks or days before this research date and should not be treated as
settled evidence:

| Direction | Representative work | Signal for us |
|---|---|---|
| Decision anchors + continuous residual | [AnchorVLA](https://arxiv.org/abs/2607.03182) | Uses compact trajectory-pattern anchor tokens for high-level decisions, then residual flow for precise trajectories. It reports `77.28` closed-loop Success Rate and `89.92` Driving Score on Bench2Drive. D0 lane/speed/lateral factors are a natural local anchor vocabulary. |
| Coupled semantics and flow planning | [VECTOR-Drive](https://arxiv.org/abs/2605.08830) | Shared attention couples language and trajectory tokens, while separate experts avoid task interference; a flow-matching action head keeps outputs continuous. |
| World tokens + diffusion | [CoWorld-VLA](https://arxiv.org/abs/2605.10426) | Semantic interaction, geometry, dynamics, and ego-goal tokens explicitly condition a diffusion planner. This is a stronger long-term design than free-form chain-of-thought. |
| Occupancy-language-action world model | [OccLLaMA](https://arxiv.org/abs/2409.03272) | Unifies semantic occupancy, language, and action tokens and predicts future occupancy. It suggests using occupancy rollouts to verify LLM intent. |
| Reason, imagine, verify, act | [Reason--Imagine--Act](https://arxiv.org/abs/2605.24004) | The LLM proposes action templates; an action-conditioned world model rolls them out and a safety scorer selects. This is a possible post-D3 interactive-planning route. |
| Learned fast/slow gate | [ASSCG](https://arxiv.org/abs/2606.25509) | Learns Query/Cache/Drop decisions instead of calling the slow LLM periodically. It reports a `60%` latency reduction and a `+2.28` score gain on AsyncDriver's nuPlan Hard20 setup. |
| Distilled real-time VLA | [RT-VLA](https://arxiv.org/abs/2606.14010) | Distills a large VLA into a compact student and reports `44.8x` vision-only and `7.9x` vision+language speedups. Again, the deployable pattern is teacher/student. |
| Planning-objective RL | [MAGNIFIED](https://arxiv.org/abs/2606.20641) | Fine-tunes tokenized trajectory generation with planning rewards instead of next-token imitation alone, reducing overlap by `10.5%` and off-road rate by `38.9%` relative to its SFT baseline. |

The common trend is a move away from unconstrained text trajectories toward
**anchors, continuous action heads, world-state tokens, planning rewards,
distillation, and fast/slow execution**.

## 4. Recommended architecture

### 4.1 Planning IR instead of free-form text

The LLM should emit a constrained intermediate representation such as:

```json
{
  "maneuver": "YIELD",
  "motion_direction": "FORWARD",
  "route_branch": "successor:lane_184",
  "yield_to": ["track_37"],
  "speed_cap_mps": [1.5, 1.5, 1.0, 0.5, 0.0, 0.0],
  "forbidden_zones": ["crane_work_zone_2"],
  "preferred_lateral_side": "RIGHT",
  "ttl_frames": 4,
  "confidence": 0.86
}
```

All fields must come from a fixed schema. Lane, actor, and zone IDs must be
validated against online inputs. Unknown or invalid fields close the semantic
gate and preserve the current fallback.

The initial maneuver vocabulary can remain small:

`KEEP`, `STOP`, `YIELD`, `FOLLOW`, `PASS_LEFT`, `PASS_RIGHT`, `TURN_LEFT`,
`TURN_RIGHT`, `REVERSE`, `DOCK`, `ENTER_WORK_ZONE`, and `EXIT_WORK_ZONE`.

### 4.2 Data flow

```text
 HD-map topology/rules -----+
 route + port task ---------+--> structured tools/serializer
 actors + occupancy --------+             |
 optional camera semantics -+             v
                                  LLM teacher / slow reasoner
                                               |
                                               v
 LiDAR/BEV/map tokens --> small intent student --> Planning IR
                                               |
 D0 candidates --> D2 top-K shortlist ---------+
                                               v
                        D3 set-aware reranker + continuous refiner
                                               |
                              deterministic validator
                                  | valid             | invalid/OOD
                                  v                   v
                              trajectory       exact D2/C2.3 fallback
```

### 4.3 Component responsibilities

1. **Structured tool layer**: exposes lane graph neighbors, speed limits,
   boundaries, actor states, occupancy risk, and candidate factors. It prevents
   the LLM from estimating geometry from prose.
2. **LLM teacher/slow reasoner**: interprets rules, job context, unusual actor
   interactions, and temporary scene semantics. It produces only Planning IR.
3. **Intent student**: predicts the same IR from frame-rate UniAD tokens. This
   can be a small Transformer or structured head; it does not need to decode
   language.
4. **D3 reranker**: receives the top 8-16 D2 candidates, exact fallback,
   agent/occupancy/map-risk features, and IR embeddings. It jointly compares
   candidates rather than independently regressing 1,000 costs.
5. **Validator and fallback**: checks schema, map compliance, collision,
   acceleration, and speed. It must preserve the exact current fallback when
   semantic guidance is absent, stale, or invalid.

## 5. Where the gain could come from

### 5.1 Likely gains

- **Turning and branch ambiguity**: explicit route branch and maneuver intent
  can target the current weak bucket without perturbing normal straight motion.
- **Port-specific operation semantics**: loading zones, crane work areas,
  loaded/unloaded vehicle priority, reverse docking, and waiting behavior are
  naturally expressed as rules and tasks, not only lane geometry.
- **Long-tail reasoning**: temporary obstacles, unusual equipment behavior,
  occlusion hypotheses, and human instructions are where pretrained language
  knowledge is most plausible.
- **Human controllability and debugging**: Planning IR is inspectable and can
  be edited or overridden without changing trajectory coordinates directly.
- **Rare-scenario supervision**: an offline teacher can label all 43,981 frames
  and generate counterfactual constraints without adding inference latency.

### 5.2 Unlikely gains

- Normal straight driving where D2 already sees sufficient geometry and ego
  state.
- Fine trajectory accuracy from free-form text generation.
- New information from captions that merely paraphrase existing boxes and map
  coordinates.
- Visual-only facts during pure-LiDAR inference when those facts leave no
  signature in the LiDAR/query representation.

## 6. Phased experiments and promotion gates

### P0: no-training semantic-value audit

Do this in parallel with D2 evaluation; it does not modify the planner.

1. Define the Planning IR schema and a deterministic serializer for map graph,
   route command, predicted actors, occupancy summaries, and D2 top-K candidate
   factors.
2. Build a balanced audit set emphasizing Turning, FrontObstacle, reverse,
   static/slow work zones, and known operation-state cases.
3. Run a local Qwen3 model offline. It may call read-only map, candidate, and
   collision tools, but it must not see GT ego future or GT candidate costs.
4. Compare D2 selection, LLM-guided selection, shuffled-IR selection, and the
   existing oracle on the unchanged validation evaluator.

Promotion requires all of the following:

- valid schema output above `99.9%` after constrained decoding/validation;
- measurable gain on Turning or the designated port long-tail split;
- no global L2 regression above `0.005 m` and no collision regression;
- shuffled IR loses the gain, proving scene-specific semantic use;
- the gain cannot be reproduced by adding the same numeric fields directly to
  a small non-LLM baseline.

That last comparison is essential. If a graph Transformer using the same
inputs matches the LLM, the LLM is useful only as an annotation teacher, not as
a runtime component.

### P1: offline teacher and compact intent student

1. Generate Planning IR labels on the training split with Qwen3-VL/Qwen3 plus
   deterministic tools and rule checks.
2. Add counterfactual candidate pairs: explain why a candidate violates a
   route, priority, or work-zone constraint. Do not train from prose alone;
   store structured labels and referenced IDs.
3. Train a compact `planning_intent_head` from LiDAR/BEV/map tokens. Reuse the
   existing LLMBridge query projection only where its same-frame controls prove
   useful.
4. Evaluate IR field accuracy, calibration, shuffle sensitivity, and downstream
   candidate-selection gain.

The LLM is absent at inference in this phase. This is the lowest-risk route and
the one best supported by DiMA and RT-VLA.

### P2: language-conditioned D3

1. Shortlist D2 top 8-16 candidates and append the byte-exact fallback.
2. Add explicit candidate-to-actor, candidate-to-occupancy, route, boundary,
   speed-limit, and Planning-IR features.
3. Jointly rerank with a set Transformer; optionally predict a bounded
   continuous residual after selection.
4. Train first with oracle/teacher IR, then replace it with student IR. This
   separates "IR is useful" from "the student can predict IR".

Required ablations are `no IR`, `oracle IR`, `student IR`, `shuffled IR`,
`stale IR`, `map off`, and `candidate off`. The exact fallback remains in every
non-oracle evaluation.

### P3: uncertainty-triggered fast/slow inference

Invoke the local LLM only when at least one calibrated trigger fires:

- small D2/D3 top-1 versus top-2 cost margin;
- high candidate entropy or disagreement between D2 and intent student;
- OOD actor/zone/task token;
- route-branch ambiguity;
- predicted collision or rule violation for all shortlisted candidates.

The slow result is cached with a short TTL. A learned Query/Cache/Drop gate can
replace fixed thresholds only after the fixed-gate system is understood.
Promotion requires a low query rate, bounded worst-case latency, no stale-IR
safety regression, and a causal gain concentrated on triggered frames.

### P4: radical VLA/world-model route

Only after P2 proves semantic value:

1. Treat lane-chain, speed-profile, and lateral-offset factors as decision
   anchors.
2. Let a VLA select or point to anchors, then use flow matching or diffusion to
   refine continuous speed/lateral residuals.
3. If interactive safety remains the bottleneck, roll candidate actions through
   a semantic occupancy/BEV world model before final D3 selection.

This is the true architecture-replacement route, but it has much higher data,
latency, closed-loop evaluation, and TensorRT costs. Training a 3B-7B VLA
end-to-end on only 44k local frames should not be the first experiment.

## 7. Failure modes to guard against

- **Language shortcut**: the model predicts common actions from command or ego
  speed and ignores scene tokens. Use query shuffle and IR shuffle controls.
- **Teacher leakage**: GT future, GT actor motion, or evaluator costs enter the
  prompt. The teacher may use only information available online.
- **Free-form hallucination**: prose mentions nonexistent lanes or actors.
  Constrained schema, ID validation, and hard fallback are mandatory.
- **Stale slow guidance**: an asynchronous decision persists after the scene
  changes. Use TTL, change detection, and immediate invalidation on safety risk.
- **Metric illusion**: open-loop L2 improves because of ego-status shortcuts
  while rule compliance or closed-loop safety does not. Add port long-tail,
  route compliance, off-map, comfort, and intervention metrics.
- **LLM theater**: a small numeric model gets the same gain. Always compare
  against a graph/set Transformer with identical structured inputs.
- **Missing semantics**: pure LiDAR cannot infer a traffic-light color, sign
  text, hand gesture, or job instruction that is never provided. Distillation
  cannot recover information absent from the student input.

## 8. Decision for this branch

1. Finish and evaluate D2 full as planned; keep D2 medium as the promoted
   reference until then.
2. Keep D3 as the next core planner because the measured bottleneck is candidate
   ranking and context-aware selection.
3. Start only P0 of the LLM route in parallel: Planning IR definition,
   structured scene serializer, and a no-training semantic-value audit.
4. Promote to P1/P2 only if scene-specific IR beats both D2 and an equal-input
   non-LLM baseline on Turning/long-tail cases.
5. Treat direct text trajectory, full VLA, and world-model planning as P4
   research, not as the next training run.

This ordering is deliberately falsifiable. It gives the LLM a chance to add
new semantics without allowing it to obscure whether D2/D3 map planning works.

## 9. Primary sources and implementations

- [Senna paper](https://arxiv.org/abs/2410.22313) and [official code](https://github.com/hustvl/Senna)
- [DriveVLM paper](https://arxiv.org/abs/2402.12289) and [project page](https://tsinghua-mars-lab.github.io/DriveVLM/)
- [GPT-Driver paper](https://arxiv.org/abs/2310.01415) and [official code](https://github.com/PointsCoder/GPT-Driver)
- [LMDrive paper](https://arxiv.org/abs/2312.07488) and [official code](https://github.com/opendilab/LMDrive)
- [Agent-Driver paper](https://arxiv.org/abs/2311.10813) and [project page](https://usc-gvl.github.io/Agent-Driver/)
- [PlanAgent paper](https://arxiv.org/abs/2406.01587)
- [AsyncDriver paper](https://arxiv.org/abs/2406.14556) and [official code](https://github.com/memberRE/AsyncDriver)
- [DiMA paper](https://arxiv.org/abs/2501.09757)
- [BEVDriver paper](https://arxiv.org/abs/2503.03074) and [official code](https://github.com/intelligent-vehicles/BEVDriver)
- [Driving with LLMs paper](https://arxiv.org/abs/2310.01957) and [official code](https://github.com/wayveai/Driving-with-LLMs)
- [EMMA paper](https://arxiv.org/abs/2410.23262)
- [AnchorVLA](https://arxiv.org/abs/2607.03182), [VECTOR-Drive](https://arxiv.org/abs/2605.08830), [CoWorld-VLA](https://arxiv.org/abs/2605.10426)
- [OccLLaMA](https://arxiv.org/abs/2409.03272), [Reason--Imagine--Act](https://arxiv.org/abs/2605.24004)
- [ASSCG](https://arxiv.org/abs/2606.25509), [RT-VLA](https://arxiv.org/abs/2606.14010), [MAGNIFIED](https://arxiv.org/abs/2606.20641)
