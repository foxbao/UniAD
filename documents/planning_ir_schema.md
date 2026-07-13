# Planning IR P0 audit protocol

> Implementation status: code complete on 2026-07-13; real D2 audit inference
> and Qwen selection are pending an available checkpoint/GPU window.

## 1. Purpose

P0 tests one narrow question without retraining UniAD:

> Can an offline LLM select a better trajectory from D2's top map candidates
> and the exact fallback using only information available online?

This is a semantic-value audit, not a runtime VLA. The LLM cannot generate XY
coordinates or modify candidate trajectories. It must point to one supplied
candidate through a strict JSON Planning IR.

## 2. Candidate audit payload

Normal D2 training and evaluation keep `audit_topk=0`. The dedicated config

```text
projects/configs/stage2_e2e_lidar/eval/
  base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval.py
```

sets `audit_topk=16` and retains:

- the 16 valid/lowest-predicted-cost map candidates available in that frame;
- the exact C2.3 fallback unconditionally;
- original candidate IDs, raw and refined trajectories, D2 probabilities,
  logits, and predicted 1s/2s/3s costs;
- map factor metadata in runtime order: lane path, speed profile, and lateral
  offset.

The fallback is not refined. Its candidate ID is the original final index in
the full D2 set. The `audit_oracle_candidate_id` produced later is only the
oracle over these 17 audited candidates, not the full D0 candidate oracle.

## 3. Leak boundary

Each exported JSONL row has two top-level branches:

```text
teacher_input   online-only data passed to Qwen
audit_labels    GT-only data used after Qwen has answered
```

`teacher_input` contains:

- ego route command, box, velocity, and pose context;
- predicted actor boxes, track IDs, classes, confidence, and motion modes;
- audited candidate trajectories, D2 scores/costs, and map factors;
- optional static semantic rule metadata supplied by the project.

`audit_labels` contains GT ego future, GT validity, future occupancy-derived
collision labels, motion/obstacle buckets, and actual candidate errors. The
teacher runner reads and serializes only `record["teacher_input"]`. Passing the
entire exported row to an LLM would violate the experiment.

## 4. Planning IR v1

Example map-candidate response:

```json
{
  "schema_version": "planning-ir/v1",
  "selected_candidate_id": 317,
  "maneuver": "YIELD",
  "lane_path_index": 4,
  "speed_profile_index": 3,
  "lateral_offset_index": 2,
  "yield_actor_id": "1087",
  "risk_flags": ["FRONT_CONFLICT"],
  "rule_ids": ["crane_zone_priority_2"],
  "confidence": 0.84,
  "ttl_frames": 3,
  "reason": "Yield to the tracked vehicle before entering the shared lane."
}
```

Required constraints:

- `selected_candidate_id` must name a valid candidate in the current frame;
- all three factor indices must exactly match the selected map candidate;
- factor indices must be `null` when fallback is selected;
- actor and rule IDs must exist in the current input;
- maneuver and risk values must come from the fixed vocabulary;
- `confidence` is in `[0,1]`, and `ttl_frames` is in `[1,10]`;
- unknown fields, duplicate IDs/flags, and malformed types are rejected.

Any parse or validation failure selects exact fallback and is counted as an
invalid teacher frame.

## 5. Commands

### 5.1 Produce the D2 audit result

Use the D2 full checkpoint after it is available. A medium checkpoint can be
used first to validate the pipeline.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4 MASTER_PORT=28830 \
  ./tools/uniad_dist_eval.sh \
  projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval.py \
  projects/work_dirs/stage2_e2e_lidar/base_e2e_lidar_plan_mapfuse_v6_d2_calibrated_cost_train/epoch_1.pth \
  5 \
  --out projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/results.pkl
```

### 5.2 Export online inputs and hidden audit labels

```bash
python tools/analysis_tools/export_planning_ir_inputs.py \
  --config projects/configs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval.py \
  --results-pkl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/results.pkl \
  --out-jsonl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/planning_ir_inputs.jsonl
```

Optional static rules use `{"rules": [...]}` JSON. Each rule needs a stable
`rule_id`; an optional `lane_ids` list limits it to candidates touching those
lanes. Do not build this file from validation GT.

### 5.3 Validate the prompt without loading Qwen

```bash
/mnt/disk1/conda_envs/uniad_train_qwen3_py39/bin/python \
  tools/analysis_tools/run_planning_ir_teacher.py \
  --input-jsonl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/planning_ir_inputs.jsonl \
  --output-jsonl /tmp/planning_ir_teacher_unused.jsonl \
  --dry-run --limit 1
```

### 5.4 Run the offline teacher

Start with a balanced or small `--limit` pilot before labeling all frames.

```bash
CUDA_VISIBLE_DEVICES=0 \
  /mnt/disk1/conda_envs/uniad_train_qwen3_py39/bin/python \
  tools/analysis_tools/run_planning_ir_teacher.py \
  --input-jsonl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/planning_ir_inputs.jsonl \
  --output-jsonl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/qwen3_4b_planning_ir.jsonl \
  --model-path /mnt/disk1/models/Qwen3-4B-Instruct-2507 \
  --device cuda:0 --limit 200
```

Use `--resume` to append missing frames to an existing output.

### 5.5 Compare selections offline

```bash
python tools/analysis_tools/eval_planning_ir_selection.py \
  --audit-jsonl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/planning_ir_inputs.jsonl \
  --teacher-jsonl projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/qwen3_4b_planning_ir.jsonl \
  --output-json projects/work_dirs/stage2_e2e_lidar/eval/base_e2e_lidar_plan_mapfuse_v6_d2_planning_ir_audit_eval/qwen3_4b_selection_summary.json
```

The report compares D2, exact fallback, teacher, and audited-candidate oracle
globally and by motion/obstacle bucket. It reports 1s/2s/3s L2, collision,
map-selection rate, useful-map rate, and invalid/missing teacher rate.

## 6. Promotion decision

P0 is positive only if teacher selection improves a designated weak/long-tail
split without material global or collision regression. It still does not prove
that an LLM belongs online. The next controls are shuffled IR and an equal-input
small graph/set baseline. If that baseline matches Qwen, use Qwen only as an
offline annotation teacher.
