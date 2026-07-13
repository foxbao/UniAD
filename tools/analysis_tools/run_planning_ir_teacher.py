#!/usr/bin/env python
"""Run a local text LLM as a constrained Planning-IR teacher."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys


# The text-only teacher is PyTorch-only. Disabling optional backends avoids
# importing incompatible user-site TensorFlow/JAX packages through Transformers.
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_FLAX', '0')


REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.analysis_tools.planning_ir_schema import (
    MANEUVERS,
    RISK_FLAGS,
    PlanningIRValidationError,
    fallback_planning_ir,
    validate_planning_ir,
)
from tools.analysis_tools.planning_ir_audit_utils import (
    extract_json_object,
    read_jsonl,
)


SYSTEM_PROMPT = """You are a safety-constrained autonomous-driving planning
teacher. Select exactly one supplied candidate. Never invent candidate, actor,
lane, or rule IDs. Return one JSON object only, with no markdown and no hidden
reasoning. The trajectory coordinates are ego-frame metres, x forward and y
left. Predicted candidate cost is a learned estimate, not ground truth."""


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a local Qwen-compatible Planning-IR teacher.')
    parser.add_argument('--input-jsonl', required=True)
    parser.add_argument('--output-jsonl', required=True)
    parser.add_argument('--model-path', default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--prompt-max-actors', type=int, default=8)
    parser.add_argument('--prompt-motion-steps', type=int, default=6)
    parser.add_argument('--max-input-tokens', type=int, default=5000)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true',
                        help='Validate inputs and print the first prompt only.')
    return parser.parse_args()


def fallback_id(teacher_input):
    for candidate in teacher_input['candidates']:
        if candidate.get('source') == 'fallback' and candidate.get(
                'valid', True):
            return int(candidate['candidate_id'])
    raise ValueError('Teacher input has no valid fallback candidate')


def rounded(value, digits=2):
    if value is None:
        return None
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, list):
        return [rounded(item, digits) for item in value]
    if isinstance(value, dict):
        return {key: rounded(item, digits) for key, item in value.items()}
    return value


def compact_teacher_input(teacher_input, max_actors=8, motion_steps=6):
    ego = teacher_input.get('ego', {})
    compact_ego = dict(
        route_command=ego.get('route_command'),
        velocity_xy_mps=rounded(ego.get('velocity_xy_mps'), digits=1),
    )
    actors = []
    for actor in teacher_input.get('actors', [])[:max_actors]:
        actors.append(dict(
            actor_id=actor.get('actor_id'),
            class_name=actor.get('class_name'),
            confidence=rounded(actor.get('confidence'), digits=2),
            center_xy_m=rounded(actor.get('center_xy_m'), digits=1),
            size_lwh_m=rounded(actor.get('size_lwh_m'), digits=1),
            yaw_rad=rounded(actor.get('yaw_rad')),
            velocity_xy_mps=rounded(
                actor.get('velocity_xy_mps'), digits=1),
            predicted_displacements_xy_m=rounded(
                (actor.get('predicted_displacements_xy_m') or [])[
                    :motion_steps], digits=1),
        ))
    candidates = []
    map_paths = {}
    for candidate in teacher_input['candidates']:
        path_index = candidate.get('path_index')
        if path_index is not None and path_index not in map_paths:
            map_paths[path_index] = [
                f'{lane.get("lane_id")}:'
                f'{"R" if lane.get("reverse") else "F"}'
                for lane in candidate.get('lane_sequence', [])
            ]
        candidates.append(dict(
            candidate_id=candidate['candidate_id'],
            source=candidate.get('source'),
            valid=candidate.get('valid', True),
            trajectory_xy_m=rounded(
                candidate.get('refined_trajectory_xy_m'), digits=1),
            predicted_horizon_costs_m=rounded(
                candidate.get('predicted_horizon_costs_m'), digits=2),
            predicted_mean_cost_m=rounded(
                candidate.get('predicted_mean_cost_m'), digits=2),
            selection_probability=rounded(
                candidate.get('selection_probability'), digits=3),
            path_index=path_index,
            speed_profile_index=candidate.get('speed_profile_index'),
            lateral_offset_index=candidate.get('lateral_offset_index'),
            lateral_offset_m=rounded(
                candidate.get('lateral_offset_m'), digits=1),
        ))
    return dict(
        ego=compact_ego,
        actors=actors,
        map_paths=[
            dict(path_index=path_index, lane_sequence=lane_sequence)
            for path_index, lane_sequence in sorted(map_paths.items())
        ],
        candidates=candidates,
        semantic_rules=rounded(teacher_input.get('semantic_rules', [])),
    )


def build_prompt(teacher_input):
    schema = dict(
        schema_version='planning-ir/v1',
        selected_candidate_id='integer from candidates',
        maneuver=list(MANEUVERS),
        lane_path_index='integer matching selected map candidate, else null',
        speed_profile_index=(
            'integer matching selected map candidate, else null'),
        lateral_offset_index=(
            'integer matching selected map candidate, else null'),
        yield_actor_id='current actor_id or null',
        risk_flags=list(RISK_FLAGS),
        rule_ids='list of current semantic rule_ids',
        confidence='number in [0,1]',
        ttl_frames='integer in [1,10]',
        reason='optional concise sentence',
    )
    return (
        'Choose the safest route-consistent candidate for the current frame. '
        'Use fallback when evidence for changing the baseline is weak. '
        'Factor indices must exactly match the selected candidate.\n\n'
        'Required output schema:\n'
        f'{json.dumps(schema, ensure_ascii=False)}\n\n'
        'Current online-only planning context:\n'
        f'{json.dumps(teacher_input, ensure_ascii=False, separators=(",", ":"))}'
    )


def apply_chat_template(tokenizer, messages):
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def load_model(model_path, device):
    if not model_path:
        raise ValueError('--model-path is required unless --dry-run is used')
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if device.startswith('cuda') else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, trust_remote_code=True)
    model.to(device)
    model.eval()
    return tokenizer, model


def generate(tokenizer, model, device, prompt, max_new_tokens,
             max_input_tokens):
    import torch

    rendered = apply_chat_template(tokenizer, [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ])
    inputs = tokenizer(rendered, return_tensors='pt').to(device)
    input_tokens = int(inputs['input_ids'].shape[1])
    if input_tokens > max_input_tokens:
        raise ValueError(
            f'compact prompt has {input_tokens} tokens, exceeding '
            f'--max-input-tokens={max_input_tokens}')
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    generated = output[0, inputs['input_ids'].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def validate_input(record):
    if record.get('input_schema_version') != 'planning-ir-audit-input/v1':
        raise ValueError('Unsupported Planning-IR audit input schema')
    teacher_input = record.get('teacher_input')
    if not isinstance(teacher_input, dict):
        raise ValueError('Record does not contain teacher_input')
    if 'audit_labels' in teacher_input:
        raise ValueError('GT audit labels leaked into teacher_input')
    candidates = teacher_input.get('candidates')
    if not isinstance(candidates, list) or not candidates:
        raise ValueError('teacher_input.candidates must be non-empty')
    fallback_id(teacher_input)
    return teacher_input


def existing_records(path):
    if not osp.exists(path):
        return []
    return read_jsonl(path)


def main():
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    if args.limit > 0:
        records = records[:args.limit]
    if not records:
        raise ValueError('Input JSONL contains no records')
    first_input = validate_input(records[0])
    first_prompt_input = compact_teacher_input(
        first_input, args.prompt_max_actors, args.prompt_motion_steps)
    if args.dry_run:
        print(SYSTEM_PROMPT)
        print(build_prompt(first_prompt_input))
        print(f'Validated {len(records)} input records; model was not loaded.')
        return

    output_path = osp.abspath(args.output_jsonl)
    prior = existing_records(output_path) if args.resume else []
    completed = {int(record['result_index']) for record in prior}
    tokenizer, model = load_model(args.model_path, args.device)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    mode = 'a' if args.resume else 'w'
    with open(output_path, mode, encoding='utf-8') as output_handle:
        for offset, record in enumerate(records, 1):
            result_index = int(record['result_index'])
            if result_index in completed:
                continue
            teacher_input = validate_input(record)
            candidates = teacher_input['candidates']
            actor_ids = [actor['actor_id']
                         for actor in teacher_input.get('actors', [])]
            rule_ids = [rule['rule_id']
                        for rule in teacher_input.get('semantic_rules', [])]
            raw_output = ''
            error = None
            valid = False
            try:
                prompt_input = compact_teacher_input(
                    teacher_input, args.prompt_max_actors,
                    args.prompt_motion_steps)
                raw_output = generate(
                    tokenizer, model, args.device, build_prompt(prompt_input),
                    args.max_new_tokens, args.max_input_tokens)
                payload = extract_json_object(raw_output)
                planning_ir = validate_planning_ir(
                    payload, candidates, actor_ids=actor_ids,
                    rule_ids=rule_ids)
                valid = True
            except (ValueError, PlanningIRValidationError) as exc:
                error = str(exc)
                planning_ir = fallback_planning_ir(fallback_id(teacher_input))
            output_record = dict(
                result_index=result_index,
                sample_token=record.get('sample_token'),
                planning_ir=planning_ir,
                valid=valid,
                error=error,
                raw_output=raw_output,
            )
            output_handle.write(json.dumps(
                output_record, ensure_ascii=False, separators=(',', ':')))
            output_handle.write('\n')
            output_handle.flush()
            print(
                f'[{offset}/{len(records)}] result_index={result_index} '
                f'valid={valid} '
                f'candidate={planning_ir["selected_candidate_id"]}')


if __name__ == '__main__':
    main()
