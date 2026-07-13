#!/usr/bin/env python
"""Run a local text LLM as a constrained Planning-IR teacher."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys


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
    parser.add_argument('--max-new-tokens', type=int, default=512)
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


def generate(tokenizer, model, device, prompt, max_new_tokens):
    import torch

    rendered = apply_chat_template(tokenizer, [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ])
    inputs = tokenizer(rendered, return_tensors='pt').to(device)
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
    if args.dry_run:
        print(SYSTEM_PROMPT)
        print(build_prompt(first_input))
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
                raw_output = generate(
                    tokenizer, model, args.device, build_prompt(teacher_input),
                    args.max_new_tokens)
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
