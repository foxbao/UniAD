"""Schema and validation for LLM-assisted planning intent."""

SCHEMA_VERSION = 'planning-ir/v1'

MANEUVERS = (
    'KEEP',
    'STOP',
    'YIELD',
    'FOLLOW',
    'PASS_LEFT',
    'PASS_RIGHT',
    'TURN_LEFT',
    'TURN_RIGHT',
    'REVERSE',
    'DOCK',
    'ENTER_WORK_ZONE',
    'EXIT_WORK_ZONE',
)

RISK_FLAGS = (
    'FRONT_CONFLICT',
    'CROSS_TRAFFIC',
    'ROUTE_AMBIGUITY',
    'WORK_ZONE',
    'BLOCKED_ROUTE',
    'OFF_MAP_RISK',
    'COLLISION_RISK',
    'UNKNOWN_ACTOR',
    'UNKNOWN_RULE',
)

_FACTOR_FIELDS = (
    'lane_path_index',
    'speed_profile_index',
    'lateral_offset_index',
)

_ALLOWED_FIELDS = {
    'schema_version',
    'selected_candidate_id',
    'maneuver',
    *_FACTOR_FIELDS,
    'yield_actor_id',
    'risk_flags',
    'rule_ids',
    'confidence',
    'ttl_frames',
    'reason',
}


class PlanningIRValidationError(ValueError):
    pass


def _require_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanningIRValidationError(f'{field} must be an integer')
    return int(value)


def _optional_int(value, field):
    if value is None:
        return None
    return _require_int(value, field)


def _string_list(value, field):
    if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value):
        raise PlanningIRValidationError(f'{field} must be a list of strings')
    if len(value) != len(set(value)):
        raise PlanningIRValidationError(f'{field} contains duplicates')
    return list(value)


def fallback_planning_ir(candidate_id, maneuver='KEEP'):
    return dict(
        schema_version=SCHEMA_VERSION,
        selected_candidate_id=int(candidate_id),
        maneuver=maneuver,
        lane_path_index=None,
        speed_profile_index=None,
        lateral_offset_index=None,
        yield_actor_id=None,
        risk_flags=[],
        rule_ids=[],
        confidence=0.0,
        ttl_frames=1,
    )


def validate_planning_ir(payload, candidates, actor_ids=(), rule_ids=(),
                         max_ttl_frames=10):
    """Validate and normalize one Planning IR response."""
    if isinstance(payload, dict) and 'planning_ir' in payload:
        payload = payload['planning_ir']
    if not isinstance(payload, dict):
        raise PlanningIRValidationError('planning_ir must be a JSON object')

    unknown = set(payload) - _ALLOWED_FIELDS
    if unknown:
        raise PlanningIRValidationError(
            f'unknown planning_ir fields: {sorted(unknown)}')
    if payload.get('schema_version') != SCHEMA_VERSION:
        raise PlanningIRValidationError(
            f'schema_version must be {SCHEMA_VERSION!r}')

    candidate_by_id = {
        int(candidate['candidate_id']): candidate
        for candidate in candidates
        if candidate.get('valid', True)
    }
    selected_id = _require_int(
        payload.get('selected_candidate_id'), 'selected_candidate_id')
    if selected_id not in candidate_by_id:
        raise PlanningIRValidationError(
            f'selected_candidate_id {selected_id} is not an allowed candidate')
    selected = candidate_by_id[selected_id]

    maneuver = payload.get('maneuver')
    if maneuver not in MANEUVERS:
        raise PlanningIRValidationError(
            f'maneuver must be one of {MANEUVERS}')

    normalized_factors = {}
    is_fallback = selected.get('source') == 'fallback'
    candidate_factor_names = {
        'lane_path_index': 'path_index',
        'speed_profile_index': 'speed_profile_index',
        'lateral_offset_index': 'lateral_offset_index',
    }
    for ir_field, candidate_field in candidate_factor_names.items():
        value = _optional_int(payload.get(ir_field), ir_field)
        expected = selected.get(candidate_field)
        if is_fallback:
            if value is not None:
                raise PlanningIRValidationError(
                    f'{ir_field} must be null for fallback')
        elif value != expected:
            raise PlanningIRValidationError(
                f'{ir_field}={value!r} does not match selected candidate '
                f'value {expected!r}')
        normalized_factors[ir_field] = value

    actor_id = payload.get('yield_actor_id')
    actor_id = None if actor_id is None else str(actor_id)
    allowed_actor_ids = {str(value) for value in actor_ids}
    if actor_id is not None and actor_id not in allowed_actor_ids:
        raise PlanningIRValidationError(
            f'yield_actor_id {actor_id!r} is not present in the current frame')

    risks = _string_list(payload.get('risk_flags', []), 'risk_flags')
    invalid_risks = set(risks) - set(RISK_FLAGS)
    if invalid_risks:
        raise PlanningIRValidationError(
            f'unknown risk flags: {sorted(invalid_risks)}')

    selected_rule_ids = _string_list(payload.get('rule_ids', []), 'rule_ids')
    invalid_rules = set(selected_rule_ids) - {str(value) for value in rule_ids}
    if invalid_rules:
        raise PlanningIRValidationError(
            f'unknown rule IDs: {sorted(invalid_rules)}')

    confidence = payload.get('confidence')
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise PlanningIRValidationError('confidence must be numeric')
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise PlanningIRValidationError('confidence must be in [0, 1]')

    ttl_frames = _require_int(payload.get('ttl_frames'), 'ttl_frames')
    if not 1 <= ttl_frames <= int(max_ttl_frames):
        raise PlanningIRValidationError(
            f'ttl_frames must be in [1, {int(max_ttl_frames)}]')

    normalized = dict(
        schema_version=SCHEMA_VERSION,
        selected_candidate_id=selected_id,
        maneuver=maneuver,
        **normalized_factors,
        yield_actor_id=actor_id,
        risk_flags=risks,
        rule_ids=selected_rule_ids,
        confidence=confidence,
        ttl_frames=ttl_frames,
    )
    reason = payload.get('reason')
    if reason is not None:
        if not isinstance(reason, str):
            raise PlanningIRValidationError('reason must be a string')
        normalized['reason'] = reason.strip()[:1000]
    return normalized
