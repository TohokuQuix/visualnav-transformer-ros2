"""Pure checks used by the offline M11-M13 design addendum."""
from __future__ import annotations

import math


def pareto_front(points):
    """Indices maximizing both observed reward and goal reduction."""
    values = [tuple(map(float, point)) for point in points]
    if any(len(point) != 2 or not all(map(math.isfinite, point)) for point in values):
        raise ValueError("finite reward/goal pairs required")
    return [i for i, point in enumerate(values)
            if not any(other[0] >= point[0] and other[1] >= point[1] and other != point
                       for j, other in enumerate(values) if j != i)]


REQUIRED_METADATA = (
    "name", "shape", "order", "unit", "frame", "timestamp_phase",
    "normalization", "clip", "missing_mask", "simulator_api", "robot_api",
    "parity_status", "role",
)
REQUIRED_FIELDS = (
    "reward_total", "reward_component_pre_dt", "reward_component_post_dt",
    "reward_component_weight", "control_dt_s", "reward_config_sha256",
    "reward_term_version", "termination_reward", "reward_component_valid",
    "path_active", "path_id_sha256", "path_publish_stamp_ns",
    "path_horizon_end", "replanning_event", "follower_mode",
    "command_current", "command_last_nonzero", "command_source",
    "command_publish_stamp_ns", "command_adopt_stamp_ns",
    "pre_command_observation_stamp_ns", "post_command_observation_stamp_ns",
    "exact_stamp_tf", "post_path_control_state", "projected_gravity",
    "base_linear_velocity", "base_angular_velocity", "full_joint_position",
    "full_joint_velocity", "previous_action", "actor_action", "actor_command_input",
    "actor_observation", "critic_observation", "controller_state",
    "raw_applied_effort", "effort_limit", "contact", "roll_pitch_tilt",
    "root_pose", "realized_xy", "dem_height_mask_history", "relative_goal",
    "candidate_path", "termination_reason", "truncation", "timeout",
    "reset", "stack_proxy", "verified_stack", "excessive_tilt",
    "infrastructure_failure", "manual_stop", "censored",
    "measurement_valid", "failure_positive_eligible",
)


def validate_logging_schema(schema):
    if schema.get("schema_version") != "future_evaluator_collection_v1":
        raise ValueError("logging schema version mismatch")
    fields = schema.get("fields", [])
    names = [field.get("name") for field in fields]
    if len(names) != len(set(names)):
        raise ValueError("duplicate logging field")
    missing = set(REQUIRED_FIELDS) - set(names)
    if missing:
        raise ValueError(f"required logging fields absent: {sorted(missing)}")
    phases = set(schema.get("step_timing", {}))
    if not {"anchor_pre_path", "pre_command_update", "post_command_update_pre_step",
            "post_step", "terminal_pre_reset"} <= phases:
        raise ValueError("step timing contract incomplete")
    for field in fields:
        if set(REQUIRED_METADATA) - set(field):
            raise ValueError(f"{field.get('name')}: metadata incomplete")
        if field["timestamp_phase"] not in phases:
            raise ValueError(f"{field['name']}: unknown timing phase")
        if field["role"] != "future_logging_only":
            raise ValueError(f"{field['name']}: must not become primary model input")
    return True
