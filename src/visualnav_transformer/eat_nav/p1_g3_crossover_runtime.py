"""Versioned P1 G3 crossover pre-Path and material contracts (Isaac-independent)."""

from __future__ import annotations

import math
import hashlib
import json
import statistics
from collections import Counter, defaultdict


SCHEMA = "eatnav_p1_g3_path_crossover_v1"
STRICT_RUN_FIELDS = ("state_group", "seed", "terrain_mesh_sha256",
                     "terrain_material_sha256", "robot_material_sha256")
STRICT_ENV_FIELDS = ("env_id", "root_state_w", "joint_position", "joint_velocity",
                     "applied_torque", "actor_pre_noise", "actor_post_noise_pre_delay",
                     "actor_input", "actor_action", "noise", "command",
                     "shared_dem_frame_sha256", "episode_length", "reset_count")
DIAGNOSTIC_ENV_FIELDS = ("contact_current", "contact_history",
                         "critic_observation", "critic_value")
NUMERIC_ENV_FIELDS = set(STRICT_ENV_FIELDS) - {"shared_dem_frame_sha256"}


def _finite(value):
    if isinstance(value, dict):
        return all(_finite(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return all(_finite(child) for child in value)
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return value is not None


def _numeric_tree(value):
    if isinstance(value, dict):
        return bool(value) and all(_numeric_tree(child) for child in value.values())
    if isinstance(value, (tuple, list)):
        return bool(value) and all(_numeric_tree(child) for child in value)
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def compare_prepath(first: dict, current: dict) -> dict:
    """The host calls this before allowing the simulator to publish any Path.

    Native contact and Critic differences remain visible diagnostics; all
    Path-independent control inputs use exact equality across fresh processes.
    """
    failures = []
    diagnostic = []
    for name in STRICT_RUN_FIELDS:
        if first.get(name) != current.get(name) or not _finite(current.get(name)):
            failures.append({"scope": "process", "signal": name})
    a, b = first.get("envs", []), current.get("envs", [])
    if len(a) != 4 or len(b) != 4:
        failures.append({"scope": "process", "signal": "four_env_completeness"})
    else:
        for env, (left, right) in enumerate(zip(a, b)):
            for name in STRICT_ENV_FIELDS:
                valid = (_numeric_tree(right.get(name)) if name in NUMERIC_ENV_FIELDS else
                         isinstance(right.get(name), list) and len(right[name]) == 4 and
                         all(isinstance(item, str) and len(item) == 64 for item in right[name]))
                if left.get(name) != right.get(name) or not valid:
                    failures.append({"scope": env, "signal": name})
            for name in DIAGNOSTIC_ENV_FIELDS:
                if not _finite(right.get(name)):
                    failures.append({"scope": env, "signal": name + "_nonfinite"})
                elif left.get(name) != right.get(name):
                    diagnostic.append({"scope": env, "signal": name,
                                       "reference": left.get(name), "current": right.get(name)})
    return {"pass": not failures, "strict_failures": failures,
            "diagnostic_differences": diagnostic,
            "strict_fields": list(STRICT_RUN_FIELDS) + list(STRICT_ENV_FIELDS),
            "diagnostic_fields": list(DIAGNOSTIC_ENV_FIELDS)}


def validate_crossover_assignment(schedule: dict) -> dict:
    if schedule.get("schema") != "eatnav_p1_g3_path_crossover_design_v1":
        raise ValueError("crossover assignment schema")
    processes = schedule.get("processes", [])
    if len(processes) != 18:
        raise ValueError("crossover process count")
    condition = defaultdict(lambda: defaultdict(list))
    pair_count = 0
    period_count = Counter()
    order_count = Counter()
    for process in processes:
        group, seed = process["state_group"], process["seed"]
        key = (group, seed)
        period_count[(group, seed, process["assignment_period"])] += 1
        order_count[(group, process["execution_index"])] += 1
        rows = process["trials"]
        if len(rows) != 4 or [row["env_id"] for row in rows] != list(range(4)):
            raise ValueError("crossover four-env assignment")
        for row in rows:
            condition[key][row["env_id"]].append(row["path_role"])
        for left, right in ((0, 1), (2, 3)):
            if rows[left]["path_role"] == rows[right]["path_role"] == "direct":
                pair_count += 1
    if len(condition) != 6 or any(len(envs) != 4 or any(
            Counter(roles) != {"direct": 1, "left_20": 1, "right_20": 1}
            for roles in envs.values()) for envs in condition.values()):
        raise ValueError("crossover Path/env balance")
    if any(value != 1 for value in period_count.values()) or len(period_count) != 18 or \
            any(value != 3 for value in order_count.values()) or len(order_count) != 6 or \
            pair_count != 12:
        raise ValueError("crossover period/order/direct duplicate balance")
    return {"pass": True, "processes": 18, "trials": 72,
            "independent_seed_conditions": 6, "direct_duplicate_pairs": pair_count}


def validate_robot_physx_material(properties) -> dict:
    """Check effective per-shape PhysX tensor after startup material randomization."""
    rows = properties.tolist() if hasattr(properties, "tolist") else properties
    if not isinstance(rows, list) or len(rows) != 4 or not all(
            isinstance(env, list) and len(env) == len(rows[0]) and len(env) > 0
            for env in rows):
        raise ValueError("effective robot material shape")
    for env in rows:
        for shape in env:
            if not isinstance(shape, list) or len(shape) != 3 or not all(
                    isinstance(x, (int, float)) and math.isfinite(x) for x in shape):
                raise ValueError("effective robot material shape/nonfinite")
            if any(abs(actual - expected) > 1e-5 for actual, expected in
                   zip(shape, (0.8, 0.6, 0.0))):
                raise ValueError("effective robot material differs from frozen startup contract")
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"),
                                       allow_nan=False).encode()).hexdigest()
    return {"env_count": 4, "shape_count": len(rows[0]), "canonical_sha256": digest,
            "properties": rows, "expected_per_shape": [0.8, 0.6, 0.0]}


EFFECT_METRICS = ("two_second_progress_m", "common_goal_progress_2s_m",
                  "tracking_mean_m", "recovery_count", "contact_force_mean_N",
                  "applied_torque_mean_abs", "contact_force_fixed_2s_mean_N",
                  "joint_torque_fixed_2s_mean_abs", "path_length_m",
                  "path_max_abs_discrete_curvature_m_inv",
                  "endpoint_common_goal_distance_m")


def paired_effects(trials: list[dict]) -> dict:
    """Calculate signed candidate−direct differences within seed/env; cluster at seed."""
    def mean_or_none(values):
        values=list(values)
        return statistics.mean(values) if values else None
    indexed = defaultdict(dict)
    for row in trials:
        key = (row["anchor_id"], row["batch_seed"], row["env_id"])
        role = row["path_role"]
        if role in indexed[key]:
            raise ValueError("duplicate Path role in seed/env")
        indexed[key][role] = row
    pair_rows = []
    for (group, seed, env), paths in sorted(indexed.items()):
        if set(paths) != {"direct", "left_20", "right_20"}:
            raise ValueError("incomplete crossover seed/env")
        direct = paths["direct"]
        for role in ("left_20", "right_20"):
            candidate = paths[role]
            differences = {}
            for metric in EFFECT_METRICS:
                a, b = direct.get(metric), candidate.get(metric)
                differences[metric] = None if a is None or b is None else float(b) - float(a)
            arrival_a, arrival_b = direct.get("arrival_time_s"), candidate.get("arrival_time_s")
            pair_rows.append({"state_group": group, "seed": seed, "env_id": env,
                              "candidate": role,
                              "candidate_right_censored": bool(candidate["right_censored"]),
                              "direct_right_censored": bool(direct["right_censored"]),
                              "arrival_time_difference_s": None if arrival_a is None or arrival_b is None
                              else float(arrival_b) - float(arrival_a),
                              "signed_candidate_minus_direct": differences})
    summary = defaultdict(dict)
    for group in sorted({row["state_group"] for row in pair_rows}):
        for role in ("left_20", "right_20"):
            subset = [row for row in pair_rows if row["state_group"] == group and
                      row["candidate"] == role]
            seeds = sorted({row["seed"] for row in subset})
            seed_means = {}
            for metric in EFFECT_METRICS:
                seed_means[metric] = [mean_or_none(
                    row["signed_candidate_minus_direct"][metric] for row in subset
                    if row["seed"] == seed and row["signed_candidate_minus_direct"][metric] is not None)
                    for seed in seeds]
            summary[group][role] = {"seed_count": len(seeds), "env_pair_count": len(subset),
                                    "seed_means": dict(zip(seeds, [
                                        {metric: seed_means[metric][i] for metric in EFFECT_METRICS}
                                        for i in range(len(seeds))])),
                                    "mean_seed_effect": {metric: mean_or_none(value for value in values if value is not None)
                                                         for metric, values in seed_means.items()},
                                    "sd_across_seeds": {metric: statistics.stdev(valid)
                                                        if len(valid := [value for value in values if value is not None]) > 1 else None
                                                        for metric, values in seed_means.items()},
                                    "censored_candidate_count": sum(row["candidate_right_censored"]
                                                                    for row in subset)}
    env_trends = {str(env): {role: {metric: mean_or_none(
        row["signed_candidate_minus_direct"][metric] for row in pair_rows
        if row["env_id"] == env and row["candidate"] == role and
        row["signed_candidate_minus_direct"][metric] is not None)
        for metric in EFFECT_METRICS} for role in ("left_20", "right_20")}
        for env in range(4)}
    return {"independent_seed_conditions": len({(row["state_group"], row["seed"])
                                                 for row in pair_rows}),
            "candidate_pairs": {role: sum(row["candidate"] == role for row in pair_rows)
                                for role in ("left_20", "right_20")},
            "pair_rows": pair_rows, "state_groups": dict(summary),
            "env_id_trends": env_trends,
            "interpretation": "descriptive_seed_clustered_not_generalization_or_navigation_quality"}
