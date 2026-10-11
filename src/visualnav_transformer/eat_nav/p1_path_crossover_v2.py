"""Offline material identity and prospective G3 Path crossover contracts.

This module does not import Isaac Lab and does not modify the frozen G3 runner.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping


MATERIAL_FIELDS = (
    "static_friction", "dynamic_friction", "restitution",
    "friction_combine_mode", "restitution_combine_mode",
    "compliant_contact_stiffness", "compliant_contact_damping",
)
COMBINE_MODES = {"average", "min", "multiply", "max"}
PATH_ROLES = ("direct", "left_20", "right_20")
CROSSOVER_PERIODS = (
    ("direct", "direct", "left_20", "right_20"),
    ("left_20", "right_20", "direct", "direct"),
    ("right_20", "left_20", "right_20", "left_20"),
)


def canonical_material(config: object) -> dict:
    """Extract only material values that Isaac Lab writes to USD.

    `func` and other Python object representations are deliberately excluded.
    Missing/invalid values fail closed; values are never filled by inference.
    """
    values = {}
    for name in MATERIAL_FIELDS:
        if isinstance(config, Mapping):
            if name not in config:
                raise ValueError("material field missing: " + name)
            value = config[name]
        else:
            if not hasattr(config, name):
                raise ValueError("material field missing: " + name)
            value = getattr(config, name)
        if name.endswith("combine_mode"):
            if str(value) not in COMBINE_MODES:
                raise ValueError("invalid material combine mode: " + name)
            values[name] = str(value)
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("invalid material value: " + name)
            values[name] = float(value)
    return values


def fingerprint_material(config: object) -> str:
    payload = {"schema": "eatnav_rigid_material_values_v1",
               "values": canonical_material(config)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def material_values_from_source(env_source: str, material_cfg_source: str) -> dict:
    """Statically read the declared terrain material and Isaac Lab defaults.

    This does not establish what PhysX applied in a historical live stage.
    """
    defaults = {}
    for node in ast.walk(ast.parse(material_cfg_source)):
        if isinstance(node, ast.ClassDef) and node.name == "RigidBodyMaterialCfg":
            for item in node.body:
                if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name) and \
                        item.target.id in MATERIAL_FIELDS:
                    defaults[item.target.id] = ast.literal_eval(item.value)
    calls = []
    for node in ast.walk(ast.parse(env_source)):
        if isinstance(node, ast.Call) and \
                (node.func.id if isinstance(node.func, ast.Name) else
                 node.func.attr if isinstance(node.func, ast.Attribute) else None) == "RigidBodyMaterialCfg":
            calls.append(node)
    if len(calls) != 1:
        raise ValueError("expected one terrain RigidBodyMaterialCfg declaration")
    values = dict(defaults)
    for keyword in calls[0].keywords:
        if keyword.arg in MATERIAL_FIELDS:
            values[keyword.arg] = ast.literal_eval(keyword.value)
    return canonical_material(values)


def read_usd_material_values(prim: object) -> dict:
    """Read a *live/composed* material prim without authoring stage changes.

    The caller must resolve the material bound to the collision mesh first.
    This function is intentionally separate from declared-config fingerprinting.
    """
    if not prim or not prim.IsValid():
        raise ValueError("invalid material prim")
    names = {
        "static_friction": "physics:staticFriction",
        "dynamic_friction": "physics:dynamicFriction",
        "restitution": "physics:restitution",
        "friction_combine_mode": "physxMaterial:frictionCombineMode",
        "restitution_combine_mode": "physxMaterial:restitutionCombineMode",
        "compliant_contact_stiffness": "physxMaterial:compliantContactStiffness",
        "compliant_contact_damping": "physxMaterial:compliantContactDamping",
    }
    values = {}
    for field, usd_name in names.items():
        attr = prim.GetAttribute(usd_name)
        if not attr or not attr.IsValid() or not attr.HasAuthoredValue():
            raise ValueError("USD material attribute missing: " + usd_name)
        values[field] = attr.Get()
    return canonical_material(values)


def reassess_direct_repeat(trials: list[dict], batches: list[dict],
                           declared_material_sha: str) -> dict:
    """Re-pair sealed v1 results without treating process-dependent SHA as physics identity."""
    if not declared_material_sha:
        raise ValueError("declared material fingerprint missing")
    indexed = defaultdict(list)
    legacy_hashes = set()
    mesh_by_group = defaultdict(set)
    for batch in batches:
        key = (batch["anchor_id"], batch["batch_seed"])
        indexed[key].append(batch["mesh_sha256"])
        legacy_hashes.add(batch["material_sha256"])
        mesh_by_group[batch["anchor_id"]].add(batch["mesh_sha256"])
    if any(len(set(meshes)) != 1 or len(meshes) != 2
           for meshes in indexed.values()) or \
            any(len(meshes) != 1 for meshes in mesh_by_group.values()):
        raise ValueError("terrain mesh drift")
    grouped = defaultdict(dict)
    for row in trials:
        key = (row["anchor_id"], row["batch_seed"], row["env_id"])
        repeat = row["repeat_index"]
        if repeat in grouped[key]:
            raise ValueError("duplicate fresh-process trial")
        grouped[key][repeat] = row
    pairs = []
    numeric_metrics = ("two_second_progress_m", "common_goal_progress_2s_m",
                       "arrival_time_s", "tracking_mean_m", "recovery_count",
                       "contact_force_mean_N", "applied_torque_mean_abs")
    for (group, seed, env), repeats in sorted(grouped.items()):
        if set(repeats) != {0, 1} or (group, seed) not in indexed:
            raise ValueError("fresh-process pair incomplete")
        a, b = repeats[0], repeats[1]
        deltas = {metric: None if a.get(metric) is None or b.get(metric) is None
                  else float(b[metric]) - float(a[metric]) for metric in numeric_metrics}
        pairs.append({"state_group": group, "seed": seed, "env_id": env,
                      "differences_repeat1_minus_repeat0": deltas,
                      "legacy_condition_keys": [a["terrain_condition_key"],
                                                b["terrain_condition_key"]]})
    return {
        "seed_env_mesh_correspondence_pairs": len(pairs),
        "actual_material_verified_pairs": 0,
        "independent_seed_conditions": len(indexed),
        "legacy_material_sha_count": len(legacy_hashes),
        "terrain_mesh_match_all": True,
        "declared_material_match_all": None,
        "declared_material_sha256": declared_material_sha,
        "historical_declared_material_values_verified": False,
        "actual_physx_material_verified": False,
        "actual_physx_material_status": "not_recorded_in_frozen_v1_artifacts",
        "pairs": pairs,
    }


def build_crossover(seeds_by_group: Mapping[str, tuple[int, ...]]) -> dict:
    """Three periods: every seed/env receives all Paths once in fresh processes."""
    processes = []
    for group, seeds in seeds_by_group.items():
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError("three unique seed bundles required per StateGroup")
        for seed_index, seed in enumerate(seeds):
            for execution_index in range(3):
                period = (execution_index + seed_index) % 3
                trials = [{"env_id": env, "path_role": CROSSOVER_PERIODS[period][env]}
                          for env in range(4)]
                processes.append({"state_group": group, "seed": seed,
                                  "execution_index": execution_index,
                                  "assignment_period": period, "trials": trials})
    return {"schema": "eatnav_p1_g3_path_crossover_design_v1",
            "status": "design_only_no_rollout_authorized",
            "processes": processes,
            "balance": dict(Counter(row["path_role"] for process in processes
                                    for row in process["trials"]))}
