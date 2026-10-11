"""Read-only v2 reanalysis of sealed direct repeats and prospective Path design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from eat_nav_p1_g3_direct_repeat_analyze import aggregate_root
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import SEEDS
from visualnav_transformer.eat_nav.p1_path_crossover_v2 import (
    build_crossover, fingerprint_material, material_values_from_source,
    reassess_direct_repeat,
)


TRACE_FIELDS = ("root_state", "post_root_state", "joint_position", "joint_velocity",
                "actor_input", "actor_action", "follower_command", "applied_command",
                "contact_net_forces_w", "contact_net_forces_w_history", "critic_observation",
                "critic_value", "reward", "applied_torque", "terminal")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, obj: dict) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(obj, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def trace_replay_audit(root: Path, pairs: list[dict]) -> dict:
    mismatches = []
    for pair in pairs:
        group, seed, env = pair["state_group"], pair["seed"], pair["env_id"]
        block = list(SEEDS[group]).index(seed)
        paths = [root / f"batch-{group}-{block}-repeat-{repeat}/output/trace-slot-{env}.jsonl"
                 for repeat in (0, 1)]
        rows = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
        if len(rows[0]) != len(rows[1]):
            mismatches.append({"pair": [group, seed, env], "signal": "trace_length"})
            continue
        for step, (a, b) in enumerate(zip(*rows)):
            first = next((field for field in TRACE_FIELDS if a.get(field) != b.get(field)), None)
            if first is not None:
                mismatches.append({"pair": [group, seed, env], "step": step,
                                   "signal": first})
                break
    return {"pairs_checked": len(pairs), "numeric_trace_exact_pairs": len(pairs) - len(mismatches),
            "first_mismatch_by_pair": mismatches}


def path_geometry(source_contract: dict) -> dict:
    by_bearing = {entry["bearing_deg"]: entry for entry in source_contract["paths"]}
    if not all(deg in by_bearing for deg in (-20, 0, 20)):
        raise ValueError("frozen Path bank lacks direct/left/right")
    result = {}
    for role, bearing in (("left_20", 20), ("direct", 0), ("right_20", -20)):
        # Positive yaw is robot-left in the documented base-frame convention.
        record = by_bearing[bearing]
        points = [(0., 0.), *(tuple(xy) for xy in record["waypoints"])]
        segments = [math.dist(a, b) for a, b in zip(points, points[1:])]
        turns = []
        for a, b, c in zip(points, points[1:], points[2:]):
            u = (b[0] - a[0], b[1] - a[1])
            v = (c[0] - b[0], c[1] - b[1])
            turns.append(math.atan2(u[0] * v[1] - u[1] * v[0],
                                    u[0] * v[0] + u[1] * v[1]))
        result[role] = {"bearing_deg": bearing, "path_sha256": record["path_sha256"],
                        "endpoint_base_xy_m": record["endpoint"], "point_count": len(record["waypoints"]),
                        "polyline_length_m": sum(segments),
                        "max_discrete_turn_rad": max(map(abs, turns), default=0.),
                        "max_discrete_curvature_per_m": max(
                            (abs(turn) / ((segments[i] + segments[i + 1]) / 2)
                             for i, turn in enumerate(turns)), default=0.)}
    return result


def prepare(root: Path, legacy_path: Path, env_cfg: Path, material_cfg: Path,
            original_g3_assignment: Path, output: Path) -> dict:
    legacy = json.loads(legacy_path.read_text())
    verified = aggregate_root(root)
    if verified != legacy:
        raise ValueError("sealed raw reaggregation differs from frozen aggregate")
    manifest = json.loads((root / "manifest.json").read_text())
    source_contract = root / "batch-first-differential-0-repeat-0/input/source-contract.json"
    if sha(source_contract) != manifest["source_input_sha256"]["source-contract.json"]:
        raise ValueError("source Path geometry SHA drift")
    g3_assignment = json.loads(original_g3_assignment.read_text())
    mapping = {"direct": "direct_duplicate_0", "left_20": "plus_20_deg",
               "right_20": "minus_20_deg"}
    geometry = path_geometry(json.loads(source_contract.read_text()))
    for role, slot in mapping.items():
        identities = {row["path_sha256"] for row in g3_assignment["trials"]
                      if row["candidate_slot"] == slot}
        if identities != {geometry[role]["path_sha256"]}:
            raise ValueError("original G3 Path SHA mismatch: " + role)
    values = material_values_from_source(env_cfg.read_text(), material_cfg.read_text())
    declared_sha = fingerprint_material(values)
    terrains = []
    for process in json.loads((root / "g3-assignment.json").read_text())["processes"]:
        batch = root / (f"batch-{process['anchor_id']}-{process['batch_index']}-"
                        f"repeat-{process['repeat_index']}")
        row = json.loads((batch / "output/terrain.json").read_text())
        terrains.append({"anchor_id": process["anchor_id"], "batch_seed": process["batch_seed"],
                         "repeat_index": process["repeat_index"],
                         "mesh_sha256": row["fingerprints"][0]["canonical_mesh_sha256"],
                         "material_sha256": row["material_sha256"]})
    reanalysis = reassess_direct_repeat(verified["trials"], terrains, declared_sha)
    reanalysis["trace_replay"] = trace_replay_audit(root, reanalysis["pairs"])
    schedule = build_crossover(SEEDS)
    result = {
        "schema": "eatnav_p1_g3_path_crossover_preparation_v2",
        "status": "offline_only_no_new_path_rollout",
        "analysis_sources": {
            "script_sha256": sha(Path(__file__)),
            "module_sha256": sha(Path(__file__).resolve().parents[1] /
                                 "src/visualnav_transformer/eat_nav/p1_path_crossover_v2.py"),
        },
        "frozen_evidence": {"manifest_sha256": sha(root / "manifest.json"),
                            "legacy_aggregate_sha256": sha(legacy_path),
                            "original_g3_assignment_sha256": sha(original_g3_assignment),
                            "source_contract_sha256": sha(source_contract)},
        "material": {"declared_terrain_values": values,
                     "canonical_declared_sha256": declared_sha,
                     "env_cfg_source_sha256": sha(env_cfg),
                     "isaac_material_cfg_source_sha256": sha(material_cfg),
                     "scope": "current_source_static_declaration_not_historical_live_PhysX_measurement",
                     "robot_contact_material": "startup_randomization_declares_fixed_0.8_static_0.6_dynamic_0_restitution; historical_live_values_not_recorded"},
        "reanalysis": reanalysis,
        "path_geometry": geometry,
        "common_goal_base_xy_m": json.loads(source_contract.read_text())["global_goal_base_xy_m"],
        "crossover_design": schedule,
    }
    write_new(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--legacy-aggregate", type=Path, required=True)
    parser.add_argument("--env-cfg", type=Path, required=True)
    parser.add_argument("--material-cfg", type=Path, required=True)
    parser.add_argument("--original-g3-assignment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(prepare(args.root, args.legacy_aggregate, args.env_cfg, args.material_cfg,
                  args.original_g3_assignment, args.output)["reanalysis"]["seed_env_mesh_correspondence_pairs"])


if __name__ == "__main__":
    main()
