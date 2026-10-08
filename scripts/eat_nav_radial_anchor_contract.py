"""Offline, candidate-free contract checks for radial settling anchors."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


CONTRACT_VERSION = "eatnav_radial_anchor_execution_v1"
FORBIDDEN = frozenset(("episode_key", "local_step", "action_prefix", "synthetic_episode"))
LOGICAL_ROW_FIELDS = frozenset(("physical_run_id", "order", "anchor_id", "candidate_path_sha256",
                                "random_seed", "candidate_retry_count", "navigation_global_goal"))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def field_hash(fields: list[str]) -> str:
    return hashlib.sha256(json.dumps(fields, separators=(",", ":")).encode()).hexdigest()


def verify_radial_source(anchor: dict, freeze: dict, pose: dict, actor: dict, frames: list[dict], record: dict,
                         expected_actor_field_hash: str) -> None:
    if any(FORBIDDEN.intersection(item) for item in (anchor, freeze, pose, actor, record)):
        raise ValueError("synthetic_episode_prefix_forbidden")
    if anchor["pose_id"] != pose["pose_id"] or anchor["pose_id"] != record["pose_id"]:
        raise ValueError("pose_identity_mismatch")
    for key in ("terrain", "terrain_seed", "source_family", "requested_xy_yaw"):
        if anchor[key] != pose[key]:
            raise ValueError(f"{key}_mismatch")
    if pose["batch_seed"] not in freeze["batch_seeds"] or pose["env_index"] != record["env_index"]:
        raise ValueError("environment_identity_mismatch")
    if freeze["planned_physics_steps"] != 200 or record["physics_step_count"] != 200 or record["control_step_count"] != 20:
        raise ValueError("settling_step_mismatch")
    if not freeze["candidate_free"] or not record["candidate_free"] or record["action"] != [0.0] * 6:
        raise ValueError("candidate_free_violation")
    if actor["shape"] != [22] or not actor["finite"] or len(actor["field_order"]) != 22 or len(actor["values"]) != 22:
        raise ValueError("actor_schema_mismatch")
    if field_hash(actor["field_order"]) != expected_actor_field_hash:
        raise ValueError("actor_field_order_hash_mismatch")
    if actor["config_identity"] != anchor["actor_config_identity"]:
        raise ValueError("actor_config_identity_mismatch")
    if len(frames) != 4:
        raise ValueError("dem_frame_count_mismatch")
    for i, frame in enumerate(frames):
        if frame["frame_index"] != i or frame["physics_step"] != 50 * (i + 1):
            raise ValueError("dem_order_mismatch")
        if frame["shape"] != [2, 20, 20] or frame["resolution_m"] != 0.1 or not frame["finite"]:
            raise ValueError("dem_schema_mismatch")
        if len(frame["source_mask"]) != 20 or any(len(row) != 20 for row in frame["source_mask"]):
            raise ValueError("dem_mask_shape_mismatch")
        if len(frame["transformed"]) != 2 or any(len(ch) != 20 or any(len(row) != 20 for row in ch) for ch in frame["transformed"]):
            raise ValueError("dem_transformed_shape_mismatch")


def reject_episode_prefix(anchor: dict) -> None:
    if anchor.get("pose_id") or anchor.get("source_shard"):
        raise ValueError("radial_anchor_incompatible_with_episode_prefix")


def output_overlay(manifest: dict, output_root: str) -> tuple[dict, list[dict]]:
    original = manifest["rows"]
    out = json.loads(json.dumps(manifest))
    diff = []
    for before, after in zip(original, out["rows"], strict=True):
        after["output_directory"] = str(Path(output_root) / "runs" / before["physical_run_id"])
        changed = {key: {"old": before.get(key), "new": after.get(key)} for key in before.keys() | after.keys() if before.get(key) != after.get(key)}
        if set(changed) != {"output_directory"} or any(before[k] != after[k] for k in LOGICAL_ROW_FIELDS):
            raise ValueError("manifest_logical_identity_changed")
        diff.append({"physical_run_id": before["physical_run_id"], "changed": changed})
    if len(out["rows"]) != 28 or any(row["order"] != i for i, row in enumerate(out["rows"])):
        raise ValueError("manifest_row_order_changed")
    return out, diff


def verify_output_overlay(original: dict, overlay: dict) -> None:
    if original.keys() != overlay.keys() or len(original['rows']) != 28 or len(overlay['rows']) != 28:
        raise ValueError('manifest_topology_changed')
    for key in original.keys() - {'rows'}:
        if original[key] != overlay[key]:
            raise ValueError('manifest_metadata_changed: '+key)
    for before, after in zip(original['rows'], overlay['rows'], strict=True):
        if before.keys() != after.keys() or any(before[k] != after[k] for k in before.keys()-{'output_directory'}):
            raise ValueError('manifest_logical_identity_changed')
        if not after['output_directory'].endswith('/'+before['physical_run_id']):
            raise ValueError('manifest_output_mapping_invalid')


def require_completion(exit_code: int, marker: dict | None, trace: list[dict]) -> None:
    if exit_code != 0 or not marker or marker.get("physics_steps") != 200:
        raise ValueError("missing_or_invalid_completion_marker")
    if not trace or trace[-1].get("physics_step") != 200:
        raise ValueError("incomplete_trace")
    if any(row.get("termination_phase") not in (None, "none", "pre_evaluation") or
           row.get("reset_counter") != 0 or row.get("episode_id") != 0 or row.get("post_reset_state")
           for row in trace):
        raise ValueError("physical_termination_or_reset_contamination")
