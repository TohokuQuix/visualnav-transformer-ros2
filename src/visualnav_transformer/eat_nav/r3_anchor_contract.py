"""Outcome-blind identity and placement contract for the frozen R3 pilot."""
from __future__ import annotations

import math

from .radial_inventory import TERRAIN_ORDER, terrain_seed

CLASSES = ("transition", "differential", "graded", "flat-only")
EXPECTED = (
    ("k_rail_step_field", .40, 2, "transition"),
    ("k_rail_field", .60, 12, "transition"),
    ("k_rail_step_field", .60, 0, "differential"),
    ("boxes", .60, 1, "differential"),
    ("k_rail_step_field", .30, 6, "graded"),
    ("k_rail_field", .50, 1, "graded"),
    ("k_rail_field", .50, 4, "flat-only"),
    ("random_rough", .60, 6, "flat-only"),
)


def build_primary(proposal: dict, virtual: list[dict]) -> list[dict]:
    """Validate every proposal value against the frozen 448-pose scan."""
    if proposal.get("rule") != "R3" or len(proposal.get("candidates", [])) != 8:
        raise ValueError("frozen R3 proposal count or rule mismatch")
    by_id = {x["pose_id"]: x for x in virtual}
    if len(by_id) != 448:
        raise ValueError("frozen virtual pose grid incomplete")
    rows = []
    for index, candidate in enumerate(proposal["candidates"]):
        terrain, radius, direction, cls = EXPECTED[index]
        pose_id = candidate["pose_id"]
        if pose_id not in by_id:
            raise ValueError("proposal pose absent from virtual scan")
        reference = by_id[pose_id]
        pairs = (
            ("terrain", terrain), ("radius_m", radius),
            ("direction_index", direction), ("classification", cls),
            ("rank", index + 1), ("seed", terrain_seed(20261007, TERRAIN_ORDER.index(terrain))),
        )
        if any(candidate.get(key) != value for key, value in pairs):
            raise ValueError("proposal identity/class/rank changed")
        for key, scan_key in (
            ("terrain", "terrain"), ("radius_m", "radius_m"),
            ("direction_index", "direction_index"), ("seed", "terrain_seed"),
            ("dem_sha256", "dem_sha256"), ("requested_xy_yaw", "requested_xy_yaw"),
            ("source_family", "source_family"), ("classification", "classification"),
        ):
            if candidate[key] != reference[scan_key]:
                raise ValueError("proposal value differs from virtual scan: " + key)
        column = TERRAIN_ORDER.index(terrain)
        env_index = direction * len(TERRAIN_ORDER) + column
        theta = direction * math.pi / 8
        if (reference.get("classification") not in CLASSES or
            candidate.get("physical_stability") != "unmeasured" or
            abs(candidate["requested_xy_yaw"][2] - theta) > 1e-6 or
            len(candidate["requested_xy_yaw"]) != 3 or
            not all(math.isfinite(v) for v in candidate["requested_xy_yaw"])):
            raise ValueError("proposal pose frame or yaw invalid")
        rows.append({
            "pose_id": pose_id, "batch_seed": 20261007,
            "terrain": terrain, "terrain_seed": candidate["seed"],
            "source_family": candidate["source_family"],
            "row": direction, "column": column, "env_index": env_index,
            "angle_index": direction, "theta_rad": theta,
            "radius_m": radius, "requested_xy_yaw": candidate["requested_xy_yaw"],
            "virtual_dem_sha256": candidate["dem_sha256"],
            "virtual_class": cls, "virtual_interaction_onset_m": candidate["interaction_onset_m"],
            "rank": index + 1,
        })
    if len({x["pose_id"] for x in rows}) != 8 or len({x["env_index"] for x in rows}) != 8:
        raise ValueError("duplicate R3 pilot pose or environment")
    return rows


def split_shards(primary: list[dict]) -> tuple[list[dict], list[dict]]:
    if len(primary) != 8 or [x["rank"] for x in primary] != list(range(1, 9)):
        raise ValueError("frozen primary rank sequence changed")
    a = primary[::2]
    b = primary[1::2]
    if ({x["virtual_class"] for x in a} != set(CLASSES) or
        {x["virtual_class"] for x in b} != set(CLASSES) or
        max(len(a), len(b)) > 5):
        raise ValueError("R3 A/B class or view assignment changed")
    return a, b


def build_reserve(r3_ordered: list[dict], primary: list[dict]) -> list[dict]:
    primary_ids = {x["pose_id"] for x in primary}
    if len(primary_ids) != 8:
        raise ValueError("primary pose set incomplete")
    result = []
    for rank, item in enumerate(r3_ordered, 1):
        if item["pose_id"] in primary_ids:
            continue
        result.append({
            "selection_rank_in_r3": rank,
            "pose_id": item["pose_id"],
            "terrain": item["terrain"],
            "source_family": item["source_family"],
            "radius_m": item["radius_m"],
            "direction_index": item["direction_index"],
            "classification": item["classification"],
            "dem_sha256": item["dem_sha256"],
            "status": "reserve_only_not_in_execution_manifest",
        })
    return result


def requested_world(row: dict, origin_xyz: list[float]) -> dict:
    if len(origin_xyz) != 3 or not all(math.isfinite(v) for v in origin_xyz):
        raise ValueError("invalid scene origin")
    x, y, yaw = row["requested_xy_yaw"]
    theta = row["theta_rad"]
    if (abs(x - (origin_xyz[0] + row["radius_m"] * math.cos(theta))) > 1e-4 or
        abs(y - (origin_xyz[1] + row["radius_m"] * math.sin(theta))) > 1e-4 or
        abs(math.atan2(math.sin(yaw-theta), math.cos(yaw-theta))) > 1e-6):
        raise ValueError("proposal requested XY/yaw differs from scene origin")
    return {"world_x": x, "world_y": y,
            "world_z_reference": float(origin_xyz[2]), "world_yaw": yaw}


def select_r3_shard_rows(manifest_poses: list[dict], primary: list[dict]) -> list[dict]:
    if not 1 <= len(manifest_poses) <= 5:
        raise ValueError("shard view count invalid")
    by_id = {x["pose_id"]: x for x in primary}
    ids = [x["pose_id"] for x in manifest_poses]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate shard pose")
    rows = []
    for item in manifest_poses:
        expected = by_id.get(item["pose_id"])
        if expected is None or item != expected:
            raise ValueError("shard pose differs from frozen primary")
        rows.append(expected)
    return rows
