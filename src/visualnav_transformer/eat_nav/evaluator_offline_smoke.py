"""Offline-only E1 Evaluator smoke inputs. Never publishes a Path or command."""

from __future__ import annotations

import numpy as np

from .portable_input_v2 import SIM_2S, SIM_24J, SIM_FLIPPERS, extract_named


def _finite_shape(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite or wrong shape {array.shape}")
    return array.astype(np.float32)


def _joint_mean(names, velocities, joints):
    raw = extract_named(names, velocities, joints)
    return float(0.05 * np.clip(np.mean(raw), -100.0, 100.0))


def anchor_features(summary):
    """Read the pre-Path anchor only. Post-follower and trace rows are ignored."""
    anchor = summary["anchor"]
    dem = _finite_shape(anchor["dem_history"], (4, 2, 20, 20), "dem_history")
    if not np.isin(dem[:, 1], (0, 1)).all():
        raise ValueError("DEM mask must be binary")
    actor = _finite_shape(anchor["policy_pre_noise"], (22,), "policy_pre_noise")
    actor_delivered = _finite_shape(anchor["policy_actor_input"], (22,), "policy_actor_input")
    names = summary["raw_joint_effort_field_order"]
    positions = _finite_shape(anchor["joint_position"], (len(names),), "joint_position")
    velocities = _finite_shape(anchor["joint_velocity"], (len(names),), "joint_velocity")
    if len(set(names)) != len(names):
        raise ValueError("joint order has duplicates")
    if any("arm" in name for name in names):
        raise ValueError("simulator arm applicability requires a frozen stowed rule")
    flippers = np.asarray(extract_named(names, positions, SIM_FLIPPERS), dtype=np.float32)
    two = np.asarray([_joint_mean(names, velocities, group) for group in SIM_2S], dtype=np.float32)
    twentyfour = np.asarray([_joint_mean(names, velocities, group) for group in SIM_24J], dtype=np.float32)
    goal = _finite_shape(summary["goal_metadata"]["goal_xy"], (2,), "relative_goal")
    path_with_origin = _finite_shape(summary["goal_metadata"]["base_path"], (9, 2), "base_path")
    if np.linalg.norm(path_with_origin[0]) > 1e-5:
        raise ValueError("candidate Path does not start at the base origin")
    fold = _finite_shape([anchor["flipper_previous_target"]], (1,), "fold_mode")
    return {
        "dem_history": dem,
        "simulator_actor22_prepath": actor_delivered,
        "simulator_actor22_pre_noise": actor,
        "relative_goal": goal,
        "candidate_path": path_with_origin[1:],
        "body_angular_velocity": actor[:3] / 0.25,
        "projected_gravity": actor[3:6],
        "flipper_fold_mode": fold,
        "flipper_position": flippers,
        "availability_mask": np.ones(10, dtype=np.float32),
        "stale_mask": np.zeros(10, dtype=np.float32),
        "arm_applicability_mask": np.asarray([0.0], dtype=np.float32),
        "right_sprocket_angular_state_scaled": two[:1],
        "left_sprocket_angular_state_scaled": two[1:],
        "simulator_actor_track_aggregate_24j": twentyfour,
        "anchor_timestamp": np.asarray([summary["logical_t0_ns"]], dtype=np.int64),
    }


def path_features(dem, path):
    """Frozen deterministic 0.1 m base-yaw raster samples along an 8-point Path."""
    dem = _finite_shape(dem, (4, 2, 20, 20), "dem_history")
    path = _finite_shape(path, (8, 2), "candidate_path")
    points = np.vstack((np.zeros((1, 2), dtype=np.float32), path))
    differences = np.diff(points, axis=0)
    lengths = np.linalg.norm(differences, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError("duplicate candidate Path point")
    arc = np.r_[0.0, np.cumsum(lengths)]
    s = np.linspace(0.0, arc[-1], 64)
    x = np.interp(s, arc, points[:, 0])
    y = np.interp(s, arc, points[:, 1])
    col = np.rint(x / 0.1 + 9.5).astype(int)
    row = np.rint(y / 0.1 + 9.5).astype(int)
    inside = (col >= 0) & (col < 20) & (row >= 0) & (row < 20)
    coverage = np.zeros(64, dtype=np.float32)
    profile = np.zeros(64, dtype=np.float32)
    valid = np.flatnonzero(inside)
    coverage[valid] = dem[-1, 1, row[valid], col[valid]]
    profile[valid] = dem[-1, 0, row[valid], col[valid]] * coverage[valid]
    tangent = np.unwrap(np.arctan2(differences[:, 1], differences[:, 0]))
    curvature = np.diff(tangent) / np.maximum((lengths[:-1] + lengths[1:]) / 2, 1e-8)
    geometry = np.asarray([arc[-1], np.max(np.abs(curvature)), np.mean(np.abs(curvature)),
                           tangent[0], tangent[-1], arc[-1] - np.linalg.norm(path[-1])], dtype=np.float32)
    return {"path_geometry": geometry, "path_dem_profile": profile,
            "path_coverage": coverage}


def group_loss(predictions, targets):
    """Huber beta=1, first mean within group, then macro mean across groups."""
    if len(predictions) != len(targets) or not predictions:
        raise ValueError("group mismatch")
    losses = []
    for prediction, target in zip(predictions, targets):
        delta = np.abs(np.asarray(prediction) - np.asarray(target))
        if delta.shape != np.asarray(target).shape or not np.isfinite(delta).all():
            raise ValueError("invalid group")
        losses.append(float(np.mean(np.where(delta <= 1, 0.5 * delta**2, delta - 0.5))))
    return float(np.mean(losses))


def audit_records(records):
    """Reject any candidate-specific change in a supposedly pre-Path source."""
    if not records:
        raise ValueError("no records")
    by_group = {}
    for record in records:
        group = record["group"]
        common = record["common"]
        if group in by_group:
            for key, original in by_group[group].items():
                if key not in common or not np.array_equal(np.asarray(common[key]), np.asarray(original)):
                    raise ValueError(f"pre-Path common field differs: {group}/{key}")
        else:
            by_group[group] = common
    return {"groups": len(by_group), "records": len(records), "common_fields": sorted(records[0]["common"])}
