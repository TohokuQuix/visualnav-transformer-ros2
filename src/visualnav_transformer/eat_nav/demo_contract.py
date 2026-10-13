"""Transport independent contracts for the ROS2/Isaac EAT-Nav demonstration.

The frozen P1 bank is used as a template.  Rotation and length scaling for an
RViz goal create demo candidates; they are not P1 experimental candidates.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .evaluator.preprocessing import body_local_path


ROLES = {"direct": 0, "left_20": 20, "right_20": -20}
CHECKPOINT_SHA256 = "329341fbf56ccb6bbc2f05b3db6b143cc1a1e9553ba2796e3dec3c785f30e470"
DATASET_SHA256 = "bcf94bec2e7578532fd85a0ba64fabb6f018f9b5ff8d92ae73dedc883ffb5a8a"
RL_CHECKPOINT_SHA256 = "b789f3ab4820a62fffeaaf5a600e5a1479cc624240e5d027772bb3fee8d51a3d"


def sensor_stamp_valid(logical_ns, actual_update_ns):
    """Check a 250 ms raycaster against the control clock with float-time slack.

    Isaac stores the sensor update time as a floating-point simulation second;
    converting it to nanoseconds can put it a few microseconds ahead of the
    integer control stamp at an otherwise equal boundary.
    """
    age_ns = int(logical_ns) - int(actual_update_ns)
    return -1_000_000 <= age_ns <= 300_000_000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bank(path):
    document = json.loads(Path(path).read_text())
    rows = {}
    for row in document["paths"]:
        angle = row["bearing_deg"]
        if angle in ROLES.values():
            points = np.asarray(row["waypoints"], dtype=np.float64)
            if points.shape != (8, 2) or not np.isfinite(points).all():
                raise ValueError("frozen bank shape/nonfinite")
            role = next(name for name, value in ROLES.items() if value == angle)
            rows[role] = points
    if set(rows) != set(ROLES):
        raise ValueError("frozen Direct/Left/Right bank incomplete")
    return rows


def candidates_for_goal(bank, robot_pose_map, goal_map, *, max_radius_m=0.75):
    """Rotate the 0.75 m P1 bank toward a local subgoal, retaining eight points."""
    pose = np.asarray(robot_pose_map, dtype=np.float64)
    goal = np.asarray(goal_map, dtype=np.float64)
    if pose.shape != (3,) or goal.shape != (2,) or not np.isfinite(pose).all() or not np.isfinite(goal).all():
        raise ValueError("invalid map pose/goal")
    relative = goal - pose[:2]
    distance = float(np.linalg.norm(relative))
    if not 0.05 <= distance or not 0 < max_radius_m <= 0.8:
        raise ValueError("goal too close or local radius invalid")
    bearing = math.atan2(relative[1], relative[0]) - pose[2]
    scale = min(distance, max_radius_m) / 0.75
    c, s = math.cos(bearing), math.sin(bearing)
    rotation = np.array([[c, -s], [s, c]])
    body = {role: (points * scale) @ rotation.T for role, points in bank.items()}
    c, s = math.cos(pose[2]), math.sin(pose[2])
    world_rotation = np.array([[c, -s], [s, c]])
    mapped = {role: points @ world_rotation.T + pose[:2] for role, points in body.items()}
    for role in ROLES:
        if not np.allclose(body_local_path(mapped[role], pose[:2], pose[2]), body[role], atol=1e-6):
            raise ValueError("map/body Path transform mismatch")
    return body, mapped, {"goal_distance_m": distance, "subgoal_distance_m": min(distance, max_radius_m),
                          "bearing_body_rad": bearing, "pilot_distribution_matched": bool(
                              abs(distance - 0.75) < 1e-6 and abs(bearing) < 1e-6)}


def validate_evaluator_sample(dem, state, paths, *, dem_stamps_ns, state_stamp_ns,
                              now_ns, max_age_ns=300_000_000):
    dem = np.asarray(dem, dtype=np.float32)
    state = np.asarray(state, dtype=np.float32)
    paths = np.asarray(paths, dtype=np.float32)
    stamps = tuple(int(x) for x in dem_stamps_ns)
    if dem.shape != (4, 2, 20, 20) or state.shape != (22,) or paths.ndim != 3 or \
            paths.shape[1:] != (8, 2):
        raise ValueError("Evaluator input shape mismatch")
    if not np.isfinite(state).all() or not np.isfinite(paths).all() or \
            not np.isfinite(dem).all() or not np.isin(dem[:, 1], (0., 1.)).all() or \
            np.any(np.mean(dem[:, 1], axis=(-2, -1)) < 0.25):
        raise ValueError("Evaluator input nonfinite/mask mismatch")
    if len(stamps) != 4 or any(b - a != 250_000_000 for a, b in zip(stamps, stamps[1:])) or \
            stamps[-1] != int(state_stamp_ns) or not 0 <= int(now_ns) - stamps[-1] <= max_age_ns:
        raise ValueError("Evaluator input timestamp mismatch/stale")


def select_best(scores, roles=("direct", "left_20", "right_20")):
    values = np.asarray(scores, dtype=np.float64)
    if values.shape != (len(roles),) or not np.isfinite(values).all():
        raise ValueError("Evaluator score shape/nonfinite")
    return roles[int(np.argmax(values))]
