"""E0 only: deterministic, outcome-free local Path geometry.

No simulator, learned model, or follower process is imported here.  The first
command diagnostic implements the fresh, forward-tracking branch of the
production follower; it is not a prediction of its later response.
"""

from __future__ import annotations

import math

import numpy as np


FAMILIES = ("cubic_bezier", "biarc", "clamped_cubic_spline")
ALPHAS = (0.15, 0.30, 0.45, 0.60)
IDS = ("direct_reference",) + tuple(f"{side}_{i}" for side in ("left", "right") for i in range(1, 5))


def _bezier(goal: np.ndarray, offset: float, t: np.ndarray) -> np.ndarray:
    distance = float(np.linalg.norm(goal))
    normal = np.array([-goal[1], goal[0]]) / distance
    p0 = np.zeros(2)
    p1 = np.array([distance * .25, 0.0])
    p2 = goal - goal / distance * distance * .25 + normal * offset
    return ((1-t)[:, None]**3*p0 + 3*((1-t)**2*t)[:, None]*p1
            + 3*((1-t)*t**2)[:, None]*p2 + t[:, None]**3*goal)


def _spline(goal: np.ndarray, offset: float, t: np.ndarray) -> np.ndarray:
    distance = float(np.linalg.norm(goal))
    normal = np.array([-goal[1], goal[0]]) / distance
    knots = np.stack((np.zeros(2), _bezier(goal, 0.0, np.array([.5]))[0] + normal * offset, goal))
    h = .5
    d0 = np.array([distance, 0.0])
    d2 = goal
    slopes = np.diff(knots, axis=0) / h
    rhs = 6 * np.stack((slopes[0]-d0, slopes[1]-slopes[0], d2-slopes[1]))
    matrix = np.array([[2*h,h,0], [h,4*h,h], [0,h,2*h]])
    second = np.linalg.solve(matrix, rhs)
    index = (t >= .5).astype(int)
    u = t - index * h
    a = (h-u)[:, None]
    b = u[:, None]
    return (second[index]*a**3/(6*h) + second[index+1]*b**3/(6*h)
            + (knots[index]-second[index]*h*h/6)*a/h
            + (knots[index+1]-second[index+1]*h*h/6)*b/h)


def _arc(start: np.ndarray, tangent: np.ndarray, end: np.ndarray, t: np.ndarray):
    delta = end-start
    cross = tangent[0]*delta[1]-tangent[1]*delta[0]
    curvature = 2*cross / float(delta @ delta)
    if abs(curvature) < 1e-10:
        points = start[None] + t[:, None]*delta
        return points, tangent, 0.0
    theta = 2*math.atan2(cross, float(tangent @ delta))
    center = start + np.array([-tangent[1], tangent[0]])/curvature
    radial = start-center
    angles = theta*t
    points = center + np.stack((np.cos(angles)*radial[0]-np.sin(angles)*radial[1],
                                np.sin(angles)*radial[0]+np.cos(angles)*radial[1]), axis=1)
    final = np.array([math.cos(theta)*tangent[0]-math.sin(theta)*tangent[1],
                      math.sin(theta)*tangent[0]+math.cos(theta)*tangent[1]])
    return points, final, curvature


def _biarc(goal: np.ndarray, offset: float, t: np.ndarray):
    distance = float(np.linalg.norm(goal))
    normal = np.array([-goal[1], goal[0]]) / distance
    join = _bezier(goal, 0.0, np.array([.5]))[0] + normal*offset
    if np.linalg.norm(join) < 1e-7 or np.linalg.norm(goal-join) < 1e-7:
        raise ValueError("degenerate_radius")
    left, tangent, k1 = _arc(np.zeros(2), np.array([1.0, 0.0]), join,
                             np.minimum(t*2, 1))
    right, _, k2 = _arc(join, tangent, goal, np.maximum(2*t-1, 0))
    points = np.where((t <= .5)[:, None], left, right)
    return points, abs(k1-k2)


def arc_resample(points: np.ndarray, count: int = 8) -> np.ndarray:
    if count < 2:
        raise ValueError("point_count must be >=2")
    cumulative = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if np.any(np.diff(cumulative) <= 1e-12):
        raise ValueError("nonmonotone_arc_length")
    target = np.linspace(0, cumulative[-1], count+1)[1:]
    return np.stack([np.interp(target, cumulative, points[:, axis]) for axis in range(2)], axis=1)


def _curve(family, goal, offset, t):
    if family == "cubic_bezier":
        return _bezier(goal, offset, t), 0.0
    if family == "clamped_cubic_spline":
        return _spline(goal, offset, t), 0.0
    return _biarc(goal, offset, t)


def _normal_peak(reference, candidate, side):
    fraction = np.linspace(0, 1, 129)
    def normalized(points):
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        return np.stack([np.interp(fraction*arc[-1], arc, points[:, i]) for i in range(2)], axis=1)
    ref = normalized(reference)
    other = normalized(candidate)
    tangent = np.gradient(ref, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.stack((-tangent[:, 1], tangent[:, 0]), axis=1)
    signed = np.sum((other-ref)*normal, axis=1)*side
    return float(np.max(signed))


def generate(family: str, goal_xy, *, lateral_scale_m: float, point_count: int = 8,
             dense_count: int = 257):
    """Return all nine common-anchor/endpoint candidates; never tune to outcomes."""
    if family not in FAMILIES:
        raise ValueError("unknown_curve_family")
    goal = np.asarray(goal_xy, dtype=float)
    if goal.shape != (2,) or not np.isfinite(goal).all() or goal[0] <= 0:
        raise ValueError("forward_goal_required")
    if not math.isfinite(lateral_scale_m) or lateral_scale_m <= 0:
        raise ValueError("positive_lateral_scale_required")
    t = np.linspace(0, 1, dense_count)
    reference, _ = _curve(family, goal, 0.0, t)
    result = []
    for candidate_id in IDS:
        side, _, rank = candidate_id.partition("_")
        if candidate_id == "direct_reference":
            target = 0.0
            offset = 0.0
        else:
            rank = int(candidate_id[-1])
            target = ALPHAS[rank-1]*lateral_scale_m
            direction = 1 if side == "left" else -1
            upper = max(target*2, 1e-4)
            for _ in range(12):
                trial, _ = _curve(family, goal, direction*upper, t)
                if _normal_peak(reference, trial, direction) >= target:
                    break
                upper *= 1.5
            else:
                result.append({"candidate_id": candidate_id, "target_normal_amplitude_m": target,
                               "offset_parameter_m": None, "dense": None, "waypoints": None,
                               "join_curvature_jump_per_m": None,
                               "generation_error": "amplitude_unreachable"})
                continue
            lower = 0.0
            for _ in range(24):
                middle = (lower+upper)*.5
                trial, _ = _curve(family, goal, direction*middle, t)
                if _normal_peak(reference, trial, direction) < target:
                    lower = middle
                else:
                    upper = middle
            offset = direction*(lower+upper)*.5
        try:
            dense, jump = _curve(family, goal, offset, t)
            path = arc_resample(dense, point_count)
            result.append({"candidate_id": candidate_id, "target_normal_amplitude_m": float(target),
                           "realized_normal_amplitude_m": _normal_peak(reference, dense, 1 if offset >= 0 else -1),
                           "offset_parameter_m": float(offset),
                           "dense": dense, "waypoints": path,
                           "join_curvature_jump_per_m": float(jump), "generation_error": None})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            result.append({"candidate_id": candidate_id, "target_normal_amplitude_m": float(target),
                           "offset_parameter_m": float(offset),
                           "dense": None, "waypoints": None,
                           "join_curvature_jump_per_m": None, "generation_error": str(exc)})
    return result


def c0_fresh_forward(waypoints, *, lookahead_m=.5, speed_mps=.3,
                     angular_gain=1.0, angular_direction_sign=.5,
                     max_angular_speed=1.6, goal_tolerance_m=.2,
                     turn_threshold_rad=.6, min_linear_speed_scale=.0):
    """Production C0 arithmetic at origin/yaw=0 in fresh controller state.

    Uses the actual eight-pose Path projection and goal-extension branches.
    TF and mutable controller state are prerequisites, not simulated here.
    """
    path = np.asarray(waypoints, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("Path needs at least two poses")
    if np.linalg.norm(path[-1]) < goal_tolerance_m:
        return {"linear_x": 0.0, "angular_z": 0.0, "carrot_xy": None, "mode": "goal"}
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    best = None
    for i, length in enumerate(lengths):
        if length < 1e-6:
            continue
        ratio = float(np.clip(-path[i] @ (path[i+1]-path[i]) / length**2, 0, 1))
        projection = path[i] + ratio*(path[i+1]-path[i])
        remaining = (1-ratio)*length + float(lengths[i+1:].sum())
        key = (float(np.linalg.norm(projection)), -remaining)
        if best is None or key[0] < best[0][0] or (abs(key[0]-best[0][0]) < 1e-9 and key[1] < best[0][1]):
            best = (key, i, projection, remaining)
    if best is None:
        raise ValueError("degenerate_Path")
    _, index, projection, remaining = best
    # Fresh controller determines forward/backward from the first 0.5 m
    # tangent of the Path at the closest segment.
    sample = min(.5, float(lengths[index:].sum()))
    tangent_sum = np.zeros(2)
    for i in range(index, len(lengths)):
        take = min(sample, lengths[i])
        tangent_sum += (path[i+1]-path[i]) / lengths[i] * take
        sample -= take
        if sample <= 0:
            break
    if tangent_sum[0] < -1e-9:
        raise ValueError("fresh_controller_would_reverse")
    terminal_sum = np.zeros(2)
    remaining_sample = .5
    for i in range(len(lengths)-1, -1, -1):
        if lengths[i] < 1e-6:
            continue
        take = min(remaining_sample, lengths[i])
        terminal_sum += (path[i+1]-path[i])/lengths[i]*take
        remaining_sample -= take
        if remaining_sample <= 0:
            break
    terminal_tangent = terminal_sum/np.linalg.norm(terminal_sum)
    if remaining <= lookahead_m:
        along = float((-path[-1]) @ terminal_tangent)
        if along >= 0:
            return {"linear_x": 0.0, "angular_z": 0.0, "carrot_xy": None,
                    "mode": "passed_goal_line"}
        carrot = path[-1] + (along+lookahead_m)*terminal_tangent
    else:
        left = lookahead_m
        current = projection
        carrot = path[-1]
        for i in range(index+1, len(path)):
            segment = float(np.linalg.norm(path[i]-current))
            if left <= segment:
                carrot = current + left/segment*(path[i]-current)
                break
            left -= segment
            current = path[i]
    error = math.atan2(carrot[1], carrot[0])
    omega = float(np.clip(angular_direction_sign*angular_gain*error,
                          -max_angular_speed, max_angular_speed))
    linear = 0.0 if abs(error) >= turn_threshold_rad else speed_mps*max(
        min_linear_speed_scale, 1-abs(error)/math.pi)
    return {"linear_x": float(linear), "angular_z": omega,
            "carrot_xy": carrot.tolist(), "mode": "tracking"}
