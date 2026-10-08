"""Outcome-blind goal-bearing candidate-bank geometry and DEM diagnostics.

The functions in this module are offline only.  A follower command computed
here is an initial-command estimate, never a simulated controller response.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math

import numpy as np

from .candidate_bank_geometry import arc_resample, c0_fresh_forward
from .candidate_bank_centerline import _has_self_intersection


BEARINGS_DEG = (-20, -10, 0, 10, 20)
CLASSES = ("transition", "differential", "graded", "flat-only")
AMPLITUDE_POOL_M = (0.04, 0.08, 0.12, 0.16, 0.20)
OFFSET_POOL_M = (0.06, 0.12, 0.18, 0.24)
FOLLOWER_CURVATURE_SCREEN_PER_M = 1.236


def goal_for_bearing(bearing_deg: int) -> np.ndarray:
    if bearing_deg not in BEARINGS_DEG:
        raise ValueError("bearing outside frozen set")
    angle = math.radians(bearing_deg)
    return np.array([0.6 * math.cos(angle), 0.6 * math.sin(angle)], dtype=float)


def freeze_anchors(rows_by_seed: dict[str, list[dict]]) -> list[dict]:
    """Use the earliest previously ranked, physically supplied pose per class."""
    if set(rows_by_seed) != {"first", "second"}:
        raise ValueError("both frozen seeds required")
    selected = []
    for seed in ("first", "second"):
        for terrain_class in CLASSES:
            eligible = [r for r in rows_by_seed[seed]
                        if r.get("virtual_class") == terrain_class
                        and r.get("physical_supply") is True
                        and r.get("classification", "settled_anchor_eligible") == "settled_anchor_eligible"
                        and r.get("settled_class", terrain_class) == terrain_class]
            if not eligible:
                raise ValueError(f"missing physical supply: {seed}/{terrain_class}")
            picked = min(eligible, key=lambda r: r["rank"])
            selected.append(dict(picked, seed_label=seed, terrain_class=terrain_class))
    if len({r["pose_id"] for r in selected}) != 8:
        raise ValueError("anchor identities are not unique")
    return selected


def _unit_and_normal(goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = float(np.linalg.norm(goal))
    if goal.shape != (2,) or not np.isfinite(goal).all() or distance <= 0 or goal[0] <= 0:
        raise ValueError("finite forward goal required")
    unit = goal / distance
    return unit, np.array([-unit[1], unit[0]])


def _bezier(controls: np.ndarray, t: np.ndarray) -> np.ndarray:
    u = 1 - t
    return (u[:, None] ** 3 * controls[0]
            + 3 * (u * u * t)[:, None] * controls[1]
            + 3 * (u * t * t)[:, None] * controls[2]
            + t[:, None] ** 3 * controls[3])


def _curve_stats(controls: np.ndarray) -> dict:
    t = np.linspace(0.0, 1.0, 513)
    derivative = 3 * ((1-t)[:, None]**2 * (controls[1]-controls[0])
                      + 2*((1-t)*t)[:, None]*(controls[2]-controls[1])
                      + t[:, None]**2*(controls[3]-controls[2]))
    second = 6 * ((1-t)[:, None]*(controls[2]-2*controls[1]+controls[0])
                  + t[:, None]*(controls[3]-2*controls[2]+controls[1]))
    speed = np.linalg.norm(derivative, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        curvature = np.abs(derivative[:, 0]*second[:, 1]-derivative[:, 1]*second[:, 0]) / speed**3
    finite = curvature[np.isfinite(curvature)]
    return {"maximum_curvature_per_m": float(np.max(finite)) if len(finite) else None,
            "curvature_continuity_max_delta_per_m": float(np.max(np.abs(np.diff(finite)))) if len(finite)>1 else None,
            "min_parametric_speed": float(np.min(speed))}


def path_sha(waypoints: np.ndarray) -> str:
    path = np.asarray(waypoints, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or not np.isfinite(path).all():
        raise ValueError("finite [N,2] Path required")
    return hashlib.sha256(json.dumps(path.tolist(), separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def bezier_candidate(bank: str, goal_xy, magnitude_m: float, side: str,
                     terminal_mode: str) -> dict:
    """Generate eight future poses, excluding the implicit robot-origin start."""
    goal = np.asarray(goal_xy, dtype=float)
    unit, normal = _unit_and_normal(goal)
    if bank not in ("A", "B") or side not in ("direct", "left", "right"):
        raise ValueError("invalid bank/side")
    if terminal_mode not in ("common", "goal_connected", "strict_goal_facing"):
        raise ValueError("invalid terminal mode")
    if not math.isfinite(magnitude_m) or magnitude_m < 0:
        raise ValueError("invalid magnitude")
    sign = 0 if side == "direct" else 1 if side == "left" else -1
    if sign and magnitude_m == 0:
        raise ValueError("non-direct requires positive magnitude")
    endpoint = goal.copy() if bank == "A" or sign == 0 else goal + sign*magnitude_m*normal
    if sign == 0:
        controls = np.stack((np.zeros(2), goal/3, 2*goal/3, goal))
    elif bank == "A":
        # Bernstein B1+B2 peaks at 0.75; this makes magnitude the signed
        # maximum lateral distance from the global-goal baseline.
        offset = sign*magnitude_m*4/3*normal
        controls = np.stack((np.zeros(2), goal/3+offset, 2*goal/3+offset, goal))
    else:
        # Both alternatives keep one task goal.  Strict terminal direction
        # points exactly from local endpoint toward it; the blended variant
        # retains forward motion while aiming toward the same task goal.
        if terminal_mode == "strict_goal_facing":
            terminal = -sign*normal
        else:
            terminal = 0.30*unit-sign*magnitude_m*normal
            terminal /= np.linalg.norm(terminal)
        controls = np.stack((np.zeros(2), goal/3+sign*magnitude_m*.4*normal,
                             endpoint-.15*terminal, endpoint))
    source = _bezier(controls, np.linspace(0, 1, 513))
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(source, axis=0), axis=1))]
    stations = np.linspace(0, arc[-1], math.ceil(arc[-1]/0.015)+1)
    dense = np.stack([np.interp(stations, arc, source[:, axis]) for axis in range(2)], axis=1)
    waypoints = arc_resample(source, 8)
    return {"bank": bank, "side": side, "magnitude_m": float(magnitude_m),
            "terminal_mode": terminal_mode, "global_goal": goal, "endpoint": endpoint,
            "controls": controls, "dense": dense, "stations_m": stations,
            "waypoints": waypoints, "arc_length_m": float(arc[-1]),
            "initial_tangent_xy": (controls[1]-controls[0])/np.linalg.norm(controls[1]-controls[0]),
            "terminal_tangent_xy": (controls[3]-controls[2])/np.linalg.norm(controls[3]-controls[2]),
            **_curve_stats(controls)}


def audit_geometry(row: dict, global_goal_xy, map_pose_xy_yaw) -> dict:
    goal = np.asarray(global_goal_xy, dtype=float)
    pose = np.asarray(map_pose_xy_yaw, dtype=float)
    curve = np.asarray(row["dense"])
    path = np.asarray(row["waypoints"])
    reasons = []
    if pose.shape != (3,) or not np.isfinite(pose).all():
        reasons.append("invalid_map_pose")
    if curve.ndim != 2 or curve.shape[1] != 2 or path.shape != (8, 2) or not np.isfinite(curve).all() or not np.isfinite(path).all():
        reasons.append("nonfinite_or_path_shape")
        return {"hard_valid": False, "invalid_reasons": reasons}
    if np.linalg.norm(curve[0]) > 1e-9:
        reasons.append("start_not_origin")
    if np.linalg.norm(path[-1]-row["endpoint"]) > 1e-6:
        reasons.append("endpoint_mismatch")
    if np.linalg.norm(row["global_goal"]-goal) > 1e-9:
        reasons.append("global_goal_mismatch")
    if not np.all(np.abs(curve) <= 1.0 + 1e-6):
        reasons.append("centerline_outside_dem")
    if row["bank"] == "B" and not np.all(np.abs(row["endpoint"]) <= 1.0 + 1e-6):
        reasons.append("endpoint_outside_dem")
    if _has_self_intersection(curve):
        reasons.append("self_intersection")
    unit, _ = _unit_and_normal(goal)
    progress = np.diff(curve, axis=0) @ unit
    if np.min(progress) < -1e-8:
        reasons.append("reverse_goal_axis")
    if np.max(np.linalg.norm(np.diff(curve, axis=0), axis=1)) > 0.02000001:
        reasons.append("dense_spacing")
    if np.any(np.linalg.norm(np.diff(np.vstack((np.zeros((1,2)), path)), axis=0), axis=1) <= 1e-8):
        reasons.append("path_order")
    if not reasons and pose.shape == (3,):
        c, s = math.cos(pose[2]), math.sin(pose[2])
        rotation = np.array([[c, -s], [s, c]])
        mapped = path @ rotation.T + pose[:2]
        restored = (mapped-pose[:2]) @ rotation
        if np.max(np.abs(restored-path)) > 1e-9:
            reasons.append("inverse_map_transform")
    return {"hard_valid": not reasons, "invalid_reasons": reasons,
            "path_sha256": path_sha(path), "maximum_dense_step_m": float(np.max(np.linalg.norm(np.diff(curve, axis=0), axis=1))),
            "minimum_goal_axis_progress_m": float(np.min(progress)),
            "map_inverse_max_error_m": float(np.max(np.abs(restored-path))) if not reasons else None}


def _sample_grid(points: np.ndarray, height: np.ndarray, mask: np.ndarray):
    cols = np.rint(points[:, 0]/.1 + 9.5).astype(int)
    rows = np.rint(points[:, 1]/.1 + 9.5).astype(int)
    inside = ((cols >= 0)&(cols < 20)&(rows >= 0)&(rows < 20)
              & (np.abs(points[:, 0]) <= 1.0)&(np.abs(points[:, 1]) <= 1.0))
    known = np.zeros(len(points), dtype=bool)
    values = np.zeros(len(points), dtype=float)
    known[inside] = mask[rows[inside], cols[inside]] > .5
    values[known] = height[rows[known], cols[known]]
    return inside, known, values, rows, cols


def corridor_class_proxy(candidate: dict, height, mask) -> dict:
    """Path-aligned three-band proxy using frozen R3 class thresholds.

    This is a diagnostic on candidate geometry, not the original axis-aligned
    20x20 anchor classifier; it must never replace the frozen anchor class.
    """
    from .radial_interaction import (DIFFERENTIAL_M, GRADE_SLOPE,
                                     HEIGHT_CHANGE_M, STEP_CHANGE_M)
    stations = np.linspace(0, min(.6, candidate['arc_length_m']), 7)
    dense_stations = candidate['stations_m']
    center = np.stack([np.interp(stations, dense_stations, candidate['dense'][:,axis])
                       for axis in range(2)], axis=1)
    tangent = np.gradient(center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:,None]
    normal = np.stack([-tangent[:,1],tangent[:,0]],axis=1)
    profiles = {}
    coverage = {}
    for name, offset in (('left',.3),('center',0.),('right',-.3)):
        inside, known, values, _, _ = _sample_grid(center+offset*normal,
                                                   np.asarray(height),np.asarray(mask))
        coverage[name] = float(known.mean())
        profiles[name] = values if known.all() else None
    result = {'class_proxy':'unknown','band_known_fraction':coverage,
              'method':'path_aligned_3_band_0_6m_proxy_not_frozen_anchor_classifier'}
    if min(coverage.values()) < .5 or any(value is None for value in profiles.values()):
        return result
    all_profiles = np.stack([profiles[name] for name in ('left','center','right')])
    lateral = float(np.max(np.ptp(all_profiles,axis=0)))
    step = float(np.max(np.abs(np.diff(all_profiles,axis=1))))
    central = profiles['center']
    delta = float(central[-1]-np.median(central[:2]))
    slope = step/max(float(stations[1]-stations[0]),1e-12)
    if lateral >= DIFFERENTIAL_M:
        cls = 'differential'
    elif step >= STEP_CHANGE_M and abs(float(central[1]-central[0])) < HEIGHT_CHANGE_M:
        cls = 'transition'
    elif abs(delta) >= HEIGHT_CHANGE_M or slope >= GRADE_SLOPE:
        cls = 'graded'
    else:
        cls = 'flat-only'
    return {**result,'class_proxy':cls,'max_lateral_difference_m':lateral,
            'maximum_step_m':step,'forward_height_change_m':delta,'maximum_slope':slope}


def corridor_audit(candidate: dict, height, mask, proxy_bboxes: dict) -> dict:
    """Descriptive raster audit; proxy rectangles are not contact geometry."""
    h = np.asarray(height, dtype=float)
    m = np.asarray(mask, dtype=float)
    if h.shape != (20,20) or m.shape != (20,20) or not np.isfinite(h).all() or not np.isfinite(m).all():
        raise ValueError("finite 20x20 DEM and mask required")
    curve = np.asarray(candidate["dense"])
    stations = np.asarray(candidate["stations_m"])
    inside, known, values, _, _ = _sample_grid(curve, h, m)
    profile = [[float(stations[i]), float(values[i])] for i in np.flatnonzero(known)]
    adjacent = np.flatnonzero(known[:-1] & known[1:])
    slope = [abs(values[i+1]-values[i])/max(stations[i+1]-stations[i],1e-12) for i in adjacent]
    steps = [abs(values[i+1]-values[i]) for i in adjacent]
    baseline = float(values[np.flatnonzero(known)[0]]) if known.any() else None
    onset = next((float(stations[i]) for i in np.flatnonzero(known)
                  if baseline is not None and abs(values[i]-baseline) >= .04), None)
    result = {"centerline_known_fraction": float(known.mean()),
              "centerline_unknown_fraction": float((inside & ~known).mean()),
              "centerline_outside_fraction": float((~inside).mean()),
              "known_height_profile_m": profile, "height_roughness_m": float(np.std(steps)) if steps else None,
              "maximum_slope": max(slope, default=None), "maximum_step_m": max(steps, default=None),
              "interaction_onset_station_m": onset, "corridors": {}}
    tangent = np.gradient(curve, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    for name, bbox in proxy_bboxes.items():
        bounds = np.asarray(bbox, dtype=float)
        if bounds.shape != (2,2) or np.any(bounds[1] <= bounds[0]):
            raise ValueError("invalid collision proxy bbox")
        xs = np.arange(bounds[0,0], bounds[1,0]+1e-9, .05)
        ys = np.arange(bounds[0,1], bounds[1,1]+1e-9, .05)
        offsets = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1,2)
        touched, known_cells = set(), set()
        corridor_profile = []
        sample_count = known_count = outside_count = 0
        for center, direction, station in zip(curve[::3], tangent[::3], stations[::3]):
            normal = np.array([-direction[1], direction[0]])
            points = center + offsets[:,0,None]*direction + offsets[:,1,None]*normal
            inside_p, known_p, values_p, rows, cols = _sample_grid(points,h,m)
            sample_count += len(points)
            known_count += int(known_p.sum())
            outside_count += int((~inside_p).sum())
            touched.update(zip(rows[inside_p].tolist(), cols[inside_p].tolist()))
            known_cells.update(zip(rows[known_p].tolist(), cols[known_p].tolist()))
            if known_p.any():
                corridor_profile.append([float(station),float(np.median(values_p[known_p]))])
        cp=np.asarray(corridor_profile)
        csteps=np.abs(np.diff(cp[:,1])) if len(cp)>1 else np.array([])
        cslopes=csteps/np.maximum(np.diff(cp[:,0]),1e-12) if len(cp)>1 else np.array([])
        result["corridors"][name] = {
            "known_fraction": known_count/sample_count,
            "unknown_fraction": 1-(known_count+outside_count)/sample_count,
            "outside_fraction": outside_count/sample_count,
            "swept_cells": sorted(touched), "known_swept_cells": sorted(known_cells),
            "known_cell_count": len(known_cells), "all_cell_count": len(touched),
            "known_corridor_profile_m": corridor_profile,
            "corridor_roughness_m":float(np.std(csteps)) if len(csteps) else None,
            "corridor_maximum_step_m":float(np.max(csteps)) if len(csteps) else None,
            "corridor_maximum_slope":float(np.max(cslopes)) if len(cslopes) else None,
            "proxy_semantics": "oriented bounding rectangle of frozen USD collision-bound corner group"}
    return result


def follower_diagnostic(candidate: dict, config: dict) -> dict:
    path = candidate["waypoints"]
    chord = path[0]
    initial_angle = math.atan2(chord[1], chord[0])
    try:
        command = c0_fresh_forward(path,
                                   lookahead_m=float(config["lookahead_distance"]),
                                   speed_mps=float(config["linear_speed"]),
                                   angular_gain=float(config["angular_gain"]),
                                   angular_direction_sign=float(config["angular_direction_sign"]),
                                   max_angular_speed=float(config["max_angular_speed"]),
                                   goal_tolerance_m=float(config["goal_tolerance"]),
                                   turn_threshold_rad=float(config["turn_in_place_heading_threshold"]),
                                   min_linear_speed_scale=float(config["min_linear_speed_scale"]))
        mode = "turn_in_place" if command["mode"] == "tracking" and command["linear_x"] == 0 else command["mode"]
        reverse = False
    except ValueError as exc:
        command = {"linear_x": None, "angular_z": None, "mode": "invalid", "error": str(exc)}
        mode = "reverse" if "reverse" in str(exc) else "invalid"
        reverse = "reverse" in str(exc)
    curvature = candidate["maximum_curvature_per_m"]
    return {"initial_chord_angle_rad": initial_angle, "initial_command_estimate": command,
            "command_mode": mode, "reverse_estimate": reverse,
            "turn_in_place_entry": mode == "turn_in_place",
            "angular_saturation_estimate": command.get("angular_z") is not None and abs(command["angular_z"]) >= float(config["max_angular_speed"])-1e-9,
            "moving_command_screen_per_m": FOLLOWER_CURVATURE_SCREEN_PER_M,
            "follower_compatible_screen": bool(curvature is not None and curvature <= FOLLOWER_CURVATURE_SCREEN_PER_M and mode == "tracking" and not reverse),
            "screen_is_not_physical_limit": True}


def pairwise_audit(left: dict, right: dict, left_dem: dict, right_dem: dict) -> dict:
    common_length = min(left["arc_length_m"], right["arc_length_m"])
    stations_full = np.linspace(0, common_length, 101)
    stations_early = np.linspace(0, min(.5, common_length), 51)
    def sample(row, query_stations):
        stations = row["stations_m"]
        curve = row["dense"]
        return np.stack([np.interp(query_stations, stations, curve[:,axis]) for axis in range(2)],axis=1)
    a,b = sample(left, stations_full),sample(right, stations_full)
    distances = np.linalg.norm(a-b,axis=1)
    early_distances = np.linalg.norm(sample(left, stations_early)-sample(right, stations_early),axis=1)
    la = left_dem["corridors"].get("main_track_sprocket", {})
    lb = right_dem["corridors"].get("main_track_sprocket", {})
    area_a = {tuple(c) for c in la.get("swept_cells", [])}
    area_b = {tuple(c) for c in lb.get("swept_cells", [])}
    known_a = {tuple(c) for c in la.get("known_swept_cells", [])}
    known_b = {tuple(c) for c in lb.get("known_swept_cells", [])}
    profile_a = {round(x/.05):y for x,y in left_dem["known_height_profile_m"]}
    profile_b = {round(x/.05):y for x,y in right_dem["known_height_profile_m"]}
    common = set(profile_a)&set(profile_b)
    terrain_rms = math.sqrt(np.mean([(profile_a[k]-profile_b[k])**2 for k in common])) if common else None
    corridor_a = {round(x/.05):y for x,y in la.get("known_corridor_profile_m",[])}
    corridor_b = {round(x/.05):y for x,y in lb.get("known_corridor_profile_m",[])}
    common_corridor = set(corridor_a)&set(corridor_b)
    corridor_rms = math.sqrt(np.mean([(corridor_a[k]-corridor_b[k])**2 for k in common_corridor])) if common_corridor else None
    return {"full_geometry_rms_m": float(np.sqrt(np.mean(distances**2))),
            "early_0_5m_geometry_rms_m": float(np.sqrt(np.mean(early_distances**2))),
            "common_absolute_arc_length_m": common_length,
            "arc_length_difference_m": abs(left["arc_length_m"]-right["arc_length_m"]),
            "swept_area_iou": len(area_a&area_b)/len(area_a|area_b) if area_a|area_b else None,
            "pairwise_exclusive_area_m2": .01*len(area_a^area_b),
            "known_exclusive_area_m2": .01*len(known_a^known_b),
            "known_terrain_profile_rms_m": terrain_rms,
            "common_known_profile_bins": len(common),
            "known_corridor_profile_rms_m": corridor_rms,
            "common_known_corridor_bins": len(common_corridor),
            "terrain_difference_requires_common_known_support": True}


def pairwise_command_estimate(first: dict, second: dict) -> dict:
    """Normalized RMS of two C0 estimates; no later follower state is inferred."""
    a,b=first['initial_command_estimate'],second['initial_command_estimate']
    if any(x.get('linear_x') is None or x.get('angular_z') is None for x in (a,b)):
        return {'initial_normalized_command_rms_estimate':None,
                'initial_linear_difference_mps':None,
                'initial_angular_difference_radps':None,
                'time_scope':'initial_command_only_not_rollout_RMS'}
    dv=float(a['linear_x']-b['linear_x'])
    dw=float(a['angular_z']-b['angular_z'])
    return {'initial_normalized_command_rms_estimate':float(math.sqrt(((dv/.3)**2+(dw/1.6)**2)/2)),
            'initial_linear_difference_mps':abs(dv),
            'initial_angular_difference_radps':abs(dw),
            'time_scope':'initial_command_only_not_rollout_RMS'}


def select_parameters(scores: list[dict]) -> dict:
    if not scores:
        raise ValueError("no parameter candidates")
    def rank(r):
        return (r.get("hard_valid_groups",0), r.get("follower_compatible",0),
                r.get("early_separation",0), r.get("full_separation",0),
                r.get("known_terrain_separation",0), -r.get("unknown_dependence",0),
                -r.get("curvature_penalty",0), r.get("side_balance",0),
                -sum(r["parameters"]))
    return max(scores, key=rank)
