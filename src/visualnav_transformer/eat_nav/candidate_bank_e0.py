"""Offline E0 audit. Missing physical limits are reported, never invented."""

from __future__ import annotations

import math

import numpy as np

from .candidate_bank_geometry import FAMILIES, c0_fresh_forward, generate
from .candidate_coverage_offline import _self_intersection


def _sample(points, distance):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return np.array([np.interp(min(distance, arc[-1]), arc, points[:, i]) for i in range(2)])


def _tangent(points, distance):
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    index = int(np.clip(np.searchsorted(arc, min(distance, arc[-1])), 1, len(points)-1))
    delta = points[index]-points[index-1]
    return delta/np.linalg.norm(delta)


def audit_path(dense, waypoints, goal, *, dem=None, resolution_m=.1,
               curvature_limit_per_m=None, path_length_limit_m=None,
               footprint_radius_m=None, unknown_is_invalid=True):
    """Geometry checks at dense spacing; null constraints remain unresolved."""
    reasons = []
    unresolved = []
    if dense is None or waypoints is None:
        return {"geometry_valid": False, "invalid_reasons": ["generation_error"]}
    dense = np.asarray(dense, dtype=float)
    waypoints = np.asarray(waypoints, dtype=float)
    goal = np.asarray(goal, dtype=float)
    if not np.isfinite(dense).all() or not np.isfinite(waypoints).all():
        return {"geometry_valid": False, "invalid_reasons": ["nonfinite"]}
    delta = np.diff(dense, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if np.any(lengths <= 1e-12):
        reasons.append("nonmonotone_arc_length")
    headings = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    curvature = np.abs(np.diff(headings)) / np.maximum(
        (lengths[1:]+lengths[:-1])*.5, 1e-12)
    max_curvature = float(curvature.max(initial=0))
    path_length = float(lengths.sum())
    if np.linalg.norm(dense[0]) > .005:
        reasons.append("start_position")
    if np.linalg.norm(dense[-1]-goal) > .02 or np.linalg.norm(waypoints[-1]-goal) > .02:
        reasons.append("common_endpoint")
    if abs(headings[0]) > .05:
        reasons.append("start_tangent")
    if np.any(delta[:, 0] < -1e-6):
        reasons.append("reverse_motion")
    if _self_intersection(dense[::4]):
        reasons.append("self_intersection")
    if curvature_limit_per_m is None:
        unresolved.append("measured_turning_envelope")
    elif max_curvature > curvature_limit_per_m:
        reasons.append("curvature_limit")
    if path_length_limit_m is None:
        unresolved.append("train_only_path_length_support")
    elif path_length > path_length_limit_m:
        reasons.append("path_length_limit")
    dem_status = {"out_of_grid": None, "unknown_crossing": None,
                  "footprint_known_fraction": None, "height_range_m": None}
    if dem is None:
        unresolved.append("DEM")
    else:
        dem = np.asarray(dem)
        if dem.ndim != 4 or dem.shape[0] != 4 or dem.shape[1] != 2:
            raise ValueError("DEM must have [4,2,H,W] layout")
        height, mask = dem[-1]
        rows = np.rint(dense[:, 1]/resolution_m + (height.shape[0]-1)/2).astype(int)
        cols = np.rint(dense[:, 0]/resolution_m + (height.shape[1]-1)/2).astype(int)
        inside = (rows >= 0) & (rows < height.shape[0]) & (cols >= 0) & (cols < height.shape[1])
        dem_status["out_of_grid"] = bool(not inside.all())
        if not inside.all():
            reasons.append("out_of_grid")
        known = inside.copy()
        known[inside] = mask[rows[inside], cols[inside]] > .5
        dem_status["unknown_crossing"] = bool(not known.all())
        if unknown_is_invalid and not known.all():
            reasons.append("unknown_cell_crossing")
        if known.any():
            values = height[rows[known], cols[known]]
            dem_status["height_range_m"] = float(values.max()-values.min())
        if footprint_radius_m is None:
            unresolved.append("full_vehicle_footprint")
        else:
            radial = int(math.ceil(footprint_radius_m/resolution_m))
            offsets = [(r, c) for r in range(-radial, radial+1) for c in range(-radial, radial+1)
                       if math.hypot(r, c)*resolution_m <= footprint_radius_m]
            samples = [(row+r, col+c) for row, col in zip(rows, cols) for r, c in offsets]
            footprint_known = [0 <= r < height.shape[0] and 0 <= c < height.shape[1]
                               and mask[r, c] > .5 for r, c in samples]
            dem_status["footprint_known_fraction"] = float(np.mean(footprint_known))
            if not all(footprint_known):
                reasons.append("footprint_unknown_or_out_of_grid")
            unresolved.append("obstacle_height_collision_threshold")
    return {"geometry_valid": not reasons, "invalid_reasons": reasons,
            "unresolved_checks": unresolved, "length_m": path_length,
            "max_abs_curvature_per_m": max_curvature,
            "total_abs_heading_change_rad": float(np.abs(np.diff(headings)).sum()),
            "start_tangent_error_rad": float(abs(headings[0])),
            "endpoint_error_m": float(np.linalg.norm(waypoints[-1]-goal)),
            "goal_entry_angle_rad": float(headings[-1]),
            "max_abs_lateral_m": float(np.abs(dense[:, 1]).max()),
            "self_intersection": "self_intersection" in reasons,
            "dem": dem_status}


def compare_family(family, goal, lateral_scale_m, *, dem=None, **limits):
    generated = generate(family, goal, lateral_scale_m=lateral_scale_m)
    reference = generated[0]
    reference_c0 = c0_fresh_forward(reference["waypoints"])
    result = []
    for candidate in generated:
        dense = candidate["dense"]
        waypoints = candidate["waypoints"]
        metric = audit_path(dense, waypoints, goal, dem=dem, **limits)
        if dense is not None:
            samples = {}
            for station in (.25, .5):
                position = _sample(dense, station)
                ref_position = _sample(reference["dense"], station)
                tangent = _tangent(dense, station)
                ref_tangent = _tangent(reference["dense"], station)
                normal = np.array([-ref_tangent[1], ref_tangent[0]])
                samples[str(station)] = {
                    "signed_reference_normal_difference_m": float((position-ref_position) @ normal),
                    "xy_difference_m": float(np.linalg.norm(position-ref_position)),
                    "heading_difference_rad": float(math.atan2(tangent[1], tangent[0])-
                                                    math.atan2(ref_tangent[1], ref_tangent[0]))}
            metric["early_path"] = samples
            stations = np.linspace(0, min(.5, metric["length_m"],
                                           float(np.linalg.norm(np.diff(reference["dense"], axis=0), axis=1).sum())), 51)
            metric["early_path_max_xy_difference_m"] = float(max(
                np.linalg.norm(_sample(dense, s)-_sample(reference["dense"], s)) for s in stations))
            metric["c0_fresh_forward_diagnostic"] = c0_fresh_forward(waypoints)
            metric["C0_exact_formula_delta_from_direct"] = {
                "linear_x_mps": metric["c0_fresh_forward_diagnostic"]["linear_x"]-reference_c0["linear_x"],
                "angular_z_radps": metric["c0_fresh_forward_diagnostic"]["angular_z"]-reference_c0["angular_z"]}
            metric["waypoints_xy_m"] = waypoints.tolist()
            metric["equal_t_waypoints_xy_m"] = dense[np.rint(np.linspace(0, len(dense)-1, 9)[1:]).astype(int)].tolist()
            metric["waypoint_resampling_rms_m"] = float(np.sqrt(np.mean((waypoints-np.asarray(
                metric["equal_t_waypoints_xy_m"]))**2)))
        result.append({"candidate_id": candidate["candidate_id"],
                       "target_normal_amplitude_m": candidate["target_normal_amplitude_m"],
                       "realized_normal_amplitude_m": candidate.get("realized_normal_amplitude_m"),
                       "offset_parameter_m": candidate["offset_parameter_m"],
                       "join_curvature_jump_per_m": candidate["join_curvature_jump_per_m"],
                       "generation_error": candidate["generation_error"], **metric})
    return {"family": family, "goal_xy_m": list(map(float, goal)),
            "lateral_scale_m": float(lateral_scale_m), "candidates": result,
            "C0_exact_fixture_parity": "unverified; fresh-forward static formula only"}
