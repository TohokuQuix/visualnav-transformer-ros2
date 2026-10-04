"""Revised E0: robot-local Path-centerline audit, with diagnostic footprint only.

This module never runs a simulator, follower process, model, or learner.
"""

from __future__ import annotations

import math

import numpy as np

from .candidate_bank_geometry import _curve, _normal_peak, arc_resample, c0_fresh_forward


def grid_bounds(cells: int, resolution_m: float) -> tuple[float, float]:
    if cells != 20 or not np.isclose(resolution_m, .1, atol=1e-12):
        raise ValueError("DEM layout differs from frozen E0 config")
    return -cells * resolution_m / 2, cells * resolution_m / 2


def inside_grid(points, *, edge_m=1.0, margin_m=1e-6):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
        raise ValueError("finite [N,2] points required")
    return np.all(np.abs(p) <= edge_m + margin_m, axis=1)


def dense_arclength(points, max_step_m: float):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("finite curve [N,2] required")
    arcs = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    if np.any(np.diff(arcs) <= 1e-12) or max_step_m <= 0:
        raise ValueError("nonmonotone curve or invalid spacing")
    stations = np.linspace(0., arcs[-1], math.ceil(arcs[-1] / max_step_m) + 1)
    dense = np.stack([np.interp(stations, arcs, p[:, i]) for i in range(2)], axis=1)
    return dense, stations


def _bezier_extrema(goal, offset):
    """Exact coordinate bounds of the underlying cubic Bezier, including between samples."""
    goal = np.asarray(goal, dtype=float)
    distance = float(np.linalg.norm(goal))
    normal = np.array([-goal[1], goal[0]]) / distance
    ctrl = np.stack((np.zeros(2), [distance * .25, 0.],
                     goal * .75 + normal * offset, goal))
    extrema = []
    for axis in range(2):
        p0, p1, p2, p3 = ctrl[:, axis]
        a = -p0 + 3*p1 - 3*p2 + p3
        b = 2*(p0 - 2*p1 + p2)
        c = p1-p0
        roots = np.roots([a, b, c]) if abs(a) > 1e-14 else (
            np.array([-c/b]) if abs(b) > 1e-14 else np.array([]))
        ts = [0., 1.] + [float(v.real) for v in roots if abs(v.imag) < 1e-12 and 0 < v.real < 1]
        values = [((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2+t**3*p3) for t in ts]
        extrema.append([float(min(values)), float(max(values))])
    return extrema


def _has_self_intersection(points):
    p = np.asarray(points)
    a, b = p[:-1], p[1:]
    def cross(u, v):
        return u[..., 0]*v[..., 1]-u[..., 1]*v[..., 0]
    delta = b-a
    first = cross(delta[:, None], a[None]-a[:, None])
    second = cross(delta[:, None], b[None]-a[:, None])
    crossed = (first*second < -1e-12) & (first.T*second.T < -1e-12)
    index = np.arange(len(delta))
    crossed &= np.abs(index[:, None]-index[None, :]) > 1
    return bool(np.any(crossed))


def generate_bank(family, goal, amplitudes, *, parameter_samples, point_count):
    """Fixed seven-candidate geometry; targets are signed normal peaks, not control offsets."""
    goal = np.asarray(goal, dtype=float)
    t = np.linspace(0., 1., parameter_samples)
    reference, _ = _curve(family, goal, 0., t)
    targets = [("direct_reference", 0., 0)] + [
        (f"{side}_{rank}", sign*float(amplitude), sign)
        for side, sign in (("left", 1), ("right", -1))
        for rank, amplitude in enumerate(amplitudes, 1)]
    rows = []
    for cid, signed_target, sign in targets:
        try:
            if sign == 0:
                offset = 0.
            else:
                target = abs(signed_target)
                low, high = 0., max(target*2, 1e-4)
                for _ in range(16):
                    trial, _ = _curve(family, goal, sign*high, t)
                    if _normal_peak(reference, trial, sign) >= target:
                        break
                    high *= 1.5
                else:
                    raise ValueError("amplitude_unreachable")
                for _ in range(32):
                    mid = (low+high)/2
                    trial, _ = _curve(family, goal, sign*mid, t)
                    if _normal_peak(reference, trial, sign) < target:
                        low = mid
                    else:
                        high = mid
                offset = sign*(low+high)/2
            curve, jump = _curve(family, goal, offset, t)
            waypoints = arc_resample(curve, point_count)
            rows.append({"candidate_id": cid, "target_signed_lateral_m": signed_target,
                         "offset_parameter_m": float(offset), "join_curvature_jump_per_m": float(jump),
                         "source_curve": curve, "waypoints": waypoints, "generation_error": None})
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            rows.append({"candidate_id": cid, "target_signed_lateral_m": signed_target,
                         "offset_parameter_m": None, "join_curvature_jump_per_m": None,
                         "source_curve": None, "waypoints": None, "generation_error": str(exc)})
    return rows


def _polygon_area(p):
    if len(p) < 3:
        return 0.
    return abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                     np.dot(p[:, 1], np.roll(p[:, 0], -1))))/2


def _clip(polygon, axis, bound, keep_greater):
    if len(polygon) == 0:
        return polygon
    output = []
    previous = polygon[-1]
    def inside(p):
        return p[axis] >= bound if keep_greater else p[axis] <= bound
    for current in polygon:
        p_in, c_in = inside(previous), inside(current)
        if p_in != c_in:
            ratio = (bound-previous[axis])/(current[axis]-previous[axis])
            output.append(previous + ratio*(current-previous))
        if c_in:
            output.append(current)
        previous = current
    return np.asarray(output).reshape(-1, 2)


def footprint_diagnostic(dense, stations, mask, bounds, *, edge_m=1., offset_step_m=.05):
    """Exact clipped AABB area per pose; unknown overlap approximated on 5 cm lattice."""
    corners = np.array([[bounds['x_min_m'], bounds['y_min_m']],
                        [bounds['x_max_m'], bounds['y_min_m']],
                        [bounds['x_max_m'], bounds['y_max_m']],
                        [bounds['x_min_m'], bounds['y_max_m']]], dtype=float)
    total_area = _polygon_area(corners)
    xs = np.arange(bounds['x_min_m'], bounds['x_max_m']+1e-12, offset_step_m)
    ys = np.arange(bounds['y_min_m'], bounds['y_max_m']+1e-12, offset_step_m)
    offsets = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    tangents = np.gradient(dense, axis=0)
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])
    coverage, unknown, outside_dist = [], [], []
    for center, angle in zip(dense, angles):
        c, s = math.cos(angle), math.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        quad = corners @ rotation.T + center
        clip = quad
        for axis, bound, greater in ((0, -edge_m, True), (0, edge_m, False),
                                     (1, -edge_m, True), (1, edge_m, False)):
            clip = _clip(clip, axis, bound, greater)
        coverage.append(_polygon_area(clip)/total_area)
        outside_dist.append(float(np.max(np.linalg.norm(np.maximum(np.abs(quad)-edge_m, 0), axis=1))))
        samples = offsets @ rotation.T + center
        in_grid = inside_grid(samples, edge_m=edge_m, margin_m=0.)
        observed = np.zeros(len(samples), dtype=bool)
        if in_grid.any():
            cols = np.minimum(np.floor((samples[in_grid, 0]+edge_m)/.1).astype(int), 19)
            rows = np.minimum(np.floor((samples[in_grid, 1]+edge_m)/.1).astype(int), 19)
            observed[in_grid] = mask[rows, cols] > .5
        unknown.append(float(np.count_nonzero(in_grid & ~observed)/len(samples)))
    coverage = np.asarray(coverage)
    unknown = np.asarray(unknown)
    outside_dist = np.asarray(outside_dist)
    first = np.flatnonzero(coverage < 1-1e-9)
    bins = np.linspace(0, len(coverage), 9, dtype=int)
    segment_coverage = []
    for i in range(8):
        selected = coverage[bins[i]:bins[i+1]]
        if len(selected) == 0:
            selected = coverage[min(bins[i], len(coverage)-1):min(bins[i], len(coverage)-1)+1]
        segment_coverage.append(float(selected.mean()))
    return {"area_coverage_mean": float(coverage.mean()),
            "area_coverage_min": float(coverage.min()),
            "segment_area_coverage_mean": segment_coverage,
            "outside_max_distance_m": float(outside_dist.max()),
            "unknown_lattice_ratio_mean": float(unknown.mean()),
            "first_outside_station_m": float(stations[first[0]]) if len(first) else None,
            "goal_neighborhood_coverage_mean": float(coverage[max(0, int(.875*len(coverage))):].mean()),
            "model": "oriented_URDF_collision_AABB; clipped_area_exact_per_pose; unknown_5cm_lattice_approx",
            "diagnostic_only": True}


def audit_candidate(row, goal, dem, bounds, config):
    reasons = []
    if row['generation_error']:
        return {"candidate_id": row['candidate_id'], "type_A_reasons": ["generation_failure:"+row['generation_error']],
                "generation_error": row['generation_error']}
    g = np.asarray(goal, dtype=float)
    curve = np.asarray(row['source_curve'])
    path = np.asarray(row['waypoints'])
    setting = config['generator']
    edge = 1.
    margin = config['dem']['boundary_margin_m']
    dense, stations = dense_arclength(curve, setting['dense_centerline_max_step_m'])
    if not inside_grid(g[None], edge_m=edge, margin_m=margin).all():
        reasons.append('goal_outside_dem_geometry')
    support = inside_grid(dense, edge_m=edge, margin_m=margin)
    if not support.all():
        reasons.append('interpolated_centerline_outside_dem_geometry')
    analytic = _bezier_extrema(g, row['offset_parameter_m']) if config['_family'] == 'cubic_bezier' else None
    if analytic and any(lo < -edge-margin or hi > edge+margin for lo, hi in analytic):
        reasons.append('analytic_bezier_centerline_outside_dem_geometry')
    if path.shape != (setting['point_count'], 2) or not np.isfinite(path).all():
        reasons.append('shape_or_nonfinite')
    if np.linalg.norm(curve[0]) > setting['start_position_tolerance_m']:
        reasons.append('start_mismatch')
    if np.linalg.norm(curve[-1]-g) > config['goal_selection']['endpoint_tolerance_m'] or np.linalg.norm(path[-1]-g) > config['goal_selection']['endpoint_tolerance_m']:
        reasons.append('endpoint_mismatch')
    delta = np.diff(dense, axis=0)
    segment_length = np.linalg.norm(delta, axis=1)
    heading = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    # The start-tangent contract is the continuous curve derivative. A 2 cm
    # secant and the first 8-waypoint chord are separate diagnostics: both can
    # deviate even when the analytic start tangent is forward.
    tip, _ = _curve(config['_family'], g, row['offset_parameter_m'], np.array([0., 1e-7]))
    true_start_tangent = math.atan2(*(tip[1]-tip[0])[::-1])
    if abs(true_start_tangent) > setting['start_tangent_tolerance_rad']:
        reasons.append('start_tangent_mismatch')
    if np.any(delta[:, 0] < -setting['forward_delta_x_tolerance_m']):
        reasons.append('unintended_reverse')
    if _has_self_intersection(dense):
        reasons.append('self_intersection')
    waypoint_segments = np.linalg.norm(np.diff(np.vstack((np.zeros((1, 2)), path)), axis=0), axis=1)
    if np.any(waypoint_segments <= 1e-12):
        reasons.append('waypoint_order_error')
    curvature = np.diff(heading)/np.maximum((segment_length[1:]+segment_length[:-1])/2, 1e-12)
    mask = np.asarray(dem[-1, 1])
    height = np.asarray(dem[-1, 0])
    center_known = np.zeros(len(dense), dtype=bool)
    sampled_height = []
    if support.any():
        cols = np.minimum(np.floor((dense[support, 0]+edge)/.1).astype(int), 19)
        rows = np.minimum(np.floor((dense[support, 1]+edge)/.1).astype(int), 19)
        center_known[support] = mask[rows, cols] > .5
        sampled_height = height[rows[mask[rows, cols] > .5], cols[mask[rows, cols] > .5]].tolist()
    footprint = footprint_diagnostic(dense, stations, mask, bounds,
                offset_step_m=config['footprint_diagnostic']['offset_grid_step_m'])
    try:
        c0 = c0_fresh_forward(path)
    except ValueError as exc:
        c0 = {"unavailable_reason": str(exc)}
    full = np.stack([np.interp(np.linspace(0, stations[-1], 129), stations, dense[:, i]) for i in range(2)], axis=1)
    early_stations = np.linspace(0, min(.5, stations[-1]), 51)
    early = np.stack([np.interp(early_stations, stations, dense[:, i]) for i in range(2)], axis=1)
    return {"candidate_id": row['candidate_id'], "type_A_reasons": sorted(set(reasons)),
            "goal_inside": bool(inside_grid(g[None], edge_m=edge, margin_m=margin)[0]),
            "centerline_inside": bool(support.all()), "dense_max_step_m": float(segment_length.max()),
            "analytic_bezier_extrema_xy_m": analytic,
            "endpoint_error_m": float(np.linalg.norm(path[-1]-g)),
            "start_tangent_error_rad": float(abs(true_start_tangent)),
            "first_dense_secant_heading_rad": float(heading[0]),
            "first_waypoint_chord_heading_rad": float(math.atan2(path[0, 1], path[0, 0])),
            "endpoint_tangent_rad": float(heading[-1]),
            "length_m": float(stations[-1]),
            "maximum_abs_curvature_per_m": float(np.abs(curvature).max(initial=0)),
            "curvature_profile_per_m": curvature[::max(1, len(curvature)//32)].tolist(),
            "signed_lateral_target_m": row['target_signed_lateral_m'],
            "maximum_abs_lateral_m": float(np.abs(dense[:, 1]).max()),
            "centerline_known_fraction": float(center_known.mean()),
            "height_range_m": float(np.ptp(sampled_height)) if sampled_height else None,
            "height_roughness_std_m": float(np.std(sampled_height)) if sampled_height else None,
            "height_slope_max_abs_per_m": float(np.max(np.abs(np.diff(sampled_height))/.1)) if len(sampled_height)>1 else None,
            "footprint": footprint, "C0_exact_diagnostic": c0,
            "waypoints_xy_m": path.tolist(), "dense_centerline_xy_m": dense.tolist(),
            "full_equal_fraction_xy_m": full.tolist(), "early_0_5m_xy_m": early.tolist(),
            "join_curvature_jump_per_m": row['join_curvature_jump_per_m'],
            "offset_parameter_m": row['offset_parameter_m']}
