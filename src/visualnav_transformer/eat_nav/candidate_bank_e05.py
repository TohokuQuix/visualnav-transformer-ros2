"""E0.5 offline feasibility calibration. No rollout or outcome labels."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET

import numpy as np

from .candidate_bank_geometry import generate
from .candidate_bank_e0 import audit_path


def waypoint_polyline_error(dense, waypoints):
    """Largest dense-curve distance from the eight-pose Path polyline."""
    dense = np.asarray(dense)
    path = np.vstack((np.zeros((1,2)),np.asarray(waypoints)))
    errors = []
    for a,b in zip(path[:-1],path[1:]):
        delta = b-a
        ratio = np.clip(((dense-a) @ delta)/(delta @ delta),0,1)
        projection = a + ratio[:,None]*delta
        errors.append(np.linalg.norm(dense-projection,axis=1))
    return float(np.min(np.stack(errors),axis=0).max())


def follower_moving_envelope(*, speed_mps: float, angular_gain: float,
                             angular_direction_sign: float, turn_threshold_rad: float,
                             max_angular_speed: float):
    """Continuous tracking branch of the unchanged production P controller.

    At/above the turn threshold the follower commands v=0; omega/v is not
    interpreted as Path curvature there. This is a command capability screen,
    not a measured vehicle guarantee.
    """
    if not 0 < turn_threshold_rad < math.pi or speed_mps <= 0:
        raise ValueError("invalid production follower parameters")
    min_tracking_v = speed_mps*(1-turn_threshold_rad/math.pi)
    max_moving_omega = min(max_angular_speed,
                           abs(angular_gain*angular_direction_sign)*turn_threshold_rad)
    return {"minimum_meaningful_tracking_speed_mps": min_tracking_v,
            "maximum_moving_angular_speed_radps": max_moving_omega,
            "maximum_command_track_curvature_per_m": max_moving_omega/min_tracking_v,
            "in_place_rotation": "allowed_for_reorientation_only; not evidence of continuous Path tracking"}


def _vec(text, default="0 0 0"):
    return np.fromstring(text if text is not None else default, sep=" ", dtype=float)


def _rotation_rpy(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                     [-sp, cp*sr, cp*cr]])


def _transform(origin):
    result = np.eye(4)
    if origin is not None:
        result[:3, :3] = _rotation_rpy(_vec(origin.get("rpy")))
        result[:3, 3] = _vec(origin.get("xyz"))
    return result


def _axis_rotation(axis, angle):
    axis = axis/np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    result = np.eye(4)
    result[:3, :3] = c*np.eye(3)+(1-c)*np.outer(axis, axis)+s*cross
    return result


def _shape_points(geometry):
    box = geometry.find("box")
    if box is not None:
        half = _vec(box.get("size"))/2
        return np.array([[x, y, z, 1] for x in (-half[0], half[0])
                         for y in (-half[1], half[1]) for z in (-half[2], half[2])])
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius, half = float(cylinder.get("radius")), float(cylinder.get("length"))/2
        return np.array([[radius*math.cos(a), radius*math.sin(a), z, 1]
                         for a in np.linspace(0, 2*math.pi, 33) for z in (-half, half)])
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius"))
        return np.array([[x, y, z, 1] for x in (-radius, radius)
                         for y in (-radius, radius) for z in (-radius, radius)])
    raise ValueError("unsupported_collision_geometry")


def urdf_collision_bounds(path, *, articulated: bool):
    """Bounds of all collision prims at zero pose or across joint angle samples.

    The articulated result is an axis-aligned union, conservative for any
    flipper posture. Joint-angle discretization is disclosed in the result.
    """
    root = ET.parse(path).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    children = {name: [] for name in links}
    child_names = set()
    for joint in root.findall("joint"):
        parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
        children[parent].append((child, joint))
        child_names.add(child)
    roots = sorted(set(links)-child_names)
    if len(roots) != 1:
        raise ValueError("URDF does not have exactly one root link")
    bounds_min = np.array([np.inf, np.inf])
    bounds_max = np.array([-np.inf, -np.inf])
    shape_count = 0
    unsupported = []
    sampled = set()

    def visit(name, transforms):
        nonlocal shape_count, bounds_min, bounds_max
        for collision in links[name].findall("collision"):
            geometry = collision.find("geometry")
            try:
                points = _shape_points(geometry)
            except ValueError:
                unsupported.append(name)
                continue
            local = _transform(collision.find("origin"))
            for transform in transforms:
                world = (transform @ local @ points.T).T[:, :2]
                bounds_min = np.minimum(bounds_min, world.min(axis=0))
                bounds_max = np.maximum(bounds_max, world.max(axis=0))
            shape_count += 1
        for child, joint in children[name]:
            kind = joint.get("type")
            if articulated and kind in ("revolute", "continuous"):
                limit = joint.find("limit")
                if kind == "revolute" and limit is not None:
                    lower, upper = float(limit.get("lower")), float(limit.get("upper"))
                    if upper-lower >= 2*math.pi:
                        angles = np.linspace(-math.pi, math.pi, 13)
                    else:
                        angles = np.linspace(lower, upper, 13)
                else:
                    angles = np.linspace(-math.pi, math.pi, 13)
                sampled.add(joint.get("name"))
            else:
                angles = (0.0,)
            origin = _transform(joint.find("origin"))
            axis_node = joint.find("axis")
            axis = _vec(axis_node.get("xyz") if axis_node is not None else "1 0 0")
            visit(child, [parent @ origin @ _axis_rotation(axis, angle)
                          for parent in transforms for angle in angles])

    visit(roots[0], [np.eye(4)])
    if unsupported or shape_count == 0:
        raise ValueError(f"unresolved collision geometry: {unsupported}")
    return {"x_min_m": float(bounds_min[0]), "x_max_m": float(bounds_max[0]),
            "y_min_m": float(bounds_min[1]), "y_max_m": float(bounds_max[1]),
            "effective_width_m": float(bounds_max[1]-bounds_min[1]),
            "half_width_m": float(max(abs(bounds_min[1]), abs(bounds_max[1]))),
            "collision_shape_count": shape_count, "articulated_union": articulated,
            "sampled_joint_count": len(sampled), "joint_angle_samples": 13 if articulated else 1}


def swept_footprint_dem(dense, dem, bounds, *, resolution_m=.1,
                        safety_margin_m=None, height_jump_limit_m=None,
                        heading_independent=False):
    """Rasterized oriented collision AABB sweep; unknown/outside always fail.

    A missing physical safety margin or obstacle threshold is reported as
    unresolved; exploratory zero-margin results cannot become E1 validity.
    """
    dense = np.asarray(dense)
    dem = np.asarray(dem)
    if dense.ndim != 2 or dense.shape[1] != 2 or dem.shape[0:2] != (4, 2):
        raise ValueError("candidate/DEM shape")
    if safety_margin_m is not None and safety_margin_m < 0:
        raise ValueError("negative safety margin")
    margin = 0 if safety_margin_m is None else safety_margin_m
    # Half-cell diagonal covers cell-center rasterization error in both axes.
    raster_margin = resolution_m/math.sqrt(2)
    step = resolution_m/2
    if heading_independent:
        radius = math.hypot(max(abs(bounds["x_min_m"]),abs(bounds["x_max_m"])),
                            max(abs(bounds["y_min_m"]),abs(bounds["y_max_m"])))+margin+raster_margin
        xs = np.arange(-radius,radius+step*.5,step)
        ys = xs
    else:
        xs = np.arange(bounds["x_min_m"]-margin-raster_margin,
                       bounds["x_max_m"]+margin+raster_margin+step*.5, step)
        ys = np.arange(bounds["y_min_m"]-margin-raster_margin,
                       bounds["y_max_m"]+margin+raster_margin+step*.5, step)
    offsets = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    if heading_independent:
        offsets = offsets[np.linalg.norm(offsets,axis=1) <= radius+1e-9]
    height, mask = dem[-1]
    # Dense source curve is <= about 1 cm/sample in this E0.5 exploration.
    tangents = np.gradient(dense, axis=0)
    angles = np.arctan2(tangents[:, 1], tangents[:, 0])
    touched = set()
    outside = False
    for point, angle in zip(dense, angles):
        c, s = math.cos(angle), math.sin(angle)
        world = point + (offsets if heading_independent else offsets @ np.array([[c, s], [-s, c]]))
        cols = np.rint(world[:, 0]/resolution_m+(mask.shape[1]-1)/2).astype(int)
        rows = np.rint(world[:, 1]/resolution_m+(mask.shape[0]-1)/2).astype(int)
        inside = (cols >= 0)&(cols < mask.shape[1])&(rows >= 0)&(rows < mask.shape[0])
        outside |= not bool(inside.all())
        touched.update(zip(rows[inside].tolist(), cols[inside].tolist()))
    unknown = int(sum(bool(mask[r, c] <= .5) for r, c in touched))
    jumps = []
    for r, c in touched:
        for rr, cc in ((r+1,c),(r,c+1)):
            if (rr,cc) in touched and mask[r,c]>.5 and mask[rr,cc]>.5:
                jumps.append(abs(float(height[r,c]-height[rr,cc])))
    max_jump = max(jumps, default=0.0)
    reasons = []
    if outside:
        reasons.append("full_footprint_out_of_dem")
    if unknown:
        reasons.append("full_footprint_unknown")
    if height_jump_limit_m is not None and max_jump > height_jump_limit_m:
        reasons.append("height_jump_exceeds_exploratory_limit")
    return {"known_geometry_screen_pass": not reasons, "reasons": reasons,
            "footprint_model": "all_yaw_collision_corner_disk" if heading_independent else "path_tangent_oriented_collision_AABB",
            "outside": outside, "unknown_cell_count": unknown,
            "touched_cell_count": len(touched), "max_adjacent_height_jump_m": max_jump,
            "raster_margin_m": raster_margin, "safety_margin_m": safety_margin_m,
            "obstacle_limit_m": height_jump_limit_m,
            "physical_feasibility_verified": False if (safety_margin_m is None or height_jump_limit_m is None) else None}


def feasible_amplitude(family, goal, lateral_scale_m, curvature_limit_per_m,
                       *, path_length_limit_m, max_target_m=None):
    """Monotone-safe scan: return largest sampled feasible target amplitude.

    It does not infer a physical turn limit; caller must supply an evidence
    based threshold. Final feasibility also requires the DEM/footprint screen.
    """
    if family not in ("cubic_bezier", "clamped_cubic_spline"):
        raise ValueError("E0.5 primary/comparison family only")
    maximum = min(float(lateral_scale_m), float(max_target_m or lateral_scale_m))
    targets = np.linspace(0, maximum, 81)
    feasible = []
    for target in targets:
        # generate uses alpha rank 1; scale it to produce this absolute target.
        bank = generate(family, goal, lateral_scale_m=max(target/.15, 1e-9))
        curve = bank[1]
        report = audit_path(curve["dense"], curve["waypoints"], goal,
                            curvature_limit_per_m=curvature_limit_per_m,
                            path_length_limit_m=path_length_limit_m)
        feasible.append(report["geometry_valid"])
    # Search is intentionally not binary: validity need not be monotone.
    good = [float(t) for t, valid in zip(targets, feasible) if valid]
    return {"maximum_sampled_feasible_amplitude_m": max(good, default=None),
            "grid_step_m": maximum/80, "nonmonotone_validity": any(
                feasible[i] and not feasible[i-1] and any(feasible[:i-1])
                for i in range(1, len(feasible))),
            "feasible_grid_count": sum(feasible), "sample_count": len(targets)}
