"""Read-only portable Evaluator source fixtures; never publishes robot commands."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence

import numpy as np


VERSION = "portable-evaluator-input-v2-proposal-1"
SIM_2S = (("chassis_r08_sprocket_axle", "chassis_r01_sprocket_axle"),
          ("chassis_l08_sprocket_axle", "chassis_l01_sprocket_axle"))
PHYSICAL_2S = (("chassis/fr_sprocket_axle", "chassis/rr_sprocket_axle"),
               ("chassis/fl_sprocket_axle", "chassis/rl_sprocket_axle"))
SIM_FLIPPERS = ("chassis_rr_flipper_joint", "chassis_rl_flipper_joint",
                "chassis_fr_flipper_joint", "chassis_fl_flipper_joint")
PHYSICAL_FLIPPERS = ("chassis/rr_flipper_joint", "chassis/rl_flipper_joint",
                     "chassis/fr_flipper_joint", "chassis/fl_flipper_joint")
SIM_24J = tuple(
    tuple([f"chassis_{side}{i:02d}_sprocket_axle" for i in range(1, 9)]
          + [f"chassis_{prefix}_subroller_{i:02d}_axle" for prefix in subrollers for i in range(1, 9)])
    for side, subrollers in (("r", ("fr", "rr")), ("l", ("fl", "rl")))
)
CORE_FIELDS = (
    ("dem_height_history", (4, 20, 20)), ("dem_valid_history", (4, 20, 20)),
    ("relative_goal", (2,)), ("candidate_path", (8, 2)),
    ("path_geometry", (6,)), ("path_dem_profile", (64,)),
    ("path_coverage", (64,)), ("body_angular_velocity", (3,)),
    ("projected_gravity", (3,)), ("flipper_position", (4,)),
    ("arm_configuration_state", (1,)),
)
TRACK_FIELD = ("track_sprocket_velocity", (2,))
ACTOR_FIELD = ("simulator_actor_track_velocity_24j", (2,))


def _finite(values: Sequence[float], *, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError("nonfinite or wrong shape")
    return array


def extract_named(names: Sequence[str], values: Sequence[float], wanted: Sequence[str]) -> list[float]:
    if len(names) != len(values):
        raise ValueError("joint names/values length mismatch")
    if len(set(names)) != len(names):
        raise ValueError("duplicate joint name")
    if len(set(wanted)) != len(wanted):
        raise ValueError("duplicate wanted joint name")
    _finite(values)
    positions = {name: i for i, name in enumerate(names)}
    absent = [name for name in wanted if name not in positions]
    if absent:
        raise ValueError("missing joint: " + ", ".join(absent))
    return _finite([values[positions[name]] for name in wanted]).tolist()


def _track_mean(names: Sequence[str], velocity: Sequence[float], groups: Sequence[Sequence[str]]) -> list[float]:
    return [float(np.clip(np.mean(extract_named(names, velocity, group)), -100., 100.) * .05)
            for group in groups]


def extract_track_2s(names: Sequence[str], velocity: Sequence[float], domain: str) -> list[float]:
    if domain == "simulator":
        return _track_mean(names, velocity, SIM_2S)
    if domain == "physical":
        return _track_mean(names, velocity, PHYSICAL_2S)
    raise ValueError("unknown domain")


def extract_track_24j(names: Sequence[str], velocity: Sequence[float]) -> list[float]:
    return _track_mean(names, velocity, SIM_24J)


def field_specs(variant: str) -> tuple[tuple[str, tuple[int, ...]], ...]:
    if variant == "Core":
        return CORE_FIELDS
    if variant == "Core+2S":
        return CORE_FIELDS + (TRACK_FIELD,)
    if variant == "Core+24J":
        return CORE_FIELDS + (ACTOR_FIELD,)
    raise ValueError("unknown variant")


def field_order_hash(variant: str) -> str:
    layout = {"version": VERSION, "variant": variant,
              "fields": [{"name": name, "shape": shape, "dtype": "float32"} for name, shape in field_specs(variant)],
              "tail": ["availability_mask (one per field)", "domain_source_mask (simulator,physical)"]}
    return hashlib.sha256(json.dumps(layout, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _field(value, stamp_ns: int | None, anchor_ns: int, max_age_ns: int, shape: tuple[int, ...], *, reason="missing"):
    if value is None:
        return {"value": None, "available": False, "reason": reason, "stamp_ns": stamp_ns, "age_ns": None}
    if stamp_ns is None:
        return {"value": None, "available": False, "reason": "missing_stamp", "stamp_ns": None, "age_ns": None}
    age = anchor_ns - int(stamp_ns)
    if age < 0 or age > max_age_ns:
        return {"value": None, "available": False, "reason": "stale", "stamp_ns": int(stamp_ns), "age_ns": age}
    array = _finite(value, shape=shape)
    return {"value": array.astype(np.float32).tolist(), "available": True, "reason": None,
            "stamp_ns": int(stamp_ns), "age_ns": age}


def _projected_gravity(orientation_xyzw):
    x, y, z, w = _finite(orientation_xyzw, shape=(4,))
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-9:
        raise ValueError("invalid quaternion")
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    # R(q)^T (0,0,-1), ROS xyzw orientation of body in world.
    return [2*(w*y-x*z), -2*(y*z+w*x), -(1-2*(x*x+y*y))]


def make_fixture(domain: str, sample: Mapping, *, max_age_ns: int) -> dict:
    if domain not in ("simulator", "physical") or max_age_ns < 0:
        raise ValueError("invalid domain or age limit")
    anchor = int(sample["anchor_stamp_ns"])
    joint_stamp = sample.get("joint_stamp_ns")
    fields = {}
    common = sample.get("common") or {}
    for name, shape in CORE_FIELDS:
        if name in ("body_angular_velocity", "projected_gravity", "flipper_position", "arm_configuration_state"):
            continue
        spec = common.get(name) or {}
        fields[name] = _field(spec.get("value"), spec.get("stamp_ns"), anchor, max_age_ns, shape)
        if fields[name]["available"] and name in ("dem_valid_history", "path_coverage"):
            array = np.asarray(fields[name]["value"])
            if not np.logical_or(array == 0, array == 1).all():
                raise ValueError("binary mask required: " + name)
    names = sample.get("joint_names") if sample.get("joint_names") is not None else []
    velocities = sample.get("joint_velocity") if sample.get("joint_velocity") is not None else []
    positions = sample.get("joint_position") if sample.get("joint_position") is not None else []
    group = SIM_FLIPPERS if domain == "simulator" else PHYSICAL_FLIPPERS
    try:
        flippers = extract_named(names, positions, group)
    except ValueError as error:
        flippers, flipper_reason = None, str(error)
    else:
        flipper_reason = "sign_or_zero_unverified" if domain == "physical" and not sample.get("flipper_calibration_verified") else "missing"
        if flipper_reason != "missing":
            flippers = None
    fields["flipper_position"] = _field(flippers, joint_stamp, anchor, max_age_ns, (4,), reason=flipper_reason)
    try:
        two = extract_track_2s(names, velocities, domain)
    except ValueError as error:
        two, track_reason = None, str(error)
    else:
        track_reason = "missing"
    fields[TRACK_FIELD[0]] = _field(two, joint_stamp, anchor, max_age_ns, (2,), reason=track_reason)
    if domain == "simulator":
        try:
            twentyfour = extract_track_24j(names, velocities)
        except ValueError as error:
            twentyfour, reason = None, str(error)
        else:
            reason = "missing"
        fields[ACTOR_FIELD[0]] = _field(twentyfour, joint_stamp, anchor, max_age_ns, (2,), reason=reason)
        fields["body_angular_velocity"] = _field(sample.get("body_angular_velocity"), sample.get("body_stamp_ns"), anchor, max_age_ns, (3,))
        fields["projected_gravity"] = _field(sample.get("projected_gravity"), sample.get("body_stamp_ns"), anchor, max_age_ns, (3,))
    else:
        imu = sample.get("imu") or {}
        covariance_orientation_bad = imu.get("orientation_covariance_0") == -1
        covariance_angular_bad = imu.get("angular_velocity_covariance_0") == -1
        verified = imu.get("extrinsic_verified") is True and imu.get("frame_id") == "chassis/body"
        imu_stamp = imu.get("stamp_ns")
        gravity = None
        if verified and not covariance_orientation_bad and imu.get("orientation_xyzw") is not None:
            gravity = _projected_gravity(imu["orientation_xyzw"])
        angular = imu.get("angular_velocity") if verified and not covariance_angular_bad else None
        fields["projected_gravity"] = _field(gravity, imu_stamp, anchor, max_age_ns, (3,),
                                             reason="covariance_invalid" if covariance_orientation_bad else "frame_or_extrinsic_unverified")
        fields["body_angular_velocity"] = _field(angular, imu_stamp, anchor, max_age_ns, (3,),
                                                  reason="covariance_invalid" if covariance_angular_bad else "frame_or_extrinsic_unverified")
    arm = sample.get("arm_configuration")
    arm_value = [1.] if arm == "fixed_stowed_pose_verified" else None
    fields["arm_configuration_state"] = _field(arm_value, sample.get("arm_stamp_ns"), anchor, max_age_ns, (1,), reason="arm_stowed_unverified")
    variants = {}
    for variant in ("Core", "Core+2S", "Core+24J"):
        spec = field_specs(variant)
        if domain == "physical" and variant == "Core+24J":
            variants[variant] = {"status": "unavailable_by_design", "tensor": None, "field_order_hash": field_order_hash(variant)}
            continue
        availability = [int(fields[name]["available"]) for name, _ in spec]
        if all(availability):
            values = np.concatenate([np.asarray(fields[name]["value"], dtype=np.float32).ravel() for name, _ in spec]
                                    + [np.asarray(availability + ([1, 0] if domain == "simulator" else [0, 1]), dtype=np.float32)])
            tensor = values.tolist()
            status = "available"
        else:
            tensor, status = None, "partial_unavailable"
        variants[variant] = {"status": status, "tensor": tensor, "availability_mask": availability,
                             "field_order_hash": field_order_hash(variant)}
    return {"domain": domain, "anchor_stamp_ns": anchor, "dtype": "float32", "fields": fields, "variants": variants}
