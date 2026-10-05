"""Pure source-fixture contract tests; no ROS node or simulator startup."""

import json
from pathlib import Path

import numpy as np
import pytest

from visualnav_transformer.eat_nav.portable_input_v2 import (
    extract_named,
    extract_track_2s,
    extract_track_24j,
    field_order_hash,
    field_specs,
    make_fixture,
)


PHYSICAL = [
    "chassis/fr_sprocket_axle", "chassis/rr_sprocket_axle",
    "chassis/fl_sprocket_axle", "chassis/rl_sprocket_axle",
]
SIM = [
    "chassis_r08_sprocket_axle", "chassis_r01_sprocket_axle",
    "chassis_l08_sprocket_axle", "chassis_l01_sprocket_axle",
]


def test_two_sprocket_order_clip_then_scale_and_name_permutation():
    names = PHYSICAL + ["unrelated"]
    velocity = [150., 50., -150., -50., 123.]
    expected = [5., -5.]  # mean first, clip mean to ±100, then scale 0.05
    assert extract_track_2s(names, velocity, "physical") == expected
    assert extract_track_2s(names[::-1], velocity[::-1], "physical") == expected
    assert extract_track_2s(SIM, [150., 50., -150., -50.], "simulator") == expected


@pytest.mark.parametrize("names,values,reason", [
    (PHYSICAL[:-1], [1., 2., 3.], "missing"),
    (PHYSICAL + PHYSICAL[:1], [1., 2., 3., 4., 5.], "duplicate"),
    (PHYSICAL, [1., 2., float("nan"), 4.], "nonfinite"),
])
def test_two_sprocket_rejects_invalid_input(names, values, reason):
    with pytest.raises(ValueError, match=reason):
        extract_track_2s(names, values, "physical")


def test_named_position_and_velocity_are_kept_distinct():
    names = ["b", "a"]
    assert extract_named(names, [2., 1.], ["a", "b"]) == [1., 2.]


def test_full_24j_order_and_missing_joint():
    names = [f"chassis_r{i:02d}_sprocket_axle" for i in range(1, 9)]
    names += [f"chassis_l{i:02d}_sprocket_axle" for i in range(1, 9)]
    for side in ("fr", "rr", "fl", "rl"):
        names += [f"chassis_{side}_subroller_{i:02d}_axle" for i in range(1, 9)]
    values = [float(i) for i in range(48)]
    result = extract_track_24j(names, values)
    assert result == extract_track_24j(names[::-1], values[::-1])
    assert len(result) == 2
    with pytest.raises(ValueError, match="missing"):
        extract_track_24j(names[:-1], values[:-1])


def test_physical_missing_or_stale_is_masked_without_zero_fill():
    sample = {"joint_names": PHYSICAL, "joint_velocity": [1., 2., 3., 4.],
              "joint_position": [0., 0., 0., 0.], "joint_stamp_ns": 100,
              "anchor_stamp_ns": 100, "imu": None, "common": {}}
    result = make_fixture("physical", sample, max_age_ns=50)
    assert result["variants"]["Core+24J"]["status"] == "unavailable_by_design"
    assert result["variants"]["Core+24J"]["tensor"] is None
    assert result["variants"]["Core+2S"]["tensor"] is None
    assert result["fields"]["track_sprocket_velocity"]["value"] == pytest.approx([0.075, 0.175])
    assert result["fields"]["body_angular_velocity"]["value"] is None
    sample["anchor_stamp_ns"] = 200
    result = make_fixture("physical", sample, max_age_ns=50)
    assert result["fields"]["track_sprocket_velocity"]["value"] is None
    assert result["fields"]["track_sprocket_velocity"]["reason"] == "stale"


def test_physical_unverified_imu_covariance_blocks_core_body_fields():
    sample = {"joint_names": PHYSICAL, "joint_velocity": [1., 2., 3., 4.],
              "joint_position": [0., 0., 0., 0.], "joint_stamp_ns": 100,
              "anchor_stamp_ns": 100,
              "imu": {"angular_velocity": [0., 0., 1.], "orientation_xyzw": [0., 0., 0., 1.],
                      "stamp_ns": 100, "frame_id": "bno055", "orientation_covariance_0": -1.,
                      "angular_velocity_covariance_0": -1., "extrinsic_verified": False},
              "common": {}}
    result = make_fixture("physical", sample, max_age_ns=50)
    assert result["fields"]["projected_gravity"]["value"] is None
    assert result["fields"]["body_angular_velocity"]["value"] is None
    assert "covariance" in result["fields"]["projected_gravity"]["reason"]


def test_simulator_trace_reproduces_frozen_actor_and_proxy():
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root / "docs/quality-identifiability-gate/track-proxy-offline-review/source-provenance.json").read_text())
    import gzip
    run = source["traces"]["expanded-train-000-left_1"]
    with gzip.open(run["trace_path"], "rt") as stream:
        step = json.load(stream)[0]
    names = json.loads(Path(run["summary_path"]).read_text())["raw_joint_effort_field_order"]
    a24 = extract_track_24j(names, step["joint_velocity"])
    a2 = extract_track_2s(names, step["joint_velocity"], "simulator")
    assert np.allclose(a24, step["pre_noise"][14:16], atol=1e-5)
    saved = json.loads((root / "docs/quality-identifiability-gate/track-proxy-offline-review/track-proxy-comparison.json").read_text())
    with gzip.open(saved["timeseries_external_path"], "rt") as stream:
        rows = (json.loads(line) for line in stream)
        row = next(r for r in rows if r["run_id"] == "expanded-train-000-left_1" and r["step"] == 0)
    assert np.allclose(a2, row["proxy2"], atol=1e-8)


def test_field_order_hash_depends_on_variant_and_order():
    assert field_order_hash("Core") != field_order_hash("Core+2S")
    assert field_order_hash("Core+2S") != field_order_hash("Core+24J")
    assert len(field_order_hash("Core+2S")) == 64


def test_complete_simulator_fixture_has_finite_float32_tensor_and_masks():
    names = [f"chassis_r{i:02d}_sprocket_axle" for i in range(1, 9)]
    names += [f"chassis_l{i:02d}_sprocket_axle" for i in range(1, 9)]
    for side in ("fr", "rr", "fl", "rl"):
        names += [f"chassis_{side}_subroller_{i:02d}_axle" for i in range(1, 9)]
    names += ["chassis_rr_flipper_joint", "chassis_rl_flipper_joint",
              "chassis_fr_flipper_joint", "chassis_fl_flipper_joint"]
    common = {}
    for field, shape in field_specs("Core"):
        if field not in ("body_angular_velocity", "projected_gravity", "flipper_position", "arm_configuration_state"):
            common[field] = {"value": np.ones(shape, dtype=np.float32).tolist(), "stamp_ns": 100}
    sample = {"joint_names": names, "joint_velocity": [1.] * len(names),
              "joint_position": [0.] * len(names), "joint_stamp_ns": 100,
              "body_angular_velocity": [0., 0., 0.], "projected_gravity": [0., 0., -1.],
              "body_stamp_ns": 100, "arm_configuration": "fixed_stowed_pose_verified",
              "arm_stamp_ns": 100, "anchor_stamp_ns": 100, "common": common}
    result = make_fixture("simulator", sample, max_age_ns=50)
    for variant in ("Core", "Core+2S", "Core+24J"):
        output = result["variants"][variant]
        expected = sum(np.prod(shape) for _, shape in field_specs(variant)) + len(field_specs(variant)) + 2
        assert output["status"] == "available"
        assert len(output["tensor"]) == expected
        assert np.asarray(output["tensor"], dtype=np.float32).dtype == np.float32
        assert np.isfinite(output["tensor"]).all()
        assert output["tensor"][-2:] == [1., 0.]


def test_simulator_fixture_accepts_numpy_joint_vectors():
    sample = {"joint_names": np.asarray(SIM), "joint_velocity": np.asarray([1., 2., 3., 4.]),
              "joint_position": np.zeros(4), "joint_stamp_ns": 100,
              "anchor_stamp_ns": 100, "common": {}}
    result = make_fixture("simulator", sample, max_age_ns=50)
    assert result["fields"]["track_sprocket_velocity"]["value"] == pytest.approx([0.075, 0.175])


def test_nonfinite_unrelated_joint_is_rejected():
    with pytest.raises(ValueError, match="nonfinite"):
        extract_track_2s(PHYSICAL + ["unrelated"], [1., 2., 3., 4., float("nan")], "physical")


def test_invalid_dem_validity_mask_is_rejected():
    sample = {"joint_names": [], "joint_velocity": [], "joint_position": [],
              "joint_stamp_ns": 100, "anchor_stamp_ns": 100,
              "common": {"dem_valid_history": {"value": np.full((4,20,20), 0.5).tolist(), "stamp_ns": 100}}}
    with pytest.raises(ValueError, match="binary mask"):
        make_fixture("simulator", sample, max_age_ns=50)


def test_verified_body_frame_imu_projects_gravity_for_positive_pitch():
    half = 2 ** -0.5
    sample = {"joint_names": PHYSICAL, "joint_velocity": [0.] * 4,
              "joint_position": [0.] * 4, "joint_stamp_ns": 100,
              "anchor_stamp_ns": 100, "common": {},
              "imu": {"orientation_xyzw": [0., half, 0., half],
                      "angular_velocity": [0., 0., 0.], "stamp_ns": 100,
                      "frame_id": "chassis/body", "extrinsic_verified": True,
                      "orientation_covariance_0": 0., "angular_velocity_covariance_0": 0.}}
    result = make_fixture("physical", sample, max_age_ns=50)
    assert result["fields"]["projected_gravity"]["value"] == pytest.approx([1., 0., 0.], abs=1e-6)


def test_complete_validated_physical_input_can_assemble_core_and_2s_only():
    names = PHYSICAL + ["chassis/rr_flipper_joint", "chassis/rl_flipper_joint",
                        "chassis/fr_flipper_joint", "chassis/fl_flipper_joint"]
    common = {name: {"value": np.ones(shape).tolist(), "stamp_ns": 100}
              for name, shape in field_specs("Core")
              if name not in ("body_angular_velocity", "projected_gravity", "flipper_position", "arm_configuration_state")}
    sample = {"joint_names": names, "joint_velocity": [1.] * len(names),
              "joint_position": [0.] * len(names), "joint_stamp_ns": 100,
              "anchor_stamp_ns": 100, "flipper_calibration_verified": True,
              "arm_configuration": "fixed_stowed_pose_verified", "arm_stamp_ns": 100,
              "imu": {"orientation_xyzw": [0., 0., 0., 1.], "angular_velocity": [0., 0., 0.],
                      "stamp_ns": 100, "frame_id": "chassis/body", "extrinsic_verified": True,
                      "orientation_covariance_0": 0., "angular_velocity_covariance_0": 0.},
              "common": common}
    result = make_fixture("physical", sample, max_age_ns=50)
    assert result["variants"]["Core"]["status"] == "available"
    assert result["variants"]["Core+2S"]["status"] == "available"
    assert result["variants"]["Core+24J"]["status"] == "unavailable_by_design"
    assert result["variants"]["Core+2S"]["tensor"][-2:] == [0., 1.]
