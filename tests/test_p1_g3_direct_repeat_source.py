"""Static/synthetic checks for the Isaac-only four-follower derivative."""

import ast
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_statistical_pilot import path_in_map


ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT / "scripts/eat_nav_p1_g3_direct_repeat_sim.py"
HOST = ROOT / "scripts/eat_nav_p1_g3_direct_repeat_host.py"


def _function(path, name, namespace):
    node = next(item for item in ast.parse(path.read_text()).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def test_four_follower_builder_uses_identical_direct_geometry_for_each_pose():
    builder = _function(SIM, "direct_paths_for_slots", {"path_in_map": path_in_map})
    rows = [{"slot": i, "path_sha256": "direct"} for i in range(4)]
    lookup = {"direct": {"waypoints": [[0, 0], [.75, 0]]}}
    poses = [[0, float(i), 0] for i in range(4)]
    paths = builder(rows, lookup, poses)
    assert len(paths) == 4
    assert paths == [[[0., float(i)], [.75, float(i)]] for i in range(4)]
    with pytest.raises(ValueError):
        builder(rows[:3], lookup, poses[:3])


def test_acceptance_rejects_replaced_path_stamp_or_missing_branch():
    accepted = _function(SIM, "accepted_events", {})
    events = [{"branch": i, "accepted": True, "stamp_ns": 2050000000,
               "tf_stamp_ns": 2050000000, "path_stamp_ns": 2000000000}
              for i in range(4)]
    assert len(accepted({"events": events}, 2050000000, 2000000000)) == 4
    events[3]["path_stamp_ns"] = 2050000000
    with pytest.raises(ValueError, match="Path"):
        accepted({"events": events}, 2050000000, 2000000000)
    events.pop()
    with pytest.raises(ValueError, match="count"):
        accepted({"events": events}, 2050000000, 2000000000)


def test_four_follower_source_has_no_inactive_env_or_monitor_branch():
    source = SIM.read_text()
    assert "poses=initial_poses,paths=paths" in source
    assert "branches=4" in source
    assert "action_batch[2:]=0" not in source
    assert "command_tensor[2:]=0" not in source
    assert "monitor_path_equal_slot_0" not in source
    assert "range(4)" in source
    assert "trace-slot-{i}.jsonl') for i in range(4)" in source
    assert "'protocol_version':'eatnav_p1_longhorizon_g3_direct_repeat_v1'" in source


def test_host_preserves_frozen_duplicate_gate_and_validates_four_traces():
    source = HOST.read_text()
    assert "'--branches','4'" in source
    assert "for index in range(4):" in source
    assert "pair_duplicate_gate(traces,summary,terrain,publication,0,1)" in source
    assert "pair_duplicate_gate(traces,summary,terrain,publication,2,3)" in source
    assert "if not duplicate['pass']:" not in source
    assert "simulator_artifact_failure" in source
    assert "seal_after_writers" in source


def test_simulator_and_host_bind_process_seed_and_input_sha_into_completion():
    sim = SIM.read_text()
    host = HOST.read_text()
    assert "process-provenance.json" in sim
    assert "'batch_seed':batch_seed" in sim
    assert "'terrain_seed':int(anchor['terrain_seed'])" in sim
    assert "'process_provenance_sha256'" in sim
    assert "('process_provenance_sha256','process-provenance.json')" in host
