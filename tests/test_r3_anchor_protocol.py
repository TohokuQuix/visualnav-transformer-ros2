"""Prelaunch parity checks for the reused first-seed measurement protocol."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import types

import numpy as np
import pytest

from visualnav_transformer.eat_nav.radial_interaction import classify_dem
from visualnav_transformer.eat_nav.r3_anchor_decision import physical_supply

ROOT = Path(__file__).resolve().parents[1]


def named_ast(path, name):
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return ast.dump(node, include_attributes=False)
    raise AssertionError(f"{name} absent from {path}")


def test_terrain_generator_dem_and_termination_capture_are_unchanged():
    old = ROOT / "scripts/eat_nav_radial_five_pose_sim.py"
    new = ROOT / "scripts/eat_nav_r3_settling_sim.py"
    for name in ("RadialTerrainGenerator", "extract_dem", "state_snapshot",
                 "measured_update", "capture_before_reset"):
        assert named_ast(old, name) == named_ast(new, name)


def test_physics_contact_history_and_reset_contract_is_frozen():
    source = (ROOT / "scripts/eat_nav_r3_settling_sim.py").read_text()
    assert "frame_steps=(4,9,14,19)" in source
    assert "control_count=6 if args.mode=='diagnostic' else 20" in source
    assert "if physics_count!=(control_step+1)*10:" in source
    assert "max_contact_data_count=view_capacity" in source
    assert "bare._reset_idx=capture_before_reset" in source
    assert "if not torch.all(root[i,7:13] == 0):" in source
    assert "terrain mesh SHA drift before physics" in source
    assert "post_reset_state':False" in source


def test_old_limits_and_classifier_sha_equal_frozen_review():
    from visualnav_transformer.eat_nav.radial_shard_inventory import settled_exceedance_reasons
    import inspect
    source = inspect.getsource(settled_exceedance_reasons)
    for threshold in (".10", ".20", ".70"):
        assert threshold in source
    classifier = ROOT / "src/visualnav_transformer/eat_nav/radial_interaction.py"
    assert hashlib.sha256(classifier.read_bytes()).hexdigest() == (
        "fc12a07a372c3597402bf6002e6ef000c63bd7a50db1717c47f2d0c18a3addf8")


def test_unknown_is_never_counted_as_interaction_supply():
    interaction = classify_dem(np.zeros((20, 20)), np.zeros((20, 20), dtype=bool), .1)
    assert interaction["class"] == "unknown"
    assert interaction["interaction_start_m"] is None
    assert not physical_supply({
        "virtual_class": "transition", "classification": "settled_anchor_eligible",
        "settled_class": "unknown", "dem_valid_fraction": 1.0, "known_onset": False})


def test_shard_b_only_after_protocol_seal_and_no_reserve_execution():
    coordinator = (ROOT / "scripts/eat_nav_r3_settling_continue.py").read_text()
    assert "completed.append(validate_seal(shard))" in coordinator
    assert 'execution["shard_order"]' in coordinator
    assert "reserve_pose_ids_excluded" in coordinator
    host = (ROOT / "scripts/eat_nav_r3_settling_run.py").read_text()
    assert "verify_shard_artifacts(ROOT/'run',pose_ids,termination_aware=True)" in host
    assert "owned_alive(item)" in host
    assert "host/container SHA mismatch" in host


def test_shard_a_launcher_failure_never_starts_shard_b(tmp_path, monkeypatch):
    source = ROOT / "scripts/eat_nav_r3_settling_continue.py"
    spec = importlib.util.spec_from_file_location("r3_continue_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    primary = {"poses": [{"pose_id": f"pose-{i}"} for i in range(8)]}
    reserve = {"poses": [{"pose_id": "reserve"}]}
    execution = {"shard_order": ["shard-A", "shard-B"],
                 "primary_pose_ids": [f"pose-{i}" for i in range(8)],
                 "reserve_pose_ids_excluded": ["reserve"],
                 "shards": [{"shard_id": "shard-A", "pose_ids": [f"pose-{i}" for i in (0,2,4,6)]},
                            {"shard_id": "shard-B", "pose_ids": [f"pose-{i}" for i in (1,3,5,7)]}]}
    for name, value in (("primary-manifest.json", primary), ("reserve-rank-freeze.json", reserve),
                        ("execution-manifest.json", execution)):
        (tmp_path / name).write_text(json.dumps(value))
    a = tmp_path / "shard-A"; a.mkdir()
    manifest = {"poses": [primary["poses"][i] for i in (0,2,4,6)]}
    freeze = {"contact_contract": {"view_capacity": 4096},
              "planned_physics_steps": 200, "terrain_mesh_sha256": "mesh",
              "limits": {"total_runtime_cap_sec": 1800}}
    for name, value in (("pose-manifest.json", manifest), ("input-freeze.json", freeze)):
        (a / name).write_text(json.dumps(value))
    (a / "prelaunch.json").write_text(json.dumps({
        "freeze_sha256": module.sha(a / "input-freeze.json"),
        "pose_manifest_sha256": module.sha(a / "pose-manifest.json")}))
    (tmp_path / "preflight.json").write_text(json.dumps({
        "execution_manifest_sha256": module.sha(tmp_path / "execution-manifest.json"),
        "primary_manifest_sha256": module.sha(tmp_path / "primary-manifest.json"),
        "reserve_manifest_sha256": module.sha(tmp_path / "reserve-rank-freeze.json"),
        "terrain_mesh_sha256": "mesh"}))
    monkeypatch.setattr(module.subprocess, "run", lambda *args, **kwargs: types.SimpleNamespace(returncode=1))
    with pytest.raises(RuntimeError, match="shard host launcher failed"):
        module.main()
    assert not (tmp_path / "shard-B" / "host-launch.json").exists()
    assert json.loads((tmp_path / "status.json").read_text())["completed_shards"] == 0
