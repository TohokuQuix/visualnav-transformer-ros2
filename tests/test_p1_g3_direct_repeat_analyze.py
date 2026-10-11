"""Offline aggregate accepts a sealed duplicate Fail as a measured outcome."""

import hashlib
import json

from eat_nav_p1_g1_lifecycle import seal_after_writers
from eat_nav_p1_g3_direct_repeat_analyze import collect_batch


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def test_collect_batch_preserves_four_trials_and_duplicate_fail(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    rows = [{"control_step": step, "label_exposure": True,
             "root_state": [step * .005, 0., 0.],
             "post_root_state": [(step + 1) * .005, 0., 0.],
             "contact_net_forces_w": [[3., 4., 0.]],
             "applied_torque": [2., -2.],
             "follower_diagnostic": {"linear_recovery": False,
                                     "angular_recovery": False, "mode": "default"}}
            for step in range(40)]
    hashes = []
    for slot in range(4):
        path = output / f"trace-slot-{slot}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
    _write(output / "process-provenance.json", {
        "anchor_id": "first-differential", "batch_index": 0,
        "repeat_index": 0, "batch_seed": 2026101101,
        "path_sha256": "direct", "checkpoint_sha256": "checkpoint"})
    provenance_sha = hashlib.sha256((output / "process-provenance.json").read_bytes()).hexdigest()
    _write(output / "complete.json", {
        "protocol_version": "eatnav_p1_longhorizon_g3_direct_repeat_v1",
        "trace_sha256": hashes, "Path_publications": 1,
        "process_provenance_sha256": provenance_sha})
    _write(output / "pilot-summary.json", {
        "anchor_id": "first-differential", "batch_index": 0, "repeat_index": 0,
        "arrivals": [{"arrival_step": None, "right_censored": True}] * 4})
    _write(output / "terrain.json", {
        "fingerprints": [{"canonical_mesh_sha256": "mesh"}] * 4,
        "material_sha256": "material", "world_tile_origins": [[0, 0, 0]] * 4})
    _write(output / "path-publication.json", {
        "trial_paths": [{"path_sha256": "direct",
                         "map_xy": [[0, 0], [.75, 0]]}] * 4})
    _write(output / "prepath-gate.json", {"pass": True})
    _write(tmp_path / "duplicate-gate.json", {"pairs": [
        {"env_pair": [0, 1], "pass": False, "failures": ["arrival_time_spread"]},
        {"env_pair": [2, 3], "pass": True, "failures": []}]})
    _write(tmp_path / "g3-host-status.json", {"failure": None,
           "native_exit": 0, "validation": {"trace_steps_per_env": [40] * 4}})
    _write(tmp_path / "cleanup-evidence.json", {"owned_residual": [],
           "bridge_residual": []})
    seal_after_writers(tmp_path, writer_state=dict.fromkeys(
        ("simulator", "heartbeat", "resource", "log"), True),
        stage="G3_batch_protocol_valid")
    result = collect_batch(tmp_path, batch_seed=2026101101)
    assert len(result["trials"]) == 4
    assert result["duplicate_gate"]["fail_pairs"] == 1
    assert result["terrain"]["mesh_sha256"] == "mesh"
    assert all(row["arrival_time_s"] is None for row in result["trials"])
    assert {row["terrain_condition_key"] for row in result["trials"]} == {"mesh:material"}
