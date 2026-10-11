"""Staging must freeze all 12 processes without touching old evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from eat_nav_p1_g3_direct_repeat_stage import stage_bundle, verify_stage
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import build_schedule


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    repo, g2, old = (tmp_path / name for name in ("repo", "g2", "old"))
    for path in (repo / "scripts", repo / "src/visualnav_transformer/eat_nav",
                 g2 / "src/visualnav_transformer/eat_nav", g2 / "input", old):
        path.mkdir(parents=True, exist_ok=True)
    scripts = ("eat_nav_p1_g3_direct_repeat_sim.py",
               "eat_nav_p1_g3_direct_repeat_host.py")
    modules = ("p1_g3_shared_dem.py", "p1_dem_capture.py",
               "p1_longhorizon_protocol.py", "p1_g3_direct_repeat.py")
    for name in scripts:
        (repo / "scripts" / name).write_text(name)
    for name in modules:
        (repo / "src/visualnav_transformer/eat_nav" / name).write_text(name)
    for name in ("freeze.json", "statistical-config.json",
                 "source-contract.json", "anchor-fixtures.json", "eat_nav.yaml"):
        (g2 / "input" / name).write_text(name)
    source_assignment = {"batch_seeds": {
        "first-differential": [2026101101, 2026101102, 2026101103],
        "second-graded": [2026101111, 2026101112, 2026101113]},
        "trials": [{"anchor_id": group, "batch_index": block,
                    "batch_seed": ((2026101101, 2026101102, 2026101103)
                                   if group == "first-differential" else
                                   (2026101111, 2026101112, 2026101113))[block],
                    "slot": slot,
                    "condition": ("A", "B", "S", "T")[slot],
                    "path_sha256": "direct" if slot == 0 else "curve"}
                   for group in ("first-differential", "second-graded")
                   for block in range(3) for slot in range(4)]}
    (g2 / "input/assignment.json").write_text(json.dumps(source_assignment))
    old_assignment = {"trials": [
        {"anchor_id": group, "batch_index": block, "slot": 0,
         "candidate_slot": "direct_duplicate_0", "path_sha256": "direct"}
        for group in ("first-differential", "second-graded") for block in range(2)]}
    (old / "g3-assignment.json").write_text(json.dumps(old_assignment))
    manifest = {
        "schema": "eatnav_p1_longhorizon_g3_direct_repeat_v1",
        "execution_mode": "direct_repeat_four_env",
        "g3_simulator_source_sha256": sha(repo / "scripts" / scripts[0]),
        "g3_host_source_sha256": sha(repo / "scripts" / scripts[1]),
        "shared_dem_module_sha256": sha(repo / "src/visualnav_transformer/eat_nav" / modules[0]),
        "direct_repeat_module_sha256": sha(repo / "src/visualnav_transformer/eat_nav" / modules[3]),
        "source_reference_assignment_sha256": sha(old / "g3-assignment.json"),
        "assignment_sha256": "filled by staging",
    }
    return repo, g2, old, manifest


def test_stages_12_immutable_process_roots_and_verifies_sha(tmp_path):
    repo, g2, old, manifest = fixture(tmp_path)
    old_sha = sha(old / "g3-assignment.json")
    root = tmp_path / "new"
    result = stage_bundle(repo, g2, old, root, manifest)
    assert result == root
    assert len(list(root.glob("batch-*/input-sha-ledger.json"))) == 12
    first = root / "batch-first-differential-0-repeat-0"
    assert verify_stage(first, json.loads((first / "input/g3-contract.json").read_text()))["pass"]
    assert sha(old / "g3-assignment.json") == old_sha
    assert json.loads((first / "input/g3-assignment.json").read_text()) == build_schedule(
        json.loads((g2 / "input/assignment.json").read_text()))
    with pytest.raises(FileExistsError):
        stage_bundle(repo, g2, old, root, manifest)
    (first / "source.py").write_text("tampered")
    with pytest.raises(ValueError, match="SHA"):
        verify_stage(first, json.loads((first / "input/g3-contract.json").read_text()))
