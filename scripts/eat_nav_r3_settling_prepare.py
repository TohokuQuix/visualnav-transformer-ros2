"""Freeze the eight-pose R3 pilot; do not start physics or reserves."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORKTREE = Path("/tmp/eatnav-vnt-r3-settling-pilot")
ROOT = Path(os.environ.get("EATNAV_R3_ROOT", "/tmp/eatnav-r3-settling-pilot-20261008"))
OFFLINE = Path("/tmp/eatnav-anchor-placement-20261008")
OLD = Path("/tmp/eatnav-radial-termination-20261008")
sys.path.insert(0, str(WORKTREE / "src"))
from visualnav_transformer.eat_nav.r3_anchor_contract import build_primary, build_reserve, requested_world, split_shards
from visualnav_transformer.eat_nav.anchor_placement_rules import select_r3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def docker_sha(path):
    p = subprocess.run(["docker", "exec", "isaac-lab-onix", "sha256sum", str(path)],
                       capture_output=True, text=True, check=True)
    return p.stdout.split()[0]


def verify_old_ledger(root, filename, expected_sha):
    path = root / filename
    if sha(path) != expected_sha:
        raise RuntimeError("old ledger SHA drift: " + str(path))
    ledger = json.loads(path.read_text())
    for item in ledger["files"]:
        old_file = root / item["path"]
        if not old_file.exists() or sha(old_file) != item["sha256"]:
            raise RuntimeError("old read-only artifact drift: " + str(old_file))
    return len(ledger["files"])


def main():
    if ROOT.exists():
        raise RuntimeError("pilot root already exists; no overwrite or automatic retry")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=WORKTREE, text=True):
        raise RuntimeError("pilot source worktree must be clean")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=WORKTREE, text=True).strip()
    container_id = subprocess.check_output(
        ["docker", "inspect", "--format", "{{.Id}}", "isaac-lab-onix"], text=True).strip()
    review = WORKTREE / "docs/eat-nav/anchor-placement-offline-20261008"
    offline_preflight = json.loads((OFFLINE / "preflight.json").read_text())
    old_count = verify_old_ledger(Path("/tmp/eatnav-radial-112-sharded-20261007"),
        "partial-output-sha-ledger.json", offline_preflight["old_ledger_sha256"])
    continuation_count = verify_old_ledger(OLD, "final-output-sha-ledger.json",
        offline_preflight["continuation_ledger_sha256"])
    proposal_path = OFFLINE / "analysis-v9/pilot-candidate-proposal.json"
    if sha(proposal_path) != sha(review / proposal_path.name):
        raise RuntimeError("offline proposal SHA differs from committed review")
    for filename in ("per-pose-review-summary.json", "r0-r3-comparison.json",
                     "source-input-provenance.json", "sha-ledger.json"):
        if sha(OFFLINE / "analysis-v9" / filename) != sha(review / filename):
            raise RuntimeError("offline review artifact SHA differs from committed copy: " + filename)
    offline_ledger = json.loads((OFFLINE / "analysis-v9/sha-ledger.json").read_text())
    for radius in (.30, .40, .50, .60):
        name = f"scan-v3/radius-{radius:.2f}.json"
        if sha(OFFLINE / name) != offline_ledger[name]:
            raise RuntimeError("frozen virtual DEM input SHA drift: " + name)
    scan_summary = json.loads((OFFLINE / "scan-v3/scan-summary.json").read_text())
    parity = json.loads((OFFLINE / "parity/parity-report.json").read_text())
    if (scan_summary["candidate_count"] != 448 or scan_summary["dem_complete"] != 448 or
        scan_summary["physics_steps_after_scene_init"] != 0 or
        parity["decision"] != "parity_pass" or parity["pass_count"] != 112):
        raise RuntimeError("offline parity/scan gate drift")
    if (sha(OFFLINE / "scan-v3/scan-summary.json") != sha(review / "scan-summary.json") or
        sha(OFFLINE / "parity/parity-report.json") != sha(review / "parity-report.json")):
        raise RuntimeError("committed offline report differs from original")
    frozen = json.loads((OFFLINE / "scan-freeze-v4.json").read_text())
    for path, expected in frozen["frozen_source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError("offline frozen source SHA drift: " + path)
    source_manifest = json.loads((OFFLINE / "inputs/pose-frames.json").read_text())
    for label, path in source_manifest["frozen_source_paths"].items():
        if docker_sha(path) != source_manifest["frozen_source_sha256"][label]:
            raise RuntimeError("container source/checkpoint SHA drift: " + label)
    old_freeze = json.loads((OLD / "cycle-1/shard-15/input-freeze.json").read_text())
    for label, path in old_freeze["source_paths"].items():
        if label in ("sim_source", "radial_shard_inventory", "radial_lean_log",
                     "radial_termination_capture"):
            continue
        if docker_sha(path) != old_freeze["source_sha256"][label]:
            raise RuntimeError("old physics/contact source SHA drift: " + label)
    if old_freeze["contact_contract"]["view_capacity"] != 4096:
        raise RuntimeError("contact capacity drift")
    for label, root in (("original", Path("/tmp/eatnav-radial-112-sharded-20261007")),
                        ("continuation", OLD)):
        pre = json.loads((root / "preflight.json").read_text())
        if not pre:
            raise RuntimeError(label + " preflight missing")
    proposal = json.loads(proposal_path.read_text())
    review_rows = json.loads((review / "per-pose-review-summary.json").read_text())
    primary = build_primary(proposal, review_rows)
    shard_a, shard_b = split_shards(primary)
    all_virtual = []
    for radius in (.30, .40, .50, .60):
        payload = json.loads((OFFLINE / f"scan-v3/radius-{radius:.2f}.json").read_text())
        all_virtual += payload["candidates"]
    if len(all_virtual) != 448:
        raise RuntimeError("full virtual candidate grid incomplete")
    by_id = {x["pose_id"]: x for x in all_virtual}
    if len(by_id) != 448:
        raise RuntimeError("virtual pose ID duplicated")
    for row in primary:
        source = by_id[row["pose_id"]]
        if (source["env_id"] != row["env_index"] or
            source["terrain_seed"] != row["terrain_seed"] or
            source["dem_sha256"] != row["virtual_dem_sha256"]):
            raise RuntimeError("primary environment/terrain/DEM identity drift: " + row["pose_id"])
        requested_world(row, source["env_origin_xyz"])
    r3 = select_r3(all_virtual)
    if len(r3) != 112:
        raise RuntimeError("frozen R3 selection count drift")
    if not {x["pose_id"] for x in primary}.issubset({x["pose_id"] for x in r3}):
        raise RuntimeError("primary pose absent from frozen R3 selection")
    reserve = build_reserve(r3, primary)
    if len(reserve) != 104:
        raise RuntimeError("reserve count drift")
    ROOT.mkdir(parents=True, exist_ok=False)
    write(ROOT / "primary-manifest.json", {
        "schema": "eatnav_r3_primary_v1", "source_commit": commit,
        "proposal_sha256": sha(proposal_path), "poses": primary,
    })
    write(ROOT / "reserve-rank-freeze.json", {
        "schema": "eatnav_r3_reserve_v1", "not_executable": True,
        "r3_rule_source_sha256": sha(WORKTREE / "src/visualnav_transformer/eat_nav/anchor_placement_rules.py"),
        "poses": reserve,
    })
    write(ROOT / "execution-manifest.json", {
        "schema": "eatnav_r3_two_shard_v1", "shard_order": ["shard-A", "shard-B"],
        "primary_pose_ids": [x["pose_id"] for x in primary],
        "shards": [
            {"shard_id": "shard-A", "pose_ids": [x["pose_id"] for x in shard_a]},
            {"shard_id": "shard-B", "pose_ids": [x["pose_id"] for x in shard_b]},
        ],
        "reserve_pose_ids_excluded": [x["pose_id"] for x in reserve],
        "candidate_free": True, "physics_steps_per_shard": 200,
    })
    source_files = [
        WORKTREE / "scripts/eat_nav_r3_settling_sim.py",
        WORKTREE / "scripts/eat_nav_r3_settling_run.py",
        WORKTREE / "scripts/eat_nav_r3_settling_continue.py",
        WORKTREE / "scripts/eat_nav_r3_settling_analyze.py",
        WORKTREE / "scripts/eat_nav_r3_settling_prepare.py",
        WORKTREE / "src/visualnav_transformer/eat_nav/r3_anchor_contract.py",
        WORKTREE / "src/visualnav_transformer/eat_nav/r3_anchor_decision.py",
        WORKTREE / "tests/test_r3_anchor_contract.py",
        WORKTREE / "tests/test_r3_anchor_decision.py",
        WORKTREE / "tests/test_r3_anchor_protocol.py",
    ]
    source_and_test_sha = {str(path): sha(path) for path in source_files}
    for name, rows in (("shard-A", shard_a), ("shard-B", shard_b)):
        sr = ROOT / name
        sr.mkdir(parents=True, exist_ok=False)
        write(sr / "pose-manifest.json", {"schema": "eatnav_r3_shard_poses_v1",
                                          "shard_id": name, "poses": rows})
        freeze = json.loads(json.dumps(old_freeze))
        freeze.update({
            "schema": "eatnav_r3_settling_freeze_v1",
            "shard_id": name, "new_top_root": str(ROOT), "inventory_root": str(ROOT),
            "expected_pose_count": 4,
            "pose_manifest_sha256": sha(sr / "pose-manifest.json"),
            "primary_manifest_path": str(ROOT / "primary-manifest.json"),
            "primary_manifest_sha256": sha(ROOT / "primary-manifest.json"),
            "terrain_mesh_sha256": scan_summary["terrain_mesh_sha256"],
            "sim_source_sha256": sha(WORKTREE / "scripts/eat_nav_r3_settling_sim.py"),
            "numeric_exceedance_stop": "none; record eligibility per pose",
            "no_further_automatic_retry": True,
            "candidate_free": True,
        })
        freeze["source_paths"]["sim_source"] = str(sr / "inputs/eat_nav_r3_settling_sim.py")
        freeze["source_sha256"]["sim_source"] = freeze["sim_source_sha256"]
        for key, filename in (
            ("radial_shard_inventory", "radial_shard_inventory.py"),
            ("radial_lean_log", "radial_lean_log.py"),
            ("radial_termination_capture", "radial_termination_capture.py"),
            ("r3_anchor_contract", "r3_anchor_contract.py"),
        ):
            path = WORKTREE / "src/visualnav_transformer/eat_nav" / filename
            freeze["source_paths"][key] = str(sr / "src/visualnav_transformer/eat_nav" / filename)
            freeze["source_sha256"][key] = sha(path)
        write(sr / "input-freeze.json", freeze)
        pythonpath = ":".join((str(sr / "src"), "/tmp/eatnav-train/src",
            "/tmp/eatnav-extended/isaac-source", "/workspace/isaaclab/source/isaaclab",
            "/workspace/isaaclab/source/isaaclab_rl", "/workspace/isaaclab/source/isaaclab_tasks",
            "/workspace/isaaclab_onix/source/onix_isaac_lab"))
        command = ["docker", "exec", "-e", "PYTHONPATH=" + pythonpath,
            "isaac-lab-onix", "/isaac-sim/python.sh",
            str(sr / "inputs/eat_nav_r3_settling_sim.py"), "--headless",
            "--freeze", str(sr / "input-freeze.json"),
            "--pose-manifest", str(sr / "pose-manifest.json"),
            "--batch-seed", "20261007", "--output", str(sr / "run"),
            "--mode", "full"]
        write(sr / "prelaunch.json", {
            "schema": "eatnav_r3_prelaunch_v1", "source_commit": commit,
            "worktree": str(WORKTREE), "shard_id": name,
            "source_and_test_sha256": source_and_test_sha,
            "freeze_sha256": sha(sr / "input-freeze.json"),
            "pose_manifest_sha256": sha(sr / "pose-manifest.json"),
            "command": command, "candidate_free": True,
        })
    write(ROOT / "preflight.json", {
        "schema": "eatnav_r3_preflight_v1", "source_commit": commit,
        "worktree": str(WORKTREE), "worktree_status": "clean",
        "container_id": container_id, "pilot_proposal_sha256": sha(proposal_path),
        "offline_scan_summary_sha256": sha(OFFLINE / "scan-v3/scan-summary.json"),
        "offline_parity_report_sha256": sha(OFFLINE / "parity/parity-report.json"),
        "primary_manifest_sha256": sha(ROOT / "primary-manifest.json"),
        "reserve_manifest_sha256": sha(ROOT / "reserve-rank-freeze.json"),
        "execution_manifest_sha256": sha(ROOT / "execution-manifest.json"),
        "source_and_test_sha256": source_and_test_sha,
        "terrain_mesh_sha256": scan_summary["terrain_mesh_sha256"],
        "old_inventory_preflight_sha256": sha("/tmp/eatnav-radial-112-sharded-20261007/preflight.json"),
        "old_continuation_preflight_sha256": sha(OLD / "preflight.json"),
        "old_ledger_verified_files": old_count,
        "continuation_ledger_verified_files": continuation_count,
        "candidate_free": True, "reserve_executed": False, "second_seed_executed": False,
    })
    print(ROOT, sha(ROOT / "preflight.json"), commit)


if __name__ == "__main__":
    main()
