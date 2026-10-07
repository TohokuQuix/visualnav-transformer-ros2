"""Sequential R3 A→B coordinator; reserve candidates never enter a run."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

WORKTREE = Path("/tmp/eatnav-vnt-r3-settling-pilot")
ROOT = Path(os.environ.get("EATNAV_R3_ROOT", "/tmp/eatnav-r3-settling-pilot-20261008"))
sys.path.insert(0, str(WORKTREE / "scripts"))
from eat_nav_r3_settling_prepare import sha, write


def validate_seal(shard: str):
    sr = ROOT / shard
    seal = json.loads((sr / "seal.json").read_text())
    status = json.loads((sr / "status.json").read_text())
    manifest = json.loads((sr / "pose-manifest.json").read_text())
    if (status.get("stage") != "full_complete" or status.get("owned_residual") is not False or
        seal.get("native_exit_code") != 0 or seal.get("shard_id") != shard or
        seal.get("pose_ids") != [x["pose_id"] for x in manifest["poses"]] or
        sha(sr / "seal.json") != status.get("seal_sha256") or
        sha(sr / "sha-ledger.json") != seal.get("run_sha_ledger_sha256")):
        raise RuntimeError("invalid protocol seal: " + shard)
    return {"shard_id": shard, "seal_sha256": sha(sr / "seal.json"),
            "pose_ids": seal["pose_ids"],
            "physical_termination_count": seal["artifact_gate"]["terminated_pose_count"]}


def main():
    pre = json.loads((ROOT / "preflight.json").read_text())
    execution = json.loads((ROOT / "execution-manifest.json").read_text())
    if (sha(ROOT / "execution-manifest.json") != pre["execution_manifest_sha256"] or
        sha(ROOT / "primary-manifest.json") != pre["primary_manifest_sha256"] or
        sha(ROOT / "reserve-rank-freeze.json") != pre["reserve_manifest_sha256"] or
        execution["shard_order"] != ["shard-A", "shard-B"] or
        len(execution["primary_pose_ids"]) != 8 or
        set(execution["primary_pose_ids"]) & set(execution["reserve_pose_ids_excluded"])):
        raise RuntimeError("frozen R3 manifest or reserve exclusion changed")
    if (ROOT / "status.json").exists():
        raise RuntimeError("pilot already launched; no automatic retry")
    started = time.monotonic()
    completed = []
    write(ROOT / "status.json", {"stage": "running", "completed_shards": 0,
                                  "current_shard": "shard-A", "decision": None})
    try:
        for shard in execution["shard_order"]:
            sr = ROOT / shard
            if (sr / "host-launch.json").exists() or (sr / "seal.json").exists():
                raise RuntimeError("shard already launched: " + shard)
            freeze = json.loads((sr / "input-freeze.json").read_text())
            prelaunch = json.loads((sr / "prelaunch.json").read_text())
            if (sha(sr / "input-freeze.json") != prelaunch["freeze_sha256"] or
                sha(sr / "pose-manifest.json") != prelaunch["pose_manifest_sha256"] or
                [x["pose_id"] for x in json.loads((sr / "pose-manifest.json").read_text())["poses"]] !=
                next(x["pose_ids"] for x in execution["shards"] if x["shard_id"] == shard) or
                freeze["contact_contract"]["view_capacity"] != 4096 or
                freeze["planned_physics_steps"] != 200 or
                freeze["terrain_mesh_sha256"] != pre["terrain_mesh_sha256"]):
                raise RuntimeError("shard frozen input drift: " + shard)
            if time.monotonic() - started > freeze["limits"]["total_runtime_cap_sec"]:
                raise RuntimeError("total runtime cap before next shard")
            write(ROOT / "status.json", {"stage": "running", "completed_shards": len(completed),
                  "current_shard": shard, "elapsed_sec": time.monotonic() - started, "decision": None})
            env = {**os.environ, "EATNAV_R3_SHARD_ROOT": str(sr)}
            with (sr / "host-runner.stdout.log").open("wb") as out:
                child = subprocess.run(
                    [sys.executable, "-u", str(WORKTREE / "scripts/eat_nav_r3_settling_run.py")],
                    cwd=WORKTREE, env=env, stdout=out, stderr=subprocess.STDOUT)
            if child.returncode != 0:
                raise RuntimeError(f"shard host launcher failed: {shard}, exit {child.returncode}")
            completed.append(validate_seal(shard))
            write(ROOT / "completed-shards.json", {"completed": completed})
            # Valid measured termination/numeric-ineligibility is not a protocol failure.
        write(ROOT / "status.json", {"stage": "measurement_complete",
              "completed_shards": 2, "elapsed_sec": time.monotonic() - started,
              "decision": "offline_analysis_pending"})
        with (ROOT / "analysis.stdout.log").open("wb") as out:
            analyzed = subprocess.run(
                [sys.executable, "-u", str(WORKTREE / "scripts/eat_nav_r3_settling_analyze.py")],
                cwd=WORKTREE, env={**os.environ, "MPLCONFIGDIR": str(ROOT / "matplotlib-cache")},
                stdout=out, stderr=subprocess.STDOUT)
        if analyzed.returncode != 0:
            raise RuntimeError("offline analysis failed after sealed measurement")
        final = json.loads((ROOT / "analysis/final-decision.json").read_text())
        write(ROOT / "status.json", {"stage": "complete", "completed_shards": 2,
              "elapsed_sec": time.monotonic() - started, "decision": final["decision"],
              "candidate_free": True, "reserve_executed": False})
    except BaseException as exc:
        current = execution["shard_order"][len(completed)] if len(completed) < 2 else None
        reason = repr(exc)
        decision = "r3_candidate_free_pilot_measurement_invalid"
        if any(word in reason.lower() for word in
               ("resource", "integrity", "sha", "seal", "cap", "timeout", "residual", "bounds")):
            decision = "resource_or_integrity_stop"
        write(ROOT / "status.json", {
            "stage": "stopped", "completed_shards": len(completed),
            "current_shard": current, "decision": decision, "error": reason,
            "traceback": traceback.format_exc(),
            "elapsed_sec": time.monotonic() - started,
            "reserve_executed": False,
        })
        raise


if __name__ == "__main__":
    main()
