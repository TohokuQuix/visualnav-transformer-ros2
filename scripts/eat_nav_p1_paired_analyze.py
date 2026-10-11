"""Read-only paired-noise P1 integrity and descriptive analysis."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

from eat_nav_p1_causal_analyze import local_xy, roll_pitch, world_to_body, yaw
from visualnav_transformer.eat_nav.p1_statistical_analysis import extract_diagnostic, extract_outcome


ROOT_NAMES = {
    ("first-differential", 0): "pilot-first-differential-b0-run-v2",
    ("second-graded", 0): "pilot-second-graded-b0-v1",
    **{(group, block): f"pilot-{group}-b{block}-v1"
       for group in ("first-differential", "second-graded") for block in (1, 2)},
}
CONTRASTS = (("B", "A"), ("S", "B"), ("T", "S"), ("T", "A"))
METRICS = (
    "reward_2s", "goal_progress_2s_m", "goal_progress_6s_m",
    "forward_displacement_2s_m", "body_forward_mean_2s_mps",
    "applied_forward_mean_2s_mps", "applied_forward_integral_2s_m",
    "forward_tracking_error_2s_mps", "zero_forward_steps_2s",
    "angular_command_abs_mean_2s_radps", "heading_change_2s_rad",
    "lateral_displacement_2s_m", "path_error_mean_2s_m",
    "contact_force_sum_mean_2s_N", "flipper_torque_abs_mean_2s_Nm",
    "roll_pitch_rms_2s_rad",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nearest_path_distance(point, path):
    best = math.inf
    for a, b in zip(path, path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        t = max(0., min(1., ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy)
                           / max(dx * dx + dy * dy, 1e-12)))
        best = min(best, math.hypot(point[0] - a[0] - t * dx,
                                    point[1] - a[1] - t * dy))
    return best


def summarize(trace, path, origin, flipper_ids):
    outcome = extract_outcome(trace, (.75, 0.), .99)
    diagnostic = extract_diagnostic(trace, (.75, 0.))
    first = trace[0]["root_state"]
    start_yaw = yaw(first[3:7])
    path_world = [(p[0] + origin[0], p[1] + origin[1]) for p in path]
    local = local_xy(trace[39]["post_root_state"][:2], first[:2], start_yaw)
    steps = trace[:40]
    body_forward = [world_to_body(row["post_root_state"][3:7],
                                  row["post_root_state"][7:10])[0] for row in steps]
    applied = [row["applied_command"][0] for row in steps]
    contact = [sum(math.sqrt(sum(c * c for c in force))
                   for force in row["contact_net_forces_w"]) for row in steps]
    torques = [statistics.mean(abs(row["applied_torque"][i]) for i in flipper_ids)
               for row in steps]
    rp = [roll_pitch(row["post_root_state"][3:7]) for row in steps]
    final_yaw = yaw(trace[39]["post_root_state"][3:7])
    return {
        "reward_2s": outcome["discounted_reward_2s"],
        "goal_progress_2s_m": outcome["global_goal_progress_2s_m"],
        "goal_progress_6s_m": diagnostic["global_goal_progress_6s_m"],
        "forward_displacement_2s_m": local[0],
        "body_forward_mean_2s_mps": statistics.mean(body_forward),
        "applied_forward_mean_2s_mps": statistics.mean(applied),
        "applied_forward_integral_2s_m": sum(applied) * .05,
        "forward_tracking_error_2s_mps": statistics.mean(
            actual - command for actual, command in zip(body_forward, applied)),
        "zero_forward_steps_2s": sum(abs(value) < 1e-9 for value in applied),
        "angular_command_abs_mean_2s_radps": statistics.mean(
            abs(row["applied_command"][2]) for row in steps),
        "heading_change_2s_rad": math.atan2(math.sin(final_yaw - start_yaw),
                                              math.cos(final_yaw - start_yaw)),
        "lateral_displacement_2s_m": local[1],
        "path_error_mean_2s_m": statistics.mean(
            nearest_path_distance(row["post_root_state"][:2], path_world)
            for row in steps),
        "contact_force_sum_mean_2s_N": statistics.mean(contact),
        "flipper_torque_abs_mean_2s_Nm": statistics.mean(torques),
        "roll_pitch_rms_2s_rad": math.sqrt(statistics.mean(r * r + p * p for r, p in rp)),
        "intervention_reasons": dict(Counter(row["intervention_reason"] for row in trace)),
        "terminal_count": sum(bool(row["terminal"]) for row in trace),
        "recovery_count": sum(bool(row["follower_diagnostic"]["angular_recovery"]
                                   or row["follower_diagnostic"]["linear_recovery"])
                              for row in trace),
    }


def analyze(root: Path):
    manifest = json.loads((root / "assignment.json").read_text())
    by_trial = {row["trial_id"]: row for row in manifest["trials"]}
    assert len(by_trial) == len(manifest["trials"]) == 24
    trials, ledger, noise_hashes = [], [], {}
    for (group, block), name in ROOT_NAMES.items():
        run = root / name
        output = run / "output"
        seal = json.loads((run / "host-seal.json").read_text())
        complete = json.loads((output / "complete.json").read_text())
        status = json.loads((run / "host-status.json").read_text())
        assert sha(output / "complete.json") == seal["complete_sha256"]
        assert sha(run / "input/freeze.json") == seal["freeze_sha256"]
        assert complete["control_steps"] == 120 and complete["Path_publications"] == 1
        assert complete["stage"] == "pilot_complete"
        assert status["native_exit"] == 0 and status["resource_failure"] is None
        assert status["peak_gpu_mib"] < 7680
        assert json.loads((output / "prepath-gate.json").read_text())["pass"]
        assert not json.loads((run / "bridge-cleanup.json").read_text())["owned_residual"]
        publication = json.loads((output / "path-publication.json").read_text())
        paths = {row["trial_id"]: row for row in publication["trial_paths"]}
        fixture = json.loads((run / "input/anchor-fixtures.json").read_text())
        names = fixture[group]["prephysics_state"]["joint_names"]
        flipper_ids = [i for i, name in enumerate(names) if "flipper_joint" in name]
        assert len(flipper_ids) == 4
        initial = json.loads((output / "initial-state.json").read_text())
        rows = [row for row in manifest["trials"] if row["anchor_id"] == group
                and row["batch_index"] == block]
        assert len(rows) == 4 and {row["condition"] for row in rows} == set("ABST")
        traces = {}
        for row in rows:
            slot = row["slot"]
            path = output / f"trace-slot-{slot}.jsonl"
            assert sha(path) == complete["trace_sha256"][slot]
            trace = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(trace) == 120 and [x["control_step"] for x in trace] == list(range(120))
            assert trace[0]["condition"] == row["condition"]
            assert paths[row["trial_id"]]["path_sha256"] == row["path_sha256"]
            assert not any(x["terminal"] for x in trace)
            assert not any(x["follower_diagnostic"]["angular_recovery"]
                           or x["follower_diagnostic"]["linear_recovery"] for x in trace)
            record = dict(row, root=str(run), trace_sha256=sha(path),
                          **summarize(trace, paths[row["trial_id"]]["map_xy"],
                                      initial[slot]["tile_origin_xy"], flipper_ids))
            assert record["trial_id"] in by_trial
            trials.append(record)
            traces[row["condition"]] = trace
        reference = traces["A"]
        t0 = reference[0]["logical_sim_time_ns"]
        for step in range(120):
            stamp = t0 + step * 50_000_000
            for condition, trace in traces.items():
                item = trace[step]
                assert item["logical_sim_time_ns"] == stamp
                assert item["follower_diagnostic"]["command_stamp_ns"] == stamp
                accepted = item["bridge_acceptance"]
                assert accepted["accepted"] and accepted["stamp_ns"] == stamp
                assert accepted["tf_stamp_ns"] == stamp
                assert item["actor_injected_noise"] == reference[step]["actor_injected_noise"]
        noise_hashes[(group, block)] = hashlib.sha256(json.dumps(
            [item["actor_injected_noise"] for item in reference], sort_keys=True
        ).encode()).hexdigest()
        ledger.append({"group": group, "block": block, "run": str(run),
                       "host_seal_sha256": sha(run / "host-seal.json"),
                       "native_exit": status["native_exit"],
                       "peak_gpu_mib": status["peak_gpu_mib"],
                       "elapsed_s": status["elapsed_s"]})
    assert len(set(noise_hashes.values())) == 6
    pairs = []
    for group in ("first-differential", "second-graded"):
        for block in range(3):
            lookup = {row["condition"]: row for row in trials
                      if row["anchor_id"] == group and row["batch_index"] == block}
            for left, right in CONTRASTS:
                pairs.append({"group": group, "block": block, "contrast": f"{left}-{right}",
                              "trial_ids": [lookup[left]["trial_id"], lookup[right]["trial_id"]],
                              "difference": {key: lookup[left][key] - lookup[right][key]
                                             for key in METRICS}})
    aggregate = []
    for group in ("first-differential", "second-graded"):
        for left, right in CONTRASTS:
            matched = [row for row in pairs if row["group"] == group
                       and row["contrast"] == f"{left}-{right}"]
            aggregate.append({"group": group, "contrast": f"{left}-{right}", "n_blocks": 3,
                              "metrics": {key: {"mean": statistics.mean(row["difference"][key]
                                                                      for row in matched),
                                                "range": [min(row["difference"][key] for row in matched),
                                                          max(row["difference"][key] for row in matched)]}
                                          for key in METRICS}})
    return {"trials": trials, "pairs": pairs, "aggregate": aggregate, "ledger": ledger,
            "paired_noise_stream_sha256": [
                {"group": group, "block": block, "sha256": digest}
                for (group, block), digest in noise_hashes.items()]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    args.output.mkdir(parents=True, exist_ok=False)
    for key, value in result.items():
        (args.output / f"{key}.json").write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
