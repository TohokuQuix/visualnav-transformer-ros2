"""Offline assignment and descriptive analysis for P1 direct Path repeats."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict


SEEDS = {
    "first-differential": (2026101101, 2026101102, 2026101103),
    "second-graded": (2026101111, 2026101112, 2026101113),
}
METRICS = ("arrival_time_s", "two_second_progress_m", "tracking_mean_m",
           "contact_force_mean_N", "applied_torque_mean_abs", "recovery_count")


def direct_pairs() -> tuple[tuple[int, int], ...]:
    return ((0, 1), (2, 3))


def validate_four_follower_readiness(ready: dict) -> dict:
    expected = {
        "follower_alive": [True] * 4,
        "path_subscribers": [1] * 4,
        "tf_ready": [True] * 4,
        "command_sequence": [0] * 4,
        "rejected": [0] * 4,
    }
    if any(ready.get(name) != value for name, value in expected.items()) or \
            len(ready.get("status_counts", [])) != 4 or \
            any(value < 1 for value in ready["status_counts"]) or \
            ready.get("clock_publishers") != 1 or \
            ready.get("use_sim_time") is not True or ready.get("protocol_errors") != []:
        raise ValueError("four-follower readiness failed")
    return {"pass": True, "branches": 4}


def validate_four_follower_final(snapshot: dict, steps: int, last_stamp_ns: int) -> dict:
    if (snapshot.get("follower_alive") != [True] * 4 or
            snapshot.get("path_subscribers") != [1] * 4 or
            snapshot.get("tf_ready") != [True] * 4 or
            snapshot.get("command_sequence") != [steps] * 4 or
            snapshot.get("rejected") != [0] * 4 or
            len(snapshot.get("status_counts", [])) != 4 or
            any(value < 1 for value in snapshot["status_counts"]) or
            snapshot.get("clock_publishers") != 1 or
            snapshot.get("clock_now_ns") != last_stamp_ns or
            snapshot.get("use_sim_time") is not True or
            snapshot.get("protocol_errors") != []):
        raise ValueError("four-follower final state invalid")
    return {"pass": True, "branches": 4, "commands_per_branch": steps}


def build_schedule(source: dict) -> dict:
    """Make two identical fresh-process repeats of each frozen G3 seed block."""
    if source.get("batch_seeds") != {key: list(value) for key, value in SEEDS.items()}:
        raise ValueError("frozen seed bundle drift")
    processes = []
    for group, seeds in SEEDS.items():
        for block, seed in enumerate(seeds):
            rows = sorted((row for row in source["trials"] if
                           row["anchor_id"] == group and row["batch_index"] == block),
                          key=lambda row: row["slot"])
            if len(rows) != 4 or [row["slot"] for row in rows] != list(range(4)):
                raise ValueError("assignment must cover four slots")
            if any(row.get("batch_seed") != seed for row in rows):
                raise ValueError("trial seed differs from frozen batch seed")
            direct_rows = [row for row in rows if row["condition"] == "A"]
            if len(direct_rows) != 1 or not direct_rows[0].get("path_sha256"):
                raise ValueError("unmodified direct reference missing")
            direct = direct_rows[0]["path_sha256"]
            for repeat in range(2):
                processes.append({
                    "process_id": f"{group}-seed-{seed}-repeat-{repeat}",
                    "anchor_id": group, "batch_index": block,
                    "batch_seed": seed, "repeat_index": repeat,
                    "trials": [{"slot": slot, "env_id": slot,
                                "trial_id": f"{group}-seed-{seed}-repeat-{repeat}-env-{slot}",
                                "candidate_slot": f"direct_repeat_{slot}",
                                "condition": "A", "path_sha256": direct}
                               for slot in range(4)],
                })
    return {"schema": "eatnav_p1_g3_direct_repeat_schedule_v1",
            "independent_condition_count": 6, "processes": processes}


def validate_terrain_support(rows: list[dict]) -> dict:
    by_group = defaultdict(lambda: {"mesh_sha256": set(), "material_sha256": set()})
    for row in rows:
        for name in ("mesh_sha256", "material_sha256"):
            if not row.get(name):
                raise ValueError("terrain fingerprint missing: " + name)
            by_group[row["anchor_id"]][name].add(row[name])
    return {group: {"same_condition": all(len(values[name]) == 1 for name in
                                          ("mesh_sha256", "material_sha256")),
                    **{name: sorted(values[name]) for name in
                       ("mesh_sha256", "material_sha256")}}
            for group, values in sorted(by_group.items())}


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _describe(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None,
                "p10": None, "p90": None, "max": None}
    if any(not math.isfinite(value) for value in values):
        raise ValueError("nonfinite metric")
    return {"count": len(values), "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "median": statistics.median(values), "p10": _quantile(values, .1),
            "p90": _quantile(values, .9), "max": max(values)}


def extract_trial(rows: list[dict], arrival: dict, path: list[list[float]],
                  origin: list[float], identity: dict) -> dict:
    """Derive separate local execution metrics from one validated 20 Hz trace."""
    if len(rows) < 40 or len(path) < 2:
        raise ValueError("incomplete two-second trajectory or Path")
    if [row["control_step"] for row in rows] != list(range(len(rows))):
        raise ValueError("noncontiguous trial trace")
    arrival_step = arrival["arrival_step"]
    if arrival_step is not None and (arrival_step < 0 or arrival_step >= len(rows)):
        raise ValueError("arrival step outside trace")
    # The row whose follower first reports Goal is the first stop-hold row.
    limit = len(rows) if arrival_step is None else arrival_step
    active = rows[:limit]

    def local_xy(state):
        return state[0] - origin[0], state[1] - origin[1]

    def goal_distance(state):
        x, y = local_xy(state)
        return math.hypot(x - path[-1][0], y - path[-1][1])

    track = []
    forces = []
    torques = []
    recoveries = 0
    recovering = False
    modes = set()
    for row in active:
        x, y = local_xy(row["root_state"])
        distances = []
        for (ax, ay), (bx, by) in zip(path, path[1:]):
            dx, dy = bx - ax, by - ay
            squared = dx * dx + dy * dy
            if squared <= 0:
                raise ValueError("zero-length Path segment")
            fraction = max(0., min(1., ((x - ax) * dx + (y - ay) * dy) / squared))
            distances.append(math.hypot(x - ax - fraction * dx,
                                        y - ay - fraction * dy))
        track.append(min(distances))
        vectors = row["contact_net_forces_w"]
        if not vectors:
            raise ValueError("contact forces missing")
        forces.append(statistics.mean(math.sqrt(sum(float(component) ** 2
                                                    for component in vector))
                                      for vector in vectors))
        torque = row.get("applied_torque")
        if not torque:
            raise ValueError("applied torque missing")
        torques.append(statistics.mean(abs(float(value)) for value in torque))
        diagnosis = row["follower_diagnostic"]
        modes.add(diagnosis["mode"])
        now = bool(diagnosis["linear_recovery"] or diagnosis["angular_recovery"])
        if now and not recovering:
            recoveries += 1
        recovering = now
    result = dict(identity)
    result.update({
        "arrival_time_s": None if arrival_step is None else arrival_step * .05,
        "right_censored": bool(arrival["right_censored"]),
        "two_second_progress_m": goal_distance(rows[0]["root_state"]) -
                                 goal_distance(rows[39]["post_root_state"]),
        "common_goal_progress_2s_m": goal_distance(rows[0]["root_state"]) -
                                      goal_distance(rows[39]["post_root_state"]),
        "tracking_mean_m": statistics.mean(track) if track else None,
        "tracking_p95_m": _quantile(track, .95) if track else None,
        "contact_force_mean_N": statistics.mean(forces) if forces else None,
        "contact_force_p95_N": _quantile(forces, .95) if forces else None,
        "contact_force_peak_N": max(forces) if forces else None,
        "applied_torque_mean_abs": statistics.mean(torques) if torques else None,
        "applied_torque_p95_abs": _quantile(torques, .95) if torques else None,
        "applied_torque_peak_abs": max(torques) if torques else None,
        "recovery_count": recoveries,
        "recovery_modes": sorted(modes),
        "exposure_steps": len(active),
    })
    if any(isinstance(value, float) and not math.isfinite(value)
           for value in result.values()):
        raise ValueError("nonfinite trial metric")
    return result


def repeatability_summary(trials: list[dict]) -> dict:
    """Keep correlated env rows nested in process and seed-condition counts."""
    by_process = defaultdict(dict)
    for row in trials:
        key = (row["anchor_id"], row["batch_seed"],
               row.get("terrain_condition_key"), row["repeat_index"])
        slot = row["env_id"]
        if slot in by_process[key]:
            raise ValueError("duplicate env in process")
        by_process[key][slot] = row
    if any(set(rows) != set(range(4)) for rows in by_process.values()):
        raise ValueError("incomplete four-env process")
    conditions = {(group, seed, terrain) for group, seed, terrain, _ in by_process}
    pairs = [(rows[left], rows[right]) for rows in by_process.values()
             for left, right in direct_pairs()]
    replays = []
    for group, seed, terrain in conditions:
        first = by_process.get((group, seed, terrain, 0))
        second = by_process.get((group, seed, terrain, 1))
        if first is not None and second is not None:
            replays.extend((first[env], second[env]) for env in range(4))
    result = {"independent_condition_count": len(conditions),
              "process_count": len(by_process), "trial_count": len(trials),
              "within_process_pairs": len(pairs),
              "fresh_process_matched_pairs": len(replays),
              "duplicate_gate": {
                  "pass_pairs": sum(bool(a["duplicate_gate_pass"]) and
                                    bool(b["duplicate_gate_pass"]) for a, b in pairs),
                  "fail_pairs": sum(not (bool(a["duplicate_gate_pass"]) and
                                     bool(b["duplicate_gate_pass"])) for a, b in pairs),
              }}
    for metric in METRICS:
        values = [float(row[metric]) for row in trials if row[metric] is not None]
        within_signed = [float(b[metric]) - float(a[metric]) for a, b in pairs
                         if a[metric] is not None and b[metric] is not None]
        fresh_signed = [float(b[metric]) - float(a[metric]) for a, b in replays
                        if a[metric] is not None and b[metric] is not None]
        result[metric] = {**_describe(values),
                          "observed_count": len(values),
                          "censored_count": len(trials) - len(values) if metric == "arrival_time_s" else 0,
                          "within_process_signed_difference": _describe(within_signed),
                          "within_process_absolute_difference": _describe([abs(x) for x in within_signed]),
                          "fresh_process_signed_difference": _describe(fresh_signed),
                          "fresh_process_absolute_difference": _describe([abs(x) for x in fresh_signed])}
    strata = {"by_state_group": defaultdict(list),
              "by_seed_bundle": defaultdict(list),
              "by_env_id": defaultdict(list)}
    for row in trials:
        strata["by_state_group"][row["anchor_id"]].append(row)
        seed_label = f'{row["anchor_id"]}:{row["batch_seed"]}'
        if row.get("terrain_condition_key") is not None:
            seed_label += ':' + str(row["terrain_condition_key"])
        strata["by_seed_bundle"][seed_label].append(row)
        strata["by_env_id"][str(row["env_id"])].append(row)
    result["strata"] = {kind: {name: {metric: _describe([
        float(row[metric]) for row in records if row[metric] is not None])
        for metric in METRICS} for name, records in sorted(groups.items())}
        for kind, groups in strata.items()}
    breakdown = {}
    for metric in METRICS:
        process_means = {}
        env_spreads = []
        for key, records in by_process.items():
            observed = [float(row[metric]) for row in records.values()
                        if row[metric] is not None]
            if len(observed) == 4:
                process_means[key] = statistics.mean(observed)
                env_spreads.append(max(observed) - min(observed))
        process_diffs = []
        seed_means = defaultdict(list)
        for group, seed, terrain in conditions:
            first = process_means.get((group, seed, terrain, 0))
            second = process_means.get((group, seed, terrain, 1))
            if first is not None and second is not None:
                process_diffs.append(second - first)
                seed_means[group].append((first + second) / 2)
        breakdown[metric] = {
            "within_process_env_spread": _describe(env_spreads),
            "fresh_process_mean_difference": _describe(process_diffs),
            "seed_bundle_means_by_group": {
                group: _describe(values) for group, values in sorted(seed_means.items())},
            "interpretation": "descriptive_only; bundled seed varies scene/reset and paired noise together",
        }
    result["variation_breakdown"] = breakdown
    return result
