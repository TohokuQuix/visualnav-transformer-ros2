"""Read-only comparison of frozen 24-joint Actor and two-sprocket track signals."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def finite_array(value, shape, name):
    array = np.asarray(value, dtype=float)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name}: invalid shape or nonfinite value {array.shape}")
    return array


def rms(array):
    return float(np.sqrt(np.mean(np.square(array))))


def corr(x, y, method):
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    value = method(x, y).statistic
    return float(value) if np.isfinite(value) else None


def metrics(x, y, eps=1e-6, deadband=1e-3):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    d = y - x
    active = (np.abs(x) > deadband) | (np.abs(y) > deadband)
    return {
        "n": int(len(x)), "actor24_mean": float(np.mean(x)), "proxy2_mean": float(np.mean(y)),
        "mean_absolute_error": float(np.mean(np.abs(d))),
        "median_absolute_error": float(np.median(np.abs(d))),
        "p95_absolute_error": float(np.quantile(np.abs(d), .95)),
        "mean_relative_absolute_error": float(np.mean(np.abs(d) / np.maximum(np.abs(x), eps))),
        "median_relative_absolute_error": float(np.median(np.abs(d) / np.maximum(np.abs(x), eps))),
        "bias_proxy_minus_actor": float(np.mean(d)), "rmse": rms(d),
        "pearson": corr(x, y, pearsonr), "spearman": corr(x, y, spearmanr),
        "sign_agreement_active": float(np.mean(np.sign(x[active]) == np.sign(y[active]))) if active.any() else None,
        "active_n": int(active.sum()),
    }


def motion(command):
    vx, _, yaw = command
    if abs(vx) < .05 and abs(yaw) < .05:
        return "stopped"
    if abs(yaw) >= .15:
        return "turning"
    if abs(vx) >= .05 and abs(yaw) < .15:
        return "straight"
    return "low_speed_other"


def distance(a, b):
    return rms(np.asarray(a) - np.asarray(b))


def json_dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n")


def save_svg(fig, path):
    fig.savefig(path)
    # Matplotlib multiline path data contains terminal spaces; keep committed SVG clean.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-output", type=Path, required=True)
    parser.add_argument("--external-output", type=Path, required=True)
    args = parser.parse_args()
    out, external = args.repo_output, args.external_output
    external.mkdir(parents=True, exist_ok=True)
    provenance = json.loads((out / "source-provenance.json").read_text())
    labels_path = out.parent / "evaluator-multihead-freeze" / "multihead-labels.jsonl.gz"
    labels = {}
    with gzip.open(labels_path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            labels[row["process_id"]] = row
    identities = {r["process_id"]: r for r in provenance["run_identity"]}
    if set(labels) != set(identities) or set(labels) != set(provenance["traces"]):
        raise ValueError("run identity mismatch")
    records = []
    byrun = {}
    maximum_reconstruction_error = 0.
    for run_id in sorted(labels):
        source = provenance["traces"][run_id]
        if sha(source["trace_path"]) != source["sha256"] or sha(source["summary_path"]) != source["summary_sha256"]:
            raise ValueError(f"source SHA mismatch: {run_id}")
        label, identity = labels[run_id], identities[run_id]
        if label["source_trace_sha256"] != source["sha256"] or label["candidate_id"] != identity["candidate_id"] or label["state_group_id"] != identity["state_group_id"]:
            raise ValueError(f"label identity mismatch: {run_id}")
        summary = json.loads(Path(source["summary_path"]).read_text())
        names = summary["raw_joint_effort_field_order"]
        if len(names) != 52 or len(set(names)) != 52:
            raise ValueError(f"bad joint order: {run_id}")
        side24 = {
            "right": [i for i, n in enumerate(names) if n.startswith(("chassis_r0", "chassis_fr_subroller", "chassis_rr_subroller")) and ("sprocket_axle" in n or "subroller_" in n)],
            "left": [i for i, n in enumerate(names) if n.startswith(("chassis_l0", "chassis_fl_subroller", "chassis_rl_subroller")) and ("sprocket_axle" in n or "subroller_" in n)],
        }
        if any(len(v) != 24 for v in side24.values()):
            raise ValueError(f"24 joint grouping failed: {run_id}: {side24}")
        side2 = {"right": [names.index("chassis_r08_sprocket_axle"), names.index("chassis_r01_sprocket_axle")],
                 "left": [names.index("chassis_l08_sprocket_axle"), names.index("chassis_l01_sprocket_axle")]}
        with gzip.open(source["trace_path"], "rt") as stream:
            trace = json.load(stream)
        if len(trace) != 120:
            raise ValueError(f"trace step count: {run_id}: {len(trace)}")
        a24, a2, commands, actions, xy = [], [], [], [], []
        for step, row in enumerate(trace):
            if row["step_identity"]["control_step"] != step or row["step_identity"]["run_id"] != run_id:
                raise ValueError(f"step identity mismatch: {run_id}/{step}")
            velocity = finite_array(row["joint_velocity"], (52,), "joint_velocity")
            observed = finite_array(row["pre_noise"], (22,), "pre_noise")
            command = finite_array(row["command"], (3,), "command")
            action = finite_array(row["actor_action"], (6,), "actor_action")
            root = finite_array(row["root_state"], (13,), "root_state")
            v24 = np.array([np.clip(np.mean(velocity[side24[s]]), -100, 100) * .05 for s in ("right", "left")])
            v2 = np.array([np.clip(np.mean(velocity[side2[s]]), -100, 100) * .05 for s in ("right", "left")])
            error = float(np.max(np.abs(v24 - observed[14:16])))
            maximum_reconstruction_error = max(maximum_reconstruction_error, error)
            if error > 1e-5:
                raise ValueError(f"Actor 24j reconstruction mismatch: {run_id}/{step}: {error}")
            a24.append(v24); a2.append(v2); commands.append(command); actions.append(action); xy.append(root[:2])
            records.append({"run_id": run_id, "state_group_id": label["state_group_id"], "terrain": label["terrain"],
                            "candidate_id": label["candidate_id"], "step": step, "motion": motion(command),
                            "actor24": v24.tolist(), "proxy2": v2.tolist(), "proxy_minus_actor": (v2-v24).tolist()})
        byrun[run_id] = {"label": label, "identity": identity, "actor24": np.array(a24), "proxy2": np.array(a2),
                         "command": np.array(commands), "action": np.array(actions), "xy": np.array(xy)}
    timeseries_path = external / "track-proxy-timeseries.jsonl.gz"
    with gzip.open(timeseries_path, "wt") as stream:
        for row in records:
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    strata = defaultdict(list)
    for row in records:
        for field in ("terrain", "state_group_id", "candidate_id", "motion"):
            strata[(field, row[field])].append(row)
    aggregate = {}
    for field, value in sorted(strata):
        rows = strata[(field, value)]
        aggregate.setdefault(field, {})[value] = {side: metrics([r["actor24"][i] for r in rows], [r["proxy2"][i] for r in rows])
                                                     for i, side in enumerate(("right", "left"))}
    aggregate["overall"] = {side: metrics([r["actor24"][i] for r in records], [r["proxy2"][i] for r in records])
                            for i, side in enumerate(("right", "left"))}
    group_results = []
    all_candidate = []
    for group in sorted({r["label"]["state_group_id"] for r in byrun.values()}):
        runs = [r for r in byrun.values() if r["label"]["state_group_id"] == group]
        direct = sorted((r for r in runs if r["label"]["candidate_id"] == "direct_reference"), key=lambda r: r["identity"]["repeat_id"])
        candidates = sorted((r for r in runs if r["label"]["candidate_id"] != "direct_reference"), key=lambda r: r["label"]["candidate_id"])
        if len(direct) != 3 or len(candidates) != 6:
            raise ValueError(f"3+6 count mismatch: {group}")
        datum = {"state_group_id": group, "terrain": runs[0]["label"]["terrain"], "candidate_count": len(candidates)}
        for key in ("actor24", "proxy2", "command", "action", "xy"):
            dup = [distance(direct[i][key], direct[j][key]) for i, j in ((0, 1), (0, 2), (1, 2))]
            direct_mean = np.mean([r[key] for r in direct], axis=0)
            cand_dist = [distance(r[key], direct_mean) for r in candidates]
            candidate_pairwise = [distance(candidates[i][key], candidates[j][key]) for i in range(6) for j in range(i + 1, 6)]
            dup_mean = float(np.mean(dup))
            datum[key] = {"duplicate_pairwise_rms": dup, "duplicate_mean_rms": float(np.mean(dup)),
                          "candidate_to_direct_rms": cand_dist, "candidate_mean_rms": float(np.mean(cand_dist)),
                          "candidate_pairwise_rms": candidate_pairwise,
                          "candidate_pairwise_mean_rms": float(np.mean(candidate_pairwise)),
                          "candidate_over_duplicate": float(np.mean(cand_dist) / dup_mean) if dup_mean > 1e-6 else None,
                          "duplicate_ratio_note": "NA: exact duplicate (zero variation)" if dup_mean <= 1e-6 else None}
        x, y = datum["actor24"]["candidate_to_direct_rms"], datum["proxy2"]["candidate_to_direct_rms"]
        comparable = [(i, j) for i in range(6) for j in range(i + 1, 6) if abs(x[i]-x[j]) > 1e-6 and abs(y[i]-y[j]) > 1e-6]
        datum["ranking"] = {"concordance": float(np.mean([np.sign(x[i]-x[j]) == np.sign(y[i]-y[j]) for i, j in comparable])) if comparable else None,
                            "comparable_pairs": len(comparable), "total_pairs": 15, "spearman": corr(x, y, spearmanr)}
        for label_key in ("reward", "direct_relative_reward", "goal", "direct_relative_goal", "critic_drawdown"):
            vals = [float(r["label"][label_key]) for r in candidates]
            datum.setdefault("outcome_relationship", {})[label_key] = {
                "candidate_values": vals, "spearman_actor24": corr(x, vals, spearmanr), "spearman_proxy2": corr(y, vals, spearmanr),
                "spearman_actor24_abs": corr(x, np.abs(vals), spearmanr), "spearman_proxy2_abs": corr(y, np.abs(vals), spearmanr)}
        datum["candidates"] = [{"candidate_id": r["label"]["candidate_id"], "process_id": r["identity"]["process_id"],
                                 "actor24_distance": x[i], "proxy2_distance": y[i],
                                 "command_distance": datum["command"]["candidate_to_direct_rms"][i],
                                 "action_distance": datum["action"]["candidate_to_direct_rms"][i],
                                 "xy_distance": datum["xy"]["candidate_to_direct_rms"][i]}
                                for i, r in enumerate(candidates)]
        all_candidate.extend(datum["candidates"])
        group_results.append(datum)
    result = {"source_provenance_sha256": sha(out / "source-provenance.json"), "analysis_config_sha256": sha(out / "analysis-config.yaml"),
              "run_count": len(byrun), "step_count": len(records), "actor24_reconstruction_max_abs_error": maximum_reconstruction_error,
              "missing_or_nonfinite": 0, "unit": "scaled rad/s (rad/s x 0.05)", "side_order": ["right", "left"],
              "timeseries_external_path": str(timeseries_path), "timeseries_sha256": sha(timeseries_path), "metrics": aggregate}
    information = {"group_count": len(group_results), "groups": group_results,
                   "candidate_distance_spearman_across_42_descriptive_only": corr([r["actor24_distance"] for r in all_candidate],
                                                                                    [r["proxy2_distance"] for r in all_candidate], spearmanr),
                   "state_group_is_independent_unit": True, "candidate_pairs_are_not_independent": True}
    json_dump(out / "track-proxy-comparison.json", result)
    json_dump(out / "candidate-information-retention.json", information)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10")
    terrains = sorted({row["terrain"] for row in records})
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, side in enumerate(("right", "left")):
        ax = axes[i]
        for t, terrain in enumerate(terrains):
            subset = [r for r in records if r["terrain"] == terrain]
            ax.scatter([r["actor24"][i] for r in subset], [r["proxy2"][i] for r in subset],
                       s=2, alpha=.2, color=colors(t), label=terrain, rasterized=True)
        ax.axline((0, 0), slope=1, color="black", linewidth=.8)
        ax.set(xlabel=f"Actor 24j {side} (rad/s × 0.05)", ylabel=f"2-sprocket {side} (rad/s × 0.05)", title=f"{side}: n=7,560")
    axes[1].legend(fontsize=7, markerscale=3, loc="best")
    save_svg(fig, out / "track-proxy-scatter.svg")
    plt.close(fig)

    example_id = sorted(k for k, r in byrun.items() if r["label"]["candidate_id"] != "direct_reference" and r["label"]["terrain"] == "boxes")[0]
    example = byrun[example_id]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for i, side in enumerate(("right", "left")):
        axes[i].plot(example["actor24"][:, i], label="Actor 24j", linewidth=1)
        axes[i].plot(example["proxy2"][:, i], label="2 sprocket", linewidth=1)
        axes[i].set(ylabel=f"{side} (rad/s × 0.05)")
        axes[i].legend()
    axes[1].set_xlabel("control step (0–119)")
    fig.suptitle(f"{example['label']['state_group_id']} / boxes / {example['label']['candidate_id']} / {example_id}")
    save_svg(fig, out / "track-proxy-timeseries.svg")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    groups = [g["state_group_id"] for g in group_results]
    x = np.arange(len(groups))
    for i, side in enumerate(("right", "left")):
        axes[0].bar(x + (i-.5)*.35, [aggregate["state_group_id"][g][side]["mean_absolute_error"] for g in groups],
                    width=.35, label=side)
    axes[0].set_xticks(x, [f"{g}\n{group_results[i]['terrain']}" for i,g in enumerate(groups)], rotation=45, ha="right", fontsize=7)
    axes[0].set(ylabel="mean |2j − 24j| (rad/s × 0.05)", title="StateGroup and terrain")
    axes[0].legend()
    axes[1].bar(x-.2, [g["actor24"]["candidate_pairwise_mean_rms"] for g in group_results], width=.4, label="24j candidate pairs")
    axes[1].bar(x+.2, [g["proxy2"]["candidate_pairwise_mean_rms"] for g in group_results], width=.4, label="2j candidate pairs")
    axes[1].scatter(x, [g["actor24"]["duplicate_mean_rms"] for g in group_results], marker="x", color="black", label="direct repeat RMS (both = 0)")
    axes[1].set_xticks(x, [f"{g}\n{group_results[i]['terrain']}" for i,g in enumerate(groups)], rotation=45, ha="right", fontsize=7)
    axes[1].set(ylabel="time-aligned RMS (rad/s × 0.05)", title="Candidate vs duplicate variation")
    axes[1].legend(fontsize=7)
    save_svg(fig, out / "track-proxy-group-variation.svg")
    plt.close(fig)
    print(json.dumps({"runs": len(byrun), "steps": len(records), "reconstruction_max": maximum_reconstruction_error,
                      "overall": aggregate["overall"], "groups": [{"id": r["state_group_id"], "rank": r["ranking"],
                      "ratio24": r["actor24"]["candidate_over_duplicate"], "ratio2": r["proxy2"]["candidate_over_duplicate"]} for r in group_results]}, indent=2))


if __name__ == "__main__":
    main()
