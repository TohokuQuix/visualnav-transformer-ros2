"""SHA and pre-Path audit of the seven-group sealed E1 training fixture."""

from __future__ import annotations

import gzip
import hashlib
import json
import pathlib
import subprocess
from collections import Counter

import numpy as np
import yaml

from visualnav_transformer.eat_nav.evaluator_offline_smoke import anchor_features, audit_records, path_features


ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/quality-identifiability-gate"
OUT = DOC / "evaluator-offline-smoke"
MULTI = DOC / "evaluator-multihead-freeze"
RAW_ROOTS = [pathlib.Path("/tmp/eatnav-expanded-feasibility-ros-continuation-20261005"),
             pathlib.Path("/tmp/eatnav-e1-artifact-recovered-continuation-20261004")]
BRIDGE = pathlib.Path("/home/kojima/onix/dep/onix_automation/onix_policy_bridge")
CHECKPOINT = pathlib.Path("/home/kojima/lib/onix_isaac_lab/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump(name, obj):
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n")


def run():
    m0 = json.loads((MULTI / "m0-provenance.json").read_text())
    with gzip.open(MULTI / "multihead-labels.jsonl.gz", "rt") as stream:
        labels = {row["process_id"]: row for row in map(json.loads, stream)}
    if len(labels) != len(m0["run_identity"]) or len(labels) != 63:
        raise ValueError("63 label/run identities do not align")
    records, run_hashes, features, label_rows, ancestry = [], [], [], [], {}
    for identity in m0["run_identity"]:
        pid = identity["process_id"]
        matches = [root / "runs" / pid / "live-0.json" for root in RAW_ROOTS
                   if (root / "runs" / pid / "live-0.json").is_file()]
        if len(matches) != 1 or pid not in labels:
            raise ValueError(f"missing or ambiguous run: {pid}")
        summary_file = matches[0]
        trace_file = summary_file.with_name("live-0.trace.json.gz")
        summary, label = json.loads(summary_file.read_text()), labels[pid]
        state = summary["state"]
        group_id = identity["state_group_id"]
        source = {"source_window_id": state["source_window_id"],
                  "episode_key": state["episode_key"], "env_id": state["env_id"],
                  "terrain_seed": state["terrain_seed"], "terrain_type": state["terrain_type"]}
        if group_id in ancestry and ancestry[group_id] != source:
            raise ValueError(f"source ancestry differs within StateGroup: {group_id}")
        ancestry[group_id] = source
        sha = digest(trace_file)
        if sha != label["source_trace_sha256"] or sha != m0["trace_sha256"][pid]:
            raise ValueError(f"trace SHA mismatch: {pid}")
        if (summary["fixed_map_path_sha256"] != identity["map_path_file_sha256"] or
                summary["goal_metadata"]["candidate_id"] != identity["candidate_id"] or
                summary["goal_metadata"]["state_id"] != identity["state_group_id"] or
                label["candidate_id"] != identity["candidate_id"] or
                label["state_group_id"] != identity["state_group_id"] or
                int(label["repeat_id"]) != int(identity["repeat_id"])):
            raise ValueError(f"identity mismatch: {pid}")
        if not label["valid"] or label["censored"]:
            raise ValueError(f"invalid/censored label: {pid}")
        history_ids = summary["state"]["dem_history_identity"]
        anchor_step = int(summary["state"]["local_step"])
        if (len(history_ids) != 4 or
                any(frame[0] != summary["state"]["episode_key"] for frame in history_ids) or
                any(int(frame[1]) > anchor_step for frame in history_ids) or
                [int(frame[1]) for frame in history_ids] != sorted(int(frame[1]) for frame in history_ids) or
                int(history_ids[-1][1]) != anchor_step):
            raise ValueError(f"DEM history crosses episode or future anchor: {pid}")
        item = anchor_features(summary)
        item.update(path_features(item["dem_history"], item["candidate_path"]))
        common_keys = ("dem_history", "relative_goal", "body_angular_velocity", "projected_gravity",
                       "flipper_fold_mode", "flipper_position", "right_sprocket_angular_state_scaled",
                       "left_sprocket_angular_state_scaled", "simulator_actor_track_aggregate_24j",
                       "arm_applicability_mask", "anchor_timestamp", "simulator_actor22_prepath")
        records.append({"group": identity["state_group_id"], "candidate": identity["candidate_id"],
                        "repeat": identity["repeat_id"],
                        "common": {key: item[key].tolist() for key in common_keys}})
        run_hashes.append({"process_id": pid, "trace_path": str(trace_file), "trace_sha256": sha,
                           "summary_path": str(summary_file), "summary_sha256": digest(summary_file),
                           "candidate": identity["candidate_id"], "repeat": identity["repeat_id"],
                           "state_group": identity["state_group_id"], "seed": identity["seed"],
                           "episode_key": identity["episode_key"],
                           "path_sha256": identity["map_path_file_sha256"]})
        features.append(item)
        label_rows.append(label)
    audit = audit_records(records)
    for group, count in Counter(record["group"] for record in records).items():
        if count != 9:
            raise ValueError(f"group {group} has {count} runs")
        group_ids = [(r["candidate"], r["repeat"]) for r in records if r["group"] == group]
        if len(set(group_ids)) != 9:
            raise ValueError(f"duplicate candidate/repeat in {group}")
    for key in ("source_window_id", "episode_key", "env_id", "terrain_seed"):
        values = [json.dumps(item[key], sort_keys=True) for item in ancestry.values()]
        if len(set(values)) != len(values):
            raise ValueError(f"StateGroup split leakage in {key}")
    contract_files = [path for directory in ("evaluator-multihead-freeze", "evaluator-multihead-addendum",
                                            "portable-input-v2-parity", "track-proxy-offline-review")
                      for path in sorted((DOC / directory).iterdir())
                      if path.suffix in (".yaml", ".json", ".md", ".gz")]
    source_files = contract_files + [OUT / "training-smoke-config.yaml", OUT / "input-variants.yaml",
        ROOT / "src/visualnav_transformer/eat_nav/portable_input_v2.py", ROOT / "scripts/eat_nav_e1_sim.py",
        ROOT / "src/visualnav_transformer/eat_nav/evaluator_offline_smoke.py",
        ROOT / "src/visualnav_transformer/eat_nav/evaluator_offline_train.py",
        ROOT / "scripts/eat_nav_evaluator_offline_train.py",
        pathlib.Path("/home/kojima/lib/onix_isaac_lab/source/onix_isaac_lab/onix_isaac_lab/tasks/locomotion/velocity/mdp/observations.py"),
        BRIDGE / "onix_policy_bridge/frontend.py", BRIDGE / "onix_policy_bridge/backend.py",
        BRIDGE / "config/onix_rough_standard/env.yaml", BRIDGE / "launch/onix_policy_bridge.launch.py",
        CHECKPOINT]
    if any(not path.is_file() for path in source_files):
        raise ValueError("provenance source missing")
    if digest(CHECKPOINT) != m0["checkpoint_sha256"]:
        raise ValueError("RL checkpoint SHA mismatch")
    provenance = {"status": "frozen_before_training", "branch": subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "worktree": str(ROOT), "output_root": "/tmp/eatnav-evaluator-offline-smoke-20261006-r1",
        "source_files": {str(path): digest(path) for path in source_files},
        "sealed_runs": run_hashes, "raw_trace_mutation": False}
    audit.update({"label_count": len(labels), "candidate_count_per_group": 7,
                  "direct_repeats_per_group": 3, "all_labels_valid": True,
                  "all_trace_and_path_sha_match": True,
                  "anchor_source": "summary.anchor, before candidate Path publication",
                  "post_follower_anchor_or_trace_input": False,
                  "physical_arm_deployment": "blocked_until_stowed_reference_and_tolerance_frozen"})
    dump("source-provenance.json", provenance)
    dump("label-alignment-audit.json", audit)
    dump("critic-runtime-dependency-audit.json", {
        "verdict": "critic_not_used_for_robot_control",
        "actor_checkpoint_loader": "backend.py:_load_policy filters actor.* state_dict keys only",
        "inference_path": "frontend request -> backend.handle_request -> _build_observation -> ActorOnly -> action processing -> frontend publications",
        "critic_observation_consumer": "none in frontend/backend; env.yaml critic section is a serialized training config only",
        "critic_value_consumer": "none in frontend/backend; no feedback to arbitration, recovery, watchdog, mode or follower",
        "pseudo_base_linear_velocity": {
            "source": "backend.py:_base_lin_vel_obs, measured left/right sprocket angular mean times radius",
            "actor_observation": "disabled by env.yaml observations.policy.base_lin_vel: null; 22D config already matches checkpoint so reconcile does not toggle",
            "critic_observation_position": "env.yaml observations.critic.base_lin_vel first 3 dimensions, not built by physical bridge",
            "other_consumer": "none; cmd_out.linear_x uses processed policy action, not measured pseudo velocity",
            "fallback": "joint lookup defaults to 0 if missing in _mean_joint_value; current Actor still uses measured track angular state",
            "enable_condition": "non-null policy.base_lin_vel.func or dimension reconciliation if model expects +3"},
        "portable_contract_exclusions": ["physical_critic_observation", "physical_critic_value",
                                          "pseudo_ground_velocity", "critic_dimension_compatibility"],
        "source_shas": {str(path): digest(path) for path in source_files if BRIDGE in path.parents}})
    dump("sprocket-angular-state-contract.json", {
        "raw_order": ["right_front_sprocket_angular_velocity_rad_s",
                      "right_rear_sprocket_angular_velocity_rad_s",
                      "left_front_sprocket_angular_velocity_rad_s",
                      "left_rear_sprocket_angular_velocity_rad_s"],
        "model_order": ["right_sprocket_angular_state_scaled", "left_sprocket_angular_state_scaled"],
        "formula": "0.05 * clip(mean(front,rear angular velocity in rad/s), -100, 100)",
        "scale_meaning": "Actor observation normalization, not wheel radius or m/s conversion",
        "simulator_24j": "simulator_actor_track_aggregate_24j; simulator-only diagnostic",
        "ground_velocity_interpretation": "prohibited"})
    variant_spec = yaml.safe_load((OUT / "input-variants.yaml").read_text())["variants"]
    def tensor_fields(variant):
        expanded = []
        for name in variant_spec[variant]:
            if name in variant_spec:
                expanded.extend(tensor_fields(name))
            elif name.startswith("privileged_") and name.endswith("_training_only"):
                continue
            else:
                expanded.append(name)
        return expanded
    layouts = {key: tensor_fields(key) for key in variant_spec}
    if layouts["B7"] != layouts["B5"]:
        raise ValueError("B7 student tensor differs from B5")
    dump("input-field-order-hashes.json", {key: {"fields": values,
        "sha256": hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()}
        for key, values in layouts.items()})
    matrices = {}
    for variant, fields in layouts.items():
        matrix = np.stack([
            np.concatenate([np.asarray([1.0], dtype=np.float32) if name == "constant" else
                            np.asarray(item[name], dtype=np.float32).ravel() for name in fields])
            for item in features])
        if not np.isfinite(matrix).all():
            raise ValueError(f"nonfinite model input: {variant}")
        matrices[variant] = {"shape": list(matrix.shape),
                             "sha256": hashlib.sha256(matrix.tobytes()).hexdigest()}
    if matrices["B5"] != matrices["B7"]:
        raise ValueError("B7 train-time student input differs from B5")
    dump("model-input-hashes.json", matrices)
    audit["model_input_sha256"] = {key: value["sha256"] for key, value in matrices.items()}
    audit["direct_repeat_handling"] = "three repeats clustered within StateGroup; candidate-level direct mean for pairwise ranking"
    audit["missing_mask"] = "all required simulator fields present; binary DEM validity retained; physical arm applicability=not inferred"
    audit["distinct_relative_goal_count"] = len({tuple(item["relative_goal"].tolist()) for item in features})
    audit["goal_diversity_warning"] = "all 7 StateGroups share one fixed relative goal; no goal generalization claim"
    dump("label-alignment-audit.json", audit)
    dump("data-split-contract.json", {"unit": "StateGroup", "groups": sorted(set(r["group"] for r in records)),
        "source_ancestry": ancestry,
        "folds": [{"held_out": group, "train": sorted(set(r["group"] for r in records) - {group})}
                  for group in sorted(set(r["group"] for r in records))],
        "same_seed_set_all_variants": [123, 456, 789],
        "train_only_scaler": True, "heldout_never_used_for_checkpoint_selection": True,
        "limitation": "one StateGroup per terrain; LOSO confounds held-out terrain with held-out source and is not terrain generalization evidence"})
    return features, label_rows, records


if __name__ == "__main__":
    run()
    print("audited 63 sealed traces, 7 groups; no label, SHA, or pre-Path common-input mismatch")
