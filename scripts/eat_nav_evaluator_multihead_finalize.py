"""Independent M0-M13 consistency checks; read-only on sealed source artifacts."""
from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import yaml

from eat_nav_evaluator_multihead_addendum import validate_logging_schema

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-freeze"
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-addendum"
SOURCE_OLD = Path("/tmp/eatnav-e1-artifact-recovered-continuation-20261004")
SOURCE_NEW = Path("/tmp/eatnav-expanded-feasibility-ros-continuation-20261005")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main():
    old_ledger = read(OLD / "artifact-sha-ledger.json")
    mismatches = [name for name, info in old_ledger["files"].items()
                  if sha(ROOT / name) != info["sha256"] or (ROOT / name).stat().st_size != info["size_bytes"]]
    m0 = read(OLD / "m0-provenance.json")
    raw_mismatch = []
    for identity in m0["run_identity"]:
        root = SOURCE_OLD if identity["state_group_id"].startswith("g0e-") else SOURCE_NEW
        path = root / "runs" / identity["process_id"] / "live-0.trace.json.gz"
        if sha(path) != m0["trace_sha256"][identity["process_id"]]:
            raw_mismatch.append(identity["process_id"])
    with gzip.open(OLD / "multihead-labels.jsonl.gz", "rt") as source:
        labels = [json.loads(line) for line in source]
    pair_file = read(OLD / "pairwise-labels.json")
    pairs = pair_file["candidate_pairs"]
    old_output = yaml.safe_load((OLD / "model-output-contract.yaml").read_text())
    old_input = read(OLD / "portable-evaluator-input-contract.json")
    new_gates = yaml.safe_load((OUT / "prospective-evaluation-gates.yaml").read_text())
    new_baselines = read(OUT / "baseline-and-ablation-contract.json")
    multiobjective = read(OUT / "multiobjective-evaluation-contract.json")
    logging = read(OUT / "future-collection-logging-schema.json")
    parity = yaml.safe_load((OUT / "logging-parity-gate.yaml").read_text())
    validate_logging_schema(logging)
    groups = Counter(row["state_group_id"] for row in labels)
    pair_groups = Counter(row["state_group_id"] for row in pairs)
    required_parity = {"command_current", "actor_command_input", "actor_observation", "critic_observation",
                       "actor_action", "root_pose", "full_joint_position", "full_joint_velocity", "contact",
                       "dem_height_mask_history", "reward_total", "reward_component_pre_dt",
                       "reward_component_post_dt", "logical_sim_time_ns", "path_id_sha256",
                       "termination_reason", "reset", "realized_xy"}
    checks = {
        "source_integrity": not mismatches and not raw_mismatch and m0["status"] == "pass" and len(m0["run_identity"]) == 63,
        "teacher_label_alignment": len(labels) == 63 and len(groups) == 7 and set(groups.values()) == {9}
            and read(OLD / "label-generation-audit.json")["old_g40_reward_exact_match"],
        "pairwise_teacher_supported_for_deterministic_smoke": len(pairs) == 147 and len(pair_groups) == 7
            and set(pair_groups.values()) == {21} and pair_file["margins"] == {"reward": 1e-6, "goal": 1e-6, "endpoint": 1e-6}
            and sum(row["reward_goal_conflict"] for row in pairs) == 66
            and multiobjective["current_reward_goal_conflict_pairs"] == 66
            and not multiobjective["current_pairwise_margin"]["physical_indifference_claim"],
        "prospective_evaluation_contract_complete": len(new_baselines["required_comparisons"]) == 11
            and "StateGroup_bootstrap_CI" in new_gates["regression_metrics_per_head"]
            and "Pareto_optimal_candidate_recall" in new_gates["multiobjective_metrics"]
            and new_gates["LOSO_role"] == "diagnostic_not_generalization_evidence"
            and new_gates["loss_precedence"] == "continuous_Huber_primary;_Bradley_Terry_auxiliary"
            and "online_selector_ready" not in new_gates["allowed_current_states"],
        "logging_contract_complete": validate_logging_schema(logging)
            and required_parity <= set(parity["comparison_signals"])
            and parity["on_fail"] == "do_not_start_collection",
        "input_contract_complete": len(old_input["fields"]) >= 12
            and bool(old_input["primary_blockers_before_real_robot_inference"])
            and all(field["timestamp"].startswith("anchor") or "anchor" in field["timestamp"] or "before publication" in field["timestamp"] or "derived at anchor" in field["timestamp"] or "last completed action before anchor" in field["timestamp"]
                    for field in old_input["fields"]),
        "safety_head_disabled": old_output["safety_head"]["enabled_for_training"] is False
            and old_output["safety_head"]["inference_use"] is False
            and old_output["safety_head"]["loss_weight"] == 0.0,
        "endpoint_not_double_weighted": "endpoint_error_2s" in new_gates["auxiliary_task_heads"]
            and "endpoint_error_2s" not in new_gates["phase_A_primary_teachers"]
            and new_baselines["phase_A_endpoint_error_role"] == "auxiliary_calibration_not_equal_weight_primary",
        "split_and_no_leakage_contract": new_gates["split"]["unit"] == "StateGroup"
            and new_gates["split"]["scaler_fit"] == "training_StateGroups_only"
            and "candidate_random_split" in new_gates["split"]["prohibit"],
        "no_unapproved_operations": new_gates["online_selection_authorized"] is False
            and new_baselines["training_run_authorized"] is False
            and logging["status"] == "proposed_logging_only_no_implementation_or_rollout",
    }
    failed = [name for name, passed in checks.items() if not passed]
    decision = ("multihead_teacher_contract_ready_for_input_integration" if not failed else
                "source_integrity_fail" if not checks["source_integrity"] else
                "teacher_label_alignment_fail" if not checks["teacher_label_alignment"] else
                "pairwise_teacher_not_supported" if not checks["pairwise_teacher_supported_for_deterministic_smoke"] else
                "prospective_evaluation_contract_incomplete" if not checks["prospective_evaluation_contract_complete"] else
                "logging_contract_incomplete" if not checks["logging_contract_complete"] else
                "input_contract_incomplete")
    save("m0-m13-consistency-audit.json", {
        "status": "pass" if not failed else "fail", "checks": checks, "first_failure": failed[0] if failed else None,
        "prior_artifacts_unchanged": not mismatches, "prior_artifact_mismatches": mismatches,
        "raw_trace_mismatches": raw_mismatch, "prior_ledger_sha256": sha(OLD / "artifact-sha-ledger.json"),
        "prior_label_sha256": sha(OLD / "multihead-labels.jsonl.gz"),
        "prior_pair_sha256": sha(OLD / "pairwise-labels.json"),
        "prior_config_sha256": sha(OLD / "multihead-analysis-config.yaml"),
        "prior_label_rows": len(labels), "prior_pair_rows": len(pairs),
        "group_sizes": dict(sorted(groups.items())), "pair_group_sizes": dict(sorted(pair_groups.items())),
        "instruction_tail_resolved": True,
        "versioned_interpretation_overrides": {
            "endpoint_error": "M0-M10 equal-weight primary proposal superseded by auxiliary calibration/reporting",
            "pairwise_Bradley_Terry": "M0-M10 equal-weight smoke proposal superseded by auxiliary ranking loss subordinate to continuous Huber",
            "pair_margin": "unchanged 1e-6 numeric floor; no physical indifference claim",
            "labels_and_split": "unchanged"
        },
        "retained_old_raw_trace_and_contracts": True,
    })
    save("final-decision.json", {
        "decision": decision, "checks": checks, "failed_checks": failed,
        "statistical_unit": "StateGroup", "current_support": "seven_train_groups_sixty_three_traces_only",
        "primary_teacher": ["discounted_reward_2s", "direct_relative_reward_2s", "goal_distance_reduction_2s",
                            "reward_and_goal_within_StateGroup_pairwise_preference_auxiliary_loss"],
        "next_design_scope": ["ROS2_portable_input_integration", "field_order_unit_timestamp_parity",
                              "logging_implementation_proposal", "logging_parity_proposal",
                              "additional_StateGroup_collection_design", "small_training_plan"],
        "does_not_authorize": ["Evaluator_training", "hard_state_rollout", "scaled_collection",
                               "online_candidate_selection", "Safety_head", "formal_validation", "real_robot_control"],
        "selector_ready": False,
    })
    if failed:
        raise ValueError(f"M13 failed at {failed[0]}")
    print(json.dumps({"decision": decision, "checks": len(checks), "prior_files": len(old_ledger["files"])}))


if __name__ == "__main__":
    main()
