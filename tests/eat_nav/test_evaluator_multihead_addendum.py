import json
from pathlib import Path

import yaml

from eat_nav_evaluator_multihead_addendum import pareto_front, validate_logging_schema


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/quality-identifiability-gate/evaluator-multihead-addendum"


def test_pareto_front_keeps_reward_goal_tradeoffs_and_drops_dominated():
    assert pareto_front([(1., 3.), (3., 1.), (2., 2.), (1., 1.)]) == [0, 1, 2]


def test_followup_contracts_keep_primary_and_auxiliary_distinct():
    gates = yaml.safe_load((OUT / "prospective-evaluation-gates.yaml").read_text())
    assert gates["phase_A_primary_teachers"] == [
        "discounted_reward_2s", "direct_relative_reward_2s",
        "goal_distance_reduction_2s", "reward_and_goal_within_StateGroup_pairwise_preference"]
    assert "endpoint_error_2s" in gates["auxiliary_task_heads"]
    assert gates["loss_precedence"] == "continuous_Huber_primary;_Bradley_Terry_auxiliary"
    assert gates["pairwise_margin_interpretation"] == "1e-6_numeric_floor_deterministic_simulator_ordering_only"
    assert gates["LOSO_role"] == "diagnostic_not_generalization_evidence"
    assert "online_selector_ready" not in gates["allowed_current_states"]
    baselines = json.loads((OUT / "baseline-and-ablation-contract.json").read_text())
    assert len(baselines["required_comparisons"]) == 11
    assert baselines["auxiliary_adoption_rule"] == "reject_if_primary_task_performance_degrades_on_independent_validation"


def test_future_logging_schema_has_all_fields_and_metadata():
    schema = json.loads((OUT / "future-collection-logging-schema.json").read_text())
    validate_logging_schema(schema)
    fields = {field["name"] for field in schema["fields"]}
    assert {"reward_total", "reward_component_pre_dt", "reward_component_post_dt",
            "path_horizon_end", "command_adopt_stamp_ns", "critic_observation",
            "full_joint_position", "previous_action", "actor_action", "measurement_valid",
            "failure_positive_eligible"} <= fields
    assert all(field["role"] == "future_logging_only" for field in schema["fields"])


def test_final_decision_requires_distinct_gate_checks():
    decision = json.loads((OUT / "final-decision.json").read_text())
    assert decision["decision"] == "multihead_teacher_contract_ready_for_input_integration"
    assert all(decision["checks"].values())
    assert "Evaluator_training" in decision["does_not_authorize"]
    audit = json.loads((OUT / "m0-m13-consistency-audit.json").read_text())
    assert audit["prior_artifacts_unchanged"] is True
    assert audit["prior_label_rows"] == 63 and audit["prior_pair_rows"] == 147
    assert audit["instruction_tail_resolved"] is True


def test_five_required_review_figures_are_valid_svg():
    names = ["conflict-pareto", "phase-heads", "evaluation-flow", "logging-timeline", "decision-flow"]
    for index, name in enumerate(names, 1):
        data = (OUT / f"fig-{index:02d}-{name}.svg").read_text()
        assert data.lstrip().startswith("<?xml") and "<svg" in data
