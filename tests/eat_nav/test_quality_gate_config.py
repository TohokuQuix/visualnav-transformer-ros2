from pathlib import Path

from visualnav_transformer.eat_nav.quality_gate_config import (
    GateSettings,
    decide_gate_a,
    decide_gate_b,
    probe_quality,
)


def test_probe_zero_keeps_quality_path_with_fixed_half():
    settings = GateSettings()
    assert probe_quality("probe_0", [0.1, 0.95], settings) == [0.5, 0.5]
    assert probe_quality("probe_q", [0.1, 0.95], settings) == [0.1, 0.95]


def test_gate_a_is_inconclusive_when_both_probes_underfit_mean():
    metrics = dict(improvement=0.04, ci_lower=0.02, shuffled=[0.01] * 10,
                   relative_rms_improvement=0.08, delta_r2=0.01,
                   terrain_robust=True, direction_agreement=True,
                   probe_0_mse=1.02, probe_q_mse=1.01, mean_mse=1.0,
                   validation_improving_at_cap=False)
    assert decide_gate_a(metrics, GateSettings()) == "inconclusive"
    metrics.update(probe_0_mse=0.8, probe_q_mse=0.7,
                   validation_improving_at_cap=True)
    assert decide_gate_a(metrics, GateSettings()) == "inconclusive"


def test_gate_a_requires_100_shuffles_when_other_conditions_pass():
    metrics = dict(improvement=0.04, ci_lower=0.02, shuffled=[0.01] * 10,
                   relative_rms_improvement=0.08, delta_r2=0.01,
                   terrain_robust=True, direction_agreement=True,
                   probe_0_mse=0.8, probe_q_mse=0.7, mean_mse=1.0,
                   validation_improving_at_cap=False)
    assert decide_gate_a(metrics, GateSettings()) == "needs_100_shuffles"
    metrics["shuffled"] = [0.01] * 100
    assert decide_gate_a(metrics, GateSettings()) == "pass"
    metrics["shuffled"][-10:] = [0.06] * 10
    assert decide_gate_a(metrics, GateSettings()) == "fail"


def test_gate_a_ten_shuffle_needs_strict_win_even_when_not_escalated():
    metrics = dict(improvement=0.01, ci_lower=-0.01, shuffled=[0.01] * 10,
                   relative_rms_improvement=0.01, delta_r2=0.0,
                   terrain_robust=True, direction_agreement=True,
                   probe_0_mse=0.8, probe_q_mse=0.79, mean_mse=1.0,
                   validation_improving_at_cap=False)
    assert decide_gate_a(metrics, GateSettings()) == "fail"


def test_gate_b_requires_all_effect_and_selection_conditions():
    good = dict(effect_fraction=0.6, correct_selection_fraction=0.95,
                direction_fraction=0.92, effect_sampling_ratio=1.1,
                seed_robust=True, training_converged=True)
    assert decide_gate_b(good, GateSettings()) == "pass"
    assert decide_gate_b(dict(good, effect_fraction=0.49), GateSettings()) == "fail"
    assert decide_gate_b(dict(good, training_converged=False), GateSettings()) == "inconclusive"


def test_frozen_configs_include_hard_constraints():
    from visualnav_transformer.eat_nav.quality_gate_config import load_gate_settings

    path = Path(__file__).parents[2] / "docs/quality-identifiability-gate/probe-config.yaml"
    settings = load_gate_settings(path)
    assert settings.fixed_probe_q == 0.5
    assert settings.initial_permutations == 10
    assert settings.final_permutations == 100
    assert settings.training_seeds == (11, 29, 47)
