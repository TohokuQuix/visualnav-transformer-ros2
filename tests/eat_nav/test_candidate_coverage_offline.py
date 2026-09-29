"""Offline proposal sets retain identity and expose meaningful path modes."""

import numpy as np
import pytest

from visualnav_transformer.eat_nav.candidate_coverage_offline import (
    candidate_sets, path_metrics, pairwise_path_rms,
)


def test_candidate_sets_keep_nested_seeds_and_fixed_bank_without_replacement():
    seeds = list(range(1001, 1033))
    generated = np.arange(32 * 8 * 2, dtype=float).reshape(32, 8, 2)
    bank = np.zeros((5, 8, 2), dtype=float)
    sets = candidate_sets(generated, seeds, bank)
    assert [r["seed"] for r in sets["nomad_k8"]] == seeds[:8]
    assert len(sets["nomad_k32"]) == 32
    assert len(sets["fixed_bank"]) == 5
    assert len(sets["hybrid_k8"]) == 13
    assert sets["hybrid_k8"][:8] == sets["nomad_k8"]
    assert sets["hybrid_k8"][8:] == sets["fixed_bank"]
    assert sets["nomad_k8"][0]["points"] == generated[0].tolist()


def test_path_metrics_detect_reverse_and_self_intersection():
    crossing = np.array([[1, 1], [0, 1], [1, 0], [0, 0], [-1, 0], [-2, 0], [-3, 0], [-4, 0]], dtype=float)
    metric = path_metrics(crossing)
    assert metric["self_intersection"] is True
    assert metric["reverse_m"] > 0
    assert metric["path_length_m"] > 4
    straight = np.stack((np.arange(1, 9), np.zeros(8)), axis=1)
    assert path_metrics(straight)["self_intersection"] is False
    assert path_metrics(straight)["reverse_m"] == 0


def test_nonfinite_candidate_is_retained_and_flagged():
    generated = np.zeros((32, 8, 2), dtype=float)
    generated[3, 0, 0] = np.nan
    sets = candidate_sets(generated, list(range(32)), np.zeros((5, 8, 2)))
    assert len(sets["nomad_k8"]) == 8
    assert sets["nomad_k8"][3]["finite"] is False
    with pytest.raises(ValueError, match="nonfinite"):
        path_metrics(generated[3])


def test_pairwise_rms_distinguishes_paths_in_meters():
    a = np.zeros((8, 2))
    b = np.full((8, 2), .2)
    assert pairwise_path_rms(a, b) == pytest.approx(.2)


def test_candidate_set_horizon_is_taken_from_checkpoint_output():
    generated = np.zeros((32, 12, 2), dtype=float)
    bank = np.zeros((5, 12, 2), dtype=float)
    sets = candidate_sets(generated, list(range(32)), bank)
    assert len(sets["nomad_k8"][0]["points"]) == 12


def test_distribution_reports_effective_rank_and_path_clusters():
    from visualnav_transformer.eat_nav.candidate_coverage_offline import path_distribution
    base = np.stack((np.arange(1, 9) * .1, np.zeros(8)), axis=1)
    paths = np.stack((base, base + [0, .2], base + [0, -.2]))
    result = path_distribution(paths, cluster_rms_m=.05)
    assert result["path_cluster_count"] == 3
    assert result["pairwise_rms_median_m"] > .1
    assert result["effective_rank"] == pytest.approx(1)
    assert result["endpoint_covariance_m2"][1][1] > 0


def test_fixed_bank_uses_nomad_seed_one_as_nominal_and_preserves_goal_endpoint():
    from visualnav_transformer.eat_nav.candidate_coverage_offline import build_candidate_records
    paths = np.zeros((32, 8, 2), dtype=float)
    paths[:, :, 0] = np.arange(1, 9) * .1
    sets = build_candidate_records(paths, list(range(1001, 1033)),
                                   goal=np.array([1., 0.]), slight_scale=.03,
                                   strong_scale=.05)
    assert [r["bank_name"] for r in sets["fixed_bank"]] == [
        "nominal", "slight_left", "slight_right", "strong_left", "strong_right"]
    assert all(r["points"][-1] == [1., 0.] for r in sets["fixed_bank"])
    assert sets["fixed_bank"][1]["points"][3][1] > 0
    assert sets["fixed_bank"][2]["points"][3][1] < 0
