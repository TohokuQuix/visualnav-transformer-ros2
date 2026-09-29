"""Offline runner retains invalid proposals and reports train-normalized OOD."""
import numpy as np
import torch
from scripts.eat_nav_candidate_coverage_offline import evaluate
from visualnav_transformer.eat_nav.candidate_coverage_offline import candidate_sets
from visualnav_transformer.eat_nav.config import EATConfig


class StatsModel:
    action_mean = torch.zeros(2)
    action_std = torch.ones(2)


def test_evaluate_keeps_all_candidates_even_when_one_is_nonfinite():
    paths = np.zeros((32, 8, 2), dtype=float)
    paths[1, 0, 0] = np.nan
    bank = np.zeros((5, 8, 2), dtype=float)
    records = candidate_sets(paths, list(range(1001, 1033)), bank)
    dem = np.ones((4, 2, 20, 20), dtype=float)
    train = torch.zeros((2, 16))
    result = evaluate(records, dem, EATConfig(),
                      {'max_step_m': 2., 'max_length_m': 8., 'max_curvature_rad': 100.},
                      train, StatsModel(), .5)
    assert len(result['candidates']) == 37
    assert result['sets']['nomad_k8']['nonfinite_count'] == 1
    assert result['sets']['nomad_k32']['candidate_ids'][1] == 'nomad-seed-1002'
    assert result['candidates'][1]['constraints']['reasons'] == ['nonfinite']
