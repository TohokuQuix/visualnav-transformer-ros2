"""Sampling must preserve nested seeds and never condition on Quality."""

import numpy as np
import torch

from visualnav_transformer.eat_nav.candidate_coverage_sampling import sample_paths


class SeedModel:
    def __init__(self):
        self.calls = []

    def sample(self, dem, goal, quality, *, seed):
        self.calls.append((quality, seed))
        return torch.full((1, 8, 2), float(seed))


def test_sampling_uses_same_dem_goal_and_quality_none_for_all_seeds():
    model = SeedModel()
    dem = torch.ones((1, 4, 2, 3, 3))
    goal = torch.tensor([[.5, 0.]])
    paths = sample_paths(model, dem, goal, [1001, 1002, 1003])
    assert paths.shape == (3, 8, 2)
    assert model.calls == [(None, 1001), (None, 1002), (None, 1003)]
    assert np.all(paths[0] == 1001)
    assert np.all(paths[2] == 1003)


def test_prepare_state_candidates_keeps_k8_nested_inside_k32():
    from visualnav_transformer.eat_nav.candidate_coverage_sampling import prepare_state_candidates
    model = SeedModel()
    dem = torch.ones((1, 4, 2, 3, 3))
    goal = torch.tensor([[.5, 0.]])
    result = prepare_state_candidates(model, dem, goal, list(range(1001, 1033)),
                                      slight_scale=.03, strong_scale=.05)
    assert len(result["nomad_k8"]) == 8
    assert len(result["nomad_k32"]) == 32
    assert result["nomad_k8"] == result["nomad_k32"][:8]
    assert all(row["points"][-1] == [.5, 0.] for row in result["fixed_bank"])
    assert all(quality is None for quality, _ in model.calls)
