"""Local invariants for the frozen-log forward evaluator."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

candidate = Path(__file__).resolve().parents[1] / 'scripts/eat_nav_forward_evaluator.py'
if not candidate.exists():
    candidate = Path(__file__).with_name('eat_nav_forward_evaluator.py')
spec = importlib.util.spec_from_file_location('forward', candidate)
forward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forward)


def test_trajectory_stats_use_increments():
    action = np.zeros((1, 8, 2), dtype=np.float32)
    action[0, :, 0] = .25
    stats = forward.trajectory_features(action)
    assert np.isclose(stats[0, 0], 2.)
    assert np.isclose(stats[0, 2], 2.)
    assert np.isclose(stats[0, 4], 2.)


def test_capacity_and_disabled_inputs():
    models = {name: forward.Evaluator(*flags) for name, flags in forward.VARIANTS.items()}
    assert len({sum(p.numel() for p in model.parameters()) for model in models.values()}) == 1
    dem = torch.zeros(2, 8, 20, 20); goal = torch.zeros(2, 2)
    action = torch.randn(2, 16); v0 = torch.randn(2, 1)
    model = models['G'].eval()
    with torch.no_grad():
        a = model(dem, goal, action, v0)
        b = model(dem, goal, action * 0, v0 * 0)
    assert torch.equal(a, b)


def test_episode_cluster_interval():
    values = np.asarray([1., 1., 3., 3.])
    episodes = np.asarray([[0, 0, 1], [0, 0, 1], [0, 0, 2], [0, 0, 2]])
    lo, hi = forward.episode_ci(values, episodes, repeats=100)
    assert 1 <= lo <= hi <= 3


def test_constant_predictor_rank_is_undefined():
    y = np.column_stack([np.arange(10, dtype=float)] * 4)
    p = np.zeros_like(y)
    scalers = {h: {'std': 1.} for h in forward.HEADS}
    m = forward.metrics(y, p, scalers)
    assert m['delta_v']['spearman'] is None
    assert m['macro_spearman'] is None


def test_pair_bootstrap_uses_both_episodes():
    left = np.asarray([[0, 0, i] for i in range(20)])
    right = np.asarray([[0, 1, i] for i in range(20)])
    ci = forward.pair_episode_ci(np.ones(20), left, right, repeats=100)
    assert ci == [1., 1.]
