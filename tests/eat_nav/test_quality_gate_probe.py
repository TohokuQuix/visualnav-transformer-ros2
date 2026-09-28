import numpy as np
import torch

from visualnav_transformer.eat_nav.quality_gate_probe import DeterministicProbe, shuffle_q_within_terrain, masked_mse


def test_probe_zero_and_q_have_identical_network_but_different_q_inputs():
    torch.manual_seed(1)
    probe0 = DeterministicProbe()
    torch.manual_seed(1)
    probeq = DeterministicProbe()
    assert sum(p.numel() for p in probe0.parameters()) == sum(p.numel() for p in probeq.parameters())
    dem = torch.zeros(2,4,2,20,20)
    goal = torch.zeros(2,2)
    fixed = probe0(dem, goal, torch.full((2,1),0.5))
    assert torch.allclose(fixed, probeq(dem, goal, torch.full((2,1),0.5)))
    changed = probeq(dem, goal, torch.tensor([[0.1],[0.9]]))
    assert not torch.allclose(changed[0],changed[1])


def test_q_permutation_preserves_each_terrain_marginal_but_changes_sample_assignment():
    q=np.array([0.1,0.2,0.3,0.4,0.6,0.7,0.8,0.9])
    terrain=np.array(['a']*4+['b']*4)
    shuffled=shuffle_q_within_terrain(q,terrain,seed=17)
    assert np.allclose(sorted(shuffled[:4]),sorted(q[:4]))
    assert np.allclose(sorted(shuffled[4:]),sorted(q[4:]))
    assert not np.array_equal(shuffled,q)


def test_masked_mse_ignores_terminal_padding():
    prediction=torch.tensor([[[1.,0.],[99.,99.]]])
    target=torch.zeros_like(prediction)
    mask=torch.tensor([[True,False]])
    assert masked_mse(prediction,target,mask).item()==0.5


def test_validation_score_is_aggregate_masked_loss_not_terrain_average():
    from visualnav_transformer.eat_nav.quality_gate_probe import aggregate_masked_mse
    prediction=torch.zeros(3,8,2)
    target=torch.zeros_like(prediction)
    target[0,0,0]=2.0
    mask=torch.zeros(3,8,dtype=torch.bool)
    mask[:,0]=True
    assert abs(aggregate_masked_mse(prediction,target,mask)-4/6)<1e-7
