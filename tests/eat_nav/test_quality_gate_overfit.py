import numpy as np
import torch
import pytest

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.quality import ActionStatistics
from visualnav_transformer.eat_nav.quality_gate_overfit import prepare_overfit_pairs, make_scratch_policy


def _sample(i, episode, q, step):
    dem=np.zeros((4,2,20,20),dtype=np.float32)
    dem[:,1]=1
    dem[:,0]=i
    action=np.zeros((8,2),dtype=np.float32)
    action[:,0]=step
    return {'episode_key':(1,episode,0),'dem':dem,'goal':np.array([0.2,0.1],dtype=np.float32),
            'quality':q,'action':action,'action_mask':np.ones(8,dtype=bool)}


def test_gate_b_has_exact_same_dem_goal_but_different_q_and_targets():
    windows=[_sample(0,0,0.1,0.01),_sample(1,1,0.9,0.1)]
    batch=prepare_overfit_pairs(windows,[(0,1)],'cpu')
    assert torch.equal(batch['dem'][0],batch['dem'][1])
    assert torch.equal(batch['goal'][0],batch['goal'][1])
    assert batch['q'][:,0].tolist()==pytest.approx([0.1,0.9])
    assert not torch.equal(batch['action'][0],batch['action'][1])


def test_gate_b_scratch_policy_seed_changes_initial_weights_without_checkpoint():
    stats=ActionStatistics(np.zeros(2,dtype=np.float32),np.ones(2,dtype=np.float32),100)
    a=make_scratch_policy(EATConfig(),stats,seed=11,device='cpu')
    b=make_scratch_policy(EATConfig(),stats,seed=29,device='cpu')
    assert not torch.equal(next(a.parameters()),next(b.parameters()))
    assert a.config.quality_enabled


def test_pair_effect_metrics_detects_correct_quality_target_selection():
    from visualnav_transformer.eat_nav.quality_gate_overfit import pair_effect_metrics
    low=np.zeros((2,8,2),dtype=np.float32)
    high=np.ones((2,8,2),dtype=np.float32)
    low[:, :, 1]=0
    high[:, :, 1]=0
    result=pair_effect_metrics(low,high,low,high)
    assert result['correct_selection_fraction']==1.0
    assert result['effect_fraction']==1.0
    assert result['forward_direction_fraction']==1.0
