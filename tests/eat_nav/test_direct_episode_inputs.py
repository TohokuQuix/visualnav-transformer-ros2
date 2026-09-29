import numpy as np
import pytest
from scripts.eat_nav_direct_episode_inputs import episode_slice, validate_episode_boundary


def test_episode_slice_skips_prior_global_steps_but_keeps_anchor_action():
    actions=np.arange(21*2*6,dtype=np.float32).reshape(21,2,6)
    commands=np.arange(21*3,dtype=np.float32).reshape(21,3)
    result=episode_slice(actions,commands,anchor_step=20,local_step=4,env_id=1)
    assert result['episode_start_step']==16
    assert result['actions'].shape==(5,2,6)
    assert np.array_equal(result['actions'][0],actions[16])
    assert np.array_equal(result['actions'][-1],actions[20])
    assert np.array_equal(result['commands'],commands[16:21])


def test_episode_slice_rejects_bad_local_step_and_shape():
    with pytest.raises(ValueError):episode_slice(np.zeros((8,2,6)),np.zeros((8,3)),anchor_step=7,local_step=8,env_id=0)
    with pytest.raises(ValueError):episode_slice(np.zeros((8,2,5)),np.zeros((8,3)),anchor_step=7,local_step=3,env_id=0)


def test_episode_boundary_must_match_recorded_episode_id():
    assert validate_episode_boundary({'step':16,'episode_id':3,'env_id':1,'pre_reset':False},16,3,1)
    with pytest.raises(ValueError):validate_episode_boundary({'step':16,'episode_id':2,'env_id':1,'pre_reset':False},16,3,1)
    with pytest.raises(ValueError):validate_episode_boundary({'step':16,'episode_id':3,'env_id':1,'pre_reset':True},16,3,1)
