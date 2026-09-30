import pytest
from visualnav_transformer.eat_nav.fresh_bridge_equivalence import compare


def _row():
    return {'command':[0.1,0,0],'follower_command':[0.1,0,0],
        'pre_noise':[1.0],'post_noise':[1.0],'actor_input':[1.0],
        'actor_latent':[2.0],'actor_action':[0.3],'processed_action':{'wheel':[0.3]},
        'critic_observation':[1.0],'critic_value':2.0,'reward':0.1,
        'root_state':[0.0,0.0,1.0],'post_root_state':[0.1,0.0,1.0],
        'joint_position':[0.2],'joint_velocity':[0.0],
        'contact_force':[[0,0,1]],'raycast_hits':[[1.0]],
        'dem':{'transformed_sha256':'abc','source_mask':[[1]]},'terminal':None,
        'logical_sim_time_ns':100,'clock_stamp_ns':100,'tf_stamp_ns':100,
        'follower_command_stamp_ns':100}


def test_identical_trace_is_equivalent():
    report=compare([_row()],[_row()],xy_rms_limit_m=.0105,numeric_limit=1e-5)
    assert report['pass']
    assert report['xy_rms_m']==0


def test_command_difference_fails_even_if_xy_matches():
    old=_row();new=_row();new['follower_command']=[0.2,0,0]
    report=compare([old],[new],xy_rms_limit_m=.0105,numeric_limit=1e-5)
    assert not report['pass']
    assert 'follower_command' in report['failures']


def test_dem_mask_and_stamp_mismatch_fail():
    old=_row();new=_row();new['dem']['source_mask']=[[0]];new['tf_stamp_ns']=101
    report=compare([old],[new],xy_rms_limit_m=.0105,numeric_limit=1e-5)
    assert not report['pass']
    assert 'dem' in report['failures']
    assert 'tf_stamp_ns' in report['failures']
