import importlib
import numpy as np
import pytest


def module():return importlib.import_module('eat_nav_moving_pose2s_predict')


def data():
    return dict(input_schema=np.array('eatnav_moving_pose2s_input_v1'),dem=np.zeros((1,4,2,20,20),np.float32),actor_state=np.zeros((1,22),np.float32),body_state=np.array([[0,0,-1,0,0,0,0,0,0]],np.float32),paths=np.zeros((1,14,8,2),np.float32),directions=np.ones((1,14,1),np.float32),candidate_mask=np.ones((1,14),bool),dem_timestamps_ns=np.array([[250000000,500000000,750000000,1000000000]]),current_time_ns=np.array([1000000000]))


def test_snapshot_requires_causal_fresh_history(tmp_path):
    m=module();a=data();p=tmp_path/'x.npz';np.savez(p,**a);assert m.load_inputs(p)['paths'].shape==(1,14,8,2)
    a['dem_timestamps_ns'][0,-1]+=50000000;np.savez(p,**a)
    with pytest.raises(ValueError,match='future|history'):m.load_inputs(p)


def test_future_label_extra_key_rejected(tmp_path):
    m=module();a=data();a['position_targets']=np.zeros((1,14,3));p=tmp_path/'x.npz';np.savez(p,**a)
    with pytest.raises(ValueError,match='unexpected'):m.load_inputs(p)


def test_missing_timestamp_metadata_rejected(tmp_path):
    m=module();a=data();a.pop('current_time_ns');a.pop('dem_timestamps_ns');p=tmp_path/'x.npz';np.savez(p,**a)
    with pytest.raises(ValueError,match='missing|time'):m.load_inputs(p)


@pytest.mark.parametrize('age,valid',[(249999999,True),(250000000,False)])
def test_exact_freshness_boundary(tmp_path,age,valid):
    a=data();a['current_time_ns']+=age;p=tmp_path/'x.npz';np.savez(p,**a)
    if valid:module().load_inputs(p)
    else:
        with pytest.raises(ValueError,match='stale'):module().load_inputs(p)


@pytest.mark.parametrize('case',['actor_extra','batch','nan_mask','command_nan','invalid_dem_nan'])
def test_cli_enforces_same_raw_contract(tmp_path,case):
    a=data()
    if case=='actor_extra':a['actor_state']=np.zeros((1,23),np.float32)
    elif case=='batch':a['body_state']=np.zeros((2,9),np.float32)
    elif case=='nan_mask':a['candidate_mask']=np.full((1,14),np.nan)
    elif case=='command_nan':a['actor_state'][0,7]=np.nan
    elif case=='invalid_dem_nan':a['dem'][0,0,0,0,0]=np.nan
    p=tmp_path/'x.npz';np.savez(p,**a)
    with pytest.raises(ValueError):module().load_inputs(p)
