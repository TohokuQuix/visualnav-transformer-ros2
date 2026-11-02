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
