"""Candidate-free DEM evidence must survive a frozen gate rejection."""
import hashlib
import json

import numpy as np
import pytest

from visualnav_transformer.eat_nav.p1_dem_capture import persist_then_gate


def _frame(env, step, height_delta=0.0, invalid=False):
    hits=np.zeros((400,3),dtype=np.float32)
    hits[:,0]=np.arange(400,dtype=np.float32)*.1+env*8
    hits[:,1]=2.0
    hits[:,2]=1.0
    if invalid:hits[0,2]=np.inf
    starts=np.zeros((400,3),dtype=np.float32)
    starts[:,0]=hits[:,0]
    starts[:,1]=hits[:,1]
    starts[:,2]=3.0
    transform=np.zeros((2,20,20),dtype=np.float32)
    transform[1]=1
    transform[0,0,0]=height_delta
    if invalid:transform[1,0,0]=0
    return {'identity':f'settling_step_{step}','timestamp_ns':step*50_000_000,
        'shape':[2,20,20],'frame':'robot_base','resolution_m':.1,
        'sensor_update_timestamp_s':step*.05,'sensor_timestamp_s':step*.05,
        'sensor_is_outdated':False,'sensor_update_period_s':.25,
        'env_origin_w':np.array([env*8,0,0],dtype=np.float32),
        'robot_pose_w':np.array([env*8,.5,.3,1,0,0,0],dtype=np.float32),
        'root_state_w':np.array([env*8,.5,.3,1,0,0,0,0,0,0,0,0,0],dtype=np.float32),
        'joint_pos':np.zeros(6,dtype=np.float32),
        'joint_vel':np.zeros(6,dtype=np.float32),
        'ray_hits_w':hits,'ray_starts_w':starts,
        'source_height':hits[:,2].reshape(20,20),
        'source_valid':np.isfinite(hits[:,2]).reshape(20,20),
        'transformed':transform}


def test_raw_frames_and_difference_maps_are_durable_before_gate_rejects(tmp_path):
    frames=[[_frame(env,step,.01 if env==1 and step==4 else 0.0,
                    invalid=env==1 and step==4) for step in (4,9,14,19)]
            for env in range(4)]
    def frozen_gate(_):
        assert (tmp_path/'prepath-dem-sha-ledger.json').is_file()
        assert (tmp_path/'prepath-dem-comparison.json').is_file()
        raise ValueError('direct DEM numeric gate')
    with pytest.raises(ValueError,match='direct DEM numeric gate'):
        persist_then_gate(tmp_path,frames,frozen_gate)
    ledger=json.loads((tmp_path/'prepath-dem-sha-ledger.json').read_text())
    assert len([x for x in ledger if x.startswith('prepath-dem-env-')])==16
    for name,sha in ledger.items():
        assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==sha
    with np.load(tmp_path/'prepath-dem-env-1-frame-0.npz') as data:
        assert np.isinf(data['ray_hits_w'][0,2])
        assert data['env_origin_w'].tolist()==[8.,0.,0.]
        assert 'root_state_w' in data and 'joint_pos' in data and 'joint_vel' in data
    metrics=json.loads((tmp_path/'prepath-dem-comparison.json').read_text())['frames'][0]
    assert metrics['height_max_abs_m']==pytest.approx(.01)
    assert metrics['height_rms_m']==pytest.approx(.0005)
    assert metrics['mask_iou']==pytest.approx(399/400)
    assert metrics['mask_difference_cells']==1
    assert metrics['height_difference_cells']==1
    assert 'ray_hit_world_max_abs_m' in metrics
    assert 'root_state_tile_local_max_abs' in metrics
    with np.load(tmp_path/'prepath-dem-diff-frame-0.npz') as data:
        assert data['height_signed'][0,0]==pytest.approx(-.01)
        assert data['mask_difference'][0,0]
        assert np.max(np.abs(data['ray_origin_tile_local_delta']))<2e-6
        assert 'ray_hit_world_delta' in data
        assert 'ray_hit_tile_local_delta' in data


def test_gate_receives_same_transformed_frames_after_persistence(tmp_path):
    frames=[[_frame(env,step) for step in (4,9,14,19)] for env in range(4)]
    seen=[]
    def frozen_gate(argument):
        seen.append(argument)
        return [{'height_rms_m':0.0}]*4
    result=persist_then_gate(tmp_path,frames,frozen_gate)
    assert len(seen)==1 and len(seen[0])==4
    assert seen[0][0][0]['transformed'].shape==(2,20,20)
    assert result[0]['height_rms_m']==0.0
