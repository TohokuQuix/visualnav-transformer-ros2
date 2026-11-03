import numpy as np
import pytest
from visualnav_transformer.eat_nav.single_pose2s_dataset import trial_samples, audit_split
from visualnav_transformer.eat_nav.single_pose2s_contract import make_manifest


def raw():
    root=np.zeros((201,13));root[:,3]=1;root[:,0]=np.arange(201)*.005
    state=np.zeros((201,27));state[:,2]=-1;state[:,25]=1
    return dict(stamps_ns=np.arange(201)*50000000,root=root,state=state,state_mask=np.ones((201,27)),state_age=np.zeros((201,27)),alive=np.ones(201,bool),dem=np.stack([np.stack([np.zeros((40,40)),np.ones((40,40)),np.ones((40,40))])]*41),dem_stamp_ns=np.arange(41)*250000000,dem_root=root[::5])


def fixture_odometry():
    stamps=np.arange(-20,201,2,dtype=np.int64)*50000000
    poses=np.zeros((len(stamps),7));poses[:,3]=1;poses[:,0]=stamps*1e-9*.1
    return dict(stamps_ns=stamps,received_ns=stamps,poses=poses)


def test_continuous_inputs_and_exact_endpoint_teachers():
    p=make_manifest()['parents'][0];rows=trial_samples(raw(),dict(parent=p,terrain=dict(geometry_sha256='abc')),odometry=fixture_odometry())
    assert len(rows)==5 and all(r['pose_mask'] for r in rows)
    np.testing.assert_allclose(rows[0]['position'],[.2,0,0])
    assert rows[1]['root0'][0]==pytest.approx(.2)
    assert rows[1]['input']['t0_ns']==2000000000
    a=raw();a['alive'][67:]=False
    rows=trial_samples(a,dict(parent=p,terrain=dict(geometry_sha256='abc')),odometry=fixture_odometry())
    assert [r['pose_mask'] for r in rows]==[True,False,False,False,False]


def test_future_input_and_cross_role_terrain_cannot_pass_audit():
    a=raw();a['dem_stamp_ns']=a['dem_stamp_ns']+1
    rows=trial_samples(a,dict(parent=make_manifest()['parents'][0],terrain=dict(geometry_sha256='abc')),odometry=fixture_odometry())
    assert not rows[0]['input_valid']
    with pytest.raises(ValueError,match='terrain'):
        audit_split([dict(parent_id='a',role='train',terrain_hash='x',seed=1),dict(parent_id='b',role='final_test',terrain_hash='x',seed=2)])


def test_known_terrain_requires_explicit_policy_and_reports_overlap():
    rows=[dict(parent_id='a',role='train',terrain_hash='x',seed=1),dict(parent_id='b',role='final_test',terrain_hash='x',seed=2)]
    r=audit_split(rows,allow_known_terrain=True)
    assert r['cross_role_terrain_overlap']==1 and r['terrain_generalization_claim']=='known_terrain_independent_runs'


def test_fastlio_sidecar_overrides_wheel_input_and_missing_is_not_filled():
    a=raw();a['state'][:,18]=99.
    stamps=np.arange(-20,201,2,dtype=np.int64)*50000000
    poses=np.zeros((len(stamps),7));poses[:,3]=1;poses[:,0]=stamps*1e-9*.2
    odometry=dict(stamps_ns=stamps,received_ns=stamps,poses=poses)
    m=dict(parent=make_manifest()['parents'][0],terrain=dict(geometry_sha256='abc'))
    rows=trial_samples(a,m,odometry=odometry)
    assert all(r['input_valid'] for r in rows)
    assert all(r['input']['state'][18]==pytest.approx(.2) for r in rows)
    assert all(r['odom_source']=='actual_FAST_LIO_pose_difference' for r in rows)
    rows=trial_samples(a,m,odometry=None)
    assert all(not r['input_valid'] for r in rows)
    assert all(r['reason']=='missing_actual_FAST_LIO' for r in rows)


def test_cross_role_raw_or_initial_state_duplicates_are_rejected():
    rows=[dict(parent_id='a',role='train',terrain_hash='x',seed=1,source_raw_sha256='same'),dict(parent_id='b',role='final_test',terrain_hash='y',seed=2,source_raw_sha256='same')]
    with pytest.raises(ValueError,match='source_raw'):audit_split(rows,allow_known_terrain=True)
    rows[0]['source_raw_sha256']='one';rows[1]['source_raw_sha256']='two'
    rows[0]['initial_state_sha256']=rows[1]['initial_state_sha256']='same-initial'
    with pytest.raises(ValueError,match='initial_state'):audit_split(rows,allow_known_terrain=True)
