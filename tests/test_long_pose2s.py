import numpy as np
import pytest
from visualnav_transformer.eat_nav.single_pose2s_contract import make_manifest
from visualnav_transformer.eat_nav.single_pose2s_collection import window_endpoints, acquisition_steps


def test_long_manifest_preserves_prefix_and_covers_all_candidates():
    old=make_manifest();new=make_manifest(trial_seconds=30)
    assert len(new['parents'])==1330 and new['planned_windows']==19950
    for a,b in zip(old['parents'],new['parents']):
        assert b['candidates'][:5]==a['candidates']
        assert len(b['candidates'])==15 and len(set(b['candidates']))==14
        assert {k:v for k,v in b.items() if k!='candidates'}=={k:v for k,v in a.items() if k!='candidates'}


def test_long_endpoints_and_terminal_masks():
    t=np.arange(601,dtype=np.int64)*50000000;a=np.ones(601,bool)
    e=window_endpoints(t,a,trial_seconds=30)
    assert len(e)==15 and e[-1]==(560,600,True)
    a[401:]=False
    assert sum(x[2] for x in window_endpoints(t,a,trial_seconds=30))==10
    with pytest.raises(ValueError):window_endpoints(t+1,a,trial_seconds=30)
    with pytest.raises(ValueError):window_endpoints(t,a)
    assert acquisition_steps(4,trial_seconds=30)[-1]==599
    assert acquisition_steps(0,trial_seconds=30)[-1]==600


def test_horizontal_overprediction_is_signed_magnitude_in_world_xy():
    from visualnav_transformer.eat_nav.long_pose2s import horizontal_overprediction
    # Body x points up: one metre of body x is not horizontal movement.
    q=np.array([2**-.5,0,2**-.5,0])
    assert horizontal_overprediction([1,0,0],[0,0,0],q)==pytest.approx(0,abs=1e-12)
    assert horizontal_overprediction([.5,0,0],[.1,0,0],[1,0,0,0])==pytest.approx(.4)
    assert horizontal_overprediction([.1,0,0],[.6,0,0],[1,0,0,0])==pytest.approx(-.5)


def test_later_interval_uses_current_input_and_endpoint_not_initial_state():
    from visualnav_transformer.eat_nav.single_pose2s_dataset import trial_samples
    p=make_manifest(trial_seconds=30)['parents'][0]
    t=np.arange(601,dtype=np.int64)*50000000
    root=np.zeros((601,13));root[:,3]=1;root[:,0]=t*1e-9*.1
    state=np.zeros((601,27));state[:,2]=-1;state[:,25]=1
    raw=dict(stamps_ns=t,root=root,state=state,state_mask=np.ones_like(state),state_age=np.zeros_like(state),alive=np.ones(601,bool),dem=np.tile(np.array([0,1,1])[None,:,None,None],(121,1,40,40)),dem_stamp_ns=t[::5],dem_root=root[::5],command=np.zeros((600,3)))
    ot=np.arange(-20,601,2)*50000000;poses=np.zeros((len(ot),7));poses[:,3]=1;poses[:,0]=ot*1e-9*.1
    odom=dict(stamps_ns=ot,received_ns=ot,poses=poses)
    meta=dict(trial_seconds=30,parent=p,terrain=dict(geometry_sha256='x'))
    rows=trial_samples(raw,meta,odometry=odom)
    assert len(rows)==15 and all(x['pose_mask'] for x in rows)
    assert rows[-1]['t0_ns']==28000000000 and rows[-1]['t2_ns']==30000000000
    assert rows[-1]['root0'][0]==pytest.approx(2.8)
    np.testing.assert_allclose(rows[-1]['position'],[.2,0,0])
    assert rows[-1]['input']['dem_stamp_ns']==28000000000
    raw['alive'][561:]=False
    rows=trial_samples(raw,meta,odometry=odom)
    assert sum(x['pose_mask'] for x in rows)==14


def test_rate_counts_windows_not_candidates_or_parents_and_empty_is_na():
    from visualnav_transformer.eat_nav.long_pose2s import rate_summary
    rows=[dict(parent_id='a',stage=0,valid=True,excess_m=.4),dict(parent_id='a',stage=1,valid=True,excess_m=-.5),dict(parent_id='b',stage=5,valid=True,excess_m=.9),dict(parent_id='b',stage=6,valid=False,excess_m=None)]
    out=rate_summary(rows)
    assert out['valid_windows']==3 and out['positive_windows']==2
    assert out['rate']==pytest.approx(2/3) and out['independent_trials']==2
    assert rate_summary(rows[:2])['rate']==.5
    assert rate_summary([rows[-1]])['rate'] is None


def test_accuracy_retains_signed_lateral_bias():
    from visualnav_transformer.eat_nav.single_pose2s_metrics import summarize
    row=dict(parent_id='a',ground_truth_valid=True,prediction_valid=True,euler_singular=False,longitudinal_cm=-2.,lateral_cm=-3.,height_cm=0.,roll_deg=0.,pitch_deg=0.,yaw_deg=0.,rotation_deg=0.,distance_3d_cm=4.)
    assert summarize([row])['lateral_signed_bias_cm']==-3.


def test_missing_time_bin_renders_na_instead_of_crashing_report():
    from eat_nav_long_pose2s_compare import format_rate
    assert format_rate(None)=='NA'
    assert format_rate(.125)=='12.50%'
