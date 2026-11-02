import copy
import math
import numpy as np
import pytest
from visualnav_transformer.eat_nav.moving_pose2s_targets import build_window


def root(x=0,tilt=0):
    return [x,0,0,math.cos(math.radians(tilt)/2),math.sin(math.radians(tilt)/2),0,0]+[0.]*6


def trace(n=80,offset=240):
    return [dict(control_step=offset+i,logical_sim_time_ns=10_000_000_000+i*50_000_000,
                 root_state=root(i*.01),post_root_state=root((i+1)*.01),
                 candidate_id=3,candidate_cycle=i//40,candidate_replan=i%40==0,
                 follower_command=[.2,0,0],body_angular_velocity_radps=[0,0,0]) for i in range(n)]


def snapshot(rows,start=0):
    return dict(root_state=rows[start]['root_state'],t0_ns=rows[start]['logical_sim_time_ns'],candidate_id=3,
                path_body_xy=np.c_[np.linspace(0,1,8),np.zeros(8)],direction=1)


def test_40_intervals_rebased_anchor_global_offset_and_t0_replan():
    rows=trace()
    y=build_window(rows,40,snapshot(rows,40))
    assert y['pose_mask'] and y['primary_eligible']
    np.testing.assert_allclose(y['pos3'],[.4,0,0])
    assert y['trajectory_mask'].sum()==40
    assert y['trajectory_time_offset_sec'][-1]==2.
    assert y['metadata']['start_control_step']==280
    assert y['event_mask4'].all()


def test_early_natural_terminal_uses_pre_reset_tip_and_stall_unknown():
    rows=trace(12)
    rows[-1]['post_root_state']=root(99)
    rows[-1]['terminal']=dict(termination_reason='tilt_over',pre_reset_state=dict(root_state=root(.12,90)))
    y=build_window(rows,0,snapshot(rows))
    assert not y['pose_mask']
    assert y['events4'][0]==1 and y['event_mask4'][0]
    assert not y['event_mask4'][2:].any()
    assert y['trajectory_position'][11,0]==pytest.approx(.12)
    assert not y['trajectory_mask'][12:].any()
    assert np.isnan(y['trajectory_position'][12:]).all()


def test_195_vs_200_terminal_endpoint():
    for count in (39,40):
        rows=trace(count)
        rows[-1]['terminal']=dict(termination_reason='tilt_over',pre_reset_state=dict(root_state=root(count*.01,90)))
        y=build_window(rows,0,snapshot(rows))
        assert bool(y['pose_mask'])==(count==40)
        assert y['events4'][0]==1


@pytest.mark.parametrize('key,value', [('candidate_replan',True),('candidate_id',4),('append_events',[{}]),
                                         ('external_intervention',True),('guard_stopped',True),('boundary_crossed',True)])
def test_internal_interventions_censor_primary(key,value):
    rows=trace(40);rows[10][key]=value
    y=build_window(rows,0,snapshot(rows))
    assert not y['primary_eligible'] and not y['pose_mask']
    assert not y['event_mask4'].any()
    assert y['trajectory_mask'].sum()<=10


def test_near_fall_five_posts_and_no_pre_window_run():
    rows=trace(40)
    for row in rows[:4]: row['post_root_state']=root(0,65)
    # Keep physical pose continuity for the synthetic tilted samples.
    for i in range(1,len(rows)): rows[i]['root_state']=rows[i-1]['post_root_state']
    assert build_window(rows,0,snapshot(rows))['events4'][1]==0
    rows[4]['post_root_state']=root(0,65);rows[5]['root_state']=rows[4]['post_root_state']
    assert build_window(rows,0,snapshot(rows))['events4'][1]==1


def test_stall_and_original_reverse_path_not_canonical_candidate_slot():
    rows=trace(40)
    for row in rows:
        row.update(root_state=root(),post_root_state=root(),follower_command=[-.1,0,.1])
    snap=snapshot(rows);snap.update(direction=-1,path_body_xy=np.c_[np.linspace(0,-1,8),np.zeros(8)])
    y=build_window(rows,0,snap)
    np.testing.assert_array_equal(y['events4'],[0,0,1,1])


@pytest.mark.parametrize('alter',[lambda r:r[3].update(logical_sim_time_ns=0),
                                  lambda r:r[3].update(root_state=root(99)),
                                  lambda r:r[3].update(root_state=[float('nan')]*13)])
def test_invalid_trace_returns_masked_partial(alter):
    rows=trace(40);snap=snapshot(rows);alter(rows)
    y=build_window(rows,0,snap)
    assert not y['pose_mask'] and not y['primary_eligible']
    assert y['trajectory_mask'].sum()<40


def test_snapshot_mismatch_rejected_without_fabricating_targets():
    rows=trace(40);snap=snapshot(rows);snap['t0_ns']+=1
    y=build_window(rows,0,snap)
    assert not y['pose_mask'] and not y['event_mask4'].any()
    assert 'snapshot_t0_mismatch' in y['reasons']['window']


@pytest.mark.parametrize('key,value',[('boundary_censor',True),('path_id','3:1'),('episode_id',99)])
def test_collector_boundary_append_path_and_episode_change_censor(key,value):
    rows=trace(40)
    for row in rows:row['episode_id']=1
    rows[10][key]=value
    y=build_window(rows,0,snapshot(rows))
    assert not y['primary_eligible'] and not y['pose_mask']


def test_trace_cutoff_not_primary_and_terminal_cannot_connect_reset():
    rows=trace(3);y=build_window(rows,0,snapshot(rows))
    assert not y['primary_eligible']
    rows=trace(40);rows[10]['terminal']=dict(termination_reason='tilt_over',pre_reset_state=dict(root_state=root(.11,90)))
    y=build_window(rows,0,snapshot(rows))
    assert y['trajectory_mask'].sum()==11 and not y['pose_mask']
    assert y['events4'][0]==1


def test_natural_tip_reason_retained_if_pre_reset_pose_missing():
    rows=trace(12)
    rows[-1]['terminal']=dict(termination_reason='tilt_over',pre_reset_state={})
    y=build_window(rows,0,snapshot(rows))
    assert not y['pose_mask']
    assert y['event_mask4'][0] and y['events4'][0]==1
    assert not y['event_mask4'][2:].any()


def test_legacy_2s_event_equivalence_and_t0_run_reset():
    from visualnav_transformer.eat_nav.fall_stall_targets import build_targets
    rows=trace(80,offset=0)
    for i,row in enumerate(rows):
        value=65 if 35<=i<44 else 0
        row['post_root_state']=root(0,value)
        row['root_state']=root(0,65 if 35<i<=44 else 0)
    snap=snapshot(rows,40)
    y=build_window(rows,40,snap)
    assert y['pose_mask'] and y['events4'][1]==0
    local=copy.deepcopy(rows[40:80])
    for i,row in enumerate(local):
        row.update(control_step=i,candidate_cycle=0,candidate_replan=False)
    old=build_targets(local,snap['path_body_xy'].tolist(),decision_stamp_ns=snap['t0_ns'])
    np.testing.assert_array_equal(y['events4'],[old['horizons']['2s']['events'][n]['label'] for n in old['event_counts']])
