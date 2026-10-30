import math
import numpy as np
import pytest
from visualnav_transformer.eat_nav.r6_targets import future_targets, TIMES_SEC


def rows(n=40, yaw=math.pi/2):
    q=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    result=[]
    for i in range(n):
        pre=[5,8+i*.05,1]+q+[0,1,0,0,0,0]
        post=[5,8+(i+1)*.05,1]+q+[0,1,0,0,0,0]
        result.append(dict(control_step=i,logical_sim_time_ns=1000000000+i*50000000,candidate_cycle=i//40,
                           root_state=pre,post_root_state=post,follower_command=[1,0,0],body_angular_velocity_radps=[0,0,0],terminal=None))
    return result


def test_pose_is_future_yaw_aligned_and_events_are_two_seconds():
    target=future_targets(rows())
    assert len(TIMES_SEC)==20
    np.testing.assert_allclose(target['trajectory'][:,0], TIMES_SEC, atol=1e-6)
    np.testing.assert_allclose(target['trajectory'][:,1:3],0,atol=1e-6)
    np.testing.assert_allclose(target['trajectory'][:,3:],np.tile([0,1,0,1,0,1],(20,1)),atol=1e-6)
    assert target['trajectory_mask'].all() and target['event_mask'].all()
    assert not target['event_targets'].any()
    assert target['contact_metadata']['available'] is False


def test_censor_masks_negatives_but_preserves_observed_tip():
    trace=rows(12)
    trace[8]['post_root_state'][3:7]=[0,1,0,0]
    trace[11]['boundary_censor']=True
    target=future_targets(trace)
    assert target['event_targets'][0]==1 and target['event_mask'][0]
    assert not target['event_mask'][2:].any()
    assert target['trajectory_mask'].sum()==5
    assert not target['trajectory'][~target['trajectory_mask']].any()


def test_terminal_snapshot_replaces_reset_and_stops_future():
    trace=rows()
    terminal=trace[9]['post_root_state'].copy()
    trace[9]['terminal']=dict(pre_reset_state=dict(root_state=terminal,termination_reasons=['tilt_over']))
    trace[9]['post_root_state']=[0]*13
    target=future_targets(trace)
    assert target['trajectory_mask'].sum()==5
    assert target['trajectory'][4,0]==pytest.approx(.5)
    assert target['event_targets'][0]==1


def test_time_gap_rejected():
    trace=rows();trace[4]['logical_sim_time_ns']+=1
    with pytest.raises(ValueError):future_targets(trace)


def test_parallel_reset_does_not_label_teleported_post_state():
    trace=rows(10)
    trace[-1]['terminal']=dict(termination_reason='parallel_env_censored_by_other_env')
    trace[-1]['post_root_state']=[0,0,0,1,0,0,0,0,0,0,0,0,0]
    target=future_targets(trace)
    assert target['trajectory_mask'].sum()==4


def test_events_after_two_seconds_do_not_contaminate_labels():
    trace=rows(60)
    trace[50]['post_root_state'][3:7]=[0,1,0,0]
    target=future_targets(trace)
    assert target['event_mask'].all()
    assert not target['event_targets'].any()


def test_contact_point_clock_retains_sub_control_interval_onset():
    trace=rows()
    for row in trace:row['obstacle_contact']=False
    trace[2]['obstacle_contact']=True
    trace[2]['obstacle_contact_time_ns']=1_125_000_000
    contact=future_targets(trace)['contact_metadata']
    assert contact['first_observed_contact_sec']==pytest.approx(.125)
    assert contact['fully_observed']


def test_new_collector_rows_derive_only_missing_cycle_metadata():
    trace=rows()
    for row in trace:row.pop('candidate_cycle')
    target=future_targets(trace)
    assert target['trajectory_mask'].all() and target['event_mask'].all()
    assert all('candidate_cycle' not in row for row in trace)
