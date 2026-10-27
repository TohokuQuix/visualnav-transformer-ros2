"""A stop must reach the command gate; switching memory must affect decisions."""
import importlib
import numpy as np
import pytest


def api():return importlib.import_module('eat_nav_risk_guard')


def fixture():
    return np.zeros((4,4)),np.ones((4,4)),[4.,0.]


def test_all_fall_risky_candidates_stop_and_release_needs_consecutive_checks():
    p,progress,goal=fixture();guard=api().RiskGuard()
    p[:,0]=.9
    decision=guard.decide(p,progress,goal,0,0.)
    assert decision['selected_index'] is None and decision['selected_candidate']=='STOP'
    assert decision['guard_reason']=='all_candidates_high_risk'
    p[:,0]=.6  # Lower than trip, still above release: must stay stopped.
    assert guard.decide(p,progress,goal,40,0.)['selected_index'] is None
    p[:,0]=.1
    assert guard.decide(p,progress,goal,80,0.)['selected_index'] is None
    assert guard.decide(p,progress,goal,120,0.)['selected_index']==0


def test_stall_risk_only_stops_when_progress_is_also_unhelpful():
    p,progress,goal=fixture();p[:,2]=.99;guard=api().RiskGuard()
    assert guard.decide(p,progress,goal,0,0.)['selected_index']==0
    progress[:,3]=0.
    assert guard.decide(p,progress,goal,40,0.)['selected_index'] is None


def test_single_safe_candidate_is_selected_instead_of_high_score_unsafe_path():
    p,progress,goal=fixture();p[:,0]=.9;p[3,0]=.1
    decision=api().RiskGuard().decide(p,progress,goal,0,0.)
    assert decision['selected_index']==3
    assert decision['guard_unsafe']==[True,True,True,False]


def test_tilt_stop_is_immediate_between_replans_and_requires_pose_recovery():
    p,progress,goal=fixture();guard=api().RiskGuard()
    guard.decide(p,progress,goal,0,0.)
    assert not guard.observe_tilt(46.,1)
    assert guard.stopped and guard.stop_reason=='measured_tilt'
    assert not guard.observe_tilt(38.,2)
    assert guard.decide(p,progress,goal,80,38.)['selected_index'] is None
    assert guard.decide(p,progress,goal,120,30.)['selected_index'] is None
    assert guard.decide(p,progress,goal,160,30.)['selected_index']==0


def test_direction_switch_waits_for_dwell_and_two_consecutive_advantages():
    p,progress,goal=fixture();guard=api().RiskGuard()
    assert guard.decide(p,progress,goal,0,0.)['selected_index']==0
    # Goal moves behind the robot; reverse is clearly preferable.
    assert guard.decide(p,progress,[-4.,0.],40,0.)['selected_index']==0
    assert guard.decide(p,progress,[-4.,0.],80,0.)['selected_index']==0
    assert guard.decide(p,progress,[-4.,0.],120,0.)['selected_index']==3


def test_safety_override_bypasses_dwell_when_current_candidate_becomes_unsafe():
    p,progress,goal=fixture();guard=api().RiskGuard()
    guard.decide(p,progress,goal,0,0.)
    p[:,0]=.9;p[3,0]=.1
    decision=guard.decide(p,progress,goal,40,0.)
    assert decision['selected_index']==3 and decision['guard_reason']=='unsafe_current_override'


def test_small_score_difference_does_not_change_candidate_after_dwell():
    p,progress,goal=fixture();guard=api().RiskGuard()
    guard.decide(p,progress,goal,0,0.)
    p[0,3]=.2  # 0.1 utility loss; other forward gains remain below 0.15 margin.
    assert guard.decide(p,progress,goal,80,0.)['selected_index']==0


def test_stop_state_is_replayable_and_invalid_prediction_is_rejected():
    p,progress,goal=fixture();guard=api().RiskGuard();guard.observe_tilt(50.,3)
    restored=api().RiskGuard.from_state(guard.snapshot())
    assert restored.snapshot()==guard.snapshot()
    assert restored.decide(p,progress,goal,40,0.)==guard.decide(p,progress,goal,40,0.)
    p[0,0]=float('nan')
    with pytest.raises(ValueError):restored.decide(p,progress,goal,80,0.)


def test_stop_reaches_zero_command_gate_and_never_advances_follower(tmp_path):
    from visualnav_transformer.eat_nav.p1_follower_core import PureFollower,compile_core
    from eat_nav_risk_navigation_v2 import candidate_paths
    follower=PureFollower(candidate_paths()[0].tolist(),library=compile_core(tmp_path))
    try:
        guard=api().RiskGuard();guard.observe_tilt(46.,0);before=follower.state()
        command,diagnostic=api().guarded_command(guard,follower,(0.,0.,0.),1_000_000_000)
        assert command==(0.,0.,0.) and diagnostic['guard_stop']
        assert follower.state()==before
        running=api().RiskGuard()
        command,diagnostic=api().guarded_command(running,follower,(0.,0.,0.),1_000_000_000)
        assert command[0]>0
    finally:follower.close()


def test_host_accepts_valid_all_stop_rollout():
    from eat_nav_risk_guard_run import validate_guard_result
    rows=[dict(guard_stopped=True,selected_candidate='STOP',follower_command=[0.,0.,0.]) for _ in range(2)]
    result=dict(selections=1,control_steps=2,guard_stop_steps=2,nonzero_command_steps=0)
    validate_guard_result(result,rows)


def test_host_rejects_nonzero_command_during_stop():
    from eat_nav_risk_guard_run import validate_guard_result
    rows=[dict(guard_stopped=True,selected_candidate='STOP',follower_command=[.1,0.,0.])]
    result=dict(selections=1,control_steps=1,guard_stop_steps=1,nonzero_command_steps=1)
    with pytest.raises(ValueError):validate_guard_result(result,rows)
