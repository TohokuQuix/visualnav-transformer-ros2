import numpy as np
import pytest
from visualnav_transformer.eat_nav.r6_evaluation import summarize_trial,aggregate_trials,ranking_key


def trace(n=40,stop=False):
    rows=[]
    for i in range(n):
        a=[i*.001,0,0,1,0,0,0,0,0,0,0,0,0];b=a.copy();b[0]+=.001
        rows.append(dict(control_step=i,logical_sim_time_ns=1000000000+i*50000000,candidate_cycle=0,
            root_state=a,post_root_state=b,follower_command=[.3,0,0],body_angular_velocity_radps=[0,0,0],
            protective_stop=stop,goal_hold=False,terminal=None,candidate_path=[[0,0],[.75,0]]))
    return rows


def test_stall_disjoint_window_seconds_and_total_stop():
    result=summarize_trial(trace(stop=True),[0,0],[2,0],duration_sec=60)
    assert result['linear_windows']==1 and result['linear_stall_sec']==2
    assert result['protective_stop_sec']==2 and result['observed_sec']==2
    assert result['residual_unavailable_sec']==58 and result['complete'] is False
    assert result['route_progress_m']==pytest.approx(.04)


def test_partial_window_unknown_and_goalhold_separate():
    rows=trace(12)
    for row in rows:row['goal_hold']=True
    result=summarize_trial(rows,[0,0],[2,0],duration_sec=60)
    assert result['linear_windows']==0 and result['unknown_stall_windows']==1
    assert result['goal_hold_sec']==pytest.approx(.6) and result['protective_stop_sec']==0


def test_fall_censor_never_becomes_success_or_low_stop_advantage():
    rows=trace(2);rows[-1]['post_root_state'][3:7]=[0,1,0,0]
    result=summarize_trial(rows,[0,0],[2,0],duration_sec=60)
    assert result['tip_count']==1 and result['residual_unavailable_sec']==pytest.approx(59.9)
    assert ranking_key(result)<ranking_key(dict(result,tip_count=2))
    with pytest.raises(ValueError):ranking_key(dict(result,residual_policy='ignore'))


def test_aggregate_keeps_denominators_and_macro_cells():
    a=summarize_trial(trace(),[0,0],[2,0],duration_sec=2);a.update(field='boxes',level=0)
    b=dict(a,field='stairs',tip_count=1)
    agg=aggregate_trials([a,b])
    assert agg['total_trials']==2 and agg['tip_count']==1 and agg['tip_rate']==.5
    assert agg['macro_mean_route_progress_m']==pytest.approx(.04)


def test_nonfall_censor_is_unknown_not_a_zero_fall_success():
 r=summarize_trial(trace(2),[0,0],[2,0],duration_sec=60)
 assert r['tip_count']==0 and r['fall_outcome_unknown']
 assert r['ranking_fall_count_upper_bound']==1
 assert ranking_key(r)>ranking_key(dict(r,complete=True,fall_outcome_unknown=False,ranking_fall_count_upper_bound=0))
