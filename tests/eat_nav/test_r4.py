import numpy as np
from eat_nav_r4 import R4Guard

def prediction():return np.zeros((4,4)),np.ones((4,4))

def test_no_measured_tilt_stop():
    g=R4Guard();p,q=prediction()
    assert g.observe_tilt(80.,0)
    assert not g.decide(p,q,[10,0],0,80.)['guard_stopped']

def test_ignored_heads_do_not_affect_selection_or_stop():
    p,q=prediction();p[:,[0,1,3]]=1.
    a=R4Guard().decide(p,q,[10,0],0,10.)
    b=R4Guard().decide(np.zeros((4,4)),q,[10,0],0,10.)
    assert a['selected_index']==b['selected_index']==0
    assert not a['guard_stopped'] and not any(a['guard_unsafe'])

def test_linear_stall_requires_low_progress_and_masks_current():
    g=R4Guard();p,q=prediction();g.decide(p,q,[10,0],0,0)
    p[0,2]=.95;q[0,3]=.05
    d=g.decide(p,q,[10,0],40,60)
    assert d['selected_index']!=0 and d['guard_reason']=='unsafe_current_override'
    assert not d['guard_stopped']

def test_linear_high_risk_with_progress_does_not_exclude():
    p,q=prediction();p[:,2]=1.
    d=R4Guard().decide(p,q,[10,0],0,60)
    assert not d['guard_stopped']

def test_linear_stop_keeps_confirmations_but_not_35_degree_gate():
    g=R4Guard();p,q=prediction();p[:,2]=1.;q[:,3]=0.
    assert g.decide(p,q,[10,0],0,65)['guard_stopped']
    p[:,2]=0.
    assert g.decide(p,q,[10,0],40,65)['guard_stopped']
    assert not g.decide(p,q,[10,0],80,65)['guard_stopped']

def test_candidate_dwell_and_direction_confirmation_kept():
    g=R4Guard();p,q=prediction();g.decide(p,q,[10,0],0,0)
    assert g.decide(p,q,[-10,0],40,0)['guard_reason']=='minimum_dwell'
    assert g.decide(p,q,[-10,0],80,0)['guard_reason']=='direction_confirmation'
    assert g.decide(p,q,[-10,0],120,0)['guard_reason']=='direction_switch_confirmed'

def test_snapshot_round_trip():
    g=R4Guard();p,q=prediction();g.decide(p,q,[10,0],0,50)
    h=R4Guard.from_state(g.snapshot())
    assert h.snapshot()==g.snapshot()
    assert h.decide(p,q,[-10,0],40,70)==g.decide(p,q,[-10,0],40,70)

def test_audit_checks_ignored_raw_prediction_provenance():
    from eat_nav_r4_audit import prediction_replay_errors
    raw,q=prediction();d=R4Guard().decide(raw,q,[10,0],0,0)
    assert prediction_replay_errors(d,d,raw)==[]
    bad=dict(d,raw_probabilities=[[.9,0,0,0]]*4)
    assert prediction_replay_errors(bad,d,raw)==['raw_risk_inference_mismatch']

def test_report_includes_linear_exposure_and_failure_reason(tmp_path):
    import json
    from eat_nav_r4_audit import report
    old=tmp_path/'old';old.mkdir();(old/'suite-results.json').write_text('[]')
    (tmp_path/'manifest.json').write_text(json.dumps({'runs':[{},{}],'fields':['k_rail_field']}))
    assignment=dict(field='k_rail_field',seed=1,model_seed=123,condition='R4')
    result=dict(control_steps=1200,route_progress_m=2.,stop_steps=0,normal_waypoints=1,laps=0,max_tilt_deg=20.,stop_reason='horizon_complete')
    audit=dict(tip_over_observed=False,linear_positive_windows=2,
        linear_exposure=dict(evaluable_running_steps=1000,unknown_running_steps=200,stop_steps=0,total_steps=1200))
    good=dict(assignment=assignment,state='completed',result=result,audit=audit,comparison_eligible=True)
    bad=dict(assignment=assignment,state='failed',error=dict(message='simulator failed',simulator={'message':'termination during settling'}))
    report(tmp_path,[good,bad],old)
    a=json.loads((tmp_path/'aggregates.json').read_text())[0]
    assert a['linear_positive_windows']==2 and a['linear_evaluable_sec']==50.
    assert a['linear_positive_sec']==4. and a['linear_unknown_running_sec']==10.
    assert 'termination during settling' in (tmp_path/'comparison-report.md').read_text()
