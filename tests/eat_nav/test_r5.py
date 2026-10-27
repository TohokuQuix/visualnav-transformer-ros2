import numpy as np

def test_linear_stall_stays_enabled():
    from eat_nav_r5 import R5Guard
    p=np.zeros((6,4));p[:,2]=1;q=np.zeros((6,4))
    assert R5Guard().decide(p,q,[1,0],0,70)['guard_stopped']

def test_turn_stall_added_to_r4_star_with_same_progress_gate():
    from eat_nav_r5 import R5Guard
    from eat_nav_r4_star import R4StarGuard
    p=np.zeros((6,4));p[:,3]=1;q=np.zeros((6,4))
    assert R5Guard().decide(p,q,[1,0],0,70)['guard_stopped']
    assert not R4StarGuard().decide(p,q,[1,0],0,70)['guard_stopped']
    q[:,3]=.051
    assert not R5Guard().decide(p,q,[1,0],0,70)['guard_stopped']

def test_tip_near_and_measured_tilt_stay_disabled():
    from eat_nav_r5 import R5Guard
    p=np.zeros((6,4));p[:,:2]=1;q=np.ones((6,4))
    d=R5Guard().decide(p,q,[1,0],0,80)
    assert not d['guard_stopped'] and not any(d['guard_unsafe'])
    assert d['risk_weights']==[0.,0.,1.,.5]
    np.testing.assert_array_equal(np.asarray(d['probabilities'])[:,:2],0)

def test_stop_release_at_high_tilt_keeps_two_confirmations():
    from eat_nav_r5 import R5Guard
    g=R5Guard();p=np.ones((6,4));q=np.zeros((6,4))
    assert g.decide(p,q,[1,0],0,75)['guard_stopped']
    p[:,2:]=0
    assert g.decide(p,q,[1,0],40,75)['guard_stopped']
    assert not g.decide(p,q,[1,0],80,75)['guard_stopped']
    assert R5Guard.from_state(g.snapshot()).snapshot()==g.snapshot()

def test_candidate_timing_matches_r4_star_when_turn_zero():
    from eat_nav_r5 import R5Guard
    from eat_nav_r4_star import R4StarGuard
    a=R5Guard();b=R4StarGuard();p=np.zeros((6,4));q=np.ones((6,4))
    for step,goal in [(0,[10,0]),(40,[-1,1]),(80,[-1,1]),(120,[-1,1])]:
        x=a.decide(p,q,goal,step,70);y=b.decide(p,q,goal,step,70)
        assert x['guard_reason']==y['guard_reason'] and x['selected_index']==y['selected_index']

def test_gui_only_adds_r5_condition_guard_and_required_turn_head():
    from pathlib import Path
    old=Path('scripts/eat_nav_r4_star_sim.py').read_text()
    expected=old.replace('R4* six-candidate GUI evaluation: linear-stall only','R5 six-candidate GUI evaluation: linear and turn stall')
    expected=expected.replace("choices=['R4*'],default='R4*'","choices=['R5'],default='R5'").replace('from eat_nav_r4_star import R4StarGuard as R4Guard,CONFIG','from eat_nav_r5 import R5Guard as R4Guard,CONFIG').replace('needed=[2]','needed=[2,3]')
    assert Path('scripts/eat_nav_r5_sim.py').read_text()==expected

def test_r5_requires_both_stall_heads():
    from eat_nav_r5_run import supported
    assert supported({'support':{'active':[False,False,True,True]}})
    assert not supported({'support':{'active':[True,True,True,False]}})
    assert not supported({'support':{'active':[True,True,False,True]}})

def test_pending_jobs_match_r4_star_and_wait_for_seal(tmp_path):
    from eat_nav_r5_run import template_jobs,ready
    jobs=template_jobs()
    assert len(jobs)==36 and {j['condition'] for j in jobs}=={'R5'}
    assert not ready(tmp_path)
    (tmp_path/'derived-seal.json').write_text('{}')
    assert ready(tmp_path)

def test_paired_report_includes_both_stalls_and_can_update(tmp_path):
    from eat_nav_r5_audit import report
    from visualnav_transformer.eat_nav.fall_stall_audit import write,read
    previous=tmp_path/'previous';previous.mkdir()
    assignment=dict(field='k_rail_field',seed=42,model_seed=123,condition='R4*')
    result=dict(control_steps=1200,route_progress_m=2.,stop_steps=40,normal_waypoints=1)
    exposure=dict(evaluable_running_steps=1000,unknown_running_steps=160)
    audit=dict(eligible=True,tip_over_observed=False,linear_positive_windows=2,turn_positive_windows=1,linear_exposure=exposure,turn_exposure=exposure)
    old=dict(assignment=assignment,result=result,audit=audit,state='completed')
    write(previous/'suite-results.json',[old])
    write(tmp_path/'manifest.json',{'fields':['k_rail_field'],'runs':[{}]})
    current=dict(old,assignment=dict(assignment,condition='R5'),comparison_eligible=True)
    report(tmp_path,[current],previous);report(tmp_path,[current],previous)
    rows=read(tmp_path/'aggregates.json')
    assert [r['condition'] for r in rows]==['R4*','R5']
    assert all(r['turn_positive_windows']==1 for r in rows)

def test_common_prediction_replay_accepts_same_ignored_nan_heads():
    from eat_nav_r5_audit import common_four_match
    p=np.zeros((6,4));p[:,:2]=np.nan;q=np.ones((6,4))
    assert common_four_match(p,q,p[:4].copy(),q[:4].copy())
    bad=p[:4].copy();bad[0,2]=.8
    assert not common_four_match(p,q,bad,q[:4])
