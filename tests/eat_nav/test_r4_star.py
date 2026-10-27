import numpy as np
import pytest

def star():
    import eat_nav_r4_star as s
    return s

def test_six_bank_keeps_first_four_and_appends_reverse_curves():
    from eat_nav_risk_navigation import candidate_paths
    from visualnav_transformer.eat_nav.bidirectional_bank import candidate
    s=star();paths=s.candidate_paths()
    assert paths.shape==(6,8,2)
    np.testing.assert_array_equal(paths[:4],candidate_paths())
    for k,a in [(4,-30),(5,30)]:np.testing.assert_array_equal(paths[k],candidate(-1,a).waypoints.astype(np.float32))
    assert s.DIRECTIONS.tolist()==[1,1,1,-1,-1,-1]

def test_common_four_selection_and_stop_equal_r4_when_extras_unsafe():
    from eat_nav_r4 import R4Guard
    s=star();old=R4Guard();new=s.R4StarGuard()
    rng=np.random.default_rng(42)
    for step in range(0,1200,40):
        p=rng.random((4,4));q=rng.random((4,4));goal=rng.uniform(-5,5,2)
        ps=np.vstack([p,[[0,0,1,0]]*2]);qs=np.vstack([q,np.zeros((2,4))])
        a=old.decide(p,q,goal,step,70);b=new.decide(ps,qs,goal,step,70)
        assert a['selected_index']==b['selected_index']
        assert a['guard_reason']==b['guard_reason']
        np.testing.assert_allclose(a['scores'],b['scores'][:4],rtol=0,atol=1e-12)

def test_new_reverse_curves_need_same_direction_confirmation():
    s=star();g=s.R4StarGuard();p=np.zeros((6,4));q=np.ones((6,4))
    assert g.decide(p,q,[10,0],0,70)['selected_index']==0
    for t in [40,80,120]:d=g.decide(p,q,[-1,1],t,70)
    assert d['selected_index'] in (4,5)
    assert d['guard_reason']=='direction_switch_confirmed'
    assert not d['guard_stopped']
    assert s.R4StarGuard.from_state(g.snapshot()).snapshot()==g.snapshot()

def test_linear_stop_still_requires_two_release_confirmations_at_high_tilt():
    s=star();g=s.R4StarGuard();p=np.ones((6,4));q=np.zeros((6,4))
    assert g.decide(p,q,[1,0],0,75)['guard_stopped']
    p[:,2]=0
    assert g.decide(p,q,[1,0],40,75)['guard_stopped']
    assert not g.decide(p,q,[1,0],80,75)['guard_stopped']

def test_reverse_curve_extension_uses_body_heading():
    s=star()
    for k in (4,5):
        path=s.candidate_paths()[k].astype(float);heading=s.ANGLES[k]
        extended,end=s.extend_path(path,heading,k)
        from eat_nav_all_fields import rotation
        expected=path[-1]+s.candidate_paths()[k].astype(float)@rotation(heading).T
        np.testing.assert_allclose(extended[8:],expected[1:])
        assert end==2*heading

def test_pairing_compares_shared_inputs_and_four_candidate_prefix():
    from eat_nav_risk_navigation import pack_inputs
    s=star();dem=np.zeros((4,2,20,20),np.float32);dem[:,1]=1
    a=pack_inputs(dem,np.zeros(22,np.float32),np.zeros(9,np.float32))
    b=s.pack_inputs(dem,np.zeros(22,np.float32),np.zeros(9,np.float32))
    assert s.compare_initial(b,a)==[]
    b['body_state'][0,0]=1
    assert 'body_state' in s.compare_initial(b,a)
    b['paths'][0,5,1,0]+=1
    assert 'appended_bank_geometry' in s.compare_initial(b,a)

def test_same_checkpoint_supports_six_and_preserves_first_four_predictions():
    from eat_nav_command_free_learning import load_checkpoint,predict
    from eat_nav_risk_navigation import pack_inputs
    from pathlib import Path
    import torch
    torch.set_num_threads(1);s=star()
    path=Path('docs/eat-nav/all-fields-learning-20261023/models/k_rail_field/seed123/checkpoint.pt')
    model,payload=load_checkpoint(path);fit=dict(payload,model=model)
    dem=np.zeros((4,2,20,20),np.float32);dem[:,1]=1
    args=(dem,np.zeros(22,np.float32),np.zeros(9,np.float32))
    p,q=predict(fit,pack_inputs(*args),[0]);ps,qs=predict(fit,s.pack_inputs(*args),[0])
    assert ps.shape==qs.shape==(1,6,4)
    assert np.isfinite(ps).all() and np.isfinite(qs).all()
    np.testing.assert_allclose(ps[:,:4],p,rtol=0,atol=1e-5)
    np.testing.assert_allclose(qs[:,:4],q,rtol=0,atol=1e-5)

def test_gui_adapter_changes_only_candidate_contract():
    from pathlib import Path
    old=Path('scripts/eat_nav_r4_sim.py').read_text()
    new=Path('scripts/eat_nav_r4_star_sim.py').read_text()
    expected=old.replace('R4 GUI evaluation:', 'R4* six-candidate GUI evaluation:').replace("choices=['R4'],default='R4'","choices=['R4*'],default='R4*'").replace('choices=range(4)','choices=range(6)')
    expected=expected.replace('from eat_nav_r4 import R4Guard,CONFIG','from eat_nav_r4_star import R4StarGuard as R4Guard,CONFIG')
    expected=expected.replace('from eat_nav_all_fields import *','from eat_nav_all_fields import *\n# Override four-candidate names AFTER the legacy wildcard import.\nfrom eat_nav_r4_star import candidate_paths,pack_inputs,CANDIDATE_IDS,ANGLES,extend_path')
    assert new==expected

def test_gate_waits_for_both_seals_and_fixed_jobs_keep_assignments():
    from eat_nav_r4_star_run import ready,fixed_jobs
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        a=Path(d)/'r4';b=Path(d)/'review';a.mkdir();b.mkdir()
        assert not ready(a,b)
        (a/'derived-seal.json').write_text('{}');assert not ready(a,b)
        (b/'derived-seal.json').write_text('{}');assert ready(a,b)
    from eat_nav_r4_run import fixed_jobs as old_jobs
    old=old_jobs();new=fixed_jobs()
    assert len(new)==len(old)==36
    for a,b in zip(old,new):
        assert b['condition']=='R4*'
        for k in ('field','seed','model_seed','model','model_sha256'):assert a[k]==b[k]

def test_report_has_candidate_usage_and_matched_r4_only(tmp_path):
    from eat_nav_r4_star_audit import report
    import json
    old=tmp_path/'old';old.mkdir()
    item=dict(condition='R4',field='k_rail_field',seed=1,model_seed=123)
    result=dict(control_steps=1200,route_progress_m=2.,stop_steps=0,normal_waypoints=1,max_tilt_deg=20.,stop_reason='horizon_complete')
    audit=dict(tip_over_observed=False,linear_positive_windows=2,linear_exposure=dict(evaluable_running_steps=1200,unknown_running_steps=0),candidate_steps={'reverse_-30':200})
    ref=dict(assignment=item,state='completed',result=result,audit=audit,comparison_eligible=True)
    (old/'suite-results.json').write_text(json.dumps([ref]))
    (tmp_path/'manifest.json').write_text(json.dumps({'runs':[{}],'fields':['k_rail_field']}))
    actual=dict(ref,assignment=dict(item,condition='R4*'))
    report(tmp_path,[actual],old)
    text=(tmp_path/'comparison-report.md').read_text()
    assert 'reverse_-30' in text and '10.00' in text
    assert [r['condition'] for r in json.loads((tmp_path/'aggregates.json').read_text())]==['R4','R4*']
