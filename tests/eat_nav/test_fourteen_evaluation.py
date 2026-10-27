import numpy as np
import pytest

@pytest.mark.parametrize('module,cls',[('eat_nav_r4_star14_eval','R4Star14Guard'),('eat_nav_r5_14','R5FourteenGuard')])
def test_last_candidate_can_be_selected_and_release_stop(module,cls):
    import importlib
    m=importlib.import_module(module);g=getattr(m,cls)();p=np.zeros((14,4));p[:,2]=1;q=np.zeros((14,4))
    assert g.decide(p,q,[1,0],0,75)['guard_stopped']
    p[13,2]=0
    assert g.decide(p,q,[1,0],40,75)['guard_stopped']
    d=g.decide(p,q,[1,0],80,75)
    assert not d['guard_stopped'] and d['selected_index']==13
    assert getattr(m,cls).from_state(g.snapshot()).snapshot()==g.snapshot()

@pytest.mark.parametrize('newmod,newcls,oldmod,oldcls',[
 ('eat_nav_r4_star14_eval','R4Star14Guard','eat_nav_r4_star','R4StarGuard'),
 ('eat_nav_r5_14','R5FourteenGuard','eat_nav_r5','R5Guard')])
def test_six_subset_decisions_unchanged_when_other_eight_unsafe(newmod,newcls,oldmod,oldcls):
    import importlib
    from eat_nav_r4_star14 import EVAL_SLOTS
    a=getattr(importlib.import_module(newmod),newcls)();b=getattr(importlib.import_module(oldmod),oldcls)()
    rng=np.random.default_rng(31)
    for step in range(0,1200,40):
        p=rng.random((6,4));q=rng.random((6,4));goal=rng.uniform(-5,5,2)
        full=np.zeros((14,4));full[:,2]=1;prog=np.zeros((14,4));full[list(EVAL_SLOTS)]=p;prog[list(EVAL_SLOTS)]=q
        x=a.decide(full,prog,goal,step,70);y=b.decide(p,q,goal,step,70)
        assert x['selected_index']==(None if y['selected_index'] is None else EVAL_SLOTS[y['selected_index']])
        assert x['guard_reason']==y['guard_reason']


def test_r4_star_ignores_turn_but_r5_uses_it_with_linear_still_enabled():
    from eat_nav_r4_star14_eval import R4Star14Guard
    from eat_nav_r5_14 import R5FourteenGuard
    p=np.zeros((14,4));p[:,3]=1;q=np.zeros((14,4))
    assert not R4Star14Guard().decide(p,q,[1,0],0,80)['guard_stopped']
    assert R5FourteenGuard().decide(p,q,[1,0],0,80)['guard_stopped']
    p[:,3]=0;p[:,2]=1
    assert R5FourteenGuard().decide(p,q,[1,0],0,80)['guard_stopped']


def test_all_reverse_indices_use_same_direction_confirmation():
    from eat_nav_r4_star14_eval import R4Star14Guard
    for target in range(7,14):
        g=R4Star14Guard();p=np.zeros((14,4));q=np.zeros((14,4));q[0,3]=1
        assert g.decide(p,q,[10,0],0,0)['selected_index']==0
        q[:,3]=0;q[target,3]=10
        assert g.decide(p,q,[-10,0],40,0)['guard_reason']=='minimum_dwell'
        assert g.decide(p,q,[-10,0],80,0)['guard_reason']=='direction_confirmation'
        assert g.decide(p,q,[-10,0],120,0)['selected_index']==target

def test_host_evaluations_use_fourteen_config_and_new_adapters():
    from eat_nav_r4_star14_run import CONFIG as r4
    from eat_nav_r5_run import CONFIG as r5
    from pathlib import Path
    assert r4.get('candidate_count')==r5.get('candidate_count')==14
    assert len(r4['candidate_ids'])==len(r5['candidate_ids'])==14
    a=Path('scripts/eat_nav_r4_star14_run.py').read_text();b=Path('scripts/eat_nav_r5_run.py').read_text()
    assert 'eat_nav_r4_star14_eval_sim.py' in a
    assert 'eat_nav_r5_14_sim.py' in b and 'eat_nav_r4_star14_eval' in b


def test_gui_adapters_change_only_candidate_axis_and_guard():
    from pathlib import Path
    for old,new,changes in [
      ('eat_nav_r4_star_sim.py','eat_nav_r4_star14_eval_sim.py',[
       ('R4* six-candidate','R4* fourteen-candidate'),('from eat_nav_r4_star import R4StarGuard as R4Guard,CONFIG','from eat_nav_r4_star14_eval import R4Star14Guard as R4Guard,CONFIG')]),
      ('eat_nav_r5_sim.py','eat_nav_r5_14_sim.py',[
       ('R5 six-candidate','R5 fourteen-candidate'),('from eat_nav_r5 import R5Guard as R4Guard,CONFIG','from eat_nav_r5_14 import R5FourteenGuard as R4Guard,CONFIG')])]:
        expected=Path('scripts',old).read_text().replace('choices=range(6)','choices=range(14)').replace('from eat_nav_r4_star import candidate_paths,pack_inputs,CANDIDATE_IDS,ANGLES,extend_path','from eat_nav_r4_star14 import candidate_paths,pack_inputs,CANDIDATE_IDS,ANGLES,extend_path')
        for a,b in changes:expected=expected.replace(a,b)
        assert Path('scripts',new).read_text()==expected
