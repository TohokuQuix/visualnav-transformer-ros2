import numpy as np


def test_train_state_selection_ignores_future_outcomes_and_caps_episodes():
    from visualnav_transformer.eat_nav.feasible_set import select_stratified_states
    rows=[]
    for terrain in ('a','b'):
        for stratum in ('open','intermediate','constrained'):
            for i in range(5):
                rows.append({'split':'train','terrain_type':terrain,'stratum':stratum,
                             'episode_key':[0,len(rows),0],'view_index':len(rows),'step':i,
                             'future_reward':float(i),'termination':'failure'})
    a=select_stratified_states(rows,{'open':2,'intermediate':1,'constrained':2},seed=7)
    b=select_stratified_states([{**r,'future_reward':-999,'termination':'success'} for r in rows],
                                {'open':2,'intermediate':1,'constrained':2},seed=7)
    assert [x['view_index'] for x in a]==[x['view_index'] for x in b]
    assert len(a)==10 and len({tuple(x['episode_key']) for x in a})==10


def test_maneuvers_share_origin_and_goal_and_are_symmetric():
    from visualnav_transformer.eat_nav.feasible_set import maneuver_bank
    nominal=np.stack([np.linspace(.1,.8,8),np.zeros(8)],axis=1)
    paths=maneuver_bank(nominal,np.array([.8,0.]),.06,.15)
    assert set(paths)=={'nominal','slight_left','slight_right','strong_left','strong_right'}
    assert all(p.shape==(8,2) and np.isfinite(p).all() and np.allclose(p[-1],[.8,0.]) for p in paths.values())
    assert np.allclose(paths['slight_left']-paths['nominal'],-(paths['slight_right']-paths['nominal']))
    assert np.allclose(paths['strong_left']-paths['nominal'],-(paths['strong_right']-paths['nominal']))
    assert paths['strong_left'][2,1]>paths['slight_left'][2,1]>0


def test_selection_does_not_fill_unsupported_stratum():
    from visualnav_transformer.eat_nav.feasible_set import select_stratified_states
    rows=[{'split':'train','terrain_type':'a','stratum':'open','episode_key':[0,i,0],'view_index':i,'step':i} for i in range(3)]
    selected=select_stratified_states(rows,{'open':2,'intermediate':1,'constrained':1},seed=1)
    assert len(selected)==2


def test_selection_rejects_heldout_even_with_eligible_stratum():
    from visualnav_transformer.eat_nav.feasible_set import select_stratified_states
    rows=[{'split':'heldout','terrain_type':'a','stratum':'open',
           'episode_key':[0,0,0],'view_index':0,'step':20}]
    import pytest
    with pytest.raises(ValueError, match='held-out'):
        select_stratified_states(rows,{'open':1},seed=123)


def test_dem_difficulty_uses_only_present_input():
    from visualnav_transformer.eat_nav.feasible_set import dem_difficulty
    dem=np.zeros((4,2,20,20),dtype=np.float32)
    dem[:,1]=1
    a=dem_difficulty(dem,np.array([.8,0.]))
    dem[:,0,8:12,8:12]=.3
    b=dem_difficulty(dem,np.array([.8,0.]))
    assert b>a
    dem[:,1,8:12,8:12]=0
    assert dem_difficulty(dem,np.array([.8,0.]))>a
