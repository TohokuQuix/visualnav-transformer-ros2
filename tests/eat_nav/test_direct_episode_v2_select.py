import pytest
from scripts.eat_nav_direct_episode_v2_select import select_cost_aware


def row(terrain, episode, local, distance, index):
    return {'split':'train','terrain_type':terrain,'episode_key':[0,index,episode],
            'view_index':index,'local_step':local,'step':local,'terrain_seed':index+10,
            'goal':[distance,0.0]}


def test_v2_excludes_long_prefix_and_short_goal_and_marks_na():
    rows=[row('boxes',0,79,.4,1),row('boxes',0,81,.8,2),
          row('rail',0,30,.29,3),row('slope',0,80,.3,4)]
    result=select_cost_aware(rows,terrains=['boxes','rail','slope'],seed=20260929,
                             min_goal=.30,max_local_step=80)
    assert {x['terrain_type'] for x in result['selected']}=={'boxes','slope'}
    assert result['na_terrains']=={'rail':'NA / no state within runtime budget'}
    assert all(x['local_step']<=80 and x['goal_distance_m']>=.30 for x in result['selected'])
    assert result['outcome_fields_used']==[]


def test_v2_rejects_nontrain_or_partial_input():
    item=row('boxes',0,10,.4,1);item['split']='validation'
    with pytest.raises(ValueError):select_cost_aware([item],terrains=['boxes'],seed=1,min_goal=.3,max_local_step=80)
