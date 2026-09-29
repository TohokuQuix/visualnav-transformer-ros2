import math
from scripts.eat_nav_seven_state_select import choose_states, rank_state


def row(terrain, episode, goal, view):
    return {'terrain_type': terrain, 'episode_key': [0, episode, 0],
            'goal': [goal, 0.0], 'view_index': view, 'split': 'train',
            'terrain_seed': episode, 'step': 50}


def test_rank_is_deterministic_and_selection_is_order_invariant():
    states=[row('a',1,.5,11), row('a',2,.7,12), row('b',3,.4,13)]
    first=choose_states(states, seed=20260929, minimum_goal_m=.3)
    reverse=choose_states(list(reversed(states)), seed=20260929, minimum_goal_m=.3)
    assert [(x['terrain_type'],x['view_index']) for x in first['selected']] == [
        (x['terrain_type'],x['view_index']) for x in reverse['selected']]
    assert first['selected'][0]['rank_sha256']==min(rank_state(x,20260929) for x in states[:2])


def test_goal_filter_and_episode_cap():
    states=[row('a',1,.2,1), row('a',2,.5,2), row('b',2,.6,3), row('b',3,.7,4)]
    result=choose_states(states, seed=1, minimum_goal_m=.3)
    assert all(x['goal_distance_m']>=.3 for x in result['selected'])
    assert len({tuple(x['episode_key']) for x in result['selected']})==len(result['selected'])
    assert len(result['selected'])==2


def test_missing_terrain_is_na_and_outcomes_are_not_read():
    states=[row('a',1,.1,1), row('b',2,.6,2)]
    states[1]['proposal_not_distinct']=True
    states[1]['termination']='failure'
    result=choose_states(states, seed=1, minimum_goal_m=.3)
    assert result['na_terrains']==['a']
    assert result['selected'][0]['terrain_type']=='b'
