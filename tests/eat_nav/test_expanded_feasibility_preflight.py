import hashlib
import json

import pytest

from eat_nav_expanded_feasibility_preflight import rank_candidate, select_states, forecast_runtime


def test_selection_is_deterministic_outcome_blind_and_independent():
    rows=[]
    for i,terrain in enumerate(('boxes','hf_pyramid_slope','k_rail_field','k_rail_step_field','pyramid_stairs')):
        for step in (20,25):
            rows.append({'episode_key':[0,i,0],'source_episode':0,'env_id':i,
                'global_step':step,'local_step':step,'prefix_steps':step+1,
                'terrain':terrain,'terrain_seed':i+100,'exclusion_reasons':[],
                'source_window_id':[[0,i,0],step]})
    rows.append({**rows[0],'critic_value':-999,'reward':123})
    selected=select_states(rows,salt='fixed',terrains=[x['terrain'] for x in rows[:10:2]],excluded_env_ids={22,78})
    assert len(selected)==5
    assert len({x['env_id'] for x in selected})==5
    assert all(x['source_episode']==0 and x['prefix_steps']<=80 for x in selected)
    first=rows[0]
    assert rank_candidate(first,'fixed')==hashlib.sha256(('fixed\n'+json.dumps(
        [first['episode_key'],20,100,'boxes'],separators=(',',':'))).encode()).hexdigest()


def test_insufficient_unique_support_stops_instead_of_relaxing():
    row={'episode_key':[0,2,0],'source_episode':0,'env_id':2,'global_step':20,
        'local_step':20,'prefix_steps':21,'terrain':'boxes','terrain_seed':11,
        'exclusion_reasons':[],'source_window_id':[[0,2,0],20]}
    with pytest.raises(ValueError,match='support'):
        select_states([row],salt='fixed',terrains=['boxes','pyramid_stairs'],excluded_env_ids=set())


def test_forecast_enforces_margin_and_caps():
    good=forecast_runtime(2772,18,50,margin=1.2,cap_sec=14400)
    assert good['point_sec']==7700
    assert good['with_margin_sec']==9240
    assert good['status']=='pass'
    assert forecast_runtime(10000,18,50,margin=1.2,cap_sec=14400)['status']=='stop'
