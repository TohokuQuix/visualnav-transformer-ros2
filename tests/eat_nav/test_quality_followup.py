import json
import numpy as np
import pytest

from visualnav_transformer.eat_nav.quality_followup import (
    is_complete_window, select_ladder_pairs, verify_stage_marker, make_stage_marker,
    stop_overfit, validate_complete_splits,
)


def window(seed, episode, mask):
    return {'terrain_seed':seed, 'episode_key':(0,seed,episode), 'action_mask':np.asarray(mask),
            'dem':np.zeros((4,2,20,20),dtype=np.float32), 'goal':np.zeros(2,dtype=np.float32),
            'action':np.zeros((8,2),dtype=np.float32), 'terrain_type':'rough', 'step':10,
            'quality':0.5}


def test_complete_window_excludes_partial_and_nonfinite():
    w=window(1,1,[True]*8)
    assert is_complete_window(w,8)
    w['action_mask'][7]=False
    assert not is_complete_window(w,8)
    w['action_mask'][7]=True
    w['action'][3,0]=np.nan
    assert not is_complete_window(w,8)


def test_complete_splits_reject_leakage_and_wrong_heldout_count():
    splits={'train':[window(1,1,[True]*8)],'val':[window(2,1,[True]*8)],'test':[window(3,1,[True]*8)]}
    with pytest.raises(ValueError,match='held-out complete count'):
        validate_complete_splits(splits,8,6208)
    splits['test'].append(window(1,2,[True]*8))
    with pytest.raises(ValueError,match='leakage'):
        validate_complete_splits(splits,8,2)


def test_ladder_selection_nested_diverse_and_capped():
    rows=[]
    for i in range(64):
        rows.append({'low_index':2*i,'high_index':2*i+1,'trajectory_rms_m':0.5+i/100,
                     'terrain_type':f't{i%7}', 'low_episode':[0,i,0], 'high_episode':[0,100+i,0]})
    selected=select_ladder_pairs(rows,cap=3)
    assert [len(selected[k]) for k in ('B1','B8','B64')]==[1,8,64]
    assert selected['B1'][0] in selected['B8']
    assert all(row in selected['B64'] for row in selected['B8'])
    assert len({row['terrain_type'] for row in selected['B8']})==7
    assert selected==select_ladder_pairs(rows,cap=3)
    rows[3]['low_index']=rows[2]['low_index']
    with pytest.raises(ValueError,match='window reused'):
        select_ladder_pairs(rows,cap=3)


def test_marker_requires_input_config_and_output_sha(tmp_path):
    out=tmp_path/'x.json';out.write_text('{}')
    marker=make_stage_marker('audit',{'source':'abc'},'cfg',{'out':out},[11])
    assert verify_stage_marker(marker,{'source':'abc'},'cfg')
    assert not verify_stage_marker(marker,{'source':'def'},'cfg')
    assert not verify_stage_marker(marker,{'source':'abc'},'new')
    out.write_text('{"changed":true}')
    assert not verify_stage_marker(marker,{'source':'abc'},'cfg')


def test_overfit_stops_on_loss_or_plateau_only_after_fifty_epochs():
    assert stop_overfit([0.3]*49,0.05)==(False,None)
    assert stop_overfit([0.3]*49+[0.04],0.05)==(True,'loss_ceiling')
    assert stop_overfit([0.2]*50,0.05)==(True,'plateau')
    assert stop_overfit(np.linspace(0.3,0.1,50).tolist(),0.05)==(False,None)
