"""Regression tests for state-level aggregation and independent candidate sets."""
import pathlib,sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))
from eat_nav_path_tracking_stages import pair_transition_counts,_macro_loso,_set_indices
from visualnav_transformer.eat_nav.feasible_transfer import cluster_labels

def test_pair_transitions_are_within_one_state():
    labels={'P':[0,1,2],'C':[0,0,1],'R':[0,0,0]}
    result=pair_transition_counts(labels,[0,1,2])
    assert result=={'pair_count':3,'P_different_C_same':1,'C_different_R_same':2,'P_C_R_all_different':0}

def test_state_macro_does_not_weight_candidate_count():
    result=_macro_loso({'state_a':1.0,'state_b':0.0})
    assert result['state_support']==2
    assert result['macro_mean']==0.5
    assert result['leave_one_state_out']=={'state_a':0.0,'state_b':1.0}

def test_generator_subset_does_not_bridge_through_bank():
    samples=[[[0.0]],[[.08]],[[.04]]] # two NoMaD, one bank
    assert len(set(cluster_labels(samples[:2],threshold=.05)))==2
    assert len(set(cluster_labels(samples,threshold=.05)))==1
    rows=[{'kind':'nomad'},{'kind':'nomad'},{'kind':'bank'}]
    assert _set_indices(rows,'nomad')==[0,1]

def test_arrival_by_two_seconds_is_subset_of_six_seconds():
    import json
    path=pathlib.Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/candidate-coverage/path-tracking-audit/endpoint-reach-timing.json')
    rows=json.loads(path.read_text())['runs']
    assert len(rows)==112
    assert all(not row['reached_by_2s'] or row['reached_by_6s'] for row in rows)
    assert sum(row['reached_by_2s'] for row in rows if row['generator']=='nomad')==28
