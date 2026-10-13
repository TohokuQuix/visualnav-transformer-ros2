import math
import hashlib
import json

import pytest

from visualnav_transformer.eat_nav.evaluator.labels import LABELS, labels_from_trace, pain_from_trace
from visualnav_transformer.eat_nav.evaluator.dataset import validate_group_split, verify_host_seal


def row(step):
    x = .01 * step
    return {'control_step':step, 'logical_sim_time_ns':2_000_000_000+step*50_000_000,
            'root_state':[x,0,0,1,0,0,0,0,0,0,0,0,0],
            'post_root_state':[x+.01,0,0,1,0,0,0,0,0,0,0,0,0],
            'follower_command':[.2,0,0], 'reward':.5,
            'applied_torque':[1.], 'contact_net_forces_w':[[1.,0,0]],
            'terminal':None, 'follower_diagnostic':{'goal_reached':step >= 25}}


def test_four_labels_and_missing_pain():
    result = labels_from_trace([row(i) for i in range(40)],
                               common_goal_xy=(1.,0.), env_origin_xy=(0.,0.))
    assert result['availability'] == [True, True, True, False]
    assert result['values'][LABELS.index('common_goal_progress')] == pytest.approx(.4)
    assert result['values'][LABELS.index('realized_reward')] == pytest.approx(20.)
    assert result['active_command_fraction'] == 1.
    assert result['arrival_step'] == 25


def test_early_censor_masks_all_fixed_window_labels():
    rows=[row(i) for i in range(12)]
    rows[-1]['terminal']={'terminated':True}
    result=labels_from_trace(rows,common_goal_xy=(1.,0.),env_origin_xy=(0.,0.))
    assert result['availability'] == [False]*4
    assert result['status'] == 'censored_before_2s'


def test_group_split_rejects_paired_leakage():
    samples=[{'sample_id':'a','group_id':'same'}, {'sample_id':'b','group_id':'same'}]
    with pytest.raises(ValueError,match='leakage'):
        validate_group_split(samples,{'train':['a'],'validation':['b'],'test':[]})
    validate_group_split(samples,{'train':['a','b'],'validation':[],'test':[]})


def test_synthetic_pain_formula_is_available_only_with_verified_provenance():
    settings={'schema':'eatnav_pain_allowables_v1','joint_names':['j'], 'body_names':['b'],
              'joint_torque_ref_Nm':{'j':2.},'body_force_ref_N':{'b':1.},
              'joint_weights':{'j':1.},'body_weights':{'b':1.},
              'combination_weights':{'joint':1.,'body':1.},'provenance':'synthetic_test'}
    p=pain_from_trace([row(0)],settings)
    assert p['joint_component_mean']==pytest.approx(.25)
    assert p['body_component_mean']==pytest.approx(1.)
    assert p['combined_mean']==pytest.approx(1.25)
    label=labels_from_trace([row(i) for i in range(40)],common_goal_xy=(1.,0.),
                            env_origin_xy=(0.,0.),pain_allowables=settings)
    assert not label['availability'][3]
    settings['provenance']='engineering_verified'
    label=labels_from_trace([row(i) for i in range(40)],common_goal_xy=(1.,0.),
                            env_origin_xy=(0.,0.),pain_allowables=settings)
    assert label['availability'][3]
    assert label['values'][3]==pytest.approx(-1.25)


def test_host_seal_read_only_verification_detects_mutation(tmp_path):
    evidence=tmp_path/'raw.txt';evidence.write_text('original')
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    ledger=tmp_path/'sha-ledger.json'
    ledger.write_text(json.dumps({'file_sha256':{'raw.txt':digest(evidence)}}))
    (tmp_path/'host-seal.json').write_text(json.dumps({'stage':'G3_rosfree_crossover_protocol_valid',
                                                        'sha_ledger_sha256':digest(ledger)}))
    assert verify_host_seal(tmp_path)[1]=='G3_rosfree_crossover_protocol_valid'
    evidence.write_text('changed')
    with pytest.raises(ValueError,match='seal'):
        verify_host_seal(tmp_path)
