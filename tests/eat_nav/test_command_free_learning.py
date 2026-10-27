"""Dropping unidentifiable command features must hold across fit and live inference."""
import importlib
import numpy as np
import pytest
import torch


def api():return importlib.import_module('eat_nav_command_free_learning')


@pytest.fixture
def short_fit(tmp_path):
    m=api();a,index,fold=m.load_current_dataset();config=m.default_config();config['epochs']=2
    fit=m.fit_current(a,index,fold,'dem_state_candidate',123,config)
    p=tmp_path/'checkpoint.pt';m.save_checkpoint(p,fit,{'fixture':True})
    return m,a,index,fold,fit,p


def test_command_values_never_enter_normalization_or_model_input():
    m=api();a,index,fold=m.load_current_dataset();ids=fold['roles']['train']['sample_indices']
    before=a['actor_state'].copy();stats=m.fit_normalization(a,ids)
    altered={k:v.copy() for k,v in a.items()};altered['actor_state'][:,6:9]=[250.,-100.,300.]
    assert m.fit_normalization(altered,ids)==stats
    first=m.prepare_inputs(a,[0,1],stats);second=m.prepare_inputs(altered,[0,1],stats)
    assert first['state'].shape==(2,28)
    for k in first:torch.testing.assert_close(first[k],second[k],rtol=0,atol=0)
    np.testing.assert_array_equal(a['actor_state'],before)


def test_retrained_checkpoint_roundtrip_and_command_invariance(short_fit):
    m,a,index,fold,fit,path=short_fit
    assert fit['model'].state_encoder[0].in_features==28
    model,payload=m.load_checkpoint(path)
    ids=fold['roles']['validation']['sample_indices']
    altered={k:v.copy() for k,v in a.items()};altered['actor_state'][:,6:9]=[.3,0.,-.4]
    for expected,actual in zip(m.predict(fit,a,ids),m.predict(dict(payload,model=model),altered,ids)):
        np.testing.assert_array_equal(expected,actual)
    assert fit['best_epoch']==min(fit['history'],key=lambda row:row['validation_bce'])['epoch']


def test_schema_prevents_legacy_checkpoint_mixing(short_fit):
    from eat_nav_risk_navigation import DEFAULT_CHECKPOINT
    from visualnav_transformer.eat_nav.fall_stall_training import load_checkpoint as load_v1
    m,a,index,fold,fit,path=short_fit
    with pytest.raises(ValueError,match='schema'):m.load_checkpoint(DEFAULT_CHECKPOINT)
    with pytest.raises(ValueError,match='schema'):load_v1(path)


def test_live_v2_uses_same_contract_and_preserves_simulator_settings(short_fit):
    from visualnav_transformer.eat_nav.fall_stall_audit import sha
    m,a,index,fold,fit,path=short_fit
    live=importlib.import_module('eat_nav_risk_navigation_v2')
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        evaluator=live.OnlineEvaluator(path,expected_sha256=sha(path))
        assert not torch.are_deterministic_algorithms_enabled()
        arrays=live.pack_inputs(a['dem'][0],a['actor_state'][0],a['body_state'][0])
        p1,progress1=evaluator.predict(arrays)
        arrays['actor_state'][:,6:9]=[.3,0.,.3]
        p2,progress2=evaluator.predict(arrays)
        np.testing.assert_array_equal(p1,p2);np.testing.assert_array_equal(progress1,progress2)
        np.testing.assert_allclose(arrays['actor_state'][:,6:9],[[.3,0.,.3]])
        with pytest.raises(ValueError,match='checkpoint SHA'):
            live.OnlineEvaluator(path,expected_sha256='0'*64)
    finally:torch.use_deterministic_algorithms(previous)


def test_comparison_conditions_come_from_sealed_30_second_reference():
    runner=importlib.import_module('eat_nav_risk_navigation_run_v2')
    reference=runner.reference_conditions()
    assert reference['seed']==2191202837
    assert reference['steps']==600 and reference['terrain']=='k_rail_step_field' and reference['level']==4
    assert reference['selection_weights']==[2.,.5,1.,.5]


def test_retraining_bundle_is_sealed_and_binds_preselected_gui_checkpoint(tmp_path,monkeypatch):
    from visualnav_transformer.eat_nav.fall_stall_audit import verify_derived,read,sha
    m=api();config=m.default_config();config['epochs']=2
    monkeypatch.setattr(m,'default_config',lambda:config)
    root=tmp_path/'artifacts';summary=m.run(root,tmp_path/'live.json')
    assert summary['learned_runs']==6 and summary['constant_runs']==2
    assert summary['input_contract']['state_dim']==28
    assert verify_derived(root)
    binding=read(root/'gui-checkpoint.json')
    assert binding['checkpoint_sha256']==sha(root/'dem_state_candidate-seed123/checkpoint.pt')
    for path in root.glob('*/fit.json'):
        fit=read(path)
        assert len(fit['normalization']['state_mean'])==28
        assert fit['normalization']['train_indices']==list(range(96))
    with pytest.raises(ValueError,match='sealed evidence'):
        m.validate_destinations(root/'nested',tmp_path/'live.json')
