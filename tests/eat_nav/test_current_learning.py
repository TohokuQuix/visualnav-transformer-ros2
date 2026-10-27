"""Current-pool fitting has real epoch progress and never selects reserved data."""
import importlib
from pathlib import Path

import numpy as np
import pytest


def api():
    return importlib.import_module('eat_nav_current_learning')


def test_epoch_progress_and_checkpoint_roundtrip(tmp_path):
    from visualnav_transformer.eat_nav.fall_stall_training import save_checkpoint,load_checkpoint,predict
    a,index,fold=api().load_current_dataset()
    config=api().default_config();config['epochs']=2
    events=[]
    fit=api().fit_current(a,index,fold,'dem_state_candidate',123,config,on_epoch=events.append)
    assert [e['epoch'] for e in events]==[1,2]
    assert all(np.isfinite(e['total']) and np.isfinite(e['validation_bce']) for e in events)
    assert fit['best_epoch']==min(events,key=lambda e:e['validation_bce'])['epoch']
    p=tmp_path/'checkpoint.pt';save_checkpoint(p,fit,{'test_fixture':True})
    model,payload=load_checkpoint(p)
    ids=fold['roles']['validation']['sample_indices']
    expected=predict(fit,a,ids);actual=predict(dict(payload,model=model),a,ids)
    for x,y in zip(expected,actual):np.testing.assert_array_equal(x,y)


def test_fixed_current_dataset_is_train_validation_only():
    a,index,fold=api().load_current_dataset()
    assert len(index)==128 and set(fold['roles'])=={'train','validation'}
    assert len(fold['roles']['train']['sample_indices'])==96
    assert len(fold['roles']['validation']['sample_indices'])==32
    assert not {index[i]['seed'] for i in fold['roles']['train']['sample_indices']}&{index[i]['seed'] for i in fold['roles']['validation']['sample_indices']}
    assert (a['event_targets'][~a['event_mask']]==-1).all()


def test_progress_is_atomic_and_identifies_offline_fit(tmp_path):
    p=tmp_path/'progress.json'
    api().write_progress(p,dict(state='running',epoch=1,epochs=300,run_index=1,total_runs=6,total=.1,validation_bce=.2))
    import json
    value=json.loads(p.read_text())
    assert value['visualization']=='offline_training_no_simulation_rollout'
    assert value['epoch']==1 and value['total']==.1
    assert not list(tmp_path.glob('*.tmp'))


def test_training_bundle_reloads_replays_and_seals(tmp_path,monkeypatch):
    from visualnav_transformer.eat_nav.fall_stall_audit import verify_derived,read
    module=api();config=module.default_config();config['epochs']=2
    monkeypatch.setattr(module,'default_config',lambda:config)
    output=tmp_path/'artifacts';progress=tmp_path/'live.json'
    summary=module.run(output,progress)
    assert summary['learned_runs']==6 and summary['constant_runs']==2
    assert summary['checkpoint_reload_predictions_exact']
    assert summary['saved_prediction_metrics_replayed']
    assert not summary['reserved_test_opened'] and not summary['simulation_rollout']
    assert len(list(output.glob('*/checkpoint.pt')))==6
    assert verify_derived(output)
    assert read(progress)['state']=='completed'
    fit=read(output/'dem_state_candidate-seed123/fit.json')
    assert fit['normalization']['train_indices']==list(range(96))
    with pytest.raises(ValueError,match='fresh training output'):
        module.run(output,progress)


def test_destinations_cannot_overwrite_evidence_or_bound_sources(tmp_path):
    module=api()
    for output,progress in ((tmp_path/'new',module.DATASET/'split.json'),
                            (module.DATASET/'new',tmp_path/'live.json'),
                            (tmp_path/'new',module.REPO/'scripts/eat_nav_current_learning.py')):
        with pytest.raises(ValueError,match='destination'):
            module.validate_destinations(output,progress)
