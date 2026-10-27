"""Training isolation, selected-epoch checkpoint and constants."""
import copy
import importlib
import numpy as np
import pytest
import torch
from test_fall_stall_learning import learning_arrays


def api():return importlib.import_module('visualnav_transformer.eat_nav.fall_stall_training')


def fixture():
    arrays=learning_arrays()
    index=[{'sample_index':i,'seed':11+(i//4)*11} for i in range(12)]
    fold={'fold':0,'roles':{role:{'sample_indices':list(range(start,start+4))} for role,start in [('train',0),('validation',4),('exploratory_holdout',8)]}}
    return arrays,index,fold


def test_holdout_changes_do_not_affect_fit_and_selected_state(tmp_path):
    arrays,index,fold=fixture();config={'epochs':4,'learning_rate':.001,'weight_decay':0.,'threads':1}
    first=api().fit_model(arrays,index,fold,'state_candidate',123,config)
    changed=copy.deepcopy(arrays);changed['actor_state'][8:]+=10000;changed['event_targets'][8:]=0;changed['progress_targets'][8:]+=300.
    second=api().fit_model(changed,index,fold,'state_candidate',123,config)
    assert first['history']==second['history'] and first['normalization']==second['normalization']
    for k,v in first['model'].state_dict().items():torch.testing.assert_close(v,second['model'].state_dict()[k],rtol=0,atol=0)
    best=min(first['history'],key=lambda r:r['validation_bce'])
    assert first['best_epoch']==best['epoch']
    path=tmp_path/'checkpoint.pt';api().save_checkpoint(path,first,{'dataset_ledger_sha256':'frozen'})
    restored,payload=api().load_checkpoint(path)
    assert payload['best_epoch']==best['epoch'] and payload['support']['active']==[True,True,True,False]
    assert payload['optimizer_state_dict']['state']
    for k,v in restored.state_dict().items():torch.testing.assert_close(v,first['model'].state_dict()[k],rtol=0,atol=0)
    probabilities,progress=api().predict(first,arrays,[8,9,10,11])
    assert np.isnan(probabilities[:,:,3]).all() and np.isfinite(progress).all()
    payload['schema']='legacy_scalar';torch.save(payload,path)
    with pytest.raises(ValueError,match='schema'):api().load_checkpoint(path)


def test_constants_ignore_unknown_labels_and_unsupported_outputs():
    arrays,index,fold=fixture();arrays['event_mask'][0,0,3,0]=False
    prior=api().constant_fit(arrays,fold,'train_prior')
    p,progress=api().predict(prior,arrays,[8,9])
    assert p[0,0,0]==pytest.approx(4/15)
    assert np.isnan(p[:,:,3]).all()
    assert progress[0,0,0]==pytest.approx(arrays['progress_targets'][:4,:,0].mean())
    free=api().constant_fit(arrays,fold,'always_event_free')
    assert api().predict(free,arrays,[8])[0][0,0,0]==0


def test_no_validation_supported_heads_aborts():
    arrays,index,fold=fixture();arrays['event_targets'][4:8,:,3,:]=0
    with pytest.raises(ValueError,match='validation'):api().fit_model(arrays,index,fold,'state_candidate',123,{'epochs':2,'learning_rate':.001,'weight_decay':0.,'threads':1})


def test_run_configuration_rejects_unimplemented_settings():
    import json
    from pathlib import Path
    config=json.loads(Path('config/eat_nav/fall_stall_baseline_v1.json').read_text())
    api().validate_config(config)
    changed=copy.deepcopy(config);changed['pair_weight']=.5
    with pytest.raises(ValueError,match='config'):api().validate_config(changed)
