"""Diagnostics use the frozen train subset and saved scales, never fit/evaluate."""
import importlib
import json
import hashlib
import numpy as np
import pytest
import torch
from test_moving_pose2s_model import arrays, module


def fixture(tmp_path):
    d=tmp_path/'dataset';d.mkdir();a=arrays(n=4)
    a['actor_state'][0,0]=2;a['actor_state'][1,0]=6
    a['actor_state'][2:,0]=999999  # excluded train row and holdout must not contribute
    np.savez_compressed(d/'samples.npz',**a)
    (d/'sample-index.json').write_text(json.dumps([{'role':r} for r in ['train','train','train','audit_eval']]))
    (d/'split.json').write_text(json.dumps({'roles':{'train':[0,1,2],'validation':[],'audit_eval':[3]}}))
    (d/'manifest.json').write_text(json.dumps({'schema':'eatnav_moving_pose2s_dataset_v1','input_schema':module().INPUT_SCHEMA,'horizon_sec':2.,'arrays':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in a.items()}}))
    binding={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in d.iterdir()}
    s=module().fit_normalization(a,[0,1]);s['state_mean'][0]=1.;s['state_std'][0]=2.  # saved values, not refit
    cp=tmp_path/'checkpoint.pt';module().save_checkpoint(cp,module().MovingPose2sEvaluator(),s,module().training_support(a,[0,1]),{'dataset_binding':binding})
    return d,cp


def tool():return importlib.import_module('eat_nav_moving_pose2s_input_diagnostics')


def test_reports_only_frozen_ids_using_saved_statistics(tmp_path):
    d,cp=fixture(tmp_path);out=tmp_path/'report.json';before=cp.read_bytes()
    report=tool().diagnose(d,cp,out)
    assert cp.read_bytes()==before
    assert report==json.loads(out.read_text())
    assert report['normalization_train_indices']==[0,1]
    assert len(report['channels']['state'])==28
    state=report['channels']['state'][0]
    assert state['label']=='actor_pre_noise[0]'
    assert state['raw']['min']==2 and state['raw']['max']==6
    assert state['standardized']['min']==.5 and state['standardized']['max']==2.5
    assert state['raw']['quantiles']['0.5']==4
    assert report['channels']['dem_height']['raw']['count']==2*4*400
    assert report['channels']['dem_valid_mask']['raw']['constant']
    assert report['channels']['position_target'][2]['standardized']['max']==0
    assert [r['source_index'] for r in report['channels']['state'][:19]]==[i for i in range(22) if i not in (6,7,8)]


@pytest.mark.parametrize('bad',['dataset_sha','nontrain_id','duplicate_id','fractional_id','missing_binding','bad_schema'])
def test_strict_binding_and_frozen_train_membership(tmp_path,bad):
    d,cp=fixture(tmp_path);p=torch.load(cp,weights_only=True)
    if bad=='dataset_sha':(d/'sample-index.json').write_text('[]')
    elif bad=='nontrain_id':p['normalization']['train_indices']=[0,3]
    elif bad=='duplicate_id':p['normalization']['train_indices']=[0,0]
    elif bad=='fractional_id':p['normalization']['train_indices']=[0,.5]
    elif bad=='missing_binding':p['metadata'].pop('dataset_binding')
    elif bad=='bad_schema':p['normalization']['source']='all_roles'
    torch.save(p,cp)
    with pytest.raises(ValueError):tool().diagnose(d,cp,tmp_path/'report.json')


def test_selected_row_reader_does_not_materialize_holdout(tmp_path):
    d,cp=fixture(tmp_path)
    selected=tool().read_selected_rows(d/'samples.npz','actor_state',[1,0])
    assert selected[:,0].tolist()==[6,2]
