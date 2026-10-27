"""Fit the sealed current development pool, with actual epoch progress.

Kept outside the collection package to preserve immutable runtime bindings.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from datetime import datetime, timezone
import numpy as np
import torch
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived
from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import validate_arrays,CANDIDATE_IDS
from visualnav_transformer.eat_nav.fall_stall_learning import (RiskEvaluator,fit_normalization,prepare_inputs,
    training_support,target_batch,event_bce,objective)
from visualnav_transformer.eat_nav.fall_stall_training import (base_seeds,constant_fit,predict,predict_scores,
    save_checkpoint,load_checkpoint,LEARNED,CONSTANTS)
from visualnav_transformer.eat_nav.fall_stall_metrics import evaluate_predictions,EVENTS

REPO=Path(__file__).resolve().parents[1]
DATASET=REPO/'docs/eat-nav/additional-development-batch02-cumulative-20261021/cumulative'
DATASET_SHA='33ed73e061c3a00c5b2bf3176935e4963daa3bcda07455d27035ed46dc42cd64'
SPEC=REPO/'docs/eat-nav/current-dataset-learning-spec-v0.1.md'


def default_config():
    return dict(schema='eatnav_current_development_training_v1',dataset_ledger_sha256=DATASET_SHA,
        optimization_seeds=[123,456,789],learned_variants=list(LEARNED),constant_variants=list(CONSTANTS),
        epochs=300,learning_rate=.001,weight_decay=0.,optimizer='Adam',batch='full_train_state_groups',
        threads=1,device='cpu',deterministic_algorithms=True,scheduler=None,event_horizon_sec=10,
        threshold=.5,pos_weight_cap=20.,progress_weight=.1,pair_weight=.1,std_floor=.001,
        validation_selection='unweighted_seed_balanced_bce_train_and_validation_supported_heads_earliest_minimum',
        formal_unseen_test=False,simulation_rollout=False)


def load_current_dataset():
    verify_derived(DATASET)
    require(sha(DATASET/'sha-ledger.json')==DATASET_SHA,'current dataset identity mismatch')
    split=read(DATASET/'split.json');index=read(DATASET/'sample-index.json')
    require(split['schema']=='eatnav_additional_fixed_development_split_v1' and
            split['reserved_test_included'] is False and set(split['roles'])=={'train','validation'},
            'only fixed development roles are permitted')
    with np.load(DATASET/'samples.npz',allow_pickle=False) as data:a={k:data[k] for k in data.files}
    validate_arrays(a)
    require(len(index)==len(a['dem'])==128,'current pool sample count mismatch')
    roles=split['roles'];train=roles['train'];val=roles['validation']
    require(len(train)==96 and len(val)==32 and sorted(train+val)==list(range(128)),
            'fixed split membership mismatch')
    require(not set(base_seeds(index,train))&set(base_seeds(index,val)),'base seed leakage')
    require(len(set(base_seeds(index,train)))==24 and len(set(base_seeds(index,val)))==8,'seed count mismatch')
    for role,ids in roles.items():
        require(all(index[i]['role']==role for i in ids),'index role mismatch')
    return a,index,{'fold':'fixed_current_development','roles':{k:{'sample_indices':v} for k,v in roles.items()}}


def serial(value):
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,np.generic):return serial(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def write_progress(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(value,visualization='offline_training_no_simulation_rollout',
                 updated_utc=datetime.now(timezone.utc).isoformat())
    name=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,suffix='.tmp',delete=False) as stream:
            name=stream.name;json.dump(serial(payload),stream,sort_keys=True,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if name and os.path.exists(name):os.unlink(name)


def fit_current(arrays,index,fold,variant,seed,config,on_epoch=None):
    if variant not in LEARNED:raise ValueError('unknown learned variant')
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(True)
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    train=fold['roles']['train']['sample_indices'];validation=fold['roles']['validation']['sample_indices']
    stats=fit_normalization(arrays,train);support=training_support(arrays,train)
    validation_support=training_support(arrays,validation)
    eligible=[a and b for a,b in zip(support['active'],validation_support['active'])]
    if not any(eligible):raise ValueError('no train/validation supported heads for epoch selection')
    inputs=prepare_inputs(arrays,train,stats);targets=target_batch(arrays,train)
    val_inputs=prepare_inputs(arrays,validation,stats);val_targets=target_batch(arrays,validation)
    train_seeds=base_seeds(index,train);val_seeds=base_seeds(index,validation)
    model=RiskEvaluator(use_dem=variant=='dem_state_candidate')
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    best_score=float('inf');best_model=best_optimizer=None;history=[];best_epoch=None
    for epoch in range(1,config['epochs']+1):
        model.train();optimizer.zero_grad(set_to_none=True);outputs=model(**inputs)
        losses=objective(outputs['risk_logits'],outputs['progress_z'],targets,train_seeds,
                         support['active'],support['pos_weight'],stats)
        if not all(torch.isfinite(v) for v in losses.values()):raise ValueError('nonfinite training loss')
        losses['total'].backward();optimizer.step();model.eval()
        with torch.no_grad():
            val=model(**val_inputs)
            score=float(event_bce(val['risk_logits'],val_targets,val_seeds,eligible))
        if not np.isfinite(score):raise ValueError('nonfinite validation score')
        history.append({'epoch':epoch,**{k:float(v.detach()) for k,v in losses.items()},'validation_bce':score})
        if on_epoch is not None:on_epoch(copy.deepcopy(history[-1]))
        if score<best_score:
            best_score=score;best_epoch=epoch
            best_model=copy.deepcopy(model.state_dict());best_optimizer=copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(best_model);model.eval()
    return {'model':model,'variant':variant,'seed':seed,'fold':fold['fold'],'config':copy.deepcopy(config),
            'normalization':stats,'support':support,'selection_heads':eligible,'best_epoch':best_epoch,
            'best_validation_bce':best_score,'optimizer_state_dict':best_optimizer,'history':history}


def source_snapshot():
    package=REPO/'src/visualnav_transformer/eat_nav'
    paths=[Path(__file__),SPEC]+[package/(name+'.py') for name in
        ('fall_stall_audit','fall_stall_dataset_tensors','fall_stall_targets','fall_stall_learning',
         'fall_stall_training','fall_stall_metrics','long_horizon_labels')]
    return {str(p.relative_to(REPO)):sha(p) for p in paths}


def save_predictions(root,fit,arrays,index,fold):
    metrics={}
    for role,entry in fold['roles'].items():
        ids=entry['sample_indices'];seeds=base_seeds(index,ids)
        p,progress=predict(fit,arrays,ids);scores=predict_scores(fit,arrays,ids)
        result=evaluate_predictions(arrays,ids,seeds,p,progress,fit['support']['active'],risk_scores=scores)
        path=root/(role+'-predictions.json')
        write(path,serial(dict(sample_indices=ids,base_seeds=seeds,candidate_ids=CANDIDATE_IDS,
                               probabilities=p,risk_scores=scores,progress_m=progress)))
        saved=read(path)
        replay=evaluate_predictions(arrays,saved['sample_indices'],saved['base_seeds'],
            np.asarray(saved['probabilities'],dtype=float),np.asarray(saved['progress_m'],dtype=float),
            fit['support']['active'],risk_scores=np.asarray(saved['risk_scores'],dtype=float))
        require(serial(result)==serial(replay),'saved prediction metric replay mismatch')
        if fit['variant'] in LEARNED:
            model,payload=load_checkpoint(root/'checkpoint.pt')
            restored=dict(payload,model=model)
            for expected,actual in zip((p,progress),predict(restored,arrays,ids)):
                np.testing.assert_array_equal(expected,actual)
            np.testing.assert_array_equal(scores,predict_scores(restored,arrays,ids))
        metrics[role]=serial(result)
    write(root/'metrics.json',metrics)
    return metrics


def validate_destinations(output,progress_path):
    allowed=(REPO/'docs/eat-nav/current-dataset-learning-20261021',Path(tempfile.gettempdir()).resolve())
    for path in (Path(output).resolve(),Path(progress_path).resolve()):
        require(any(base in path.parents for base in allowed),
                'destination must be a new current-learning run path or temporary path')
    require(Path(output).resolve()!=Path(progress_path).resolve() and
            Path(output).resolve() not in Path(progress_path).resolve().parents,
            'live progress must be outside sealed output')


def run(output,progress_path):
    output=Path(output).resolve();progress_path=Path(progress_path).resolve()
    validate_destinations(output,progress_path)
    require(not output.exists(),'fresh training output required')
    arrays,index,fold=load_current_dataset();config=default_config();snapshot=source_snapshot()
    provenance=dict(dataset=str(DATASET),dataset_ledger_sha256=DATASET_SHA,source_sha256=snapshot,
                    reserved_test_opened=False,python_version=sys.version,numpy_version=np.__version__,
                    torch_version=str(torch.__version__),started_utc=datetime.now(timezone.utc).isoformat())
    output.mkdir(parents=True)
    write(output/'config.json',config);write(output/'split.json',fold);write(output/'provenance.json',provenance)
    # These hashes are fixed before the first update and recorded in every checkpoint.
    provenance['config_sha256']=sha(output/'config.json')
    provenance['split_sha256']=sha(output/'split.json')
    write(output/'run-start.json',provenance)
    results=[];total=len(LEARNED)*len(config['optimization_seeds']);number=0
    try:
        for variant in config['learned_variants']:
            for seed in config['optimization_seeds']:
                number+=1;name=f'{variant}-seed{seed}';root=output/name;root.mkdir()
                def progress(event):
                    state=dict(event,state='running',variant=variant,seed=seed,run_index=number,
                               total_runs=total,epochs=config['epochs'],completed_runs=number-1)
                    write_progress(progress_path,state)
                    if event['epoch']==1 or event['epoch']%25==0:
                        print(f"[{number}/{total}] {name} epoch {event['epoch']}/{config['epochs']} "
                              f"loss={event['total']:.5f} val_BCE={event['validation_bce']:.5f}",flush=True)
                fit=fit_current(arrays,index,fold,variant,seed,config,on_epoch=progress)
                save_checkpoint(root/'checkpoint.pt',fit,provenance)
                write(root/'fit.json',serial({k:v for k,v in fit.items() if k not in ('model','optimizer_state_dict')}))
                metrics=save_predictions(root,fit,arrays,index,fold)
                results.append(dict(run=name,variant=variant,seed=seed,best_epoch=fit['best_epoch'],
                                    best_validation_bce=fit['best_validation_bce'],metrics=metrics))
                print(f"SAVED {name} best_epoch={fit['best_epoch']} val_BCE={fit['best_validation_bce']:.5f}",flush=True)
        for variant in config['constant_variants']:
            root=output/variant;root.mkdir();fit=constant_fit(arrays,fold,variant)
            write(root/'fit.json',serial(fit));metrics=save_predictions(root,fit,arrays,index,fold)
            results.append(dict(run=variant,variant=variant,seed=None,metrics=metrics))
        verify_derived(DATASET)
        require(sha(DATASET/'sha-ledger.json')==DATASET_SHA,'dataset changed during fit')
        require(source_snapshot()==snapshot,'training source changed during fit')
        require(sha(output/'config.json')==provenance['config_sha256'] and
                sha(output/'split.json')==provenance['split_sha256'],'run binding changed')
        summary=dict(schema='eatnav_current_training_results_v1',state='completed',results=results,
            learned_runs=total,constant_runs=len(CONSTANTS),reserved_test_opened=False,simulation_rollout=False,
            checkpoint_reload_predictions_exact=True,saved_prediction_metrics_replayed=True,
            source_and_dataset_unchanged=True,completed_utc=datetime.now(timezone.utc).isoformat(),
            formal_comparison='N/A; validation only; rare-event coverage insufficient; reserved test pending')
        write(output/'summary.json',serial(summary));seal_derived(output)
        write_progress(progress_path,dict(state='completed',completed_runs=total,total_runs=total,
                        epochs=config['epochs'],output=str(output),artifact_ledger_sha256=sha(output/'sha-ledger.json')))
        print('COMPLETED '+str(output)+' ledger='+sha(output/'sha-ledger.json'),flush=True)
        return summary
    except BaseException as exc:
        write_progress(progress_path,dict(state='failed',run_index=number,total_runs=total,
                                         error_type=type(exc).__name__,error=str(exc)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--progress',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.progress)

