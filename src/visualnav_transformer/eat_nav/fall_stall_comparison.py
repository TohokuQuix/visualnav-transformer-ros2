"""New-directory comparison export, saved-prediction replay and integrity checks."""
from __future__ import annotations
import json
import platform
import sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from .fall_stall_audit import read,write,sha,seal_derived,verify_derived
from .fall_stall_dataset import load_dataset
from .fall_stall_metrics import evaluate_predictions,EVENTS
from .fall_stall_training import (fit_model,constant_fit,predict,base_seeds,save_checkpoint,
                                  load_checkpoint,validate_config,predict_scores)
from .fall_stall_learning import training_support

RUN_SCHEMA='eatnav_fall_stall_comparison_v1'


def serial(value):
    if isinstance(value,np.ndarray):return serial(value.tolist())
    if isinstance(value,list):return [serial(v) for v in value]
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def deserialize(value):return np.asarray(value,dtype=float)


def source_files(repo):
    names=['fall_stall_learning.py','fall_stall_metrics.py','fall_stall_training.py',
           'fall_stall_comparison.py','fall_stall_dataset.py','fall_stall_dataset_tensors.py',
           'fall_stall_dataset_splits.py','fall_stall_audit.py','fall_stall_targets.py']
    files=[repo/'src/visualnav_transformer/eat_nav'/n for n in names]
    files.extend([repo/'scripts/eat_nav_fall_stall_baseline.py',
                  repo/'config/eat_nav/fall_stall_baseline_v1.json',
                  repo/'docs/eat-nav/fall-stall-learning-spec-v0.1.md'])
    return {str(p.relative_to(repo)):sha(p) for p in files}


def run_comparison(dataset,config_path,output,repo):
    dataset=Path(dataset).resolve();config_path=Path(config_path).resolve();output=Path(output).resolve();repo=Path(repo).resolve()
    config=read(config_path);validate_config(config)
    if sha(dataset/'sha-ledger.json')!=config['dataset_ledger_sha256']:raise ValueError('frozen dataset ledger SHA mismatch')
    arrays,index,split=load_dataset(dataset)
    frozen=source_files(repo);frozen_config_sha=sha(config_path)
    output.mkdir(parents=True,exist_ok=False)
    manifest={'schema':RUN_SCHEMA,'started_utc':datetime.now(timezone.utc).isoformat(),
              'dataset_path':str(dataset),'dataset_ledger_sha256':sha(dataset/'sha-ledger.json'),
              'config_sha256':frozen_config_sha,'source_sha256':frozen,
              'python':sys.version,'torch':str(torch.__version__),'numpy':np.__version__,
              'platform':platform.platform(),'device':'cpu','formal_unseen_test':False,
              'status':'running','runs':[]}
    (output/'frozen-config.json').write_bytes(config_path.read_bytes());write(output/'frozen-split.json',split)
    write(output/'started-manifest.json',manifest)
    # Keep a byte copy of sources so the run can be audited after workspace changes.
    for name in frozen:
        destination=output/'sources'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_bytes((repo/name).read_bytes())
    for fold in split['folds']:
        jobs=[(v,None) for v in config['constant_variants']]
        jobs.extend((v,s) for v in config['learned_variants'] for s in config['optimization_seeds'])
        for variant,seed in jobs:
            run_id=f"fold{fold['fold']}-{variant}"+(f'-seed{seed}' if seed is not None else '')
            print(f'Running {run_id}',flush=True)
            fit=constant_fit(arrays,fold,variant) if seed is None else fit_model(arrays,index,fold,variant,seed,config)
            directory=output/'runs'/run_id;directory.mkdir(parents=True)
            ids=fold['roles']['exploratory_holdout']['sample_indices'];seeds=base_seeds(index,ids)
            p,progress=predict(fit,arrays,ids);scores=predict_scores(fit,arrays,ids)
            metrics=evaluate_predictions(arrays,ids,seeds,p,progress,fit['support']['active'],risk_scores=scores)
            prediction={'sample_indices':ids,'base_seeds':seeds,
                        'sample_ids':[index[i]['sample_id'] for i in ids],
                        'candidate_ids':[[c['candidate'] for c in index[i]['candidates']] for i in ids],
                        'risk_probabilities':serial(p),'risk_ordering_scores':serial(scores),'progress_m':serial(progress),
                        'events':arrays['event_targets'][ids,:,3,:].tolist(),
                        'event_mask':arrays['event_mask'][ids,:,3,:].tolist(),
                        'progress_targets_m':arrays['progress_targets'][ids].tolist(),
                        'progress_mask':arrays['progress_mask'][ids].tolist(),
                        'candidate_mask':arrays['candidate_mask'][ids].tolist()}
            write(directory/'holdout-predictions.json',prediction);write(directory/'holdout-metrics.json',metrics)
            metadata={k:serial(v) for k,v in fit.items() if k not in ('model','optimizer_state_dict','history')}
            write(directory/'fit.json',metadata);write(directory/'history.json',fit['history'])
            if seed is not None:
                save_checkpoint(directory/'checkpoint.pt',fit,{'dataset_ledger_sha256':manifest['dataset_ledger_sha256'],
                    'config_sha256':frozen_config_sha,'source_sha256':frozen,'split':fold})
                restored,payload=load_checkpoint(directory/'checkpoint.pt')
                restored_fit=dict(payload,model=restored)
                rp,rg=predict(restored_fit,arrays,ids)
                np.testing.assert_array_equal(rp,p);np.testing.assert_array_equal(rg,progress)
            replay=evaluate_predictions(arrays,ids,seeds,deserialize(prediction['risk_probabilities']),
                                        deserialize(prediction['progress_m']),fit['support']['active'],risk_scores=deserialize(prediction['risk_ordering_scores']))
            if replay!=metrics:raise ValueError('saved prediction metric replay differs')
            manifest['runs'].append({'run_id':run_id,'variant':variant,'fold':fold['fold'],'seed':seed,
                                     'best_epoch':fit['best_epoch'],'train_supported':fit['support']['active']})
    if source_files(repo)!=frozen or sha(config_path)!=frozen_config_sha:raise ValueError('code/config changed during run')
    load_dataset(dataset)
    if sha(dataset/'sha-ledger.json')!=manifest['dataset_ledger_sha256']:raise ValueError('dataset changed during run')
    manifest.update(status='complete',finished_utc=datetime.now(timezone.utc).isoformat(),
                    checkpoint_roundtrips=18,metric_replays=24,dataset_unchanged=True)
    write(output/'run-manifest.json',manifest);seal_derived(output)
    return manifest


def verify_comparison(output,dataset):
    output=Path(output);dataset=Path(dataset);verify_derived(output)
    manifest=read(output/'run-manifest.json');config=read(output/'frozen-config.json');validate_config(config)
    torch.set_num_threads(config['threads']);torch.use_deterministic_algorithms(config['deterministic_algorithms'])
    if manifest.get('schema')!=RUN_SCHEMA or manifest.get('status')!='complete':raise ValueError('incomplete comparison')
    if sha(dataset/'sha-ledger.json')!=manifest['dataset_ledger_sha256']:raise ValueError('dataset ledger mismatch')
    arrays,index,split=load_dataset(dataset);checks=0
    if sha(output/'frozen-config.json')!=manifest['config_sha256']:raise ValueError('frozen config SHA mismatch')
    if read(output/'frozen-split.json')!=split:raise ValueError('frozen split mismatch')
    for name,digest in manifest['source_sha256'].items():
        if sha(output/'sources'/name)!=digest:raise ValueError('source snapshot SHA mismatch')
    expected={(f,v,s) for f in config['folds'] for v in config['learned_variants'] for s in config['optimization_seeds']}
    expected|={(f,v,None) for f in config['folds'] for v in config['constant_variants']}
    actual={(r['fold'],r['variant'],r['seed']) for r in manifest['runs']}
    if actual!=expected or len(manifest['runs'])!=24:raise ValueError('run coverage mismatch')
    for run in manifest['runs']:
        directory=output/'runs'/run['run_id'];saved=read(directory/'holdout-predictions.json');fit=read(directory/'fit.json')
        ids=split['folds'][run['fold']]['roles']['exploratory_holdout']['sample_indices']
        if ids!=saved['sample_indices']:raise ValueError('saved split mismatch')
        support=training_support(arrays,split['folds'][run['fold']]['roles']['train']['sample_indices'])
        if support!=fit['support']:raise ValueError('saved training support mismatch')
        p=deserialize(saved['risk_probabilities']);progress=deserialize(saved['progress_m'])
        metric=evaluate_predictions(arrays,ids,base_seeds(index,ids),p,progress,support['active'],risk_scores=deserialize(saved['risk_ordering_scores']))
        if metric!=read(directory/'holdout-metrics.json'):raise ValueError('metric replay mismatch')
        if run['seed'] is not None:
            model,payload=load_checkpoint(directory/'checkpoint.pt')
            history=read(directory/'history.json')
            if fit['config']!=config or payload['config']!=config or len(history)!=config['epochs']:
                raise ValueError('checkpoint/config/epoch count mismatch')
            best=min(history,key=lambda row:row['validation_bce'])
            if payload['history']!=history or best['epoch']!=fit['best_epoch'] or best['epoch']!=payload['best_epoch']:
                raise ValueError('validation selection/history mismatch')
            if payload['provenance']['dataset_ledger_sha256']!=manifest['dataset_ledger_sha256'] or payload['provenance']['config_sha256']!=manifest['config_sha256']:
                raise ValueError('checkpoint provenance mismatch')
            actual_p,actual_g=predict(dict(payload,model=model),arrays,ids)
        else:actual_p,actual_g=predict(constant_fit(arrays,split['folds'][run['fold']],run['variant']),arrays,ids)
        verified_fit=dict(payload,model=model) if run['seed'] is not None else constant_fit(arrays,split['folds'][run['fold']],run['variant'])
        np.testing.assert_array_equal(predict_scores(verified_fit,arrays,ids),deserialize(saved['risk_ordering_scores']))
        np.testing.assert_array_equal(actual_p,p);np.testing.assert_array_equal(actual_g,progress);checks+=1
    return {'verified_runs':checks,'checkpoint_roundtrips':18,'seal':'pass','dataset_ledger_sha256':manifest['dataset_ledger_sha256']}
