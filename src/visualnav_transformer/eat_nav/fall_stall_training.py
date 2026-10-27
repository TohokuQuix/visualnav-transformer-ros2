"""Deterministic full-batch fitting; holdout is absent from the selection path."""
from __future__ import annotations
import copy
import random
import numpy as np
import torch
from .fall_stall_learning import (RiskEvaluator,MODEL_SCHEMA,fit_normalization,prepare_inputs,
                                  training_support,target_batch,event_bce,objective)

CHECKPOINT_SCHEMA='eatnav_fall_stall_checkpoint_v1'
LEARNED=('state_candidate','dem_state_candidate')
CONSTANTS=('always_event_free','train_prior')


def base_seeds(index,indices):return [index[i]['seed'] for i in indices]


def fit_model(arrays,index,fold,variant,seed,config):
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
        if score<best_score:
            best_score=score;best_epoch=epoch
            best_model=copy.deepcopy(model.state_dict());best_optimizer=copy.deepcopy(optimizer.state_dict())
    model.load_state_dict(best_model);model.eval()
    return {'model':model,'variant':variant,'seed':seed,'fold':fold['fold'],'config':copy.deepcopy(config),
            'normalization':stats,'support':support,'selection_heads':eligible,'best_epoch':best_epoch,
            'best_validation_bce':best_score,'optimizer_state_dict':best_optimizer,'history':history}


def constant_fit(arrays,fold,variant):
    if variant not in CONSTANTS:raise ValueError('unknown constant variant')
    train=fold['roles']['train']['sample_indices'];stats=fit_normalization(arrays,train)
    support=training_support(arrays,train)
    probabilities=[c['positive']/(c['positive']+c['negative']) if variant=='train_prior' and a else 0.
                   for c,a in zip(support['counts'],support['active'])]
    return {'variant':variant,'fold':fold['fold'],'seed':None,'normalization':stats,'support':support,
            'constant_probabilities':probabilities,'best_epoch':None,'history':[]}


def predict(fit,arrays,indices):
    stats=fit['normalization'];shape=(len(indices),arrays['paths'].shape[1],4)
    if fit['variant'] in CONSTANTS:
        probabilities=np.broadcast_to(np.asarray(fit['constant_probabilities']),shape).copy()
        progress=np.broadcast_to(np.asarray(stats['progress_mean']),shape).copy()
    else:
        fit['model'].eval()
        with torch.no_grad():
            outputs=fit['model'](**prepare_inputs(arrays,indices,stats))
            probabilities=outputs['risk_logits'].sigmoid().numpy().astype(float)
            progress=outputs['progress_z'].numpy()*np.asarray(stats['progress_std'])+np.asarray(stats['progress_mean'])
    probabilities[:,:,~np.asarray(fit['support']['active'])]=np.nan
    probabilities[~arrays['candidate_mask'][indices]]=np.nan
    progress[:,:,np.asarray(stats['progress_count'])==0]=np.nan
    progress[~arrays['candidate_mask'][indices]]=np.nan
    return probabilities,progress


def save_checkpoint(path,fit,provenance):
    payload={k:copy.deepcopy(v) for k,v in fit.items() if k!='model'}
    payload.update(schema=CHECKPOINT_SCHEMA,model_schema=MODEL_SCHEMA,torch_version=str(torch.__version__),
                   state_dict=fit['model'].state_dict(),provenance=copy.deepcopy(provenance))
    torch.save(payload,path)


def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    if payload.get('schema')!=CHECKPOINT_SCHEMA or payload.get('model_schema')!=MODEL_SCHEMA:
        raise ValueError('checkpoint schema mismatch (scalar/legacy models are incompatible)')
    if payload.get('variant') not in LEARNED:raise ValueError('checkpoint variant mismatch')
    model=RiskEvaluator(use_dem=payload['variant']=='dem_state_candidate')
    model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    return model,payload


def validate_config(config):
    expected={
        'schema':'eatnav_fall_stall_baseline_config_v1',
        'dataset_ledger_sha256':'6d8dae5aff3611f89171a03c41cc6b2bb498da8e1a3f9962ced3956fac6ffcaa',
        'folds':[0,1,2],'optimization_seeds':[123,456,789],
        'learned_variants':list(LEARNED),'constant_variants':list(CONSTANTS),
        'epochs':300,'learning_rate':.001,'weight_decay':0.,'optimizer':'Adam',
        'batch':'full_train_state_groups','threads':1,'device':'cpu','deterministic_algorithms':True,
        'scheduler':None,'event_horizon_sec':10,'threshold':.5,'pos_weight_cap':20.,
        'progress_weight':.1,'pair_weight':.1,'std_floor':.001,
        'validation_selection':'unweighted_seed_balanced_bce_train_and_validation_supported_heads_earliest_minimum',
        'pr_auc':'tie_aware_noninterpolated_average_precision','formal_unseen_test':False}
    if config!=expected:raise ValueError('config differs from implemented, approved baseline v1 contract')


def predict_scores(fit,arrays,indices):
    """Keep raw logit ordering, including when the sigmoid numerically saturates."""
    if fit['variant'] in CONSTANTS:return predict(fit,arrays,indices)[0]
    fit['model'].eval()
    with torch.no_grad():
        scores=fit['model'](**prepare_inputs(arrays,indices,fit['normalization']))['risk_logits'].numpy().astype(float)
    scores[:,:,~np.asarray(fit['support']['active'])]=np.nan
    scores[~arrays['candidate_mask'][indices]]=np.nan
    return scores
