"""Command-free state contract v2; preserves all frozen v1 sources/evidence."""
from __future__ import annotations
import argparse,copy,json,os,random,sys,tempfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,sha,require,verify_derived,seal_derived
from visualnav_transformer.eat_nav.fall_stall_learning import (RiskEvaluator as LegacyEvaluator,
    fit_normalization as legacy_normalization,indices_for,training_support,target_batch,event_bce,objective)
from visualnav_transformer.eat_nav.fall_stall_training import base_seeds,LEARNED,CONSTANTS,constant_fit as legacy_constant_fit
from visualnav_transformer.eat_nav.fall_stall_metrics import evaluate_predictions,EVENTS
from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import CANDIDATE_IDS
from eat_nav_current_learning import load_current_dataset,serial,write_progress,DATASET,DATASET_SHA

REPO=Path(__file__).resolve().parents[1]
SPEC=REPO/'docs/eat-nav/command-free-learning-spec-v0.1.md'
RUN_ROOT=REPO/'docs/eat-nav/command-free-learning-20261021'
MODEL_SCHEMA='eatnav_fall_stall_model_v2_command_free'
CHECKPOINT_SCHEMA='eatnav_fall_stall_checkpoint_v2_command_free'
NORMALIZATION_SCHEMA='eatnav_fall_stall_normalization_v2_command_free'
KEEP_INDICES=tuple(i for i in range(31) if i not in (6,7,8))
INPUT_CONTRACT=dict(schema='eatnav_fall_stall_state_without_command_v2',
                    excluded_actor_indices=[6,7,8],state_keep_indices=list(KEEP_INDICES),state_dim=28)


def default_config():
    from eat_nav_current_learning import default_config as old_config
    return dict(old_config(),schema='eatnav_current_development_training_v2_command_free',
                input_contract=copy.deepcopy(INPUT_CONTRACT),initialization='v1_shared_weights_keep_state_columns')


def fit_normalization(arrays,train_indices):
    stats=legacy_normalization(arrays,train_indices)
    for field in ('state_mean','state_std'):stats[field]=[stats[field][i] for i in KEEP_INDICES]
    stats.update(schema=NORMALIZATION_SCHEMA,input_contract=copy.deepcopy(INPUT_CONTRACT))
    return stats


def prepare_inputs(arrays,indices,stats):
    require(stats.get('schema')==NORMALIZATION_SCHEMA and stats.get('source')=='train_split_only' and
            stats.get('input_contract')==INPUT_CONTRACT,'normalization/input schema mismatch')
    require(len(stats['state_mean'])==len(stats['state_std'])==28,'normalization state dimension mismatch')
    ids=indices_for(arrays,indices);dem=arrays['dem'][ids].copy();mask=dem[:,:,1]
    dem[:,:,0]=np.where(mask==1,(dem[:,:,0]-stats['dem_mean'])/stats['dem_std'],0)
    # Drop the three raw command fields BEFORE subtraction/division.
    state=np.concatenate((arrays['actor_state'][ids],arrays['body_state'][ids]),axis=1)[:,KEEP_INDICES]
    state=(state-np.asarray(stats['state_mean']))/np.asarray(stats['state_std'])
    paths=(arrays['paths'][ids]-np.asarray(stats['path_mean']))/np.asarray(stats['path_std'])
    inputs=dict(dem=torch.from_numpy(dem.reshape(len(ids),8,20,20)),state=torch.from_numpy(state.astype(np.float32)),
        paths=torch.from_numpy(paths.astype(np.float32)),directions=torch.from_numpy(arrays['directions'][ids].copy()),
        candidate_mask=torch.from_numpy(arrays['candidate_mask'][ids].copy()))
    require(all(torch.isfinite(v).all() for v in inputs.values()),'nonfinite normalized input')
    return inputs


class RiskEvaluator(LegacyEvaluator):
    def __init__(self,use_dem=True):
        super().__init__(use_dem=use_dem)
        original=self.state_encoder[0]
        replacement=nn.Linear(28,32)
        with torch.no_grad():
            replacement.weight.copy_(original.weight[:,KEEP_INDICES]);replacement.bias.copy_(original.bias)
        self.state_encoder[0]=replacement

    def forward(self,dem,state,paths,directions,candidate_mask):
        batch,candidates=paths.shape[:2]
        if candidates<1 or paths.shape!=(batch,candidates,8,2) or directions.shape!=(batch,candidates,1) or \
                state.shape!=(batch,28) or dem.shape!=(batch,8,20,20) or candidate_mask.shape!=(batch,candidates):
            raise ValueError('command-free risk model input shape mismatch')
        if not candidate_mask.any(dim=1).all():raise ValueError('all candidates invalid')
        shared=self.state_encoder(state)
        if self.use_dem:shared=torch.cat((self.dem_encoder(dem),shared),dim=-1)
        candidate=torch.cat((paths.reshape(batch,candidates,16),directions),dim=-1)
        features=self.fusion(torch.cat((shared[:,None].expand(-1,candidates,-1),self.path_encoder(candidate)),dim=-1))
        return dict(risk_logits=self.risk_head(features),progress_z=self.progress_head(features))


def constant_fit(arrays,fold,variant):
    fit=legacy_constant_fit(arrays,fold,variant)
    fit['normalization']=fit_normalization(arrays,fold['roles']['train']['sample_indices'])
    return fit


def save_checkpoint(path,fit,provenance):
    payload={k:copy.deepcopy(v) for k,v in fit.items() if k!='model'}
    payload.update(schema=CHECKPOINT_SCHEMA,model_schema=MODEL_SCHEMA,input_contract=copy.deepcopy(INPUT_CONTRACT),
        torch_version=str(torch.__version__),state_dict=fit['model'].state_dict(),provenance=copy.deepcopy(provenance))
    torch.save(payload,path)


def load_checkpoint(path):
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(payload.get('schema')==CHECKPOINT_SCHEMA and payload.get('model_schema')==MODEL_SCHEMA and
            payload.get('input_contract')==INPUT_CONTRACT,'checkpoint schema mismatch: expected command-free v2')
    require(payload.get('variant') in LEARNED,'checkpoint variant mismatch')
    require(payload['normalization'].get('schema')==NORMALIZATION_SCHEMA and
            payload['normalization'].get('input_contract')==INPUT_CONTRACT,'checkpoint normalization schema mismatch')
    model=RiskEvaluator(use_dem=payload['variant']=='dem_state_candidate')
    model.load_state_dict(payload['state_dict'],strict=True);model.eval()
    return model,payload




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

def predict_scores(fit,arrays,indices):
    """Keep raw logit ordering, including when the sigmoid numerically saturates."""
    if fit['variant'] in CONSTANTS:return predict(fit,arrays,indices)[0]
    fit['model'].eval()
    with torch.no_grad():
        scores=fit['model'](**prepare_inputs(arrays,indices,fit['normalization']))['risk_logits'].numpy().astype(float)
    scores[:,:,~np.asarray(fit['support']['active'])]=np.nan
    scores[~arrays['candidate_mask'][indices]]=np.nan
    return scores

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



def source_snapshot():
    from eat_nav_current_learning import source_snapshot as legacy_snapshot
    result=legacy_snapshot()
    for path in [Path(__file__),SPEC,REPO/'scripts/eat_nav_risk_navigation.py',
        REPO/'scripts/eat_nav_risk_navigation_run.py',REPO/'scripts/eat_nav_risk_navigation_v2.py',
        REPO/'scripts/eat_nav_risk_navigation_sim_v2.py',REPO/'scripts/eat_nav_risk_navigation_run_v2.py']:
        result[str(path.relative_to(REPO))]=sha(path)
    return result


def validate_destinations(output,progress_path):
    output=Path(output).resolve();progress_path=Path(progress_path).resolve()
    for path in (output,progress_path):
        require(RUN_ROOT in path.parents or Path(tempfile.gettempdir()).resolve() in path.parents,
                'destination outside command-free learning outputs')
        require(not any((p/'derived-seal.json').is_file() for p in path.parents),
                'destination within sealed evidence')
    require(output!=progress_path and output not in progress_path.parents,'progress must be outside sealed output')




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
        checkpoint=output/'dem_state_candidate-seed123/checkpoint.pt'
        write(output/'gui-checkpoint.json',dict(input_contract=INPUT_CONTRACT,
            relative_checkpoint='dem_state_candidate-seed123/checkpoint.pt',checkpoint_sha256=sha(checkpoint),
            optimization_seed=123,selection='preselected_same_dem_seed123_as_v1_not_posthoc_validation_winner'))
        verify_derived(DATASET)
        require(sha(DATASET/'sha-ledger.json')==DATASET_SHA,'dataset changed during fit')
        require(source_snapshot()==snapshot,'training source changed during fit')
        require(sha(output/'config.json')==provenance['config_sha256'] and
                sha(output/'split.json')==provenance['split_sha256'],'run binding changed')
        summary=dict(schema='eatnav_current_training_results_v2_command_free',input_contract=INPUT_CONTRACT,state='completed',results=results,
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
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--progress',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.progress)
