"""Censor-aware v2 dataset and source-weighted training for each field."""
from __future__ import annotations
import copy,json,random
from pathlib import Path
import numpy as np
import torch
from eat_nav_all_fields import compare_anchor,FIELDS
from eat_nav_risk_navigation import candidate_paths,CANDIDATE_IDS
from eat_nav_command_free_learning import RiskEvaluator,fit_normalization,prepare_inputs,save_checkpoint,load_checkpoint,predict,predict_scores
from eat_nav_current_learning import load_current_dataset,serial
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,verify_derived,seal_derived,sha
from visualnav_transformer.eat_nav.fall_stall_targets import build_targets,EVENT_NAMES
from visualnav_transformer.eat_nav.fall_stall_dataset_tensors import pack_groups,validate_arrays
from visualnav_transformer.eat_nav.fall_stall_learning import training_support,target_batch,objective,event_bce
from visualnav_transformer.eat_nav.fall_stall_metrics import evaluate_predictions
KRAIL_SEEDS=[123,456,789,101112,131415,161718]

def npz(path):
    with np.load(path,allow_pickle=False) as data:return {k:data[k] for k in data.files}

def teacher_rows(rows):
    # A crossing interval can contain an out-of-field impact; censor before it.
    accepted=[]
    for row in rows:
        if row.get('boundary_censor'):break
        accepted.append(row)
    return accepted

def mix_terms(new,old,available):
    result={}
    for k in ('event','progress','pair'):
        result[k]=(2*new[k]+old[k])/3 if available[k] else old[k]
    result['total']=result['event']+.1*result['progress']+.1*result['pair']
    return result

def coverage(arrays,index,roles):
    result={}
    for role,ids in roles.items():
        result[role]={}
        for j,name in enumerate(EVENT_NAMES):
            y=arrays['event_targets'][ids,:,3,j];m=arrays['event_mask'][ids,:,3,j]
            result[role][name]=dict(positive=int(((y==1)&m).sum()),negative=int(((y==0)&m).sum()),unknown=int((~m).sum()),
                positive_seeds=sorted({index[i]['seed'] for i in ids if np.any((arrays['event_targets'][i,:,3,j]==1)&arrays['event_mask'][i,:,3,j])}))
    return result

def field_dataset(root,field):
    root=Path(root);dest=root/'datasets'/field
    if dest.exists():
        verify_derived(dest)
        if (dest/'unavailable.json').exists():return None,None,None
        return npz(dest/'samples.npz'),read(dest/'sample-index.json'),read(dest/'split.json')
    manifest=read(root/'manifest.json');items=[(i,r) for i,r in enumerate(manifest['collect']) if r['field']==field]
    grouped={}
    for i,item in items:grouped.setdefault((item['seed'],item['anchor']),[]).append((i,item))
    groups=[];index=[];audit=[]
    for (seed,anchor),entries in grouped.items():
        outputs=[root/f'collect-{i:03d}'/'sim-output' for i,_ in entries]
        if not all((p/'complete.json').exists() for p in outputs):
            audit.append(dict(seed=seed,anchor=anchor,status='failed_run'));continue
        complete=[read(p/'complete.json') for p in outputs]
        if not all(c['anchor_available'] for c in complete):audit.append(dict(seed=seed,anchor=anchor,status='anchor_unavailable',outputs=[str(p) for p in outputs]));continue
        anchors=[npz(p/'anchor-state.npz') for p in outputs];inputs=[npz(p/'initial-input.npz') for p in outputs]
        differences=[dict(candidate=j,anchor=compare_anchor(a,anchors[0]),inputs=compare_anchor(inputs[j],inputs[0])) for j,a in enumerate(anchors)]
        if any(d['anchor'] or d['inputs'] for d in differences):audit.append(dict(seed=seed,anchor=anchor,status='anchor_mismatch',differences=differences));continue
        candidates={};valid=True;target_evidence=[]
        for slot,p in enumerate(outputs):
            verify_derived(p)
            rows=teacher_rows([json.loads(s) for s in (p/'trace.jsonl').read_text().splitlines()])
            if not rows:valid=False;break
            targets=build_targets(rows,candidate_paths()[slot].tolist())
            candidates[CANDIDATE_IDS[slot]]=dict(path=candidate_paths()[slot].tolist(),direction=-1 if slot==3 else 1,targets=targets)
            target_evidence.append(dict(candidate=slot,trace_sha256=sha(p/'trace.jsonl'),targets=targets))
        if not valid:audit.append(dict(seed=seed,anchor=anchor,status='no_branch_observation'));continue
        initial=dict(dem=inputs[0]['dem'][0],actor_pre_noise=inputs[0]['actor_state'][0],root=read(outputs[0]/'anchor.json')['root'])
        groups.append(dict(initial=initial,candidates=candidates))
        index.append(dict(sample_id=f'{field}/{seed}/anchor{anchor}',seed=seed,anchor=anchor,role=entries[0][1]['role'],field=field,
            sources=[str(p) for p in outputs]))
        audit.append(dict(seed=seed,anchor=anchor,status='included',differences=differences,targets=target_evidence))
    dest.mkdir(parents=True)
    write(dest/'anchor-audit.json',audit)
    if not groups:
        write(dest/'unavailable.json',dict(field=field,reason='no_valid_training_groups'));seal_derived(dest);return None,None,None
    arrays=pack_groups(groups);roles={r:[i for i,g in enumerate(index) if g['role']==r] for r in ('train','validation')}
    np.savez_compressed(dest/'samples.npz',**arrays);write(dest/'sample-index.json',index);write(dest/'split.json',roles)
    write(dest/'coverage.json',coverage(arrays,index,roles));seal_derived(dest)
    return arrays,index,roles

def available_terms(arrays,ids,active):
    e=arrays['event_mask'][ids,:,3,:]&np.asarray(active)[None,None,:]
    return dict(event=bool(e.any()),pair=bool(e.any()),progress=bool(arrays['progress_mask'][ids].any()))

def fit_field(new,index,roles,seed,output,*,epochs=300):
    output=Path(output)
    if output.exists():verify_derived(output);return output/'checkpoint.pt'
    old,old_index,old_fold=load_current_dataset();n=len(new['dem'])
    arrays={k:np.concatenate((new[k],old[k]),axis=0) for k in new};validate_arrays(arrays)
    all_index=index+old_index;train_old=[n+i for i in old_fold['roles']['train']['sample_indices']]
    val_old=[n+i for i in old_fold['roles']['validation']['sample_indices']]
    train=roles['train']+train_old
    if not roles['train']:return None
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    stats=fit_normalization(arrays,train);support=training_support(arrays,train);active=support['active']
    model=RiskEvaluator(use_dem=True);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    train_ids=[roles['train'],train_old];val_ids=[roles['validation'],val_old]
    prepared=[prepare_inputs(arrays,ids,stats) for ids in train_ids]
    targets=[target_batch(arrays,ids) for ids in train_ids]
    seeds=[[all_index[i]['seed'] for i in ids] for ids in train_ids]
    val_inputs=[prepare_inputs(arrays,ids,stats) if ids else None for ids in val_ids]
    val_targets=[target_batch(arrays,ids) if ids else None for ids in val_ids]
    val_seeds=[[all_index[i]['seed'] for i in ids] for ids in val_ids]
    new_available=available_terms(arrays,roles['train'],active)
    val_available=bool(roles['validation']) and bool((arrays['event_mask'][roles['validation'],:,3,:]&np.asarray(active)).any())
    best=float('inf');best_state=None;best_epoch=None;best_optimizer=None;history=[]
    output.mkdir(parents=True)
    for epoch in range(1,epochs+1):
        model.train();optimizer.zero_grad(set_to_none=True);losses=[]
        for x,y,s in zip(prepared,targets,seeds):
            v=model(**x);losses.append(objective(v['risk_logits'],v['progress_z'],y,s,active,support['pos_weight'],stats))
        loss=mix_terms(losses[0],losses[1],new_available)
        if not all(torch.isfinite(v) for v in loss.values()):raise ValueError('nonfinite loss')
        loss['total'].backward();optimizer.step();model.eval()
        with torch.no_grad():
            old_score=float(event_bce(model(**val_inputs[1])['risk_logits'],val_targets[1],val_seeds[1],active))
            new_score=float(event_bce(model(**val_inputs[0])['risk_logits'],val_targets[0],val_seeds[0],active)) if val_available else None
            score=(2*new_score+old_score)/3 if new_score is not None else old_score
        if not np.isfinite(score):raise ValueError('nonfinite validation')
        record=dict(epoch=epoch,**{k:float(v.detach()) for k,v in loss.items()},validation_bce=score,new_validation_bce=new_score,existing_validation_bce=old_score)
        history.append(record)
        if score<best:best=score;best_epoch=epoch;best_state=copy.deepcopy(model.state_dict());best_optimizer=copy.deepcopy(optimizer.state_dict())
        if epoch%10==0:print('TRAINING '+json.dumps(dict(model=str(output),seed=seed,**record)),flush=True)
    model.load_state_dict(best_state);model.eval()
    fit=dict(model=model,variant='dem_state_candidate',seed=seed,fold='fixed_field_development',
        config=dict(epochs=epochs,learning_rate=.001,source_weights=[2/3,1/3],initialization='fresh_seeded_28dim',threads=1),
        normalization=stats,support=support,selection_heads=active,best_epoch=best_epoch,best_validation_bce=best,
        optimizer_state_dict=best_optimizer,history=history)
    save_checkpoint(output/'checkpoint.pt',fit,dict(reserved_test_opened=False,new_members=index,existing_dataset='fixed_batch01_batch02_development',validation_fallback=not val_available))
    restored,payload=load_checkpoint(output/'checkpoint.pt');restored_fit=dict(payload,model=restored)
    metrics={}
    for name,ids in [('new_train',roles['train']),('new_validation',roles['validation']),('existing_train',train_old),('existing_validation',val_old)]:
        if not ids:metrics[name]=dict(status='N/A_no_groups');continue
        p,progress=predict(fit,arrays,ids);scores=predict_scores(fit,arrays,ids)
        p2,q2=predict(restored_fit,arrays,ids);np.testing.assert_array_equal(p,p2);np.testing.assert_array_equal(progress,q2)
        metrics[name]=serial(evaluate_predictions(arrays,ids,[all_index[i]['seed'] for i in ids],p,progress,active,risk_scores=scores))
        write(output/(name+'-predictions.json'),serial(dict(ids=ids,probabilities=p,progress=progress,scores=scores)))
    write(output/'metrics.json',metrics);write(output/'training.json',serial(dict(history=history,best_epoch=best_epoch,best_validation_bce=best,
        train_indices=train,validation_indices=val_ids,new_available_terms=new_available,validation_fallback=not val_available,training_support=support,
        normalization=stats,new_coverage=coverage(new,index,roles),checkpoint_sha256=sha(output/'checkpoint.pt'))))
    seal_derived(output);return output/'checkpoint.pt'
