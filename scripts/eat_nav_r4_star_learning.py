"""Censor-aware v2 dataset and source-weighted training for each field."""
from __future__ import annotations
import copy,json,random
from pathlib import Path
import numpy as np
import torch
from eat_nav_all_fields import compare_anchor,FIELDS
from eat_nav_r4_star import candidate_paths,CANDIDATE_IDS,compare_initial
from eat_nav_command_free_learning import RiskEvaluator,fit_normalization,prepare_inputs,save_checkpoint,load_checkpoint,predict,predict_scores
from eat_nav_current_learning import load_current_dataset,serial
from visualnav_transformer.eat_nav.fall_stall_audit import read,write,verify_derived,seal_derived,sha
from visualnav_transformer.eat_nav.fall_stall_targets import build_targets,EVENT_NAMES
from eat_nav_r4_star_tensors import pack_groups,validate_arrays
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

def available_terms(arrays,ids,active):
    e=arrays['event_mask'][ids,:,3,:]&np.asarray(active)[None,None,:]
    return dict(event=bool(e.any()),pair=bool(e.any()),progress=bool(arrays['progress_mask'][ids].any()))

def fit_field(new,index,roles,seed,output,*,epochs=300):
    output=Path(output)
    if output.exists():verify_derived(output);return output/'checkpoint.pt'
    old,old_index,old_fold=load_current_dataset();old=pad_four(old);n=len(new['dem'])
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

CANDIDATE_ARRAY_KEYS=('paths','directions','candidate_mask','event_targets','event_mask','progress_targets','progress_mask','event_free_10s','event_free_mask')

def pad_four(arrays):
    result={}
    for name,value in arrays.items():
        if name not in CANDIDATE_ARRAY_KEYS:result[name]=value.copy();continue
        if value.shape[1]!=4:raise ValueError('four candidate source required')
        fill=-1 if name in ('event_targets','event_free_10s') else 0
        padded=np.full((value.shape[0],6,*value.shape[2:]),fill,dtype=value.dtype);padded[:,:4]=value;result[name]=padded
    return result

def merge_slot(arrays,index,slot,group):
    if slot not in (4,5):raise ValueError('only appended reverse slots may change')
    packed=pack_groups([group])
    for key in CANDIDATE_ARRAY_KEYS:arrays[key][index,slot]=packed[key][0,slot]

def field_dataset(root,field,baseline):
    root=Path(root);baseline=Path(baseline);dest=root/'datasets'/field
    if dest.exists():
        verify_derived(dest)
        if (dest/'unavailable.json').exists():return None,None,None
        return npz(dest/'samples.npz'),read(dest/'sample-index.json'),read(dest/'split.json')
    original=baseline/'datasets'/field;verify_derived(original);dest.mkdir(parents=True)
    if (original/'unavailable.json').exists():
        write(dest/'unavailable.json',dict(reason='baseline has no included groups'));seal_derived(dest);return None,None,None
    old=npz(original/'samples.npz');arrays=pad_four(old);index=copy.deepcopy(read(original/'sample-index.json'));roles=read(original/'split.json')
    jobs=read(root/'plan.json')['collect'];audit=[]
    ledger=read(root/'collection-progress.json')['results']
    if len({r['index'] for r in ledger})!=len(ledger):raise ValueError('duplicate collection ledger index')
    attempts={r['index']:r for r in ledger}
    for i,member in enumerate(index):
        reference=Path(member['sources'][0]);reference=reference if reference.is_absolute() else Path.cwd()/reference
        initial=npz(reference/'initial-input.npz');anchor=npz(reference/'anchor-state.npz')
        for slot in (4,5):
            matches=[(j,r) for j,r in enumerate(jobs) if r['field']==field and r['seed']==member['seed'] and r['anchor']==member['anchor'] and r['candidate']==slot]
            if len(matches)!=1:raise ValueError('new branch assignment mismatch')
            j,item=matches[0];out=root/'collection-runtime'/f'collect-{j:03d}'/'sim-output'
            record=dict(seed=member['seed'],anchor=member['anchor'],candidate=slot,source=str(out))
            attempt=attempts.get(j,{})
            if attempt.get('state')!='completed' or (out/'failure.json').exists() or (out.parent/'host-failure.json').exists() or not (out/'complete.json').exists():
                record['status']='failed_run';audit.append(record);continue
            if attempt.get('result',{}).get('stop_reason') not in ('horizon_complete','natural_termination','boundary_censor'):
                record['status']='unusable_outcome';audit.append(record);continue
            verify_derived(out);complete=read(out/'complete.json')
            if not complete['anchor_available']:record['status']='anchor_unavailable';audit.append(record);continue
            if complete['candidate']!=slot or complete['seed']!=member['seed']:raise ValueError('new branch identity mismatch')
            diff=compare_anchor(npz(out/'anchor-state.npz'),anchor)+compare_initial(npz(out/'initial-input.npz'),initial)
            if diff:record.update(status='anchor_mismatch',differences=diff);audit.append(record);continue
            rows=teacher_rows([json.loads(s) for s in (out/'trace.jsonl').read_text().splitlines()])
            if not rows:record['status']='no_branch_observation';audit.append(record);continue
            targets=build_targets(rows,candidate_paths()[slot].tolist())
            group=dict(initial=dict(dem=initial['dem'][0],actor_pre_noise=initial['actor_state'][0],root=read(reference/'anchor.json')['root']),
                candidates={CANDIDATE_IDS[slot]:dict(path=candidate_paths()[slot].tolist(),direction=-1,targets=targets)})
            merge_slot(arrays,i,slot,group);member['sources'].append(str(out));record.update(status='included',trace_sha256=sha(out/'trace.jsonl'),targets=targets);audit.append(record)
    validate_arrays(arrays)
    for key in old:
        actual=arrays[key][:,:4] if key in CANDIDATE_ARRAY_KEYS else arrays[key]
        np.testing.assert_array_equal(actual,old[key])
    # Require both appended paths to have actual train targets (unknown does not count).
    ids=roles['train'];known=(arrays['event_mask'][ids].any(axis=(2,3))|arrays['progress_mask'][ids].any(axis=2)) if ids else np.zeros((0,6),bool)
    learned_slots=[bool(known[:,k].any()) for k in (4,5)]
    np.savez_compressed(dest/'samples.npz',**arrays);write(dest/'sample-index.json',index);write(dest/'split.json',roles)
    write(dest/'anchor-audit.json',audit);write(dest/'coverage.json',coverage(arrays,index,roles))
    write(dest/'added-candidate-support.json',dict(train_slots_with_targets=learned_slots,train_candidate_counts=arrays['candidate_mask'][ids].sum(axis=0).tolist(),baseline_ledger_sha256=sha(original/'sha-ledger.json')))
    if not all(learned_slots):write(dest/'unavailable.json',dict(reason='both appended curves need actual training targets',train_slots_with_targets=learned_slots))
    seal_derived(dest)
    return (arrays,index,roles) if all(learned_slots) else (None,None,None)
