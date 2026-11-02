"""Reproducible endpoint/event training; audit evaluation never enters selection."""
from __future__ import annotations
import argparse,collections,hashlib,json,random,time
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.moving_pose2s_model import (MovingPose2sEvaluator,fit_normalization,prepare_inputs,training_support,objective,transfer_features,save_checkpoint,load_checkpoint,predict)

OLD_SHA='c29bef5a56947db03f91ae2261dd8d259762479ebd9d38553017f3d5bd3c0421'
OLD_PATH=Path('/home/ubuntu/visualnav-transformer-ros2/docs/eat-nav/r4-star-fourteen-training-six-evaluation-20261026/models/random_rough/seed123/checkpoint.pt')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def verify_roles(index,roles):
    seen={};ids=[]
    for role,members in roles.items():
        if role not in ('train','validation'):raise ValueError('audit/test must never enter training')
        for i in members:
            parent=index[i].get('parent_id',str((index[i].get('field'),index[i].get('seed'))))
            if parent in seen and seen[parent]!=role:raise ValueError('parent leakage across training roles')
            seen[parent]=role;ids.append(i)
    if len(ids)!=len(set(ids)):raise ValueError('duplicate training role member')
    if not roles.get('train') or not roles.get('validation'):raise ValueError('train and validation required')


def select_training_roles(index,roles,variant):
    if variant not in ('initial','moving','final'):raise ValueError('unknown training variant')
    result={r:[i for i in roles[r] if variant!='initial' or index[i].get('is_initial',False)] for r in ('train','validation')}
    verify_roles(index,result);return result


def hierarchical_weights(index,ids):
    fields=collections.defaultdict(lambda:collections.defaultdict(lambda:collections.defaultdict(list)))
    for ordinal,i in enumerate(ids):
        row=index[i];parent=row.get('parent_id',str(row.get('seed')))
        state=row.get('state_signature',str((row.get('episode_id',parent),row.get('anchor'),row.get('level'),row.get('t0_ns'))))
        fields[row['field']][parent][state].append(ordinal)
    weights=np.zeros(len(ids),np.float32)
    for parents in fields.values():
        for states in parents.values():
            for members in states.values():
                weights[members]=1/(len(fields)*len(parents)*len(states)*len(members))
    return weights


def component_weights(arrays,index,ids):
    ids=np.asarray(ids,dtype=int);cm=arrays['candidate_mask'][ids].astype(bool)
    def one(mask):
        out=np.zeros(mask.shape,np.float32);positions=np.argwhere(mask)
        if len(positions):
            expanded=[dict(index[ids[i]]) for i,k in positions]
            out[tuple(positions.T)]=hierarchical_weights(expanded,list(range(len(expanded))))
        return out
    pm=arrays['pose_mask'][ids].astype(bool)&cm
    em=arrays['event_mask'][ids].astype(bool)&cm[...,None]
    return dict(position=one(pm),regularizer=one(cm),event=np.stack([one(em[...,h]) for h in range(4)],-1))


def fit(dataset,output,*,variant='moving',epochs=150,batch_size=64,device='cpu',threads=8,seed=123,time_budget_sec=7200,transfer_checkpoint=OLD_PATH):
    dataset=Path(dataset);output=Path(output)
    if output.exists():raise FileExistsError('fresh training output required; resume needs explicit checkpoint verification')
    with np.load(dataset/'samples.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    index=json.loads((dataset/'sample-index.json').read_text());split=json.loads((dataset/'split.json').read_text());roles=select_training_roles(index,split['roles'],variant)
    torch.set_num_threads(threads);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    stats=fit_normalization(arrays,roles['train']);support=training_support(arrays,roles['train']);model=MovingPose2sEvaluator()
    if sha(transfer_checkpoint)!=OLD_SHA:raise ValueError('transfer checkpoint SHA mismatch')
    transferred=transfer_features(model,torch.load(transfer_checkpoint,map_location='cpu',weights_only=True));model.to(device)
    head=[];backbone=[]
    for name,p in model.named_parameters():(head if name.startswith(('position_head.','rotation_head.','event_head.')) else backbone).append(p)
    for p in backbone:p.requires_grad_(False)
    optimizer=torch.optim.Adam([{'params':head,'lr':.001},{'params':backbone,'lr':.0001}],weight_decay=1e-5)
    weights={r:component_weights(arrays,index,ids) for r,ids in roles.items()};train=np.asarray(roles['train']);val=np.asarray(roles['validation']);rng=np.random.default_rng(seed)
    config=dict(variant=variant,epochs=epochs,batch_size=batch_size,device=device,threads=threads,seed=seed,warmup_epochs=5,learning_rate_head=.0003,learning_rate_backbone=.0001,weight_decay=1e-5,gradient_clip_norm=5.,early_stop_min_epochs=30,patience=20,time_budget_sec=time_budget_sec,loss='Lpos+.5Lrot+.5unweightedBCE+.001Lreg',event_pos_weight=1.,selection='validation_pose_plus_rotation_plus_unweighted_event',normalization_source='train_only',transfer_sha256=OLD_SHA)
    binding={name:sha(dataset/name) for name in ('samples.npz','sample-index.json','split.json','manifest.json')}
    output.mkdir(parents=True);write(output/'config.json',config);write(output/'dataset-binding.json',dict(dataset=str(dataset.resolve()),files_sha256=binding,roles=roles,transfer=transferred))
    best=float('inf');best_epoch=None;history=[];start=time.monotonic();stop_reason='epochs_completed';checkpoint=output/'checkpoint.pt'
    for epoch in range(1,epochs+1):
        if epoch==6:
            for p in backbone:p.requires_grad_(True)
            optimizer.param_groups[0]['lr']=.0003
        model.train();sums=collections.defaultdict(float);seen=0
        for positions in np.array_split(rng.permutation(len(train)),max(1,int(np.ceil(len(train)/batch_size)))):
            ids=train[positions];optimizer.zero_grad(set_to_none=True)
            batchweights={k:v[positions] for k,v in weights['train'].items()};batchweights['global_scale']=len(train)/len(ids)
            losses=objective(model(**prepare_inputs(arrays,ids,stats,device)),arrays,ids,stats,support,batchweights)
            if not all(torch.isfinite(v).all() for v in losses.values()):raise ValueError('nonfinite train objective')
            losses['total'].backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            for k,v in losses.items():sums[k]+=float(v.detach())*len(ids)
            seen+=len(ids)
        model.eval();valoutputs=[]
        with torch.inference_mode():
            for offset in range(0,len(val),batch_size):
                ids=val[offset:offset+batch_size];valoutputs.append(model(**prepare_inputs(arrays,ids,stats,device)))
            joined={k:torch.cat([o[k] for o in valoutputs],dim=0) for k in valoutputs[0]}
            v=objective(joined,arrays,val,stats,support,weights['validation'])
        score=float(v['position']+.5*v['rotation']+.5*v['event'])
        if not np.isfinite(score):raise ValueError('nonfinite validation objective')
        if score<best:
            best=score;best_epoch=epoch
            save_checkpoint(checkpoint,model,stats,support,dict(config=config,best_epoch=best_epoch,validation_objective=best,dataset_binding=binding,role_counts={r:len(v) for r,v in roles.items()},audit_evaluation_opened=False,transfer=transferred))
        row=dict(epoch=epoch,train={k:v/seen for k,v in sums.items()},validation_objective=score,best_epoch=best_epoch,elapsed_sec=time.monotonic()-start);history.append(row)
        write(output/'progress.json',dict(state='training',**row,planned_epochs=epochs))
        if epoch==1 or epoch%10==0:print(json.dumps(row),flush=True)
        if epoch>=30 and epoch-best_epoch>=20:stop_reason='validation_early_stop';break
        if time.monotonic()-start>=time_budget_sec:stop_reason='walltime_early_stop';break
    restored,payload=load_checkpoint(checkpoint,device='cpu');probe=roles['validation'][:min(4,len(val))];prediction=predict(restored,stats,arrays,probe,support)
    if not np.isfinite(prediction['delta_position_body_m']).all():raise ValueError('checkpoint reload inference nonfinite')
    result=dict(state='completed',variant=variant,epochs_completed=len(history),best_epoch=best_epoch,best_validation_objective=best,stop_reason=stop_reason,elapsed_sec=time.monotonic()-start,checkpoint_sha256=sha(checkpoint),role_counts={r:len(v) for r,v in roles.items()},audit_evaluation_opened=False,support=support,device=device,checkpoint_cpu_reload_verified=True)
    write(output/'training.json',dict(result,history=history));write(output/'progress.json',result)
    torch.save(dict(epoch=len(history),optimizer_state_dict=optimizer.state_dict(),last_state_dict=model.cpu().state_dict(),config=config),output/'training-resume-local.pt')
    print(json.dumps(result),flush=True);return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--variant',choices=('initial','moving','final'),default='moving');p.add_argument('--epochs',type=int,default=150);p.add_argument('--batch-size',type=int,default=64);p.add_argument('--device',default='cpu');p.add_argument('--threads',type=int,default=8);p.add_argument('--seed',type=int,default=123);p.add_argument('--time-budget-sec',type=int,default=7200);p.add_argument('--transfer-checkpoint',type=Path,default=OLD_PATH)
    a=p.parse_args();fit(a.dataset,a.output,variant=a.variant,epochs=a.epochs,batch_size=a.batch_size,device=a.device,threads=a.threads,seed=a.seed,time_budget_sec=a.time_budget_sec,transfer_checkpoint=a.transfer_checkpoint)
if __name__=='__main__':main()
