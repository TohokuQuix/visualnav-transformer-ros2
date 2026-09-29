"""Frozen DDPM trace and sampling-only candidate coverage measurement."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from eat_nav_phase_c_coverage import atomic,digest,write_marker,marker_valid
from eat_nav_phase_c_coverage_stage1 import residual,clusters_pairwise,command_clusters,follower_commands
from eat_nav_phase_c_coverage_evaluator import FrozenEvaluator,score_spread


def distribution_metrics(paths):
    x=np.asarray(paths,dtype=float)
    if x.ndim!=3 or x.shape[2]!=2:raise ValueError('paths must be [K,T,2]')
    y=x.reshape(len(x),-1)
    pair=np.sqrt(np.mean((y[:,None]-y[None,:])**2,axis=-1))
    upper=pair[np.triu_indices(len(x),1)]
    centered=y-y.mean(0)
    singular=np.linalg.svd(centered,compute_uv=False)
    variance=singular**2
    effective=float((variance.sum()**2)/(np.square(variance).sum())) if variance.sum()>0 else 0.
    cov=np.cov(x[:,-1,:].T).tolist() if len(x)>1 else [[0.,0.],[0.,0.]]
    r=np.stack([residual(p) for p in x])
    shape=np.sqrt(np.mean((r[:,None]-r[None,:])**2,axis=(2,3)))
    return {'pairwise_rms_median_m':float(np.median(upper)) if len(upper) else 0.,
            'pairwise_rms_p95_m':float(np.quantile(upper,.95)) if len(upper) else 0.,
            'effective_rank':effective,'endpoint_covariance_m2':cov,
            'shape_spread_m':float(np.median(shape[np.triu_indices(len(x),1)])) if len(x)>1 else 0.,
            'candidate_variance':float(np.var(y,axis=0).mean()),
            'lateral_covariance_m2':float(np.var(x[:,-1,1])),
            'endpoint_spread_m':float(np.std(x[:,-1,:],axis=0).sum())}


def coverage_decision(rows,spec):
    c=spec['coverage'];n=len(rows)
    if not n:return {'pass':False,'reason':'no_states'}
    command=sum(r['command_cluster_count']>=2 for r in rows)/n
    paths=sum(r['path_cluster_count']>=3 for r in rows)/n
    terrains=len({r['terrain_type'] for r in rows if r['command_cluster_count']>=2})
    ood=sum(r['ood_count'] for r in rows)/max(1,sum(r['candidate_count'] for r in rows))
    nonfinite=sum(r['nonfinite_count'] for r in rows)
    shape=any(r['shape_spread_m']>=c.get('min_shape_spread_m',.01) for r in rows)
    plausible=all(r['evaluator_input_plausible'] for r in rows)
    checks={'command_fraction':command>=c['min_fraction_two_command_clusters'],
            'path_fraction':paths>=c['min_fraction_three_path_clusters'],
            'terrain_count':terrains>=c['min_terrain_count_with_two_command_clusters'],
            'ood_fraction':ood<=c['max_train_p95_ood_fraction'],
            'nonfinite':nonfinite<=c['max_nonfinite'],
            'shape':shape or not c['shape_not_endpoint_only'],
            'evaluator_input':plausible or not c['evaluator_input_plausibility_required']}
    return {'pass':all(checks.values()),'checks':checks,'command_fraction':command,'path_fraction':paths,
            'terrain_count':terrains,'ood_fraction':ood,'nonfinite_count':nonfinite}


def sampled_trace(model,dem,goal,seed,initial_scale=1.):
    context=model.encode(dem,goal,None)
    generator=torch.Generator(device=context.device).manual_seed(int(seed))
    shape=(len(dem),model.config.len_traj_pred,2)
    action=torch.randn(shape,dtype=context.dtype,device=context.device,generator=generator)*initial_scale
    model.scheduler.set_timesteps(model.scheduler.config.num_train_timesteps,device=context.device)
    noisy=[action[0].detach().cpu().numpy()];predictions=[];updated=[]
    for timestep in model.scheduler.timesteps:
        predicted=model.noise_pred_net(sample=action,timestep=timestep,global_cond=context)
        action=model.scheduler.step(predicted,timestep,action,generator=generator).prev_sample
        predictions.append(predicted[0].detach().cpu().numpy())
        updated.append(action[0].detach().cpu().numpy());noisy.append(action[0].detach().cpu().numpy())
    delta=action*model.action_std+model.action_mean
    return torch.cumsum(delta,dim=1)[0].detach().cpu().numpy(), np.stack(noisy),np.stack(predictions),np.stack(updated)


def ood_distances(paths,model,train_norm):
    p=np.asarray(paths,dtype=np.float32)
    delta=np.diff(np.concatenate((np.zeros((len(p),1,2),dtype=np.float32),p),axis=1),axis=1)
    mean=model.action_mean.detach().cpu().numpy();std=model.action_std.detach().cpu().numpy()
    normalized=torch.from_numpy(((delta-mean)/std).reshape(len(p),-1)).to(train_norm.device)
    with torch.no_grad():
        return (torch.cdist(normalized,train_norm).min(dim=1).values/np.sqrt(normalized.shape[1])).cpu().numpy()


def examine(paths,terrain,model,train_norm,ood_limit,cluster_spec,client):
    x=np.asarray(paths,dtype=float)
    nonfinite=int((~np.isfinite(x)).reshape(len(x),-1).any(axis=1).sum())
    if nonfinite:return {'terrain_type':terrain,'candidate_count':len(x),'nonfinite_count':nonfinite,
                         'command_cluster_count':0,'path_cluster_count':0,'ood_count':len(x),
                         'shape_spread_m':0.,'evaluator_input_plausible':False}
    ood=ood_distances(x,model,train_norm)
    commands=follower_commands(x,client)
    metrics=distribution_metrics(x)
    return {'terrain_type':terrain,'candidate_count':len(x),'nonfinite_count':0,
            'path_cluster_count':clusters_pairwise(x,cluster_spec['path_rms_m']),
            'command_cluster_count':command_clusters(commands,cluster_spec['command_linear_mps'],cluster_spec['command_angular_radps']),
            'ood_count':int(np.sum(ood>ood_limit)),'ood_distances':ood.tolist(),
            'follower_commands':commands,'evaluator_input_plausible':bool(np.isfinite(x).all() and np.max(np.abs(x))<10),
            **metrics}


def run(args):
    spec=yaml.safe_load(args.coverage_config.read_text());root=args.output
    frozen=json.loads((root/'source-provenance.json').read_text())
    if frozen['input_sha256']['coverage_config']!=digest(args.coverage_config):raise ValueError('config changed')
    ids=json.loads((root/'diagnostic-state-ids.json').read_text())
    states=[('train',row) for row in ids['train']]+[('val',row) for row in ids['val']]
    if len(states)!=spec['states']['train_count']+spec['states']['validation_count'] or ids['heldout_ids']:
        raise ValueError('diagnostic state split mismatch')
    cfg=load_eat_config(args.reference_config)
    views={s:CompleteOnlyView(EATDataset(args.manifest,s,cfg,split_seed=123),cfg.len_traj_pred) for s in ['train','val']}
    if len(views['train'])!=49598 or len(views['val'])!=6171:raise ValueError('split count mismatch')
    model,_,_=load_checkpoint(args.checkpoint,cfg)
    model=model.to('cuda').eval()
    train_actions=np.stack([views['train'][i]['action'] for i in range(len(views['train']))]).astype(np.float32)
    mean=model.action_mean.detach().cpu().numpy();std=model.action_std.detach().cpu().numpy()
    train_norm=torch.from_numpy(((train_actions-mean)/std).reshape(len(train_actions),-1)).to('cuda')
    thresholds=json.loads(args.prior_thresholds.read_text())
    if not np.allclose(mean,thresholds['train_action_mean'],atol=1e-6):raise ValueError('action normalization mismatch')
    ood_limit=float(thresholds['train_ood_nearest_l2_per_dimension_p95'])
    evaluator=FrozenEvaluator('cuda')
    client=FollowerBridgeClient(timeout=45.)
    try:
        with torch.inference_mode():
            for state_i,(split,row) in enumerate(states):
                prior=root/'sampling-states'/f'{state_i:03d}.json'
                trace_prior=root/'trace-states'/f'{state_i:03d}.npz'
                if prior.exists():
                    saved=json.loads(prior.read_text())
                    if saved['split']!=split or saved['state']!=row or not trace_prior.exists() or saved['trace_file_sha256']!=digest(trace_prior):
                        raise ValueError('partial or incompatible sampling state; refusing resume')
                    print('SAMPLING_RESUME_SKIP',state_i,flush=True)
                    continue
                view=views[split];w=view.dataset.windows[view.indices[row['view_index']]]
                if list(w['episode_key'])!=row['episode_key'] or w['step']!=row['step']:
                    raise ValueError('state identity mismatch')
                dem=torch.from_numpy(w['dem'][None]).to('cuda');goal=torch.from_numpy(w['goal'][None]).to('cuda')
                raw=[];noisy=[];pred=[];updated=[]
                for seed in ids['diffusion_seeds']:
                    p,n,e,u=sampled_trace(model,dem,goal,seed)
                    raw.append(p);noisy.append(n);pred.append(e);updated.append(u)
                paths=np.stack(raw)
                if state_i==0:
                    for k in [0,7,31]:
                        baseline=model.sample(dem,goal,None,seed=ids['diffusion_seeds'][k])[0].cpu().numpy()
                        if not np.allclose(paths[k],baseline,atol=1e-6):raise ValueError('trace differs from ordinary sampling')
                trace_file=root/'trace-states'/f'{state_i:03d}.npz'
                trace_file.parent.mkdir(parents=True,exist_ok=True)
                if not trace_file.exists():
                    temp=trace_file.with_name(trace_file.name+'.tmp')
                    norm_all=np.stack(noisy)
                    denorm_delta=norm_all*std+mean
                    cumul_waypoint=np.cumsum(denorm_delta,axis=2)
                    with temp.open('wb') as stream:np.savez_compressed(stream,paths=paths,noisy=norm_all,predicted_noise=np.stack(pred),updated=np.stack(updated),denormalized_delta=denorm_delta,cumulative_waypoint=cumul_waypoint)
                    temp.replace(trace_file)
                conditions={'baseline_k8':paths[:8], 'baseline_k32':paths}
                for scale in spec['sampling']['initial_noise_scales_diagnostic']:
                    if scale==1.:continue
                    conditions[f'noise_scale_{scale}']=np.stack([sampled_trace(model,dem,goal,seed,scale)[0] for seed in ids['diffusion_seeds'][:8]])
                row_out={'split':split,'state':row,'state_index':state_i,'conditions':{},'trace_file_sha256':digest(trace_file)}
                for name,candidates in conditions.items():
                    row_out['conditions'][name]=examine(candidates,row['terrain_type'],model,train_norm,ood_limit,spec['clusters'],client)
                row_out['trace_stats']=[]
                for t in range(noisy[0].shape[0]):
                    norm=np.stack(noisy)[:,t]
                    denorm=norm*std+mean
                    cumul=np.cumsum(denorm,axis=1)
                    row_out['trace_stats'].append({'step':t,'normalized':distribution_metrics(np.cumsum(norm,axis=1)),
                                                   'waypoint':distribution_metrics(cumul),
                                                   'denormalized_delta_variance':float(np.var(denorm,axis=0).mean()),
                                                   'predicted_noise_variance':float(np.var(np.stack(pred)[:,t-1],axis=0).mean()) if t else None,
                                                   'update_variance':float(np.var(np.stack(updated)[:,t-1],axis=0).mean()) if t else None,
                                                   'evaluator_score_spread':score_spread(evaluator.predict(w['dem'],w['goal'],cumul))})
                var0=row_out['trace_stats'][0]['waypoint']['candidate_variance']
                for stat in row_out['trace_stats']:
                    stat['variance_retention_ratio']=stat['waypoint']['candidate_variance']/var0 if var0 else None
                atomic(root/'sampling-states'/f'{state_i:03d}.json',row_out)
                atomic(root/'status.json',{'stage':'sampling','completed_states':state_i+1,'total_states':len(states),'formal_phase_c_allowed':False})
                print('SAMPLING_STATE',state_i+1,len(states),flush=True)
    finally:client.close()
    rows=[json.loads((root/'sampling-states'/f'{i:03d}.json').read_text()) for i in range(len(states))]
    trace={'state_count':len(rows),'rows':[{'split':r['split'],'state':r['state'],'trace_stats':r['trace_stats'],'trace_file_sha256':r['trace_file_sha256']} for r in rows]}
    atomic(root/'denoising-variance-trace.json',trace)
    atomic(root/'sampling-grid-results.json',{'state_count':len(rows),'conditions':list(rows[0]['conditions']),'rows':[{'split':r['split'],'state':r['state'],'conditions':r['conditions']} for r in rows]})
    decisions={name:coverage_decision([r['conditions'][name] for r in rows],spec) for name in rows[0]['conditions']}
    selected=[n for n in ['baseline_k8','baseline_k32'] if decisions[n]['pass']]
    choice=min(selected,key=lambda n:(decisions[n]['ood_fraction'],-decisions[n]['command_fraction'],-decisions[n]['path_fraction'],n!='baseline_k8')) if selected else None
    atomic(root/'sampling-coverage-decision.json',{'decision':'pass' if choice else 'fail','selected':choice,
        'conditions':decisions,'noise_scale_nonunit_diagnostic_only':True,'heldout_used':False})
    stage_input=hashlib.sha256((digest(args.coverage_config)+digest(root/'diagnostic-state-ids.json')+digest(Path(__file__))).encode()).hexdigest()
    write_marker(root/'sampling.done.json',[root/'denoising-variance-trace.json',root/'sampling-grid-results.json',root/'sampling-coverage-decision.json'],stage_input)


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','reference-config','coverage-config','checkpoint','prior-thresholds','output']:
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
if __name__=='__main__':main()
