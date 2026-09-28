"""Validation-selected post-hoc deterministic Probe comparison and permutation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.relative_quality_probe import quality_tensor,tensorize,train_relative_probe,predict,permutation_comparison
from visualnav_transformer.eat_nav.quality_gate_evaluation import episode_cluster_ci,trajectory_errors

ROOT=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
OUT=ROOT/'output';SPEC=Path('/tmp/eatnav-train/relative-quality-config.yaml')
CKPT=Path('/tmp/eatnav-train/pilot-quality-best.pt')
KINDS=('probe_0','probe_abs','probe_adv','probe_drawdown','probe_multi')

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def write(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+'\n');tmp.replace(path)
def setup():
    s=yaml.safe_load(SPEC.read_text());checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
    if checkpoint['split_hash']!=s['split_hash']:raise ValueError('checkpoint split mismatch')
    stats=checkpoint['action_statistics'];mean=np.asarray(stats['mean'],dtype=np.float32);std=np.asarray(stats['std'],dtype=np.float32)
    if mean.shape!=(2,) or std.shape!=(2,) or not (std>0).all():raise ValueError('action statistics mismatch')
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    return s,mean,std,device

def load(split,mean,std,device,limit=None):
    raw=np.load(ROOT/'splits'/f'{split}.npz')
    if limit is not None:raw={key:raw[key][:limit] for key in raw.files}
    data=tensorize(raw,mean,std,device)
    return raw,data

def fingerprint():
    return {name:sha(path) for name,path in [('config',SPEC),('train',ROOT/'splits/train.npz'),
            ('val',ROOT/'splits/val.npz'),('probe_code',Path('/tmp/eatnav-train/src/visualnav_transformer/eat_nav/relative_quality_probe.py'))]}

def mean_baseline(train,val):
    mean=train['action'].mean(dim=0)
    loss=float((val['action']-mean).square().mean())
    return mean,loss

def train(smoke=False):
    s,mean,std,device=setup();start=time.monotonic();torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
    train_raw,train_data=load('train',mean,std,device,limit=100 if smoke else None)
    val_raw,val_data=load('val',mean,std,device,limit=50 if smoke else None)
    mean_action,mean_val=mean_baseline(train_data,val_data)
    if smoke:s=dict(s,probe_max_epochs=2,probe_patience=3)
    schedule=[('probe_0',s['probe_seeds'][0])] if smoke else [(kind,seed) for kind in KINDS for seed in s['probe_seeds']]
    result={'mode':'smoke' if smoke else 'train','device':device,'train_windows':len(train_data['dem']),
            'validation_windows':len(val_data['dem']),'mean_trajectory_validation_mse':mean_val,
            'mean_trajectory_train_action':mean_action.cpu().numpy().tolist(),
            'runs':[],'completed':False,'heldout_used':False,'input_fingerprint':fingerprint()}
    target=ROOT/'smoke-probe' if smoke else ROOT/'probes'
    for kind,seed in schedule:
        run_dir=target/f'{kind}-seed{seed}'
        tq=quality_tensor(train_raw,kind,device);vq=quality_tensor(val_raw,kind,device)
        out=train_relative_probe(train_data,val_data,tq,vq,kind=kind,seed=seed,settings=s,
                                  output=run_dir,fingerprint=result['input_fingerprint'])
        out['checkpoint_sha256']=sha(out['checkpoint']);result['runs'].append(out)
        write(target/'training-summary.json',result)
        print('RELATIVE_PROBE_DONE',kind,seed,out['best_epoch'],round(out['best_validation_loss'],6),flush=True)
    result.update({'completed':True,'seconds':time.monotonic()-start,
                   'peak_torch_gpu_bytes':int(torch.cuda.max_memory_allocated()) if device=='cuda' else None})
    write(target/'training-summary.json',result)
    if not smoke:write(OUT/'relative-quality-probe-training-summary.json',result)
    print('RELATIVE_PROBE_TRAIN_COMPLETE',result['mode'],round(result['seconds'],2),flush=True)


def predict_metric(run,raw,data,mean,std,s):
    q=quality_tensor(raw,run['kind'],data['dem'].device)
    pred=predict(run['checkpoint'],data,q,s['probe_batch_size']).detach().cpu().numpy()
    target=data['action'].detach().cpu().numpy();diff=np.square(pred-target)
    mse=diff.mean(axis=(1,2));physical_pred=pred*std+mean;physical_target=target*std+mean
    errors=trajectory_errors(physical_pred,physical_target,np.ones((len(pred),8),dtype=bool))
    return {'mse':mse,'rms':errors['rms_m'],'physical_pred':physical_pred,
            'total_mse':float(diff.mean()),'total_rms_m':float(errors['rms_m'].mean())}

def evaluate():
    s,mean,std,device=setup();train_raw,train_data=load('train',mean,std,device)
    test_raw,test_data=load('test',mean,std,device);val_raw,val_data=load('val',mean,std,device)
    summary=json.loads((OUT/'relative-quality-probe-training-summary.json').read_text())
    if not summary['completed'] or summary['input_fingerprint']!=fingerprint():raise ValueError('Probe input fingerprint mismatch')
    mean_action=torch.tensor(summary['mean_trajectory_train_action'],device=device)
    test_target=test_data['action'].cpu().numpy();mean_target=np.asarray(summary['mean_trajectory_train_action'])
    mean_test=float(np.square(test_target-mean_target).mean());sst=float(np.square(test_target-mean_target).sum())
    runs=[];bykind={name:[] for name in KINDS}
    for run in summary['runs']:
        if sha(run['checkpoint'])!=run['checkpoint_sha256']:raise ValueError('Probe checkpoint SHA mismatch')
        result=predict_metric(run,test_raw,test_data,mean,std,s)
        entry={'kind':run['kind'],'seed':run['seed'],'best_epoch':run['best_epoch'],
               'validation_loss':run['best_validation_loss'],'heldout_window_mse':float(result['mse'].mean()),
               'heldout_physical_rms_m':result['total_rms_m'],
               'heldout_r2':float(1-result['mse'].sum()*16/sst),
               'parameter_count':run['parameter_count'],'underfit_cap':run['validation_improving_at_cap']}
        runs.append(entry);bykind[run['kind']].append((run,result))
        print('RELATIVE_HELDOUT',run['kind'],run['seed'],round(entry['heldout_window_mse'],6),flush=True)
    comparison={};episodes=test_data['episode']
    baseline={r['seed']:result for r,result in bykind['probe_0']}
    absolute={r['seed']:result for r,result in bykind['probe_abs']}
    for kind in KINDS:
        if kind=='probe_0':continue
        rows=[]
        for run,result in bykind[kind]:
            base=baseline[run['seed']];abs_result=absolute[run['seed']]
            rows.append({'seed':run['seed'],'paired_mse_gain':base['mse']-result['mse'],
                         'paired_rms_gain':base['rms']-result['rms'],
                         'paired_mse_vs_abs':abs_result['mse']-result['mse'],
                         'rms':result['total_rms_m'],'base_rms':base['total_rms_m'],
                         'abs_rms':abs_result['total_rms_m'],'mse':result['total_mse']})
        gain=np.mean([r['paired_mse_gain'] for r in rows],axis=0)
        rms_gain=np.mean([r['paired_rms_gain'] for r in rows],axis=0)
        abs_gain=np.mean([r['paired_mse_vs_abs'] for r in rows],axis=0)
        ci,_=episode_cluster_ci(gain,episodes,seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])
        ci_abs,_=episode_cluster_ci(abs_gain,episodes,seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])
        base_rms=np.mean([r['base_rms'] for r in rows]);rms=np.mean([r['rms'] for r in rows]);abs_rms=np.mean([r['abs_rms'] for r in rows])
        delta_r2=float(np.mean([r['paired_mse_gain'].sum()*16/sst for r in rows]))
        terrain={}
        for t in sorted(set(test_data['terrain'])):
            ix=np.flatnonzero(test_data['terrain']==t);bins=np.minimum((np.asarray(test_raw['quality_abs'])[ix]*4).astype(int),3)
            supported=all(int(np.sum(bins==b))>=s['terrain_q_min_support'] for b in range(4))
            terrain[t]={'count':len(ix),'q_bin_counts':{str(b):int(np.sum(bins==b)) for b in range(4)},
                        'status':'supported' if supported else 'insufficient support',
                        'paired_mse_gain':float(gain[ix].mean()) if supported else None}
        comparison[kind]={'heldout_mse_gain_over_probe_0':float(gain.mean()),'gain_ci95_episode':ci,
                          'gain_over_abs_ci95_episode':ci_abs,
                          'physical_rms_gain_over_probe_0':float(base_rms-rms),
                          'relative_physical_rms_gain':float((base_rms-rms)/base_rms),
                          'relative_physical_rms_gain_vs_abs':float((abs_rms-rms)/abs_rms),
                          'delta_r2':delta_r2,'by_terrain':terrain,
                          'validation_better_than_mean':all(r['best_validation_loss']<summary['mean_trajectory_validation_mse']*(1-s['min_probe_validation_gain_over_mean']) for r in summary['runs'] if r['kind']==kind),
                          'validation_improving_at_cap':any(r['validation_improving_at_cap'] for r in summary['runs'] if r['kind']==kind)}
    result={'heldout_window_count':len(test_target),'mean_trajectory_test_mse':mean_test,
            'mean_trajectory_val_mse':summary['mean_trajectory_validation_mse'],
            'runs':runs,'comparison':comparison,'heldout_used_for_checkpoint_selection':False,
            'posthoc_future_quality_input':True,'action_normalization':'frozen train-only EAT checkpoint statistics'}
    write(OUT/'relative-quality-probe-results.json',result)
    print('RELATIVE_EVALUATION_COMPLETE',json.dumps({k:{m:v[m] for m in ('relative_physical_rms_gain','delta_r2')} for k,v in comparison.items()}),flush=True)


def permutation():
    s,mean,std,device=setup();test_raw,test_data=load('test',mean,std,device)
    summary=json.loads((OUT/'relative-quality-probe-training-summary.json').read_text())
    evaluated=json.loads((OUT/'relative-quality-probe-results.json').read_text())
    terrain=np.asarray(test_raw['terrain']);target=test_data['action']
    output={};rng=np.random.default_rng(s['permutation_seed'])
    for kind in KINDS[1:]:
        runs=[r for r in summary['runs'] if r['kind']==kind]
        observed=evaluated['comparison'][kind]['heldout_mse_gain_over_probe_0']
        comparison=evaluated['comparison'][kind]
        other_conditions=(comparison['gain_ci95_episode'][0]>0 and comparison['relative_physical_rms_gain']>=s['min_relative_rms_gain']
                          and comparison['delta_r2']>=s['min_delta_r2'])
        near=(comparison['relative_physical_rms_gain']>=s['min_relative_rms_gain']-s['permutation_boundary_relative_rms_margin']
              or comparison['delta_r2']>=s['min_delta_r2']-s['permutation_boundary_r2_margin'])
        count=s['final_permutations'] if other_conditions or near else s['initial_permutations']
        original=quality_tensor(test_raw,kind,device);shuffled_effect=[]
        for b in range(count):
            perm=np.arange(len(terrain))
            for t in np.unique(terrain):
                ix=np.flatnonzero(terrain==t);perm[ix]=rng.permutation(ix)
            q=original[torch.from_numpy(perm).to(device)]
            shuffled_mse=[]
            for run in runs:
                pred=predict(run['checkpoint'],test_data,q,s['probe_batch_size'])
                shuffled_mse.append(float((pred-target).square().mean()))
            shuffled_effect.append(float(evaluated['mean_trajectory_test_mse']-np.mean(shuffled_mse)))
        # Compare model Quality effect on its own fixed DEM/goal/target, not mean-trajectory performance.
        original_mse=float(np.mean([r['heldout_window_mse'] for r in evaluated['runs'] if r['kind']==kind]))
        shuffled_mse=[float(evaluated['mean_trajectory_test_mse']-x) for x in shuffled_effect]
        comparison_result=permutation_comparison(original_mse,shuffled_mse)
        output[kind]={'shuffle_count':count,**comparison_result,
                      'terrain_preserved':True,'dem_goal_teacher_episode_fixed':True,
                      'expanded_100':count==s['final_permutations']}
        print('RELATIVE_PERMUTATION',kind,count,output[kind]['mean_observed_minus_shuffle_gain'],flush=True)
    write(OUT/'relative-quality-permutation.json',output)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['smoke','train','evaluate','permutation'],required=True)
    a=p.parse_args();{'smoke':lambda:train(True),'train':train,'evaluate':evaluate,'permutation':permutation}[a.stage]()
if __name__=='__main__':main()
