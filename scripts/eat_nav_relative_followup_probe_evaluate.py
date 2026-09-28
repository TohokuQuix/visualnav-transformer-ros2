"""One-time held-out paired evaluation of frozen reference and new Probes."""
from __future__ import annotations
from collections import defaultdict
import gzip
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.relative_quality_probe import tensorize,quality_tensor,predict
from visualnav_transformer.eat_nav.relative_quality_followup_probe import followup_quality_tensor
from visualnav_transformer.eat_nav.relative_quality_followup_train import predict_followup
from visualnav_transformer.eat_nav.quality_gate_evaluation import episode_cluster_ci,trajectory_errors
from eat_nav_relative_followup_probe_train import sha,write,OLD,ROOT,OUT,SPEC,CKPT


def main():
    s=yaml.safe_load(SPEC.read_text());torch.set_num_threads(4)
    checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
    mean=np.asarray(checkpoint['action_statistics']['mean'],dtype=np.float32)
    std=np.asarray(checkpoint['action_statistics']['std'],dtype=np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    raw_file=OLD/'splits/test.npz'
    if sha(raw_file)!=s['source_aligned_split_sha256']['test']:raise ValueError('heldout feature SHA mismatch')
    raw=np.load(raw_file);data=tensorize(raw,mean,std,device)
    if len(data['dem'])!=s['complete_aligned']['test']:raise ValueError('heldout window count mismatch')
    prior=json.loads((OLD/'output/relative-quality-probe-training-summary.json').read_text())
    current=json.loads((OUT/'probe-ablation-training-summary.json').read_text())
    if not current['completed'] or current['heldout_used']:raise ValueError('training stage incomplete or heldout used')
    if current['input_fingerprint']['config']!=sha(SPEC):raise ValueError('training config SHA mismatch')
    if current['prior_reference_fingerprint']!=prior['input_fingerprint']:
        raise ValueError('reference Probe fingerprint mismatch')
    allruns=prior['runs']+current['runs']
    if len(allruns)!=36:raise ValueError('expected 15 reference plus 21 new Probe runs')
    expected=set(s['probe_kinds'])|set(s['reference_kinds'])
    if {r['kind'] for r in allruns}!=expected:raise ValueError('Probe kind mismatch')
    target=data['action'].detach().cpu().numpy();mean_train=np.asarray(prior['mean_trajectory_train_action'])
    sst=float(np.square(target-mean_train).sum());mean_test=float(np.square(target-mean_train).mean())
    out_runs=[];perwindow={};bykind=defaultdict(dict)
    for run in allruns:
        if sha(run['checkpoint'])!=run['checkpoint_sha256']:raise ValueError('checkpoint SHA mismatch')
        if run['kind'] in s['probe_kinds']:
            q=followup_quality_tensor(raw,run['kind'],device)
            prediction=predict_followup(run['checkpoint'],data,q,s['probe_batch_size'])
        else:
            q=quality_tensor(raw,run['kind'],device)
            prediction=predict(run['checkpoint'],data,q,s['probe_batch_size'])
        pred=prediction.detach().cpu().numpy();mse=np.square(pred-target).mean(axis=(1,2))
        physical=trajectory_errors(pred*std+mean,target*std+mean,np.ones((len(pred),8),dtype=bool))
        key=f"{run['kind']}-seed{run['seed']}";perwindow[key+'_mse']=mse.astype(np.float32)
        perwindow[key+'_rms']=physical['rms_m'].astype(np.float32)
        entry={'kind':run['kind'],'seed':run['seed'],'best_epoch':run['best_epoch'],
               'validation_loss':run['best_validation_loss'],'heldout_window_mse':float(mse.mean()),
               'heldout_physical_rms_m':float(physical['rms_m'].mean()),
               'heldout_r2':float(1-mse.sum()*16/sst),'parameter_count':run['parameter_count'],
               'underfit_cap':run['validation_improving_at_cap'],
               'checkpoint_sha256':run['checkpoint_sha256']}
        out_runs.append(entry);bykind[run['kind']][run['seed']]=entry
        print('FOLLOWUP_HELDOUT',key,round(entry['heldout_window_mse'],6),flush=True)
    np.savez_compressed(ROOT/'probe-ablation-window-errors.npz',**perwindow)
    episodes=[tuple(e) for e in raw['episode']];terrain=np.asarray(raw['terrain']);qabs=np.asarray(raw['quality_abs'])
    tilt=set()
    with gzip.open(OLD/'output/relative-quality-values.jsonl.gz','rt') as source:
        for line in source:
            row=json.loads(line)
            if row['split']=='test' and row['primary_available'] and 'tilt_over' in row['episode_terminal_reason']:
                tilt.add(tuple(row['episode_key']))
    unique=list(dict.fromkeys(episodes))
    comparison={}
    for kind in s['probe_kinds']:
        gain=[];rms_gain=[];against={ref:[] for ref in ('probe_abs','probe_adv','probe_drawdown','probe_multi')}
        for seed in s['probe_seeds']:
            base=perwindow[f'probe_0-seed{seed}_mse'];own=perwindow[f'{kind}-seed{seed}_mse']
            gain.append(base-own)
            rms_gain.append(perwindow[f'probe_0-seed{seed}_rms']-perwindow[f'{kind}-seed{seed}_rms'])
            for ref in against:against[ref].append(perwindow[f'{ref}-seed{seed}_mse']-own)
        gain=np.mean(gain,axis=0);rms_gain=np.mean(rms_gain,axis=0)
        ci,_=episode_cluster_ci(gain,episodes,seed=s['bootstrap_seed'],repeats=s['bootstrap_repeats'])
        ref_ci={ref:episode_cluster_ci(np.mean(rows,axis=0),episodes,seed=s['bootstrap_seed'],
                                      repeats=s['bootstrap_repeats'])[0] for ref,rows in against.items()}
        new_ref={'probe_delta_drawdown':'probe_delta','probe_relative_triplet':'probe_delta_drawdown',
                 'probe_multi_reward_free':'probe_relative_triplet','probe_multi_no_v0':'probe_adv'}
        if kind in new_ref:
            ref=new_ref[kind]
            paired=np.mean([perwindow[f'{ref}-seed{seed}_mse']-perwindow[f'{kind}-seed{seed}_mse']
                            for seed in s['probe_seeds']],axis=0)
            ref_ci[ref]=episode_cluster_ci(paired,episodes,seed=s['bootstrap_seed'],
                                            repeats=s['bootstrap_repeats'])[0]
        base_rms=np.mean([bykind['probe_0'][seed]['heldout_physical_rms_m'] for seed in s['probe_seeds']])
        own_rms=np.mean([bykind[kind][seed]['heldout_physical_rms_m'] for seed in s['probe_seeds']])
        byterrain={}
        for t in sorted(set(terrain)):
            ix=np.flatnonzero(terrain==t);bins=np.minimum((qabs[ix]*4).astype(int),3)
            supported=all(int(np.sum(bins==i))>=s['min_terrain_q_bin'] for i in range(4))
            byterrain[t]={'count':len(ix),'q_bin_counts':{str(i):int(np.sum(bins==i)) for i in range(4)},
                          'status':'supported' if supported else 'insufficient support',
                          'paired_mse_gain':float(gain[ix].mean()) if supported else None}
        leave=[float(gain[np.array([e!=key for e in episodes])].mean()) for key in unique]
        non_tilt=np.array([e not in tilt for e in episodes])
        comparison[kind]={'heldout_mse_gain_over_probe_0':float(gain.mean()),
          'gain_ci95_episode':ci,'gain_over_reference_ci95_episode':ref_ci,
          'relative_physical_rms_gain':float((base_rms-own_rms)/base_rms),
          'delta_r2':float(gain.sum()*16/sst),
          'validation_better_than_mean':all(bykind[kind][seed]['validation_loss']<
                    current['mean_trajectory_validation_mse']*(1-s['min_validation_gain_over_mean']) for seed in s['probe_seeds']),
          'validation_improving_at_cap':any(bykind[kind][seed]['underfit_cap'] for seed in s['probe_seeds']),
          'by_terrain':byterrain,'minimum_leave_one_episode_out_gain':float(min(leave)),
          'tilt_excluded_gain':float(gain[non_tilt].mean()),
          'per_seed_mse_gain':{str(seed):float((perwindow[f'probe_0-seed{seed}_mse']-
                                              perwindow[f'{kind}-seed{seed}_mse']).mean()) for seed in s['probe_seeds']}}
    result={'heldout_window_count':len(target),'heldout_evaluated_once':True,
            'heldout_used_for_checkpoint_selection':False,'train_mean_heldout_mse':mean_test,
            'runs':out_runs,'comparison':comparison,'common_support_terrains':
              sorted(set.intersection(*(set(json.loads((OLD/'output/relative-quality-summary.json').read_text())[split]['by_terrain'])
                                       for split in ('train','val','test')))),
            'posthoc_quality_input':True,'action_normalization':'frozen train-only EAT checkpoint statistics'}
    write(OUT/'probe-ablation-results.json',result)
    print('FOLLOWUP_EVALUATION_COMPLETE',flush=True)


if __name__=='__main__':main()
