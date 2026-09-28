"""Paired-noise Quality sweep and target selection for scratch overfit models."""
from __future__ import annotations
import argparse
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import torch
from visualnav_transformer.eat_nav.config import load_eat_config, assert_checkpoint_compatible
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.quality_gate_config import load_gate_settings, decide_gate_b
from visualnav_transformer.eat_nav.quality_gate_overfit import prepare_overfit_pairs,make_scratch_policy,pair_effect_metrics


def rms(a,b):
    return np.sqrt(np.square(a-b).sum(axis=-1).mean(axis=-1))


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','overfit-config','matched-pairs','runs-summary','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    config=load_eat_config(a.quality_config);settings=load_gate_settings(a.overfit_config)
    pair_report=json.loads(a.matched_pairs.read_text())
    runs=json.loads(a.runs_summary.read_text())['runs']
    train=EATDataset(a.manifest,'train',config)
    windows=[train[i] for i in range(len(train))]
    ids=[(row['low_index'],row['high_index']) for row in pair_report['selected_pairs']]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    batch=prepare_overfit_pairs(windows,ids,device)
    dem=batch['dem'][0::2];goal=batch['goal'][0::2]
    qlow=batch['q'][0::2];qhigh=batch['q'][1::2]
    targetlow=torch.cumsum(batch['action'][0::2],dim=1).cpu().numpy()
    targethigh=torch.cumsum(batch['action'][1::2],dim=1).cpu().numpy()
    q_values=[.1,.3,.5,.7,.9,.95]
    results=[];sweep_results=[]
    for run in runs:
        seed=run['training_seed']
        model=make_scratch_policy(config,train.stats,seed=seed,device=device)
        checkpoint=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
        assert_checkpoint_compatible(config,checkpoint['config_metadata'])
        if checkpoint['seed']!=seed or checkpoint['epoch']!=run['epochs_ran']:
            raise ValueError('overfit checkpoint seed/epoch mismatch')
        model.load_state_dict(checkpoint['state_dict'],strict=True)
        model.eval()
        low_out=[];high_out=[];mid_out=[];sweep=[]
        for diffusion_seed in settings.diffusion_seeds:
            generator=torch.Generator(device=device).manual_seed(diffusion_seed)
            initial=torch.randn((len(dem),config.len_traj_pred,2),device=device,generator=generator)
            with torch.no_grad():
                low=model.sample(dem,goal,qlow,seed=diffusion_seed,initial_noise=initial).cpu().numpy()
                high=model.sample(dem,goal,qhigh,seed=diffusion_seed,initial_noise=initial).cpu().numpy()
                outputs=[]
                for q in q_values:
                    quality=torch.full((len(dem),1),q,device=device)
                    outputs.append(model.sample(dem,goal,quality,seed=diffusion_seed,
                                                initial_noise=initial).cpu().numpy())
            low_out.append(low);high_out.append(high)
            sweep.append(np.stack(outputs,axis=0));mid_out.append(outputs[2])
        low_out=np.stack(low_out);high_out=np.stack(high_out)
        sweep=np.stack(sweep)
        effect=pair_effect_metrics(low_out.reshape(-1,config.len_traj_pred,2),high_out.reshape(-1,config.len_traj_pred,2),
                                   np.tile(targetlow,(len(settings.diffusion_seeds),1,1)),
                                   np.tile(targethigh,(len(settings.diffusion_seeds),1,1)))
        stochastic=np.concatenate([rms(mid_out[i],mid_out[j]) for i,j in combinations(range(len(mid_out)),2)])
        sampling=float(np.median(stochastic))
        effect['sampling_variation_m']=sampling
        effect['effect_sampling_ratio']=effect['generated_effect_m']/max(sampling,1e-12)
        effect['training_seed']=seed
        effect['training_converged']=run['training_converged']
        effect['fixed_bank_loss_final']=run['final_fixed_bank_loss']
        effect['checkpoint']=run['checkpoint']
        results.append(effect)
        progress=sweep[:,:,:,-1,0]
        rho=[];monotonic=[]
        for noise_i in range(progress.shape[0]):
            for pair_i in range(progress.shape[2]):
                values=progress[noise_i,:,pair_i]
                correlation=spearmanr(q_values,values).statistic if np.std(values)>1e-8 else 0.0
                rho.append(float(correlation) if np.isfinite(correlation) else 0.0)
                dif=np.diff(values)
                monotonic.append(bool(np.all(dif>=0) or np.all(dif<=0)))
        sweep_results.append({'training_seed':seed,'q_values':q_values,
                              'forward_progress_median_m':[float(np.median(progress[:,j])) for j in range(len(q_values))],
                              'adjacent_trajectory_rms_m':[float(np.median(rms(sweep[:,j],sweep[:,j+1])))
                                                            for j in range(len(q_values)-1)],
                              'median_forward_spearman':float(np.median(rho)),
                              'monotonic_forward_fraction':float(np.mean(monotonic))})
        print('OVERFIT_EVALUATED',seed,effect['effect_fraction'],effect['correct_selection_fraction'],
              effect['effect_sampling_ratio'],flush=True)
    merged={name:float(np.median([row[name] for row in results])) for name in
            ('effect_fraction','correct_selection_fraction','direction_fraction','effect_sampling_ratio',
             'low_to_low_m','low_to_high_m','high_to_high_m','high_to_low_m',
             'generated_effect_m','target_effect_m')}
    per_seed_pass=[row['effect_fraction']>=settings.min_b_effect_fraction
                   and row['correct_selection_fraction']>=settings.min_b_selection_fraction
                   and row['direction_fraction']>=settings.min_b_direction_fraction
                   and row['effect_sampling_ratio']>settings.min_b_sampling_ratio for row in results]
    decision={'effect_fraction':merged['effect_fraction'],
              'correct_selection_fraction':merged['correct_selection_fraction'],
              'direction_fraction':merged['direction_fraction'],
              'effect_sampling_ratio':merged['effect_sampling_ratio'],
              'low_target_closer':merged['low_to_low_m']<merged['low_to_high_m'],
              'high_target_closer':merged['high_to_high_m']<merged['high_to_low_m'],
              'seed_robust':sum(per_seed_pass)>=2,
              'training_converged':all(row['training_converged'] for row in results)}
    status=decide_gate_b(decision,settings)
    report={'status':status,'decision_metrics':decision,'median_metrics':merged,
            'per_seed':results,'per_seed_full_criteria_pass':per_seed_pass,
            'q_sweep':sweep_results,'pair_count':len(ids),'diffusion_seeds':list(settings.diffusion_seeds),
            'scratch_training':True,'target_source_split':'train'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print('GATE_B_DONE',json.dumps({'status':status,'decision':decision}),flush=True)

if __name__=='__main__':main()
