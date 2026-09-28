"""Frozen-model terrain-preserving Quality permutations for new Probes."""
from __future__ import annotations
import json
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.relative_quality_probe import tensorize,permutation_comparison
from visualnav_transformer.eat_nav.relative_quality_followup_probe import FollowupProbe,followup_quality_tensor
from visualnav_transformer.eat_nav.relative_quality_followup_permutation import terrain_permutation
from eat_nav_relative_followup_probe_train import sha,write,OLD,ROOT,OUT,SPEC,CKPT


def main():
    s=yaml.safe_load(SPEC.read_text());torch.set_num_threads(4)
    checkpoint=torch.load(CKPT,map_location='cpu',weights_only=True)
    mean=np.asarray(checkpoint['action_statistics']['mean'],dtype=np.float32)
    std=np.asarray(checkpoint['action_statistics']['std'],dtype=np.float32)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if sha(OLD/'splits/test.npz')!=s['source_aligned_split_sha256']['test']:
        raise ValueError('heldout feature SHA mismatch')
    raw=np.load(OLD/'splits/test.npz');data=tensorize(raw,mean,std,device)
    training=json.loads((OUT/'probe-ablation-training-summary.json').read_text())
    result=json.loads((OUT/'probe-ablation-results.json').read_text())
    terrain=np.asarray(raw['terrain']);target=data['action'];rng=np.random.default_rng(s['permutation_seed'])
    output={}
    for kind in s['probe_kinds']:
        comparison=result['comparison'][kind]
        other=(comparison['gain_ci95_episode'][0]>0 and
               comparison['relative_physical_rms_gain']>=s['min_relative_rms_gain'] and
               comparison['delta_r2']>=s['min_delta_r2'])
        near=(comparison['relative_physical_rms_gain']>=s['min_relative_rms_gain']-s['permutation_boundary_relative_rms_margin']
              or comparison['delta_r2']>=s['min_delta_r2']-s['permutation_boundary_r2_margin'])
        count=s['final_permutations'] if other or near else s['initial_permutations']
        q=followup_quality_tensor(raw,kind,device)
        models=[]
        for run in training['runs']:
            if run['kind']!=kind:continue
            if sha(run['checkpoint'])!=run['checkpoint_sha256']:
                raise ValueError('Probe checkpoint SHA mismatch')
            model=FollowupProbe(q.shape[1]).to(device).eval()
            saved=torch.load(run['checkpoint'],map_location='cpu',weights_only=True)
            model.load_state_dict(saved['model']);models.append(model)
        if len(models)!=len(s['probe_seeds']):raise ValueError('Probe seed count mismatch')
        shuffle_mse=[]
        with torch.no_grad():
            for index in range(count):
                perm=terrain_permutation(terrain,rng)
                quality=q[torch.from_numpy(perm).to(device)]
                losses=[]
                for model in models:
                    total=0.;n=0
                    for i in range(0,len(q),s['probe_batch_size']):
                        sl=slice(i,i+s['probe_batch_size'])
                        pred=model(data['dem'][sl],data['goal'][sl],quality[sl]);y=target[sl]
                        total+=float((pred-y).square().sum());n+=y.numel()
                    losses.append(total/n)
                shuffle_mse.append(float(np.mean(losses)))
        observed=float(np.mean([r['heldout_window_mse'] for r in result['runs'] if r['kind']==kind]))
        compared=permutation_comparison(observed,shuffle_mse)
        output[kind]={'shuffle_count':count,**compared,
                      'empirical_one_sided_p':(1+sum(x<=observed for x in shuffle_mse))/(count+1),
                      'terrain_preserved':True,'dem_goal_teacher_episode_fixed':True,
                      'expanded_100':count==s['final_permutations']}
        write(OUT/'probe-ablation-permutation.json',output)
        print('FOLLOWUP_PERMUTATION',kind,count,round(compared['mean_observed_minus_shuffle_gain'],6),flush=True)


if __name__=='__main__':main()
