"""Train-only natural-history candidate audit; frozen states and diffusion seeds."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView

SEEDS = tuple(range(1001, 1009))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');tmp.replace(path)

def select_states(view, count=32):
    # Stable hash ranking, one state per episode when support allows.
    by_terrain={}
    for view_index,index in enumerate(view.indices):
        w=view.dataset.windows[index]
        key=tuple(w['episode_key'])
        rank=int(hashlib.sha256((repr(key)+':'+str(w['step'])).encode()).hexdigest(),16)
        by_terrain.setdefault(w['terrain_type'],{}).setdefault(key,[]).append((rank,view_index,w))
    pools={terrain:sorted((min(rows,key=lambda r:r[0]) for rows in episodes.values()),key=lambda r:r[0])
           for terrain,episodes in by_terrain.items()}
    selected=[]
    terrains=sorted(pools)
    while len(selected)<count:
        changed=False
        for terrain in terrains:
            if pools[terrain] and len(selected)<count:
                _,view_index,w=pools[terrain].pop(0)
                selected.append({'view_index':view_index,'terrain_type':terrain,'terrain_seed':int(w['terrain_seed']),
                                 'episode_key':list(w['episode_key']),'step':int(w['step']),
                                 'goal':w['goal'].astype(float).tolist()})
                changed=True
        if not changed:break
    if len(selected)!=count:raise ValueError(f'only {len(selected)} unique-episode states')
    assert len({tuple(x['episode_key']) for x in selected})==count
    return selected

def pairwise_metrics(paths):
    x=np.asarray(paths,dtype=float)
    delta=x[:,None]-x[None,:]
    ix=np.triu_indices(len(x),1)
    rms=np.sqrt(np.mean(delta**2,axis=(2,3)))[ix]
    endpoints=x[:,-1]
    endpoint_dist=np.linalg.norm(endpoints[:,None]-endpoints[None,:],axis=-1)[ix]
    diff=np.diff(np.concatenate((np.zeros((len(x),1,2)),x),axis=1),axis=1)
    heading=np.arctan2(diff[:,-1,1],diff[:,-1,0])
    heading_diff=np.abs(np.arctan2(np.sin(heading[:,None]-heading[None,:]),np.cos(heading[:,None]-heading[None,:])))[ix]
    length=np.linalg.norm(diff,axis=-1).sum(axis=1)
    curvature=np.diff(np.unwrap(np.arctan2(diff[:,:,1],diff[:,:,0]),axis=1),axis=1)
    curvature=np.abs(curvature).sum(axis=1)
    lateral=x[:,-1,1]
    forward=np.clip(diff[:,:,0],0,None).sum(axis=1)
    reverse=np.clip(-diff[:,:,0],0,None).sum(axis=1)
    return {'trajectory_rms':rms.tolist(),'endpoint_distance_m':endpoint_dist.tolist(),
            'endpoint_heading_difference_rad':heading_diff.tolist(),
            'path_length_difference_m':np.abs(length[:,None]-length[None,:])[ix].tolist(),
            'curvature_difference_rad':np.abs(curvature[:,None]-curvature[None,:])[ix].tolist(),
            'lateral_displacement_difference_m':np.abs(lateral[:,None]-lateral[None,:])[ix].tolist(),
            'forward_m':forward.tolist(),'reverse_m':reverse.tolist()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['select','generate'],required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--reference',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--follower-config',type=Path,default=Path('/workspace/onix_dep/onix_automation/teaching_playback/simple_trajectory_follower/config/onix.yaml'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    config=load_eat_config(a.config)
    dataset=EATDataset(a.manifest,'train',config,split_seed=123)
    view=CompleteOnlyView(dataset,config.len_traj_pred)
    if len(view)!=49598:raise ValueError('train window count mismatch')
    states_path=a.output/'candidate-states.json'
    if a.stage=='select':
        if states_path.exists():raise FileExistsError(states_path)
        states=select_states(view)
        atomic(states_path,{'source_manifest_sha256':sha(a.manifest),'config_sha256':sha(a.config),
                            'split':'train','state_count':len(states),'states':states,'diffusion_seeds':SEEDS})
        actions=np.stack([view[i]['action'] for i in range(len(view))]).astype(np.float32)
        mean=actions.reshape(-1,2).mean(axis=0);std=actions.reshape(-1,2).std(axis=0)
        if not np.allclose(mean,[0.028116857632994652,-4.290757715352811e-05],atol=1e-6):
            raise ValueError('train action normalization differs from frozen audit')
        normalized=torch.from_numpy(((actions-mean)/std).reshape(len(actions),-1)).to('cuda' if torch.cuda.is_available() else 'cpu')
        query=normalized[:256];pool=normalized[256:]
        nearest=torch.cat([torch.cdist(query[i:i+32],pool).min(dim=1).values.cpu() for i in range(0,256,32)]).numpy()/np.sqrt(16)
        follower=yaml.safe_load(a.follower_config.read_text())['/**']['ros__parameters']
        thresholds={'source_split':'complete_only_train','follower_config_sha256':sha(a.follower_config),
                    'follower_lookahead_m':float(follower['lookahead_distance']),
                    'follower_goal_tolerance_m':float(follower['goal_tolerance']),
                    'follower_turn_in_place_heading_threshold_rad':float(follower['turn_in_place_heading_threshold']),
                    'path_rms_min_m':0.05,'endpoint_min_m':0.05,
                    'command_linear_min_mps':0.03,'command_angular_min_radps':0.05,
                    'two_second_min_yaw_rad':0.1,'two_second_min_lateral_m':0.03,
                    'train_ood_nearest_l2_per_dimension_p95':float(np.quantile(nearest,0.95)),
                    'train_ood_calibration_query_count':256,'train_action_mean':mean.tolist(),'train_action_std':std.tolist()}
        atomic(a.output/'candidate-thresholds.json',thresholds)
        print('FIXED_STATES',sha(states_path),len(states),'THRESHOLDS',sha(a.output/'candidate-thresholds.json'),flush=True);return
    frozen=json.loads(states_path.read_text())
    if frozen['source_manifest_sha256']!=sha(a.manifest) or frozen['config_sha256']!=sha(a.config):
        raise ValueError('frozen state source mismatch')
    thresholds=json.loads((a.output/'candidate-thresholds.json').read_text())
    if thresholds['source_split']!='complete_only_train':raise ValueError('candidate thresholds must be train-only')
    model,_,_=load_checkpoint(a.reference,config)
    device='cuda' if torch.cuda.is_available() else 'cpu';model=model.to(device).eval()
    train_actions=np.stack([view[i]['action'] for i in range(len(view))]).astype(np.float32)
    mean=np.asarray(thresholds['train_action_mean'],dtype=np.float32)
    std=np.asarray(thresholds['train_action_std'],dtype=np.float32)
    train_normalized=torch.from_numpy(((train_actions-mean)/std).reshape(len(view),-1)).to(device)
    all_paths=[];rows=[]
    with torch.inference_mode():
        for row in frozen['states']:
            w=view[int(row['view_index'])]
            if not np.array_equal(w['goal'],np.asarray(row['goal'],dtype=np.float32)):
                raise ValueError('frozen state goal mismatch')
            if w['dem'].shape!=(4,2,20,20):raise ValueError('natural DEM history shape mismatch')
            dem=torch.from_numpy(w['dem'][None]).to(device)
            goal=torch.from_numpy(w['goal'][None]).to(device)
            paths=np.stack([model.sample(dem,goal,None,seed=seed)[0].cpu().numpy() for seed in SEEDS])
            all_paths.append(paths)
            metrics=pairwise_metrics(paths)
            candidate_delta=np.diff(np.concatenate((np.zeros((8,1,2),dtype=np.float32),paths),axis=1),axis=1)
            normalized=torch.from_numpy(((candidate_delta-mean)/std).reshape(8,-1)).to(device)
            ood=(torch.cdist(normalized,train_normalized).min(dim=1).values/np.sqrt(16)).cpu().numpy()
            distances=np.asarray(metrics['trajectory_rms'])
            parent=list(range(8))
            def find(i):
                while parent[i]!=i:i=parent[i]
                return i
            pair=0
            for i in range(8):
                for j in range(i+1,8):
                    if distances[pair]<thresholds['path_rms_min_m']:
                        parent[find(j)]=find(i)
                    pair+=1
            rows.append({'state':row,'pairwise':metrics,'path_cluster_count':len({find(i) for i in range(8)}),
                         'ood_nearest_l2_per_dimension':ood.astype(float).tolist(),
                         'ood_count_over_train_p95':int((ood>thresholds['train_ood_nearest_l2_per_dimension_p95']).sum())})
    path=a.output/'candidate-paths.npz'
    if path.exists():raise FileExistsError(path)
    np.savez_compressed(path,paths=np.asarray(all_paths),seeds=np.asarray(SEEDS))
    atomic(a.output/'candidate-diversity-offline.json',{'states_sha256':sha(states_path),
           'reference_sha256':sha(a.reference),'paths_sha256':sha(path),'thresholds_sha256':sha(a.output/'candidate-thresholds.json'),'rows':rows,
           'natural_history':True,'heldout_used':False})
    print('CANDIDATES_GENERATED',len(rows),sha(path),flush=True)
if __name__=='__main__':main()
