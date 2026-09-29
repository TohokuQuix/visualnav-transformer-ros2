"""Train-only conditional teacher trajectory neighborhood audit."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import yaml
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.relative_quality_matching import goal_direction_delta_deg
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from eat_nav_phase_c_coverage import atomic,digest,write_marker,marker_valid,require_train_only


def python_index(value):
    return int(value)


def residual(path):
    x=np.asarray(path,dtype=float)
    return x-(np.arange(1,len(x)+1)/len(x))[:,None]*x[-1]


def matching_neighbor(a,b,cfg):
    if a['terrain_type']!=b['terrain_type'] or tuple(a['episode_key'])==tuple(b['episode_key']):return None
    goal_a=np.asarray(a['goal']);goal_b=np.asarray(b['goal'])
    distance=float(np.linalg.norm(goal_a-goal_b))
    if distance>cfg['goal_distance_m']:return None
    bearing=goal_direction_delta_deg(goal_a,goal_b)
    if bearing>cfg['goal_direction_deg']:return None
    dem_a=np.asarray(a['dem']);dem_b=np.asarray(b['dem'])
    mask_a=dem_a[:,1]>.5;mask_b=dem_b[:,1]>.5
    common=mask_a&mask_b;union=mask_a|mask_b
    iou=float(common.sum()/max(int(union.sum()),1))
    if iou<cfg['min_mask_iou'] or not common.any():return None
    heights_a=dem_a[:,0][common];heights_b=dem_b[:,0][common]
    rmse=float(np.sqrt(np.mean((heights_a-heights_b)**2)))
    if not np.isfinite(rmse) or rmse>cfg['max_dem_rmse_m']:return None
    roughness=abs(float(np.std(heights_a))-float(np.std(heights_b)))
    if roughness>cfg['roughness_max_abs_difference_m']:return None
    return {'goal_distance_m':distance,'bearing_difference_deg':bearing,'mask_iou':iou,'height_rmse_m':rmse,'roughness_difference_m':roughness}


def path_features(path):
    x=np.asarray(path,dtype=float)
    increments=np.diff(np.vstack([np.zeros((1,2)),x]),axis=0)
    headings=np.unwrap(np.arctan2(increments[:,1],increments[:,0]))
    r=residual(x)
    return {'endpoint_m':x[-1].tolist(),'endpoint_heading_rad':float(headings[-1]),
            'length_m':float(np.linalg.norm(increments,axis=1).sum()),
            'curvature_rad':float(np.abs(np.diff(headings)).sum()),
            'lateral_m':float(x[-1,1]),'forward_m':float(np.clip(increments[:,0],0,None).sum()),
            'reverse_m':float(np.clip(-increments[:,0],0,None).sum()),
            'residual_rms_m':float(np.sqrt(np.mean(r*r)))}


def clusters_pairwise(points, threshold):
    x=np.asarray(points,dtype=float)
    parent=list(range(len(x)))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if np.sqrt(np.mean((x[i]-x[j])**2))<threshold:parent[find(j)]=find(i)
    return len({find(i) for i in range(len(x))})


def command_clusters(commands,linear_threshold,angular_threshold):
    x=np.asarray(commands,dtype=float)
    parent=list(range(len(x)))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if abs(x[i,0]-x[j,0])<linear_threshold and abs(x[i,2]-x[j,2])<angular_threshold:
                parent[find(j)]=find(i)
    return len({find(i) for i in range(len(x))})


def follower_commands(paths,client):
    commands=[]
    for first in range(0,len(paths),8):
        batch=list(paths[first:first+8])
        batch += [batch[-1]]*(8-len(batch))
        ros_paths=[[[0.,0.]]+np.asarray(x,dtype=float).tolist() for x in batch]
        poses=[[0.,0.,0.]]*8
        client.request('reset',poses=poses,paths=ros_paths)
        steps=[]
        for _ in range(3):
            response=client.request('step',poses=poses,timeout_sec=5.0)
            if not response['fresh']:raise RuntimeError('stale follower command')
            steps.append(response['commands'])
        commands.extend(np.median(np.asarray(steps),axis=0)[:min(8,len(paths)-first)].tolist())
    return commands


def audit(args):
    spec=yaml.safe_load(args.coverage_config.read_text())
    out=args.output
    frozen=json.loads((out/'source-provenance.json').read_text())
    if frozen['input_sha256']['coverage_config']!=digest(args.coverage_config):raise ValueError('config changed after freeze')
    source_hash=digest(Path(__file__))
    stage_input=hashlib.sha256((digest(out/'diagnostic-state-ids.json')+source_hash+digest(args.coverage_config)).encode()).hexdigest()
    if marker_valid(out/'teacher-audit.done.json',stage_input):print('STAGE1_ALREADY_DONE',flush=True);return
    if (out/'teacher-audit.done.json').exists():raise ValueError('invalid stage1 marker; investigate before resume')
    cfg=load_eat_config(args.reference_config)
    train=CompleteOnlyView(EATDataset(args.manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(train)!=spec['reference']['complete_train_windows']:raise ValueError('train count mismatch')
    match=spec['matching']; require_train_only({'split':match['split'].replace('complete_only_','')})
    by_terrain=defaultdict(list)
    for i in range(len(train)):
        w=train.dataset.windows[train.indices[i]]
        by_terrain[w['terrain_type']].append(i)
    trees={t:(np.asarray(ids),cKDTree(np.stack([train.dataset.windows[train.indices[i]]['goal'] for i in ids])))
           for t,ids in by_terrain.items()}
    anchor_ids=[]
    per_terrain=max(1,match['anchor_count']//len(trees))
    for t,(ids,_) in sorted(trees.items()):
        ranks=sorted(ids,key=lambda i:hashlib.sha256(f'{t}:{train.identities[i]}'.encode()).digest())
        anchor_ids.extend(ranks[:per_terrain])
    anchor_ids=anchor_ids[:match['anchor_count']]
    neighborhoods=[]
    for count,anchor in enumerate(anchor_ids):
        anchor=python_index(anchor)
        a=train.dataset.windows[train.indices[anchor]]
        ids,tree=trees[a['terrain_type']]
        nearby=tree.query_ball_point(a['goal'],match['goal_distance_m'])
        candidates=[]
        for j in nearby:
            other=int(ids[j])
            if other==anchor:continue
            b=train.dataset.windows[train.indices[other]]
            met=matching_neighbor(a,b,match)
            if met is not None:candidates.append((met['goal_distance_m'],other,met))
        candidates.sort(key=lambda r:(r[0],r[1]))
        neighbors=candidates[:match['max_neighbors_per_anchor']-1]
        if len(neighbors)+1<match['min_neighborhood_size']:continue
        members=[anchor]+[n[1] for n in neighbors]
        if len(set(train.identities[i] for i in members))!=len(members):raise ValueError('duplicate window')
        windows=[train.dataset.windows[train.indices[i]] for i in members]
        if len({tuple(w['episode_key']) for w in windows})<2:raise ValueError('same-episode neighborhood')
        paths=np.stack([np.cumsum(w['action'],axis=0) for w in windows])
        if not np.isfinite(paths).all():raise ValueError('nonfinite teacher path')
        neighborhoods.append({'anchor_view_index':anchor,'member_view_indices':members,
            'terrain_type':a['terrain_type'],'episode_keys':[list(w['episode_key']) for w in windows],
            'quality_raw':[float(w['quality_raw']) for w in windows],
            'matches':[n[2] for n in neighbors],
            'path_features':[path_features(p) for p in paths],
            'path_cluster_count':clusters_pairwise(paths,spec['clusters']['path_rms_m'])})
        if count%100==0:print('MATCHED_ANCHORS',count,len(neighborhoods),flush=True)
    print('NEIGHBORHOODS',len(neighborhoods),'anchors',len(anchor_ids),flush=True)
    client=FollowerBridgeClient(timeout=45.0)
    try:
        for n,row in enumerate(neighborhoods):
            progress=out/'teacher-command-progress'/f'{n:04d}.json'
            if progress.exists():
                cmd=json.loads(progress.read_text())['commands']
            else:
                paths=[np.cumsum(train.dataset.windows[train.indices[i]]['action'],axis=0) for i in row['member_view_indices']]
                cmd=follower_commands(paths,client)
                atomic(progress,{'commands':cmd,'anchor_view_index':row['anchor_view_index']})
            row['follower_commands']=cmd
            row['command_cluster_count']=command_clusters(cmd,spec['clusters']['command_linear_mps'],spec['clusters']['command_angular_radps'])
            if n%10==0:print('FOLLOWER_NEIGHBORHOOD',n+1,len(neighborhoods),flush=True)
    finally:client.close()
    target=out/'train-neighborhoods.jsonl.gz'
    temp=target.with_name(target.name+'.tmp')
    with gzip.open(temp,'wt') as stream:
        for row in neighborhoods:stream.write(json.dumps(row,sort_keys=True)+'\n')
    temp.replace(target)
    counts=Counter(row['command_cluster_count'] for row in neighborhoods)
    summary={'status':'complete','anchor_count':len(anchor_ids),'neighborhood_count':len(neighborhoods),
        'window_count_by_neighborhood':[len(r['member_view_indices']) for r in neighborhoods],
        'command_mode_count_histogram':dict(counts),
        'fraction_two_or_more_command_modes':sum(r['command_cluster_count']>=2 for r in neighborhoods)/max(len(neighborhoods),1),
        'terrain':{t:{'count':sum(r['terrain_type']==t for r in neighborhoods),
                      'multimodal':sum(r['terrain_type']==t and r['command_cluster_count']>=2 for r in neighborhoods)} for t in sorted(trees)},
        'data_multimodal_gate_threshold':match['data_multimodal_min_fraction'],
        'no_neighborhoods':not bool(neighborhoods),'source_split':'complete_only_train'}
    summary['classification']='insufficient_matching_support' if not neighborhoods else ('data_multimodal' if summary['fraction_two_or_more_command_modes']>=match['data_multimodal_min_fraction'] else 'data_unimodal')
    atomic(out/'teacher-multimodality.json',summary)
    write_marker(out/'teacher-audit.done.json',[target,out/'teacher-multimodality.json'],stage_input)


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--reference-config',type=Path,required=True);p.add_argument('--coverage-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
if __name__=='__main__':main()
