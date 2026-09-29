"""Train-only PCA residual proposals with frozen constraints and coverage gate."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from eat_nav_phase_c_coverage import atomic,digest,write_marker,require_train_only
from eat_nav_phase_c_coverage_stage1 import residual
from eat_nav_phase_c_coverage_sampling import examine,coverage_decision


def fit_basis(train_paths,component_count,quantile):
    x=np.asarray(train_paths,dtype=np.float64)
    r=np.stack([residual(p) for p in x]).reshape(len(x),-1)
    centered=r-r.mean(axis=0)
    _,_,vh=np.linalg.svd(centered,full_matrices=False)
    basis=vh[:component_count].reshape(component_count,x.shape[1],2)
    basis[:,-1]=0
    coefficients=centered@vh[:component_count].T
    scale=np.quantile(np.abs(coefficients),quantile,axis=0)
    return basis,scale


def farthest_raw_index(raw):
    x=np.asarray(raw,dtype=float)
    distances=np.sqrt(np.mean((x-x[:1])**2,axis=(1,2)))
    distances[0]=-np.inf
    return int(np.argmax(distances))


def proposals(raw,basis,scale):
    x=np.asarray(raw,dtype=float);b=np.asarray(basis,dtype=float);a=np.asarray(scale,dtype=float)
    out=[x[0],x[farthest_raw_index(x)]]
    for i in range(len(b)):
        out.extend([x[0]+a[i]*b[i],x[0]-a[i]*b[i]])
    return np.stack(out)


def features(path):
    p=np.asarray(path,dtype=float)
    inc=np.diff(np.vstack((np.zeros((1,2)),p)),axis=0)
    angle=np.unwrap(np.arctan2(inc[:,1],inc[:,0]))
    return {'max_increment':float(np.linalg.norm(inc,axis=1).max()),
            'length':float(np.linalg.norm(inc,axis=1).sum()),
            'curvature':float(np.abs(np.diff(angle)).sum()),
            'forward':float(np.clip(inc[:,0],0,None).sum()),
            'reverse':float(np.clip(-inc[:,0],0,None).sum())}


def crossing(p):
    points=np.vstack((np.zeros((1,2)),p))
    def orient(a,b,c):return np.cross(b-a,c-a).item()
    for i in range(len(points)-1):
        for j in range(i+2,len(points)-1):
            if i==0 and j==len(points)-2:continue
            a,b=points[i:i+2];c,d=points[j:j+2]
            if orient(a,b,c)*orient(a,b,d)<0 and orient(c,d,a)*orient(c,d,b)<0:return True
    return False


def constraints(path,base,goal,limits,config,spec):
    f=features(path)
    issue=[]
    if not np.isfinite(path).all():issue.append('nonfinite');return issue
    if np.linalg.norm(path[-1]-base[-1])>spec['structured']['endpoint_tolerance_m']:issue.append('endpoint_changed')
    if np.linalg.norm(path[-1]-goal)>limits['goal_endpoint_p99']:issue.append('goal_endpoint')
    for metric in ('max_increment','length','curvature','forward','reverse'):
        if f[metric]>limits[metric+'_p99']:issue.append(metric)
    radius=min(config.dem_height_cells,config.dem_width_cells)*config.dem_resolution_m/2
    if np.any(np.abs(path)>radius):issue.append('dem_bounds')
    if crossing(path):issue.append('self_intersection')
    return issue


def run(args):
    root=args.output;spec=yaml.safe_load(args.coverage_config.read_text())
    sampling=json.loads((root/'sampling-coverage-decision.json').read_text())
    if sampling['decision']=='pass':
        atomic(root/'structured-proposal-results.json',{'status':'not_run','reason':'sampling_coverage_pass'})
        atomic(root/'structured-coverage-decision.json',{'decision':'not_run','reason':'sampling_coverage_pass'})
        return
    cfg=load_eat_config(args.reference_config)
    train=CompleteOnlyView(EATDataset(args.manifest,'train',cfg,split_seed=123),cfg.len_traj_pred)
    if len(train)!=49598:raise ValueError('train count mismatch')
    require_train_only({'split':'train'})
    train_paths=np.stack([np.cumsum(train[i]['action'],axis=0) for i in range(len(train))])
    basis,scale=fit_basis(train_paths,spec['structured']['components'],spec['structured']['coefficient_absolute_quantile'])
    train_features=[features(x) for x in train_paths]
    limits={key+'_p99':float(np.quantile([f[key] for f in train_features],.99)) for key in train_features[0]}
    goals=np.stack([train[i]['goal'] for i in range(len(train))])
    limits['goal_endpoint_p99']=float(np.quantile(np.linalg.norm(train_paths[:,-1]-goals,axis=1),.99))
    basis_path=root/'residual-basis.npz';temp=basis_path.with_name(basis_path.name+'.tmp')
    with temp.open('wb') as stream:np.savez_compressed(stream,basis=basis,scale=scale,limits_json=json.dumps(limits),split='train',train_count=len(train),train_source_sha256=digest(args.manifest))
    temp.replace(basis_path)
    model,_,_=load_checkpoint(args.checkpoint,cfg);model=model.to('cuda').eval()
    mean=model.action_mean.detach().cpu().numpy();std=model.action_std.detach().cpu().numpy()
    actions=np.stack([train[i]['action'] for i in range(len(train))])
    train_norm=torch.from_numpy(((actions-mean)/std).reshape(len(actions),-1).astype(np.float32)).to('cuda')
    thresholds=json.loads(args.prior_thresholds.read_text())
    ood_limit=thresholds['train_ood_nearest_l2_per_dimension_p95']
    states=json.loads((root/'diagnostic-state-ids.json').read_text())
    rows=[];client=FollowerBridgeClient(timeout=45.)
    try:
        for index,(split,state) in enumerate([('train',x) for x in states['train']]+[('val',x) for x in states['val']]):
            raw=np.load(root/'trace-states'/f'{index:03d}.npz')['paths'][:8]
            candidate=proposals(raw,basis,scale)
            assert len(candidate)==8 and np.array_equal(candidate[0],raw[0])
            goal=np.asarray(state['goal'])
            violation=[constraints(p,candidate[0],goal,limits,cfg,spec) for p in candidate]
            # Frozen fallback: invalid candidates stay counted and are never replaced.
            metrics=examine(candidate,state['terrain_type'],model,train_norm,ood_limit,spec['clusters'],client)
            metrics['invalid_candidate_count']=sum(bool(x) for x in violation)
            metrics['invalid_reasons']=violation
            metrics['split']=split;metrics['state_index']=index
            rows.append(metrics)
            atomic(root/'structured-states'/f'{index:03d}.json',metrics)
            print('STRUCTURED_STATE',index+1,len(states['train'])+len(states['val']),flush=True)
    finally:client.close()
    decision=coverage_decision(rows,spec)
    if sum(x['invalid_candidate_count'] for x in rows):decision['pass']=False;decision['constraint_failure']=True
    atomic(root/'structured-proposal-results.json',{'status':'complete','basis_sha256':digest(basis_path),
        'basis_type':'global_complete_train_residual_pca','train_count':len(train),'scale_quantile':spec['structured']['coefficient_absolute_quantile'],
        'limits':limits,'rows':rows})
    atomic(root/'structured-coverage-decision.json',{'decision':'pass' if decision['pass'] else 'fail',**decision})
    write_marker(root/'structured.done.json',[basis_path,root/'structured-proposal-results.json',root/'structured-coverage-decision.json'],digest(args.coverage_config))


def main():
    p=argparse.ArgumentParser()
    for name in ['manifest','reference-config','coverage-config','checkpoint','prior-thresholds','output']:
        p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
if __name__=='__main__':main()
