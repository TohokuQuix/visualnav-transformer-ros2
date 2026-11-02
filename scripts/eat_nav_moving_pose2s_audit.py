#!/usr/bin/env python3
"""Verify immutable packed dataset and report coverage without fitting statistics."""
from collections import Counter,defaultdict
import argparse
import json
from pathlib import Path
import numpy as np
from eat_nav_moving_pose2s_dataset import verify_dataset,read_json,write_json,load_npz

EVENTS=('tip_over','near_fall','linear_stall','turn_stall')


def audit_dataset(dataset):
    p=Path(dataset);manifest=verify_dataset(p);index=read_json(p/'sample-index.json');a=load_npz(p/'samples.npz');split=read_json(p/'split.json')['roles']
    n=len(index);seen=[];parents=defaultdict(set);signatures=defaultdict(set);encoder_train=set();eval_parents=set();geometry=defaultdict(lambda:defaultdict(set))
    strata=defaultdict(list);coverage=Counter()
    for role,indices in split.items():
        for i in indices:
            if index[i]['role']!=role:
                raise ValueError('split index role mismatch')
            seen.append(i)
    if sorted(seen)!=list(range(n)):
        raise ValueError('split does not partition all records')
    for i,r in enumerate(index):
        parents[r['parent_id']].add(r['role']);signatures[r['initial_state_signature']].add(r['role'])
        if r['original_role']=='train': encoder_train.add(r['parent_id'])
        if r['role']!='train': eval_parents.add(r['parent_id'])
        if r['geometry_sha256']:geometry[r['field']][r['role']].add(r['geometry_sha256'])
        else:coverage['geometry_hash_missing']+=1
        for k in ('dem','actor_state','body_state','paths','directions'):
            if not np.isfinite(a[k][i]).all():raise ValueError('nonfinite feature')
        for key in ('position_targets','rotation_targets'):
            if a['pose_mask'][i].any() and not np.isfinite(a[key][i]).all():raise ValueError('nonfinite valid pose')
        if not np.isfinite(a['event_targets'][i][a['event_mask'][i]]).all():raise ValueError('nonfinite known event')
        if not np.isfinite(a['trajectory_position'][i][a['trajectory_mask'][i]]).all():raise ValueError('nonfinite observed trajectory')
        if not a['candidate_mask'][i,0]:raise ValueError('executed candidate input invalid')
        state='initial' if r['is_initial'] else 'moving'
        speed='v<0.05' if r['speed_mps']<.05 else 'v0.05-0.2' if r['speed_mps']<.2 else 'v>=0.2'
        pitch='pitch<-10' if r['pitch_deg']< -10 else 'pitch>10' if r['pitch_deg']>10 else 'pitch[-10,10]'
        for dimension,value in [('role',r['role']),('field',r['field']),('level',str(r['level'])),('state',state),('speed',speed),('pitch',pitch),('direction',str(r['direction'])),('yawrate', '|wz|>=0.2' if abs(r['yawrate_radps'])>=.2 else '|wz|<0.2')]:
            strata[(dimension,value)].append(i)
    bad_parents={k:sorted(v) for k,v in parents.items() if len(v)>1};bad_signatures={k:sorted(v) for k,v in signatures.items() if len(v)>1}
    if bad_parents or bad_signatures or encoder_train&eval_parents:
        raise ValueError('train/eval ancestry or initial-state leakage')
    def summary(indices):
        ids=np.asarray(indices,int);m=a['event_mask'][ids,0];y=a['event_targets'][ids,0];events={}
        for j,name in enumerate(EVENTS):
            positive=m[:,j]&(y[:,j]==1);negative=m[:,j]&(y[:,j]==0)
            events[name]=dict(positive=int(positive.sum()),negative=int(negative.sum()),unknown=int((~m[:,j]).sum()),
                independent_positive_parents=len({index[i]['parent_id'] for i in ids[positive]}))
        return dict(records=len(ids),pose=int(a['pose_mask'][ids].sum()),partial=int((~a['pose_mask'][ids]).sum()),parents=len({index[i]['parent_id'] for i in ids}),events=events)
    overlap={}
    for field,g in geometry.items():
        overlap[field]=dict(train_validation=len(g['train']&g['validation']),train_audit_eval=len(g['train']&g['audit_eval']),
            validation_audit_eval=len(g['validation']&g['audit_eval']),counts={k:len(v) for k,v in g.items()})
    return dict(schema='eatnav_moving_pose2s_audit_v1',manifest_counts=manifest['counts'],total=summary(list(range(n))),
        strata=[dict(dimension=k[0],value=k[1],**summary(v)) for k,v in sorted(strata.items())],
        leakage=dict(parent_role_collisions=bad_parents,initial_signature_role_collisions=bad_signatures,encoder_train_eval_parent_overlap=sorted(encoder_train&eval_parents),
            geometry_overlap=overlap,geometry_missing=int(coverage['geometry_hash_missing']),
            geometry_interpretation='Geometry overlap is known shared terrain, not unseen-terrain evaluation. Missing hashes do not establish unseen terrain.'),
        hashes_verified=True,source_hashes_verified=True,reserved_test_opened=False,normalization_fitted=False,
        limitations=['Only executed candidates have targets; windows are correlated within parents.',
            'DEM history is causal; exhaustive historical raw-ray reconstruction is not performed.',
            'Pose supervision is complete_2s_only; missing teachers retain explicit masks.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',required=True);parser.add_argument('--output');args=parser.parse_args()
    report=audit_dataset(args.dataset)
    if args.output:
        out=Path(args.output)
        if out.exists():
            if read_json(out)!=report:raise ValueError('existing audit differs; refusing overwrite')
        else:write_json(out,report)
    print(json.dumps(dict(total=report['total'],leakage=report['leakage']),indent=2))

if __name__=='__main__':main()
