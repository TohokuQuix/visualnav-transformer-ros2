"""Real simple_trajectory_follower command-distinctness precheck."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import yaml

from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):
            h.update(block)
    return h.hexdigest()


def atomic(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    temp.replace(path)


def clusters(command,linear_min,angular_min):
    parent=list(range(len(command)))
    def root(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(command)):
        for j in range(i+1,len(command)):
            if abs(command[i,0]-command[j,0])<linear_min and abs(command[i,2]-command[j,2])<angular_min:
                parent[root(j)]=root(i)
    return len({root(i) for i in range(len(command))})


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--maneuvers',type=Path,required=True)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--spec',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    bank=json.loads(a.bank.read_text());spec=yaml.safe_load(a.spec.read_text())
    with np.load(a.maneuvers) as data:
        paths=data['paths'];names=list(data['names'])
    if sha(a.maneuvers)!=bank['paths_sha256'] or names!=spec['maneuvers']['names']:
        raise ValueError('frozen maneuver source mismatch')
    if paths.shape!=(len(bank['states']),5,8,2):
        raise ValueError('frozen maneuver bank shape mismatch')
    limits=spec['command_distinctness']
    source={'maneuvers_sha256':sha(a.maneuvers),'bank_sha256':sha(a.bank),
            'spec_sha256':sha(a.spec),'precheck_source_sha256':sha(Path(__file__))}
    folder=a.output/'precheck-states'
    folder.mkdir(exist_ok=True,parents=True)
    bridge=FollowerBridgeClient(timeout=45.0)
    try:
        for index,state in enumerate(bank['states']):
            marker=folder/f'{index:03d}.json'
            if marker.exists():
                previous=json.loads(marker.read_text())
                if previous['source']!=source or previous['state']!=state['state']:
                    raise ValueError(f'resume precheck marker mismatch: {marker}')
                continue
            proposals=paths[index]
            full=np.concatenate((proposals,proposals[:3]),axis=0)
            path_json=[[[0.,0.]]+candidate.astype(float).tolist() for candidate in full]
            poses=[[0.,0.,0.]]*8
            reset=bridge.request('reset',poses=poses,paths=path_json)
            repeat=[]
            for _ in range(3):
                response=bridge.request('step',poses=poses,timeout_sec=5.)
                if not response['fresh']:
                    raise RuntimeError(f'stale follower command for state {index}')
                repeat.append(response['commands'])
            command=np.median(np.asarray(repeat,dtype=float),axis=0)
            if command.shape!=(8,3) or not np.isfinite(command).all():
                raise ValueError('follower command shape/nonfinite')
            duplicate_rms=float(np.sqrt(np.mean((command[5:]-command[:3])**2)))
            pairwise=[]
            for i in range(5):
                for j in range(i+1,5):
                    linear=abs(command[i,0]-command[j,0])
                    angular=abs(command[i,2]-command[j,2])
                    pairwise.append({'i':names[i],'j':names[j],'linear_delta_mps':float(linear),
                                     'angular_delta_radps':float(angular),
                                     'distinct':bool(linear>=limits['initial_linear_min_mps'] or angular>=limits['initial_angular_min_radps'])})
            key_pairs={frozenset((p['i'],p['j'])):p['distinct'] for p in pairwise}
            nominal_left_right=all(key_pairs[frozenset(pair)] for pair in
                                   [('nominal','slight_left'),('nominal','slight_right'),('slight_left','slight_right')])
            result={'source':source,'state':state['state'],'state_index':index,'split':state['split'],
                    'follower_startup':reset,'command_repeats':repeat,'initial_commands':command[:5].tolist(),
                    'duplicate_branch_command_rms':duplicate_rms,'pairwise':pairwise,
                    'cluster_count':clusters(command[:5],limits['initial_linear_min_mps'],limits['initial_angular_min_radps']),
                    'nominal_left_right_distinct':nominal_left_right,
                    'status':'distinct' if nominal_left_right else 'proposal_not_distinct'}
            atomic(marker,result)
            print('FOLLOWER_PRECHECK',index+1,'of',len(paths),result['status'],'clusters',result['cluster_count'],flush=True)
    finally:
        bridge.close()
    rows=[json.loads((folder/f'{i:03d}.json').read_text()) for i in range(len(paths))]
    atomic(a.output/'command-distinctness.json',{'source':source,'rows':rows,
        'train_count':sum(r['split']=='train' for r in rows),
        'validation_count':sum(r['split']=='val' for r in rows),
        'proposal_not_distinct_count':sum(r['status']=='proposal_not_distinct' for r in rows),
        'computed_before_counterfactual_outcomes':True})
    print('FOLLOWER_PRECHECK_COMPLETE',len(rows),flush=True)

if __name__=='__main__':main()
