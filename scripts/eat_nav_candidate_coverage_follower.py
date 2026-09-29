"""Measure initial commands for every frozen candidate through the real ROS2 follower."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient
from scripts.eat_nav_candidate_coverage_select import atomic_once,digest


def command_cluster_count(commands, linear_limit=.03, angular_limit=.05):
    x=np.asarray(commands,dtype=float);parent=list(range(len(x)))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(len(x)):
        for j in range(i+1,len(x)):
            if abs(x[i,0]-x[j,0])<linear_limit and abs(x[i,2]-x[j,2])<angular_limit:
                parent[find(j)]=find(i)
    return len({find(i) for i in range(len(x))})


def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();selection=json.loads(a.selection.read_text())
    if digest(a.selection)!='1bb7865e81d2f0f52fff550037bb2e07b560b73f5de553b6183203bdaa969ddc':
        raise ValueError('frozen state selection changed')
    poses=[[0.,0.,0.]]*8
    client=FollowerBridgeClient(timeout=45.)
    try:
        for selected in selection['selected']:
            source=a.offline/f"{selected['state_id']}.json";state=json.loads(source.read_text())
            if state['state']!=selected:raise ValueError('offline state identity mismatch')
            target=a.output/f"{selected['state_id']}-follower.json"
            if target.exists():
                previous=json.loads(target.read_text())
                if previous['offline_sha256']!=digest(source):raise ValueError('follower resume source changed')
                print('FOLLOWER_RESUME_SKIP',selected['state_id'],flush=True);continue
            candidates=state['candidates'];rows=[]
            for first in range(0,len(candidates),8):
                group=candidates[first:first+8]
                if any(not r['finite'] for r in group):raise ValueError('nonfinite candidate Path in follower diagnostic')
                paths=[r['ros_base_path'] for r in group]
                paths += [paths[-1]]*(8-len(paths))
                attempts=0
                while True:
                    try:
                        startup=client.request('reset',poses=poses,paths=paths)
                        samples=[]
                        for _ in range(3):
                            response=client.request('step',poses=poses,timeout_sec=5.)
                            if not response['fresh']:raise RuntimeError('follower command timeout')
                            samples.append(response['commands'])
                        break
                    except (RuntimeError,TimeoutError) as exc:
                        attempts+=1
                        print('FOLLOWER_BATCH_RETRY',selected['state_id'],first,attempts,repr(exc),flush=True)
                        if attempts>1:raise
                values=np.asarray(samples,dtype=float)[:,:len(group)]
                for index,item in enumerate(group):
                    rows.append({'candidate_id':item['candidate_id'],'initial_command':np.median(values[:,index],axis=0).tolist(),
                                 'static_pose_command_samples':values[:,index].tolist(),
                                 'follower_reset_startup':startup['readiness']['event'],
                                 'infrastructure_retry_count':attempts})
            if len(rows)!=37:raise ValueError('candidate command count mismatch')
            by_id={r['candidate_id']:r for r in rows}
            sets={}
            for name,group in state['sets'].items():
                ids=group['candidate_ids']
                sets[name]={'command_cluster_count':command_cluster_count([by_id[cid]['initial_command'] for cid in ids]),
                            'candidate_ids':ids}
            result={'state':selected,'offline_sha256':digest(source),'selection_sha256':digest(a.selection),
                    'pose_protocol':'three static zero-map-pose samples per candidate; full moving command series comes from rollout',
                    'candidates':rows,'sets':sets,'candidate_outcomes_used':False}
            atomic_once(target,result)
            print('FOLLOWER_STATE',selected['state_id'],'K8_commands',sets['nomad_k8']['command_cluster_count'],flush=True)
    finally:client.close()
    atomic_once(a.output/'follower-status.json',{'stage':'follower_initial_commands_complete','state_count':7,
                                             'candidate_outcomes_used':False})

if __name__=='__main__':main()
