"""Measure frozen NoMaD Path diversity at the actual ROS2 follower boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from visualnav_transformer.eat_nav.phase_c_bridge_client import FollowerBridgeClient


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clusters(dist,threshold):
    n=len(dist);parent=list(range(n))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for i in range(n):
        for j in range(i+1,n):
            if dist[i,j]<threshold: parent[find(j)]=find(i)
    return len({find(i) for i in range(n)})

def atomic(path,obj):
    p=Path(path);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n');t.replace(p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true')
    a=p.parse_args();root=a.output
    offline=json.loads((root/'candidate-diversity-offline.json').read_text())
    thresholds=json.loads((root/'candidate-thresholds.json').read_text())
    frozen=json.loads((root/'candidate-states.json').read_text())
    data=np.load(root/'candidate-paths.npz');paths=data['paths']
    if paths.shape!=(32,8,8,2) or len(frozen['states'])!=32:raise ValueError('frozen candidate shape mismatch')
    source={'offline_sha256':sha(root/'candidate-diversity-offline.json'),
            'paths_sha256':sha(root/'candidate-paths.npz'),
            'thresholds_sha256':sha(root/'candidate-thresholds.json')}
    result_path=root/'candidate-diversity.json'
    if result_path.exists() and not a.resume:raise FileExistsError(result_path)
    result=json.loads(result_path.read_text()) if result_path.exists() else {'source':source,'rows':[]}
    if result['source']!=source:raise ValueError('resume source mismatch')
    bridge=FollowerBridgeClient(timeout=45.0)
    poses=[[0.0,0.0,0.0]]*8
    try:
        for state_index in range(len(paths)):
            if state_index<len(result['rows']):continue
            candidate_paths=[[[0.0,0.0]]+paths[state_index,seed_index].astype(float).tolist()
                             for seed_index in range(8)]
            bridge.request('reset',poses=poses,paths=candidate_paths)
            commands=[]
            for _ in range(3):
                response=bridge.request('step',poses=poses,timeout_sec=5.0)
                if not response['fresh']:raise RuntimeError(f'stale follower command at state {state_index}')
                commands.append(response['commands'])
            command=np.median(np.asarray(commands,dtype=float),axis=0)
            linear=np.abs(command[:,None,0]-command[None,:,0])
            angular=np.abs(command[:,None,2]-command[None,:,2])
            collapsed=(linear<thresholds['command_linear_min_mps'])&(angular<thresholds['command_angular_min_radps'])
            command_dist=np.maximum(linear/thresholds['command_linear_min_mps'],
                                    angular/thresholds['command_angular_min_radps'])
            report={'state_index':state_index,'state':frozen['states'][state_index],
                    'follower_commands':command.tolist(),
                    'follower_linear_pairwise_difference_mps':linear[np.triu_indices(8,1)].tolist(),
                    'follower_angular_pairwise_difference_radps':angular[np.triu_indices(8,1)].tolist(),
                    'indistinguishable_command_pair_count':int(collapsed[np.triu_indices(8,1)].sum()),
                    'command_cluster_count':clusters(command_dist,1.0),
                    'path_cluster_count':offline['rows'][state_index]['path_cluster_count'],
                    'ood_count_over_train_p95':offline['rows'][state_index]['ood_count_over_train_p95']}
            result['rows'].append(report)
            atomic(result_path,result)
            print('FOLLOWER_CANDIDATE_STATE',state_index+1,'of',len(paths),'command_clusters',report['command_cluster_count'],flush=True)
    finally:
        bridge.close()
    result['state_count']=len(result['rows'])
    result['path_single_cluster_count']=sum(x['path_cluster_count']==1 for x in result['rows'])
    result['command_single_cluster_count']=sum(x['command_cluster_count']==1 for x in result['rows'])
    result['out_of_distribution_candidate_count']=sum(x['ood_count_over_train_p95'] for x in result['rows'])
    atomic(result_path,result)
    print('FOLLOWER_DIVERSITY_COMPLETE',json.dumps({k:result[k] for k in ('state_count','path_single_cluster_count','command_single_cluster_count')},sort_keys=True),flush=True)
if __name__=='__main__':main()
