"""Train-set-only paired candidate ranking diagnostic for the pilot checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from visualnav_transformer.eat_nav.evaluator.inference import load_checkpoint
from visualnav_transformer.eat_nav.evaluator.labels import LABELS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(dataset, checkpoint):
    dataset=Path(dataset);checkpoint=Path(checkpoint)
    metadata=json.loads((dataset/'metadata.json').read_text())
    if sha(dataset/'samples.npz')!=metadata['arrays_sha256']:
        raise ValueError('dataset SHA changed')
    api,payload=load_checkpoint(checkpoint,device='cpu',
                                expected_dataset_sha256=metadata['arrays_sha256'])
    if payload['dataset_metadata_sha256']!=sha(dataset/'metadata.json'):
        raise ValueError('dataset metadata changed')
    column=LABELS.index(payload['label_type'])
    with np.load(dataset/'samples.npz',allow_pickle=False) as arrays:
        dem,state,paths=[arrays[key].copy() for key in ('dem','state','path')]
        labels,masks=arrays['label'].copy(),arrays['label_mask'].copy()
    entries=metadata['samples']
    roles=('direct','left_20','right_20')
    groups=[]
    for env in sorted({item['env_id'] for item in entries}):
        indices=[next(i for i,item in enumerate(entries)
                      if item['env_id']==env and item['path_role']==role) for role in roles]
        if not np.array_equal(dem[indices],np.broadcast_to(dem[indices[0]],dem[indices].shape)) or \
                not np.array_equal(state[indices],np.broadcast_to(state[indices[0]],state[indices].shape)):
            raise ValueError('paired pre-Path input differs')
        if not masks[indices,column].all():
            raise ValueError('paired label missing')
        candidate_paths=paths[indices][None]
        scores=api.predict(dem[indices[0]][None],state[indices[0]][None],candidate_paths)[0]
        truth=labels[indices,column]
        selected=int(np.argmax(scores));oracle=int(np.argmax(truth))
        pairs=[(i,j) for i in range(3) for j in range(i+1,3)]
        agreements=[bool(np.sign(scores[i]-scores[j])==np.sign(truth[i]-truth[j]))
                    for i,j in pairs]
        groups.append({'env_id':env,'roles':roles,'sample_ids':[entries[i]['sample_id'] for i in indices],
                       'predicted_scores':scores.tolist(),'teacher_labels':truth.tolist(),
                       'predicted_order':[roles[i] for i in np.argsort(-scores)],
                       'teacher_order':[roles[i] for i in np.argsort(-truth)],
                       'pairwise_ranking_agreement':sum(agreements)/len(agreements),
                       'selected_role':roles[selected],'oracle_role':roles[oracle],
                       'regret':float(truth[oracle]-truth[selected])})
    dem_one=dem[:1];state_one=state[:1];paths_one=paths[:1,None]
    start=time.perf_counter()
    for _ in range(100):api.predict(dem_one,state_one,paths_one)
    latency_ms=(time.perf_counter()-start)*10
    return {'schema':'eatnav_baseline_ranking_diagnostic_v1',
            'interpretation':'train-set-only, one independent paired StateGroup-seed; no generalization claim',
            'checkpoint_sha256':sha(checkpoint),'dataset_sha256':metadata['arrays_sha256'],
            'label_type':payload['label_type'],'candidate_count':3,
            'independent_group_count':1,'env_display_groups':groups,
            'mean_pairwise_ranking_agreement':float(np.mean([g['pairwise_ranking_agreement'] for g in groups])),
            'mean_regret':float(np.mean([g['regret'] for g in groups])),
            'cpu_single_candidate_latency_ms_mean_100_calls':latency_ms}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=evaluate(args.dataset,args.checkpoint)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2)
        stream.write('\n')
    print(json.dumps({'output':str(args.output),'mean_pairwise_ranking_agreement':
                      result['mean_pairwise_ranking_agreement'],'mean_regret':result['mean_regret']},sort_keys=True))


if __name__=='__main__':main()
