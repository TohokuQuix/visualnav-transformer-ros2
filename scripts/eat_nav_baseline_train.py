"""Train-only small-sample Baseline Evaluator smoke, without a generalization claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from visualnav_transformer.eat_nav.evaluator.dataset import SCHEMA as DATASET_SCHEMA, validate_group_split
from visualnav_transformer.eat_nav.evaluator.labels import LABELS, SCHEMA as LABEL_SCHEMA
from visualnav_transformer.eat_nav.evaluator.model import BaselineEvaluator
from visualnav_transformer.eat_nav.evaluator.preprocessing import SCHEMA as INPUT_SCHEMA, fit_normalization, prepare_inputs


CHECKPOINT_SCHEMA='eatnav_baseline_checkpoint_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_torch_save(payload,path):
    path=Path(path)
    temp=path.with_name(path.name+'.tmp')
    if path.exists() or temp.exists():
        raise FileExistsError(path)
    with temp.open('xb') as stream:
        torch.save(payload,stream)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def train(dataset, config_path, output):
    dataset,config_path,output=Path(dataset),Path(config_path),Path(output)
    if output.exists():
        raise FileExistsError(output)
    config=json.loads(config_path.read_text())
    if config.get('schema')!='eatnav_baseline_pilot_config_v1' or config['label_type'] not in LABELS:
        raise ValueError('pilot config')
    metadata=json.loads((dataset/'metadata.json').read_text())
    if (metadata.get('schema')!=DATASET_SCHEMA or
            metadata.get('source_protocol')!='eatnav_p1_g3_rosfree_crossover_v1' or
            metadata['independent_group_count']!=1):
        raise ValueError('pilot dataset/source contract')
    if sha(dataset/'samples.npz')!=metadata['arrays_sha256']:
        raise ValueError('dataset arrays SHA')
    validate_group_split(metadata['samples'],metadata['split'])
    if metadata['split']['validation'] or metadata['split']['test'] or \
            len(metadata['split']['train'])!=metadata['sample_count']:
        raise ValueError('pilot is train-only')
    with np.load(dataset/'samples.npz',allow_pickle=False) as arrays:
        dem,state,paths=[arrays[key].copy() for key in ('dem','state','path')]
        labels,masks=arrays['label'].copy(),arrays['label_mask'].copy()
    column=LABELS.index(config['label_type'])
    selected=np.flatnonzero(masks[:,column])
    if len(selected)<2 or not np.isfinite(labels[selected,column]).all():
        raise ValueError('label unavailable/insufficient')
    if len(selected)!=len(labels):
        raise ValueError('pilot requires complete selected label')
    dem,state,paths=dem[selected],state[selected],paths[selected]
    raw_label=labels[selected,column]
    normal=fit_normalization(dem,state,paths[:,None])
    prepared=prepare_inputs(dem,state,paths[:,None],normal)
    mean=float(raw_label.mean());std=max(float(raw_label.std()),1e-6)
    target=(raw_label-mean)/std
    seed=int(config['seed'])
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(1)
    device=torch.device(config.get('device','cpu'))
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    model=BaselineEvaluator(**config['model']).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=float(config['learning_rate']),
                               weight_decay=float(config['weight_decay']))
    inputs={key:torch.from_numpy(value).to(device) for key,value in prepared.items()}
    y=torch.from_numpy(target.astype(np.float32)).to(device)
    batch_size=int(config['batch_size'])
    if batch_size<1 or int(config['max_epochs'])<1:
        raise ValueError('training length/batch')
    start=time.monotonic()
    history=[]
    for epoch in range(int(config['max_epochs'])):
        model.train()
        order=torch.randperm(len(y),device=device)
        for indices in order.split(batch_size):
            prediction=model(inputs['dem'][indices],inputs['state'][indices],
                             inputs['paths'][indices]).squeeze(1)
            loss=F.mse_loss(prediction,y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss=float(F.mse_loss(model(inputs['dem'],inputs['state'],inputs['paths']).squeeze(1),y))
        history.append(train_loss)
        if not np.isfinite(train_loss):
            raise ValueError('nonfinite training loss')
    elapsed=time.monotonic()-start
    # Identical pre-Path inputs with different outcomes impose a nonzero floor.
    buckets={}
    for i in range(len(y)):
        key=hashlib.sha256(dem[i].tobytes()+state[i].tobytes()+paths[i].tobytes()).hexdigest()
        buckets.setdefault(key,[]).append(float(target[i]))
    floor=float(np.mean([(value-float(np.mean(buckets[key])))**2
                         for key,values in buckets.items() for value in values]))
    source_files=[Path(__file__),Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/evaluator/model.py',
                  Path(__file__).resolve().parents[1]/'src/visualnav_transformer/eat_nav/evaluator/preprocessing.py']
    payload={'schema':CHECKPOINT_SCHEMA,'input_schema':INPUT_SCHEMA,'label_schema':LABEL_SCHEMA,
             'label_type':config['label_type'],'label_direction':'higher_is_better',
             'model_config':model.config,'model_state':model.cpu().state_dict(),
             'optimizer_state':optimizer.state_dict(),'normalization':normal,
             'label_transform':{'mean':mean,'std':std},
             'dataset_sha256':sha(dataset/'samples.npz'),
             'dataset_metadata_sha256':sha(dataset/'metadata.json'),
             'split_manifest':metadata['split'],'independent_group_count':1,
             'training_seed':seed,'training_config':config,'training_config_sha256':sha(config_path),
             'loss_history':history,'source_sha256':{str(p.name):sha(p) for p in source_files},
             'git_commit':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True,
                                         check=True).stdout.strip(),
             'torch_version':torch.__version__}
    output.mkdir(parents=True)
    checkpoint=output/'checkpoint.pt'
    _atomic_torch_save(payload,checkpoint)
    result={'schema':'eatnav_baseline_pilot_result_v1','checkpoint_sha256':sha(checkpoint),
            'checkpoint_path':str(checkpoint),'dataset_sha256':payload['dataset_sha256'],
            'label_type':config['label_type'],'sample_count':len(y),
            'unique_input_count':len(buckets),'independent_group_count':1,
            'train_loss_initial':history[0],'train_loss_final':history[-1],
            'identical_input_mse_floor':floor,'loss_history':history,
            'elapsed_s':elapsed,'parameter_count':sum(p.numel() for p in model.parameters()),
            'validation_status':'unavailable_one_independent_group',
            'interpretation':'training overfit/pipeline smoke only'}
    with (output/'result.json').open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=train(args.dataset,args.config,args.output)
    print(json.dumps({k:result[k] for k in ('checkpoint_path','checkpoint_sha256',
          'train_loss_initial','train_loss_final','identical_input_mse_floor','elapsed_s')},sort_keys=True))


if __name__=='__main__':main()
