"""Reproduce frozen Phase B GT held-out metrics before counterfactual interpretation."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch
import yaml
from eat_nav_forward_evaluator import (
    Evaluator,HEADS,VARIANTS,load_split,metrics,predict,sha,tensors,
)

ROOT=Path('/tmp/eatnav-candidate-coverage/feasible-transfer')
PHASE=Path('/tmp/eatnav-quality-identifiability-gate/forward-evaluator')
BASE=Path('/tmp/eatnav-train')
SCALERS=Path('/tmp/eatnav-quality-identifiability-gate/relative-quality/output/relative-quality-extraction.json')


def write_once(path,value):
    content=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if path.exists():
        if path.read_text()!=content:raise FileExistsError(f'prior result differs: {path}')
        return
    temp=path.with_name(path.name+'.tmp');temp.write_text(content);temp.replace(path)


def run():
    start=time.time();source=json.loads((ROOT/'source-provenance.json').read_text())
    expected=source['container']
    for key,item in expected.items():
        if sha(item['path'])!=item['sha256']:raise ValueError(f'Stage 0 source changed: {key}')
    spec=yaml.safe_load((BASE/'forward-evaluator-config.yaml').read_text())
    prior=json.loads((PHASE/'heldout-results.json').read_text())
    scalers=json.loads(SCALERS.read_text())['train_only_scalers']
    test=load_split('test',spec)
    y=np.column_stack([test['z_'+h] for h in HEADS])
    device='cuda' if torch.cuda.is_available() else 'cpu'
    arrays=tensors(test,device)
    comparisons={};max_abs=0.
    for seed in (11,29,47):
        path=PHASE/'checkpoints'/f'GT-{seed}'/'best.pt'
        saved=torch.load(path,map_location=device,weights_only=True)
        if saved['variant']!='GT' or saved['seed']!=seed or saved['source_split_sha256']!=spec['source_split_sha256']:
            raise ValueError('Phase B checkpoint metadata mismatch')
        model=Evaluator(*VARIANTS['GT']).to(device).eval()
        model.load_state_dict(saved['model'])
        with torch.inference_mode():pred=predict(model,arrays,spec['batch_size'])
        computed=metrics(y,pred,scalers);recorded=prior['metrics'][f'GT-{seed}']
        differences={'aggregate_mse':abs(computed['aggregate_mse']-recorded['aggregate_mse'])}
        for head in HEADS:
            for metric in ('r2','spearman','mae_raw','rmse_raw','standardized_mse'):
                a=computed[head][metric];b=recorded[head][metric]
                if a is None or b is None:
                    if a!=b:raise ValueError(f'Phase B null metric mismatch {seed} {head} {metric}')
                    continue
                differences[f'{head}.{metric}']=abs(a-b)
        worst=max(differences.values());max_abs=max(max_abs,worst)
        comparisons[str(seed)]={'checkpoint_sha256':sha(path),'metric_abs_difference':differences,
                                'max_abs_difference':worst,'sample_count':len(test['index'])}
    tolerance=1e-5
    if max_abs>tolerance:
        write_once(ROOT/'phase-b-reproduction-failed.json',{'status':'fail','max_abs_difference':max_abs,
            'tolerance':tolerance,'seeds':comparisons,'heldout_used_for_model_selection':False})
        raise ValueError(f'Phase B held-out metrics do not reproduce: max diff {max_abs}')
    write_once(ROOT/'phase-b-reproduction.json',{'status':'pass','tolerance':tolerance,
        'max_abs_difference':max_abs,'seeds':comparisons,'source_split_sha256':spec['source_split_sha256'],
        'heldout_used_for_model_selection':False,'elapsed_sec':time.time()-start,
        'peak_gpu_bytes':int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
    write_once(ROOT/'stage-2a-complete.json',{'stage':'2a','status':'complete',
        'source_sha256':sha(ROOT/'source-provenance.json'),
        'output_sha256':sha(ROOT/'phase-b-reproduction.json'),'unix_sec':time.time()})
    print('FEASIBLE_TRANSFER_PHASE_B_REPRODUCED',max_abs,'sec',round(time.time()-start,1),flush=True)

if __name__=='__main__':run()
