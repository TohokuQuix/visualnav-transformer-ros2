"""Trace q signal from token MLP through transformer/U-Net to trajectories."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.quality_gate_layers import trace_network_layers


def main():
    p=argparse.ArgumentParser()
    for name in ('manifest','quality-config','checkpoint','prior-report','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    expected='fcaf3b9397f3344a0017a52e324ad11ce879c29d9737e1b924d3800293e00987'
    if sha!=expected:raise ValueError('pilot Quality checkpoint SHA mismatch')
    config=load_eat_config(a.quality_config)
    dataset=EATDataset(a.manifest,'test',config)
    prior=json.loads(a.prior_report.read_text())
    indices=prior['selection']['indices']
    rows=[dataset[i] for i in indices]
    if not all(row['action_mask'].all() for row in rows):raise ValueError('layer inputs must have eight valid points')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model,_,_=load_checkpoint(a.checkpoint,config)
    model=model.to(device).eval()
    dem=torch.from_numpy(np.stack([row['dem'] for row in rows])).to(device)
    goal=torch.from_numpy(np.stack([row['goal'] for row in rows])).to(device)
    action=torch.from_numpy(np.stack([row['action'] for row in rows])).to(device)
    result=trace_network_layers(model,dem,goal,action,seed=11)
    result.update({'checkpoint_sha256':sha,'heldout_split':'test','window_indices':indices,
                   'episode_ids':[list(row['episode_key']) for row in rows],
                   'terrain_seeds':[int(row['terrain_seed']) for row in rows],
                   'diffusion_seed':11,'input_count':len(rows)})
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('LAYER_DONE',json.dumps({k:result[k] for k in ('input_count','final_effect_sampling_ratio','q_context_delta_norm','goal_0p1m_context_delta_norm')}),flush=True)

if __name__=='__main__':main()
