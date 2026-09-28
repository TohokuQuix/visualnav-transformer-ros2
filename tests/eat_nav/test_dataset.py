import gzip
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.config import EATConfig
from visualnav_transformer.eat_nav.dataset import EATDataset, assign_terrain_splits


def pilot(tmp_path: Path, irregular=False):
    rows=[]
    for seed in range(10):
        for ep in range(2):
            end=95 if ep==0 else 99
            for step in range(end+1):
                dem = None
                if step % 5 == 0:
                    grid=np.ones((2,20,20),dtype=float)
                    grid[0] *= seed / 10
                    dem={'transformed':grid.tolist(),'error':None}
                rows.append({'session_id':0,'env_id':seed,'episode_id':ep,'terrain_seed':seed,'terrain_type':'stairs' if seed%2 else 'flat',
                             'step':ep*100+step,'timestamp_ns':round((ep*100+step+1)*.05*1e9)+(100000000 if irregular and seed==0 and ep==0 and step==20 else 0),
                             'pose':[100*ep+step*.01,0.,0.,1.,0.,0.,0.],'yaw':0.,
                             'critic_value':float(seed+ep+step/100),'reward':0.,'command':[.2,0.,0.],
                             'terminated':False,'truncated':step==end and ep==0,'termination_reason':['time_out'] if step==end and ep==0 else [],
                             'dem':dem})
    with gzip.open(tmp_path/'chunk-000000.jsonl.gz','wt') as f:
        for r in rows:f.write(json.dumps(r)+'\n')
    (tmp_path/'manifest.json').write_text(json.dumps({'metadata':{'num_envs':10,'step_dt':.05},'chunks':[{'file':'chunk-000000.jsonl.gz','rows':len(rows)}]}))
    return tmp_path/'manifest.json'


def test_grouped_split_does_not_leak_terrain_seed():
    split=assign_terrain_splits(range(10), seed=123)
    assert set(split)==set(range(10))
    assert {v for v in split.values()}=={'train','val','test'}
    assert split==assign_terrain_splits(range(10), seed=123)


def test_history_future_base_frame_and_variable_length(tmp_path):
    manifest=pilot(tmp_path)
    ds=EATDataset(manifest,'train',EATConfig())
    item=ds[0]
    assert item['dem'].shape==(4,2,20,20)
    assert item['action'].shape==(8,2)
    np.testing.assert_allclose(item['action'][0],[.05,0],atol=1e-5)
    np.testing.assert_allclose(item['goal'],[.4,0],atol=1e-5)
    assert item['critic_raw'].shape==(40,)
    ds12=EATDataset(manifest,'train',replace(EATConfig(),context_size=5,len_traj_pred=12))
    assert ds12[0]['dem'].shape==(6,2,20,20)
    assert ds12[0]['action'].shape==(12,2)


def test_terminal_padding_stays_within_episode(tmp_path):
    manifest=pilot(tmp_path)
    ds=EATDataset(manifest,'train',EATConfig())
    sample=next(ds[i] for i,w in enumerate(ds.windows) if w['episode_id']==0 and w['local_step']==85)
    assert sample['action_mask'].sum()==2
    np.testing.assert_allclose(sample['action'][2:], 0)
    assert sample['termination_reason']==('time_out',)
    assert sample['critic_mask'].sum()==11


def test_timestamp_irregularity_rejects_window(tmp_path):
    manifest=pilot(tmp_path,irregular=True)
    ds=EATDataset(manifest,'train',EATConfig())
    assert not any(w['terrain_seed']==0 and w['episode_id']==0 and w['local_step']==20 for w in ds.windows)


def test_terminal_reason_retained_when_last_point_is_terminal(tmp_path):
    ds=EATDataset(pilot(tmp_path),'train',EATConfig())
    item=next(ds[i] for i,w in enumerate(ds.windows) if w['episode_id']==0 and w['local_step']==55)
    assert item['action_mask'].all()
    assert item['termination_reason']==('time_out',)


def test_goal_radius_is_independent_config(tmp_path):
    ds=EATDataset(pilot(tmp_path),'train',replace(EATConfig(),goal_max_radius_m=.2))
    assert np.isclose(np.linalg.norm(ds[0]['goal']),.2)
