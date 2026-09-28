"""Episode-safe EAT windows from raw Isaac chunk manifests."""

from __future__ import annotations

from collections import defaultdict
import bisect
import gzip
import json
import math
from pathlib import Path

import numpy as np

from .config import EATConfig
from .quality import fit_train_statistics, p10


def assign_terrain_splits(terrain_seeds, seed=123):
    seeds=np.asarray(sorted(set(int(s) for s in terrain_seeds)),dtype=np.int64)
    if len(seeds)<3:
        raise ValueError('at least three terrain seeds required')
    rng=np.random.default_rng(seed)
    shuffled=rng.permutation(seeds)
    n_train=max(1,int(round(len(seeds)*.8)))
    n_val=max(1,int(round(len(seeds)*.1)))
    n_train=min(n_train,len(seeds)-2)
    n_val=min(n_val,len(seeds)-n_train-1)
    return {int(s):('train' if i<n_train else 'val' if i<n_train+n_val else 'test') for i,s in enumerate(shuffled)}


def _records(manifest):
    manifest=Path(manifest)
    data=json.loads(manifest.read_text())
    episodes=defaultdict(list)
    for chunk in data['chunks']:
        with gzip.open(manifest.parent/chunk['file'],'rt') as stream:
            for line in stream:
                r=json.loads(line)
                d=r.get('dem')
                transformed=None if not d or d.get('transformed') is None else np.asarray(d['transformed'],dtype=np.float32)
                compact={'timestamp_ns':r['timestamp_ns'],'step':r['step'],'pose':np.asarray(r['pose'][:2],dtype=np.float32),
                         'yaw':r['yaw'],'critic_value':r['critic_value'],'reward':r['reward'],
                         'terminated':r['terminated'],'truncated':r['truncated'],
                         'termination_reason':tuple(r['termination_reason']),'dem':transformed,
                         'terrain_seed':r['terrain_seed'],'terrain_type':r['terrain_type']}
                episodes[(r['session_id'],r['env_id'],r['episode_id'])].append(compact)
    for rows in episodes.values():
        rows.sort(key=lambda r:r['timestamp_ns'])
    return episodes


def _nearest(times,target,tolerance_ns):
    k=bisect.bisect_left(times,target)
    choices=[j for j in (k-1,k) if 0<=j<len(times)]
    if not choices:
        return None
    index=min(choices,key=lambda j:abs(times[j]-target))
    return index if abs(times[index]-target)<=tolerance_ns else None


def _build_window(key,rows,index,config):
    dt_ns=round(config.trajectory_interval_sec*1e9)
    hist_ns=round(config.dem_history_interval_sec*1e9)
    tol_ns=round(min(config.trajectory_interval_sec,config.dem_history_interval_sec,.05)*.4*1e9)
    times=[r['timestamp_ns'] for r in rows]
    now=rows[index]
    history=[]
    for offset in range(config.context_size,-1,-1):
        j=_nearest(times,now['timestamp_ns']-offset*hist_ns,tol_ns)
        if j is None or rows[j]['dem'] is None:
            return None
        history.append(rows[j]['dem'])
    if any(x.shape!=(2,config.dem_height_cells,config.dem_width_cells) for x in history):
        return None
    last=rows[-1]
    is_terminal=last['terminated'] or last['truncated']
    horizon_ns=config.len_traj_pred*dt_ns
    if not is_terminal and now['timestamp_ns']+horizon_ns>last['timestamp_ns']+tol_ns:
        return None
    point_positions=[]
    mask=[]
    c,s=math.cos(now['yaw']),math.sin(now['yaw'])
    for point in range(1,config.len_traj_pred+1):
        target=now['timestamp_ns']+point*dt_ns
        j=_nearest(times,target,tol_ns)
        valid=j is not None and j>index and target<=last['timestamp_ns']+tol_ns
        if not valid and not is_terminal:
            return None
        pose=rows[j]['pose'] if valid else last['pose']
        dx,dy=pose-now['pose']
        point_positions.append([c*dx+s*dy,-s*dx+c*dy])
        mask.append(valid)
    xy=np.asarray(point_positions,dtype=np.float32)
    action=np.diff(np.vstack((np.zeros((1,2),dtype=np.float32),xy)),axis=0)
    action[~np.asarray(mask,dtype=bool)]=0
    goal=xy[np.flatnonzero(mask)[-1]] if any(mask) else np.zeros(2,dtype=np.float32)
    radius=float(np.linalg.norm(goal))
    if radius>config.goal_max_radius_m:goal=goal*(config.goal_max_radius_m/radius)
    critic_values=[r['critic_value'] for r in rows[index:] if r['timestamp_ns']<now['timestamp_ns']+horizon_ns and r['timestamp_ns']<=last['timestamp_ns']]
    if not critic_values:
        return None
    raw=np.zeros(round(config.trajectory_horizon_sec/.05),dtype=np.float32)
    crit_mask=np.zeros_like(raw,dtype=bool)
    count=min(len(raw),len(critic_values))
    raw[:count]=critic_values[:count]
    crit_mask[:count]=True
    return {'episode_key':key,'episode_id':key[2],'local_step':index,'terrain_seed':now['terrain_seed'],
            'terrain_type':now['terrain_type'],'step':now['step'],'dem':np.stack(history),
            'goal':goal.astype(np.float32),'action':action.astype(np.float32),
            'action_mask':np.asarray(mask,dtype=bool),'critic_raw':raw,
            'critic_mask':crit_mask,'quality_raw':p10(raw[crit_mask]),
            'termination_reason':last['termination_reason'] if (is_terminal and now['timestamp_ns']+horizon_ns>=last['timestamp_ns']-tol_ns) else ()}


class EATDataset:
    def __init__(self,manifest:Path,split:str,config:EATConfig,split_seed=123):
        if split not in ('train','val','test'):
            raise ValueError('split must be train, val or test')
        episodes=_records(manifest)
        self.split_map=assign_terrain_splits([rows[0]['terrain_seed'] for rows in episodes.values()],split_seed)
        self.config=config
        all_windows=[]
        for key,rows in sorted(episodes.items()):
            for index,r in enumerate(rows):
                if r['dem'] is None:
                    continue
                window=_build_window(key,rows,index,config)
                if window is not None:
                    all_windows.append(window)
        train=[w for w in all_windows if self.split_map[w['terrain_seed']]=='train']
        self.stats,self.ecdf=fit_train_statistics([dict(w,critic_raw=w['critic_raw'][w['critic_mask']]) for w in train])
        self.windows=[w for w in all_windows if self.split_map[w['terrain_seed']]==split]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self,index):
        window=dict(self.windows[index])
        window['quality']=self.ecdf.transform(window['quality_raw'])
        return window
