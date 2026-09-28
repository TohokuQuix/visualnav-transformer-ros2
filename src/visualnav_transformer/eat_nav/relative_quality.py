"""Timestamp-aligned, post-hoc Critic and return labels for complete EAT windows."""
from __future__ import annotations

import bisect
import math
from typing import Mapping, Sequence

import numpy as np


def aligned_endpoint(rows: Sequence[Mapping], anchor_index: int, horizon_sec: float,
                     tolerance_sec: float) -> int:
    """Find the same-episode pose/reward row at anchor timestamp + horizon."""
    if anchor_index < 0 or anchor_index >= len(rows) or horizon_sec <= 0 or tolerance_sec < 0:
        raise ValueError('invalid anchor or horizon')
    times=[int(row['timestamp_ns']) for row in rows]
    target=times[anchor_index]+round(horizon_sec*1e9)
    k=bisect.bisect_left(times,target)
    choices=[j for j in (k-1,k) if anchor_index<j<len(rows)]
    if not choices:raise ValueError('endpoint unavailable')
    j=min(choices,key=lambda v:abs(times[v]-target))
    if abs(times[j]-target)>round(tolerance_sec*1e9):raise ValueError('endpoint timestamp mismatch')
    return j


def _robust_slope(values: np.ndarray, step_sec: float) -> float:
    """Median pairwise Critic slope (Theil-Sen) in value units per second."""
    x=np.asarray(values,dtype=np.float64)
    a,b=np.triu_indices(len(x),1)
    return float(np.median((x[b]-x[a])/((b-a)*step_sec)))


def compute_relative_window(rows: Sequence[Mapping], anchor_index: int, endpoint_index: int,
                            *, gamma: float, step_sec: float) -> dict:
    """Align V(s), r(s,a,s'), and post-step poses without using next-episode rows.

    The collector writes pre-action V and post-action reward/pose into one row.
    Anchor pose at row i is therefore the state valued in row i+1.
    """
    i,j=anchor_index,endpoint_index
    if not (0<=i<j<len(rows)) or not (0<gamma<=1) or step_sec<=0:
        raise ValueError('invalid window or RL discount')
    period_ns=round(step_sec*1e9)
    keys=[tuple(r['episode_key']) for r in rows[i:j+2] if 'episode_key' in r]
    if keys and len(set(keys))!=1:raise ValueError('episode boundary in value/reward alignment')
    for k in range(i+1,j+1):
        if int(rows[k]['step'])!=int(rows[k-1]['step'])+1 or int(rows[k]['timestamp_ns'])-int(rows[k-1]['timestamp_ns'])!=period_ns:
            raise ValueError('control cadence mismatch')
        if k<j and (rows[k]['terminated'] or rows[k]['truncated']):raise ValueError('terminal inside window')
    if i+1>=len(rows):raise ValueError('initial value unavailable')
    v0=float(rows[i+1]['critic_value'])
    rewards=np.asarray([rows[k]['reward'] for k in range(i+1,j+1)],dtype=np.float64)
    values=np.asarray([rows[k]['critic_value'] for k in range(i+1,j+1)],dtype=np.float64)
    if not np.isfinite(v0) or not np.isfinite(rewards).all() or not np.isfinite(values).all():
        raise ValueError('nonfinite Critic or reward')
    n=j-i
    discounted=float(np.dot(np.power(gamma,np.arange(n,dtype=np.float64)),rewards))
    terminal=bool(rows[j]['terminated']);timeout=bool(rows[j]['truncated'])
    if terminal and timeout:raise ValueError('ambiguous terminal and timeout')
    if terminal or timeout:
        if j+1<len(rows):raise ValueError('episode boundary after terminal expected')
        vend=None
        if terminal:
            ret=discounted;bootstrap='true_terminal_zero'
        else:
            ret=discounted+(gamma**n)*float(rows[j]['critic_value'])
            bootstrap='ppo_timeout_preterminal'
        vmin=delta=drawdown=recovery=slope=None
    else:
        if j+1>=len(rows):raise ValueError('nonterminal endpoint value unavailable')
        if int(rows[j+1]['step'])!=int(rows[j]['step'])+1 or int(rows[j+1]['timestamp_ns'])-int(rows[j]['timestamp_ns'])!=period_ns:
            raise ValueError('endpoint value cadence mismatch')
        vend=float(rows[j+1]['critic_value'])
        if not math.isfinite(vend):raise ValueError('nonfinite endpoint Critic')
        series=np.concatenate((values,[vend]))
        vmin=float(series.min());delta=vend-v0;drawdown=vmin-v0;recovery=vend-vmin
        slope=_robust_slope(series,step_sec)
        ret=discounted+(gamma**n)*vend;bootstrap='endpoint_value'
    return {'start_timestamp_ns':int(rows[i]['timestamp_ns']),'end_timestamp_ns':int(rows[j]['timestamp_ns']),
            'reward_steps':n,'v0':v0,'v_end':vend,'v_min':vmin,'delta_v':delta,'drawdown':drawdown,
            'recovery':recovery,'critic_slope_per_sec':slope,'return_h':ret,'advantage_h':ret-v0,
            'bootstrap_kind':bootstrap,'primary_available':not (terminal or timeout),
            'reward_sequence':rewards.tolist(),'critic_sequence':values.tolist()+([] if vend is None else [vend])}


def fit_train_scalers(rows: Sequence[Mapping], names: Sequence[str]) -> dict[str,dict[str,float]]:
    """Fit z-score parameters using train labels only; reject missing input."""
    result={}
    for name in names:
        values=np.asarray([row[name] for row in rows],dtype=np.float64)
        if not len(values) or not np.isfinite(values).all():raise ValueError(f'invalid train label {name}')
        result[name]={'mean':float(values.mean()),'std':float(max(values.std(),1e-6)),'count':len(values)}
    return result


def normalize_labels(row: Mapping, scalers: Mapping[str,Mapping[str,float]]) -> dict[str,float]:
    return {name:(float(row[name])-values['mean'])/values['std'] for name,values in scalers.items()}
