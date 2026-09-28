"""Compute reproducibility diagnostics from immutable Phase C trace arrays."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np

TOLERANCES = {
    'command': 1e-5, 'pre_noise': 1e-5, 'post_noise': 1e-5,
    'actor_input': 1e-5, 'actor_latent': 1e-5, 'actor_action': 1e-5,
    'processed_action': 1e-5, 'joint_target': 1e-5,
    'root_state': 1e-4, 'joint_position': 1e-5, 'joint_velocity': 1e-5,
    'contact_force': 1.0, 'raycast': 1e-4, 'critic_obs': 1e-5,
    'critic_value': 1e-4, 'reward': 1e-4, 'xy': 1e-4,
    'contact_max': 1.0, 'dem_height': 1e-4,
}

def signal_report(array, tol):
    x = np.asarray(array, dtype=np.float64)
    if x.ndim < 2 or x.shape[1] < 2:
        raise ValueError('trace must have time and branch axes')
    delta = x[:, 1:] - x[:, :1]
    finite = np.isfinite(delta)
    abs_delta = np.where(finite, np.abs(delta), 0)
    by_step = abs_delta.reshape(len(x), -1).max(axis=1)
    first = np.flatnonzero(by_step > tol)
    per_branch = np.sqrt(np.mean(np.where(finite, delta*delta, 0), axis=tuple([0]+list(range(2,delta.ndim)))))
    ids = np.arange(1, x.shape[1])
    branch_corr = float(np.corrcoef(ids, per_branch)[0,1]) if np.std(per_branch) > 0 and len(ids)>2 else None
    return {'max_abs': float(abs_delta.max()), 'rms': float(np.sqrt(np.mean(np.where(finite, delta*delta, 0)))),
            'first_step': int(first[0]) if len(first) else None,
            'previous_step_rms': float(np.sqrt(np.mean(delta[first[0]-1]**2))) if len(first) and first[0] else None,
            'tolerance': tol, 'branch_rms': per_branch.tolist(), 'branch_id_correlation': branch_corr,
            'nonfinite_count': int((~finite).sum())}

def trace_report(path):
    with np.load(path) as data:
        return {key: signal_report(data[key], TOLERANCES[key]) for key in TOLERANCES if key in data}

def atomic_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n'); tmp.replace(path)

def main():
    p=argparse.ArgumentParser(); p.add_argument('trace',type=Path); p.add_argument('output',type=Path)
    a=p.parse_args(); report=trace_report(a.trace); atomic_json(a.output,report)
    print(json.dumps({k:{'first_step':v['first_step'],'rms':v['rms']} for k,v in report.items()},sort_keys=True))
if __name__=='__main__': main()
