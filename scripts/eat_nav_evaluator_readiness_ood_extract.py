"""Extract P/C/R two-second 8-point paths for frozen train-only OOD screen."""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
import os
import numpy as np
from visualnav_transformer.eat_nav.command_preview_long_horizon import post_root
from visualnav_transformer.eat_nav.feasible_transfer import integrate_commands, map_xy_to_base

ROOT = Path(os.environ.get('EAT_READINESS_ROOT', '/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot'))
DOC = Path('/tmp/eatnav-vnt-candidate-coverage/docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot')
INPUT = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')


def main():
    if json.loads((ROOT/'status.json').read_text())['stage'] != 'RAW_COMPLETE':
        raise RuntimeError('OOD extraction requires all raw outcomes')
    runs = json.loads((DOC/'schedule.json').read_text())['runs']
    rows = []
    for entry in runs:
        if entry['kind'] == 'anchor' or (entry['kind'] == 'nominal' and entry['repeat'] != 0):
            continue
        index = entry['state_index']
        folder = ROOT/'runs'/entry['run_id']
        summary = json.loads((folder/'live-0.json').read_text())
        with gzip.open(folder/'live-0.trace.json.gz', 'rt') as stream:
            trace = json.load(stream)
        planned = np.asarray(json.loads((INPUT/f'state-{index:03d}'/f"{entry['candidate_id']}-base-path.json").read_text()), dtype=float)[1:]
        if planned.shape != (8,2):
            raise ValueError('planned path shape mismatch')
        first = trace[:40]
        command = np.asarray([r['command'] for r in first], dtype=float)
        integrated = integrate_commands(command, dt_sec=.05)
        realized_map = np.asarray([post_root(r)[:2] for r in first], dtype=float)
        root = summary['post_follower_anchor']['root_state']
        w,x,y,z = map(float,root[3:7])
        yaw = math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        realized = map_xy_to_base(realized_map, anchor_xy=root[:2], yaw_rad=yaw)
        if len(first) < 40:
            integrated = np.concatenate((integrated,np.repeat(integrated[-1:],40-len(first),axis=0)))
            realized = np.concatenate((realized,np.repeat(realized[-1:],40-len(first),axis=0)))
        paths = {'P':planned,'C':integrated[np.arange(4,40,5)],
                 'R':realized[np.arange(4,40,5)]}
        if any(v.shape!=(8,2) or not np.isfinite(v).all() for v in paths.values()):
            raise ValueError('P/C/R path shape/nonfinite')
        rows.append({'run_id':entry['run_id'],'state_id':entry['state_id'],
                     'candidate_id':entry['candidate_id'],
                     'paths':{k:v.tolist() for k,v in paths.items()}})
    if len(rows)!=98:
        raise ValueError('98 nominal/candidate P/C/R rows required')
    target=ROOT/'analysis/ood-paths.json';target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        if json.loads(target.read_text())!=rows:raise FileExistsError('OOD paths changed')
    else:
        tmp=target.with_suffix('.tmp');tmp.write_text(json.dumps(rows,sort_keys=True,separators=(',',':'))+'\n');tmp.replace(target)
    print('READINESS_OOD_PATHS_EXTRACTED',len(rows),flush=True)


if __name__=='__main__':
    main()
