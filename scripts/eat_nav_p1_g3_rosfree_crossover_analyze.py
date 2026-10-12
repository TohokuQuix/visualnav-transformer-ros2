"""Read-only period-zero analysis for the independent ROS-free crossover."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_crossover_v2_analyze import _trial
from visualnav_transformer.eat_nav.p1_g3_rosfree_crossover_evidence import (
    SCHEMA, validate_output,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_trial_assignment(rows):
    if ([row['env_id'] for row in rows] != list(range(4)) or
            [row['path_role'] for row in rows] !=
            ['direct','direct','left_20','right_20']):
        raise ValueError('period-zero frozen assignment mismatch')


def analyze(root):
    root=Path(root)
    manifest=json.loads((root/'manifest.json').read_text())
    if (manifest['schema'] != SCHEMA or
            manifest.get('authorized_process_ids') !=
            ['batch-first-differential-0-period-0'] or
            sha(Path(__file__)) != manifest['analysis_source_sha256']):
        raise ValueError('ROS-free crossover analysis provenance mismatch')
    batch=root/'batch-first-differential-0-period-0'
    seal=verify_seal(batch)
    if seal['stage'] != 'G3_rosfree_crossover_protocol_valid':
        raise ValueError('period zero was not protocol-valid')
    assignment=json.loads((root/'g3-assignment.json').read_text())
    if sha(root/'g3-assignment.json') != manifest['assignment_sha256']:
        raise ValueError('assignment SHA drift')
    assigned=assignment['processes'][0]['trials']
    validate_trial_assignment(assigned)
    output=batch/'output'
    control=validate_output(output,manifest,assigned)
    publication=json.loads((output/'path-activation.json').read_text())['trial_paths']
    summary=json.loads((output/'pilot-summary.json').read_text())
    terrain=json.loads((output/'terrain.json').read_text())
    trials=[]
    for env in range(4):
        path=output/f'trace-slot-{env}.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        identity={'anchor_id':'first-differential','batch_seed':2026101101,
                  'env_id':env,'execution_index':0,'assignment_period':0,
                  'path_sha256':assigned[env]['path_sha256']}
        trials.append(_trial(rows,summary['arrivals'][env],publication[env],
                             terrain['world_tile_origins'][env],identity))
    validate_trial_assignment(trials)
    return {'schema':SCHEMA+'_period_zero_analysis_v1',
            'status':'period_zero_only_no_candidate_minus_direct_pairing',
            'independent_conditions':1,'trial_count':4,
            'manifest_sha256':sha(root/'manifest.json'),
            'host_seal_sha256':sha(batch/'host-seal.json'),
            'control_validation':control,'trials':trials,
            'interpretation':'Local arrival is separate from common-goal progress.'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.root)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,sort_keys=True,indent=2,allow_nan=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(args.output)


if __name__=='__main__':main()
