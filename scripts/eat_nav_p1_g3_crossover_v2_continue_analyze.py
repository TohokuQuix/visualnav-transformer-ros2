"""Read-only integrated analysis of two sealed old periods and one continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_crossover_v2_analyze import analyze
from eat_nav_p1_g3_crossover_v2_continuation import SCHEMA, sha, verify_prior_pair, write_new


def analyze_continuation(root: Path) -> dict:
    root=Path(root)
    provenance=json.loads((root/'continuation-manifest.json').read_text())
    contract=json.loads((root/'manifest.json').read_text())
    if provenance['schema']!=SCHEMA or \
            sha(Path(__file__))!=provenance['continuation_analyzer_source_sha256'] or \
            sha(root/'manifest.json')!=provenance['period2_manifest_sha256']:
        raise ValueError('continuation manifest drift')
    verify_prior_pair(root,provenance['prior_periods'])
    period2=root/'batch-first-differential-0-period-2'
    if verify_seal(period2)['stage']!='G3_crossover_batch_protocol_valid' or \
            sha(period2/'host-runner.py')!=provenance['period2_host_source_sha256'] or \
            sha(period2/'source.py')!=provenance['period2_simulator_source_sha256']:
        raise ValueError('period 2 not protocol-valid with frozen source')
    result=analyze(root,contract)
    if result['trial_count']!=12 or result['paired']['independent_seed_conditions']!=1:
        raise ValueError('one-seed 12-trial crossover incomplete')
    grouped={}
    for trial in result['trials']:
        key=(trial['anchor_id'],trial['batch_seed'],trial['env_id'])
        grouped.setdefault(key,[]).append(trial['path_role'])
    if len(grouped)!=4 or any(set(roles)!={'direct','left_20','right_20'} or
                              len(roles)!=3 for roles in grouped.values()):
        raise ValueError('Direct/Left/Right coverage incomplete by env')
    result['cross_attempt']={
        'schema':SCHEMA,'single_source_across_periods_claimed':False,
        'period0_1_manifest_sha256':provenance['original_manifest_sha256'],
        'period2_manifest_sha256':provenance['period2_manifest_sha256'],
        'source_by_period':{
            '0':{'host_sha256':provenance['prior_periods'][0]['host_source_sha256'],
                 'simulator_sha256':provenance['prior_periods'][0]['simulator_source_sha256']},
            '1':{'host_sha256':provenance['prior_periods'][1]['host_source_sha256'],
                 'simulator_sha256':provenance['prior_periods'][1]['simulator_source_sha256']},
            '2':{'host_sha256':provenance['period2_host_source_sha256'],
                 'simulator_sha256':provenance['period2_simulator_source_sha256']}},
        'comparison_scope':'first-differential / one seed bundle / four env IDs; descriptive only',
        'geometry_warning':'curved and direct endpoints differ; local arrival alone is not navigation quality'}
    return result


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    result=analyze_continuation(args.root)
    write_new(args.output,result)


if __name__=='__main__':main()
