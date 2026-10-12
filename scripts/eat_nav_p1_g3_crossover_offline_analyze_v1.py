"""Separate 12-trial dataset for two host-valid and one offline-audited period."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_crossover_v2_continuation import sha, verify_prior_pair, write_new
from eat_nav_p1_g3_crossover_v2_analyze import _trial
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment
from visualnav_transformer.eat_nav.p1_g3_crossover_runtime import paired_effects


SCHEMA = 'eatnav_p1_g3_crossover_offline_integrated_v1'


def require_audit_pass(result: dict) -> None:
    if result.get('decision') != 'audit_pass' or not result.get('checks') or \
            any(value != 'pass' for value in result['checks'].values()):
        raise ValueError('period 2 audit_pass required for integrated analysis')
    if result.get('original_period2_stage_preserved') != \
            'G3_crossover_batch_failed_or_inconclusive':
        raise ValueError('original period 2 failed stage must remain explicit')


def analyze(root: Path, audit_root: Path) -> dict:
    root, audit_root = Path(root), Path(audit_root)
    audit_path = audit_root/'audit-result.json'
    audit = json.loads(audit_path.read_text())
    require_audit_pass(audit)
    manifest = json.loads((root/'continuation-manifest.json').read_text())
    contract = json.loads((root/'manifest.json').read_text())
    verify_prior_pair(root, manifest['prior_periods'])
    assignment = json.loads((root/'g3-assignment.json').read_text())
    if sha(root/'g3-assignment.json') != manifest['assignment_sha256']:
        raise ValueError('frozen assignment SHA drift')
    trials = []
    provenance = []
    for period in range(3):
        batch = root/f'batch-first-differential-0-period-{period}'
        seal = verify_seal(batch)
        expected_stage = ('G3_crossover_batch_protocol_valid' if period < 2 else
                          'G3_crossover_batch_failed_or_inconclusive')
        if seal['stage'] != expected_stage:
            raise ValueError('original period seal changed')
        if period == 2 and sha(batch/'host-seal.json') != audit['period2_original_host_seal_sha256']:
            raise ValueError('original period 2 seal SHA changed')
        process = next(p for p in assignment['processes'] if p['anchor_id']=='first-differential'
                       and p['batch_index']==0 and p['execution_index']==period)
        frozen = next(p['trials'] for p in contract['processes'] if
                      p['anchor_id']=='first-differential' and p['batch_index']==0 and
                      p['execution_index']==period)
        validate_frozen_assignment(frozen, process['trials'], stage='prelaunch')
        output = batch/'output'
        marker = json.loads((output/'complete.json').read_text())
        summary = json.loads((output/'pilot-summary.json').read_text())
        publication = json.loads((output/'path-publication.json').read_text())
        terrain = json.loads((output/'terrain.json').read_text())
        validate_frozen_assignment(frozen, publication['trial_paths'], stage='post_path')
        for env in range(4):
            path = output/f'trace-slot-{env}.jsonl'
            if sha(path) != marker['trace_sha256'][env]:
                raise ValueError('trace SHA mismatch')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if len(rows) != marker['control_steps']:
                raise ValueError('trace step count mismatch')
            declared = publication['trial_paths'][env]
            identity = {'anchor_id':process['anchor_id'],'batch_seed':process['batch_seed'],
                        'env_id':env,'execution_index':period,
                        'assignment_period':process['assignment_period'],
                        'path_sha256':declared['path_sha256']}
            trial = _trial(rows, summary['arrivals'][env], declared,
                           terrain['world_tile_origins'][env], identity)
            trial['evidence_class'] = ('original_host_protocol_valid' if period < 2 else
                                       'offline_audit_pass_candidate')
            trials.append(trial)
        provenance.append({'period':period,'original_stage':seal['stage'],
                           'seal_sha256':sha(batch/'host-seal.json'),
                           'host_source_sha256':sha(batch/'host-runner.py'),
                           'simulator_source_sha256':sha(batch/'source.py'),
                           'manifest_sha256':sha(batch/'input/g3-contract.json'),
                           'trace_sha256':marker['trace_sha256']})
    if len(trials) != 12:
        raise ValueError('12-trial coverage incomplete')
    paired = paired_effects(trials)
    return {'schema':SCHEMA,'analysis_source_sha256':sha(Path(__file__)),
            'audit_result_sha256':sha(audit_path),
            'criteria_sha256':audit['criteria_sha256'],
            'assignment_sha256':manifest['assignment_sha256'],
            'independent_unit':'one StateGroup × one seed bundle; four env IDs nested',
            'original_host_protocol_valid_trials':8,
            'offline_audit_pass_candidate_trials':4,
            'original_period2_seal_remains_failed':True,
            'trial_count':len(trials),'provenance_by_period':provenance,
            'paired':paired,'trials':trials,
            'interpretation':'Descriptive Path response for one seed; different endpoints and geometry prevent navigation-quality or population superiority claims.'}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--audit-root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source_root, args.audit_root)
    write_new(args.audit_root/'integrated-dataset-v1.json', result)
    print('integrated', result['trial_count'], 'trials')


if __name__ == '__main__':
    main()
