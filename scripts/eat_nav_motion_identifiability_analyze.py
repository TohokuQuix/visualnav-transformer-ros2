"""Read sealed B-MI trials and report seed-clustered candidate motion effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from visualnav_transformer.eat_nav.motion_identifiability import (
    SCHEMA, CANDIDATES, candidate_decision, seed_effects)
from eat_nav_motion_identifiability_host import sha, verify_sim_seal
from eat_nav_motion_identifiability_schedule import assignment_rows, FROZEN


def analyze(root: Path) -> dict:
    root = Path(root)
    frozen = json.loads((root/'frozen-provenance.json').read_text())
    if frozen['schema'] != SCHEMA or frozen['manifest_sha256'] != sha(FROZEN):
        raise ValueError('schedule provenance mismatch')
    assignments = assignment_rows(json.loads(FROZEN.read_text()))
    if frozen['assignments'] != assignments:
        raise ValueError('assignment provenance mismatch')
    rows = []
    for ordinal, assignment in enumerate(assignments):
        name = f"{ordinal:02d}-seed{assignment['seed']}-r{assignment['repeat']}-{assignment['candidate']}"
        attempt = root/name
        if not json.loads((attempt/'decision.json').read_text())['pass']:
            raise ValueError(f'failed process {name}')
        verify_sim_seal(attempt/'sim-output')
        manifest = json.loads((attempt/'manifest.json').read_text())
        if any(manifest[key] != assignment[key] for key in
               ('seed', 'terrain_type', 'terrain_level', 'condition', 'repeat')):
            raise ValueError(f'input mismatch {name}')
        if manifest['protocol_sha256'] != frozen['protocol_sha256'] or \
                manifest['source_sha256'] != frozen['sim_source_sha256'] or \
                manifest['host_source_sha256'] != frozen['host_source_sha256'] or \
                manifest['analysis_source_sha256'] != frozen['analysis_source_sha256']:
            raise ValueError(f'source/protocol SHA mismatch {name}')
        labels = json.loads((attempt/'sim-output/motion-labels.json').read_text())['labels']
        if len(labels) != 4:
            raise ValueError('four env labels required')
        for env, label in enumerate(labels):
            if not label['available']:
                raise ValueError('incomplete 2 s label')
            rows.append({'seed': assignment['seed'], 'condition': assignment['condition'],
                         'repeat': assignment['repeat'], 'env': env,
                         'candidate': assignment['candidate'],
                         'xy': label['delta_body_xy_m'],
                         'yaw': label['delta_body_yaw_rad'],
                         'seal_sha256': sha(attempt/'outer-seal.json')})
    decisions = {}
    for candidate in CANDIDATES:
        if candidate.endswith('direct'):
            continue
        estimate = seed_effects(rows, candidate)
        decisions[candidate] = {**estimate, 'decision': candidate_decision(estimate)}
    direction = {}
    for role in ('forward', 'reverse'):
        states = [decisions[key]['decision']['status'] for key in decisions if key.startswith(role)]
        direction[role] = ('pass' if 'pass' in states else
                           'fail' if all(state == 'fail' for state in states) else 'inconclusive')
    overall = ('pass' if all(value == 'pass' for value in direction.values()) else
               'fail' if all(value == 'fail' for value in direction.values()) else 'inconclusive')
    return {'schema': SCHEMA, 'protocol_sha256': frozen['protocol_sha256'],
            'manifest_sha256': frozen['manifest_sha256'], 'row_count': len(rows),
            'independent_terrain_seeds': 6, 'candidate_decisions': decisions,
            'direction_decisions': direction, 'mi_a_decision': overall,
            'rows': rows}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    output = args.root/'mi_a_paired_motion_results.json'
    if output.exists():
        raise FileExistsError('analysis output already exists')
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    print(json.dumps({'decision': result['mi_a_decision'],
                      'direction': result['direction_decisions'],
                      'rows': result['row_count']}, sort_keys=True))
