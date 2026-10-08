#!/usr/bin/env python3
"""Apply frozen P1 role taxonomy to eight SHA-verified physical anchors only."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from scripts.eat_nav_curvature_attainable import sha
from scripts.eat_nav_local_primitive_review import verify_inputs, terrain_support_for_anchor
from visualnav_transformer.eat_nav.goal075_offline import class_proxy_interval
from visualnav_transformer.eat_nav.local_motion_primitives import BEARINGS_DEG, DISTANCE_M, constant_curvature_primitive
from visualnav_transformer.eat_nav.role_aware_terrain import classify_role, select_five


ROOT = Path(__file__).resolve().parents[1]
BACKUP = Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')
PRIOR = Path('/tmp/eatnav-local-primitives-offline-20261008-r4')
OUTPUT = Path('/tmp/eatnav-local-primitive-p1-20261008-r1')
TAXONOMY = ROOT/'docs/eat-nav/local-primitive-p1-role-aware-20261008/role-aware-terrain-taxonomy.json'
TAXONOMY_SHA = 'b787e879bc0417132ab79b429d90ad44031dbdd075297b05d34aa6d798a1c60a'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')


def verify_prior(root):
    ledger = read(root/'artifact-sha-ledger.json')['files']
    for relative, entry in ledger.items():
        path = root/relative
        if sha(path) != entry['sha256'] or path.stat().st_size != entry['size_bytes']:
            raise ValueError('previous primitive review artifact SHA/size mismatch: '+relative)
    decision = read(root/'final-decision.json')
    if decision['decision'] != 'local_motion_primitive_terrain_support_fail':
        raise ValueError('prior decision unexpectedly changed')
    return {'root': str(root), 'artifact_ledger_sha256': sha(root/'artifact-sha-ledger.json'),
            'verified_file_count': len(ledger), 'prior_decision': decision['decision']}


def run():
    if sha(TAXONOMY) != TAXONOMY_SHA or sha(OUTPUT/'role-aware-terrain-taxonomy.json') != TAXONOMY_SHA:
        raise ValueError('taxonomy SHA changed after freeze')
    if sorted(OUTPUT.iterdir()) != [OUTPUT/'role-aware-terrain-taxonomy.json']:
        raise FileExistsError('P1 gate output root is not fresh')
    prior = verify_prior(PRIOR)
    frozen = verify_inputs(BACKUP, Path('/home/kojima/eatnav-backups'))
    anchors = frozen['anchors']
    if len(anchors) != 8:
        raise ValueError('eight physical anchors required')
    previous_paths = read(PRIOR/'primitive-path-definitions.json')['candidates']
    geometry = read(PRIOR/'primitive-geometry-audit.json')
    if not geometry['geometry_gate'] or not geometry['follower_gate'] or geometry['hard_valid_count'] != 40 or geometry['follower_compatible_count'] != 40:
        raise ValueError('previous five-path geometry or follower gate failed')
    curves = [constant_curvature_primitive(DISTANCE_M, bearing) for bearing in BEARINGS_DEG]
    if [(x['bearing_deg'], x['path_sha256']) for x in previous_paths] != [(c['bearing_deg'], c['path_sha256']) for c in curves]:
        raise ValueError('regenerated primitive Path SHA differs from frozen definition')
    geometry_by_anchor = {}
    for record in geometry['records']:
        geometry_by_anchor.setdefault(record['anchor_id'], []).append(record)
    rows = []
    for anchor in anchors:
        aid = anchor['anchor_id']
        records = geometry_by_anchor[aid]
        terrain = terrain_support_for_anchor(curves, anchor['dem'][0], anchor['dem'][1], anchor['proxy_bboxes'])
        valid_pairs = [p for p in terrain['pairs'] if p['primary_separated']]
        maximum = max(terrain['pairs'], key=lambda p: p['primary']['known_corridor_profile_rms_m'] or 0)
        paths = []
        for curve in curves:
            classifier = class_proxy_interval({'dense': curve['dense'], 'stations_m': curve['stations_m']},
                                              anchor['dem'][0], anchor['dem'][1], .6)
            coverage = terrain['candidate_coverage'][curve['variant_id']]
            known = (classifier['class_proxy'] != 'unknown' and
                     all(v == 1.0 for v in classifier['band_known_fraction'].values()) and
                     coverage['primary_centerline_known_fraction'] == 1.0)
            paths.append({'candidate_id': curve['variant_id'],
                          'path_sha256': curve['path_sha256'],
                          'class_proxy': classifier['class_proxy'],
                          'class_diagnostic': classifier,
                          'known': bool(known),
                          'known_fraction': min(classifier['band_known_fraction'].values()),
                          'onset_m': coverage['interaction_onset_primary_m'],
                          'slope': coverage['primary_main_track_slope'],
                          'step_m': coverage['primary_main_track_step_m'],
                          'roughness_m': coverage['primary_main_track_roughness_m']})
        row = {'anchor_id': aid, 'seed_label': anchor['seed_label'],
               'terrain': anchor['terrain'], 'terrain_seed': anchor['terrain_seed'],
               'source_family': anchor['source_family'],
               'selection_rank': anchor['selection_rank'],
               'pose_id': anchor['pose_id'], 'requested_xy_yaw': anchor['requested_xy_yaw'],
               'settled_map_xy_yaw': anchor['settled_map_xy_yaw'],
               'dem_history_sha256': anchor['dem_history_sha256'],
               'dem_file_sha256': anchor['dem_file_sha256'],
               'source_path_shas': anchor['source_path_shas'],
               'physical_eligible': bool(anchor['physical_supply']),
               'hard_valid': all(x['hard_valid'] for x in records),
               'follower_compatible': all(x['follower_compatible'] for x in records),
               'paths': paths, 'pair_count': len(valid_pairs),
               'max_pair_rms_m': maximum['primary']['known_corridor_profile_rms_m'] or 0,
               'max_pair': {'first': maximum['first'], 'second': maximum['second'],
                            'primary': maximum['primary']},
               'primary_pair_metrics': [{'first': p['first'], 'second': p['second'],
                                        'primary': p['primary']} for p in valid_pairs]}
        rows.append(classify_role(row))
    selection = select_five(rows)
    source = {'schema': 'eatnav_p1_role_gate_source_provenance_v1',
              'base_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'branch': subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              'worktree': str(ROOT), 'taxonomy_sha256': TAXONOMY_SHA,
              'previous_primitive_review': prior,
              'prior_source_provenance_sha256': sha(PRIOR/'source-provenance.json'),
              'current_source_provenance': frozen['provenance'],
              'role_source_sha256': sha(ROOT/'src/visualnav_transformer/eat_nav/role_aware_terrain.py'),
              'gate_runner_sha256': sha(Path(__file__)),
              'candidate_outcome_read': False}
    gate = {'schema': 'eatnav_p1_role_aware_terrain_gate_v1',
            'taxonomy_sha256': TAXONOMY_SHA, 'primary_interval_arc_m': [0, .6],
            'candidate_outcome_read': False, 'anchors': rows,
            'selection': selection, 'pass': selection['pass'],
            'prior_gate_decision_preserved': prior['prior_decision']}
    write(OUTPUT/'source-provenance.json', source)
    write(OUTPUT/'role-aware-terrain-gate.json', gate)
    write(OUTPUT/'input-freeze.json', {'taxonomy_sha256': TAXONOMY_SHA,
                                      'previous_primitive_review_ledger_sha256': prior['artifact_ledger_sha256'],
                                      'physical_anchor_count': len(rows),
                                      'primitive_path_sha256': {x['variant_id']: x['path_sha256'] for x in curves},
                                      'outcome_access': False})
    if not selection['pass']:
        decision = 'role_aware_terrain_support_insufficient'
        write(OUTPUT/'selected-five-anchors.json', {'selected': [], 'status': 'not_selected_gate_failed'})
        write(OUTPUT/'p1-run-manifest.json', {'processes': [], 'status': 'not_created_gate_failed'})
    else:
        decision = 'role_gate_passed_pending_x1_x2'
    write(OUTPUT/'final-decision.json', {'decision': decision,
                                         'previous_decision_preserved': prior['prior_decision'],
                                         'candidate_rollout_started': False})
    report = [
        '# P1 role-aware terrain gate', '',
        f'Taxonomy SHA: `{TAXONOMY_SHA}`.',
        f'Previous gate decision: `{prior["prior_decision"]}` (unchanged).',
        f'New gate: **{"pass" if selection["pass"] else "fail"}**.',
        'Only SHA-verified physical anchor, DEM, Path and source records were used; no candidate Outcome was read.', '',
        '| Anchor | Seed | Source family | Primary role | Secondary flags | Known paths | Differential pairs |',
        '|---|---|---|---|---|---:|---:|',
    ]
    for row in rows:
        flags = ', '.join(k for k,v in row['flags'].items() if v and k != row['primary_role']) or '—'
        report.append(f"| {row['anchor_id']} | {row['seed_label']} | {row['source_family']} | {row['primary_role'] or 'none'} | {flags} | {sum(p['known'] for p in row['paths'])}/5 | {row['pair_count']} |")
    report += ['', f'Missing primary roles: `{selection["missing_roles"]}`.',
               f'Decision: `{decision}`.',
               'P1 simulator and follower were not started.' if not selection['pass'] else 'Proceed to frozen five-anchor manifest before X1.']
    (OUTPUT/'report.md').write_text('\n'.join(report)+'\n')
    files = {str(p.relative_to(OUTPUT)): {'sha256': sha(p), 'size_bytes': p.stat().st_size}
             for p in OUTPUT.rglob('*') if p.is_file() and p.name != 'artifact-sha-ledger.json'}
    write(OUTPUT/'artifact-sha-ledger.json', {'files': files})
    return selection


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
