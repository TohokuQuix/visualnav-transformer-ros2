"""Sealed read-only Phase A/D audit of the frozen P1 pre-Path runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from itertools import combinations

from visualnav_transformer.eat_nav.p1_batch_golden import verify_backup
from visualnav_transformer.eat_nav.p1_prepath_divergence import (
    gate_checks, initial_state_contract, pair_metrics,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text())


def write(root: Path, name: str, value):
    (root / name).write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')


def verify_batch(root: Path) -> dict:
    ledger = load(root / 'sha-ledger.json')
    for row in ledger['files']:
        path = root / row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'batch SHA failure: {path}')
    seal = load(root / 'seal.json')
    if sha(root / 'sha-ledger.json') != seal['sha_ledger_sha256']:
        raise ValueError('batch ledger seal mismatch')
    if sha(root / 'prepath-decision.json') != seal['decision_sha256']:
        raise ValueError('batch decision seal mismatch')
    return {'seal_sha256': sha(root / 'seal.json'), 'verified_files': len(ledger['files'])}


def audit(golden: Path, v1: Path, v2: Path, tolerance: dict, tolerance_path: Path) -> dict:
    seals = {'golden': {'seal_sha256': sha(golden / 'backup-seal.json'),
                        'verified_files': verify_backup(golden)['verified_file_count']},
             'v1': verify_batch(v1), 'v2': verify_batch(v2)}
    plan = load(v2 / 'input/plan.json')
    fixtures = load(v2 / 'input/anchor-fixtures.json')
    records = load(v2 / 'prepath-run/prepath-records.json')
    d1 = load(v2 / 'prepath-audit.json')
    by_index = {r['target_env_index']: r for r in records}
    if len(by_index) != 28 or len(plan['targets']) != 28 or len(d1['rows']) != 28:
        raise ValueError('target record count mismatch')
    field_rows, terrain_rows = [], []
    for name in fixtures:
        target = next(t for t in plan['targets'] if t['anchor_id'] == name)
        pose = target['source_pose_id']
        source = load(golden / 'runs' / target['physical_run_id'] / 'run/reset-pre-physics.json')
        saved = source['radial'][pose]
        fixture = fixtures[name]['prephysics_state']
        if saved != fixture:
            raise ValueError(f'source fixture mismatch: {name}')
        contract = initial_state_contract(fixture)
        field_rows.append({'anchor_id': name, 'source_prephysics_fixture_exact': True,
                           'source_reset_to_radial_z_exact': source['reset'][pose]['root_state_w'][2] == saved['root_state_w'][2],
                           'source_z_m': saved['root_state_w'][2],
                           'source_root_velocity': saved['root_state_w'][7:13],
                           'fields': contract})
        mesh_file = golden / 'runs' / target['physical_run_id'] / 'run/terrain-mesh-identity.json'
        terrain_rows.append({'anchor_id': name,
                             'source_whole_scene_mesh': load(mesh_file),
                             'source_terrain_metadata': source['terrain_metadata'][pose],
                             'target_terrain_metadata': by_index[target['target_env_index']]['terrain_metadata'],
                             'source_tile_origin_xy': target['source_tile_origin_xy'],
                             'target_tile_origin_xy': by_index[target['target_env_index']]['target_tile_origin_xy'],
                             'tile_local_mesh_sha256': None,
                             'collision_mesh_identity': 'unresolved_no_tile_vertices_or_faces_saved_in_batch_v2',
                             'material_identity': 'source_and_batch_share_cfg_factory_but_live_material_not_saved'})
    direct = []
    for group in sorted(set(t['anchor_id'] for t in plan['targets'])):
        rows = [t for t in plan['targets'] if t['anchor_id'] == group and '-b+00-r0' in t['physical_run_id']]
        if len(rows) != 3:
            raise ValueError(f'expected direct triplicate: {group}')
        for a, b in combinations(rows, 2):
            metrics = pair_metrics(by_index[a['target_env_index']], by_index[b['target_env_index']])
            checks = gate_checks(metrics, tolerance)
            direct.append({'anchor_id': group, 'a': a['physical_run_id'],
                           'b': b['physical_run_id'], 'metrics': metrics,
                           'diagnostic_frozen_criteria_checks': checks,
                           'diagnostic_all_pass': all(checks.values())})
    d1_rows = d1['rows']
    return {
        'schema': 'p1_prepath_static_audit_v1', 'reference_seals': seals,
        'input_sha256': {'v2_plan': sha(v2 / 'input/plan.json'),
                         'v2_fixtures': sha(v2 / 'input/anchor-fixtures.json'),
                         'v2_records': sha(v2 / 'prepath-run/prepath-records.json'),
                         'frozen_tolerance': sha(tolerance_path)},
        'initial_state_fields': field_rows, 'terrain_provenance': terrain_rows,
        'D1': {'question': 'batch target versus historical golden',
               'pass_count': sum(all(x['checks'].values()) for x in d1_rows),
               'target_count': len(d1_rows),
               'fail_count_by_field': {k: sum(not r['checks'][k] for r in d1_rows)
                                       for k in d1_rows[0]['checks']},
               'rows': d1_rows},
        'D2': {'question': 'within-batch direct clone pairwise at step 200',
               'pair_count': len(direct),
               'diagnostic_all_pass_count': sum(x['diagnostic_all_pass'] for x in direct),
               'pairs': direct},
        'temporal_limit': 'batch v2 only saves final step 200; no batch post-reset/step-0 or per-step traces, so earliest divergence step and causal ordering cannot be measured offline',
        'gate_consequence': 'original frozen pre-Path gate remains failed; no Path publication authorized',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--v1', type=Path, required=True)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--tolerance', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = audit(args.golden, args.v1, args.v2,
                   load(args.tolerance)['pre_path_tolerance'], args.tolerance)
    write(args.output, 'phase-a-static-audit.json', result)
    ledger = {'files': [{'path': 'phase-a-static-audit.json',
                         'sha256': sha(args.output / 'phase-a-static-audit.json'),
                         'bytes': (args.output / 'phase-a-static-audit.json').stat().st_size}],
              'source_sha256': sha(Path(__file__)),
              'candidate_Path_publications': 0,
              'simulator_processes': 0}
    write(args.output, 'sha-ledger.json', ledger)
    write(args.output, 'seal.json', {'schema': 'p1_prepath_phase_a_seal_v1',
                                    'ledger_sha256': sha(args.output / 'sha-ledger.json')})
