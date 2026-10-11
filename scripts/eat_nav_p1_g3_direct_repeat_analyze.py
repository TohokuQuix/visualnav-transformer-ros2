"""Read-only analysis of sealed four-follower direct Path repeat batches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_direct_repeat_stage import verify_stage, durable_json
from visualnav_transformer.eat_nav.p1_g3_direct_repeat import (
    extract_trial, repeatability_summary, validate_terrain_support,
)


PROTOCOL = 'eatnav_p1_longhorizon_g3_direct_repeat_v1'


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_batch(batch: Path, *, batch_seed: int) -> dict:
    batch = Path(batch)
    seal = verify_seal(batch)
    if seal['stage'] != 'G3_batch_protocol_valid':
        raise ValueError('direct repeat batch seal is not protocol-valid')
    status = json.loads((batch / 'g3-host-status.json').read_text())
    cleanup = json.loads((batch / 'cleanup-evidence.json').read_text())
    if status.get('failure') is not None or status.get('native_exit') != 0 or \
            status.get('validation') is None or cleanup.get('owned_residual') or \
            cleanup.get('bridge_residual'):
        raise ValueError('direct repeat host or cleanup invalid')
    output = batch / 'output'
    marker = json.loads((output / 'complete.json').read_text())
    provenance_path = output / 'process-provenance.json'
    if digest(provenance_path) != marker['process_provenance_sha256']:
        raise ValueError('direct repeat process provenance SHA mismatch')
    provenance = json.loads(provenance_path.read_text())
    summary = json.loads((output / 'pilot-summary.json').read_text())
    publication = json.loads((output / 'path-publication.json').read_text())
    terrain = json.loads((output / 'terrain.json').read_text())
    duplicate = json.loads((batch / 'duplicate-gate.json').read_text())
    if marker.get('protocol_version') != PROTOCOL or marker.get('Path_publications') != 1 or \
            len(marker.get('trace_sha256', [])) != 4 or \
            len(publication.get('trial_paths', [])) != 4 or \
            len(summary.get('arrivals', [])) != 4 or \
            len(terrain.get('fingerprints', [])) != 4 or \
            not json.loads((output / 'prepath-gate.json').read_text())['pass'] or \
            len(duplicate.get('pairs', [])) != 2 or \
            [pair['env_pair'] for pair in duplicate['pairs']] != [[0, 1], [2, 3]]:
        raise ValueError('direct repeat completion or paired gate invalid')
    paths = publication['trial_paths']
    if len({path['path_sha256'] for path in paths}) != 1:
        raise ValueError('direct Path SHA differs across env')
    if (provenance['anchor_id'] != summary['anchor_id'] or
            provenance['batch_index'] != summary['batch_index'] or
            provenance['repeat_index'] != summary['repeat_index'] or
            provenance['batch_seed'] != batch_seed or
            provenance['path_sha256'] != paths[0]['path_sha256']):
        raise ValueError('direct repeat process provenance identity mismatch')
    meshes = {row['canonical_mesh_sha256'] for row in terrain['fingerprints']}
    if len(meshes) != 1:
        raise ValueError('direct repeat within-batch terrain mesh drift')
    mesh_sha = next(iter(meshes))
    terrain_key = mesh_sha + ':' + terrain['material_sha256']
    trials = []
    for slot in range(4):
        path = output / f'trace-slot-{slot}.jsonl'
        if digest(path) != marker['trace_sha256'][slot]:
            raise ValueError('direct repeat trace SHA mismatch')
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        pair = duplicate['pairs'][slot // 2]
        identity = {'anchor_id': summary['anchor_id'], 'batch_seed': batch_seed,
                    'batch_index': summary['batch_index'],
                    'repeat_index': summary['repeat_index'], 'env_id': slot,
                    'terrain_condition_key': terrain_key,
                    'path_sha256': paths[slot]['path_sha256'],
                    'duplicate_gate_pass': bool(pair['pass'])}
        trials.append(extract_trial(rows, summary['arrivals'][slot],
                                    paths[slot]['map_xy'],
                                    terrain['world_tile_origins'][slot], identity))
    return {'batch': batch.name, 'trials': trials,
            'duplicate_gate': {'pass_pairs': sum(pair['pass'] for pair in duplicate['pairs']),
                               'fail_pairs': sum(not pair['pass'] for pair in duplicate['pairs'])},
            'terrain': {'anchor_id': summary['anchor_id'], 'batch_seed': batch_seed,
                        'mesh_sha256': mesh_sha,
                        'material_sha256': terrain['material_sha256']},
            'host_seal_sha256': digest(batch / 'host-seal.json'),
            'process_provenance_sha256': digest(provenance_path)}


def aggregate_root(root: Path) -> dict:
    root = Path(root)
    stage = json.loads((root / 'stage-complete.json').read_text())
    manifest = json.loads((root / 'manifest.json').read_text())
    if digest(Path(__file__)) != manifest['analysis_source_sha256']:
        raise ValueError('direct-repeat analysis source SHA drift')
    assignment = json.loads((root / 'g3-assignment.json').read_text())
    if stage['schema'] != PROTOCOL or stage['process_count'] != 12 or \
            stage['trial_count'] != 48 or digest(root / 'manifest.json') != stage['manifest_sha256'] or \
            digest(root / 'g3-assignment.json') != stage['assignment_sha256'] or \
            manifest['assignment_sha256'] != stage['assignment_sha256'] or \
            len(assignment['processes']) != 12:
        raise ValueError('direct repeat staged manifest drift')
    batches = []
    for process in assignment['processes']:
        name = (f"batch-{process['anchor_id']}-{process['batch_index']}-"
                f"repeat-{process['repeat_index']}")
        batch = root / name
        if digest(batch / 'input-sha-ledger.json') != stage['batch_input_ledgers'][name]:
            raise ValueError('direct repeat staged ledger drift')
        verify_stage(batch, manifest)
        result = collect_batch(batch, batch_seed=process['batch_seed'])
        if result['terrain']['anchor_id'] != process['anchor_id'] or \
                any(row['repeat_index'] != process['repeat_index'] for row in result['trials']):
            raise ValueError('direct repeat batch identity mismatch')
        batches.append(result)
    trials = [trial for batch in batches for trial in batch['trials']]
    terrains = validate_terrain_support([batch['terrain'] for batch in batches])
    result = {'schema': PROTOCOL + '_aggregate_v1',
              'interpretation': 'descriptive_repeatability_not_path_effect_or_detection_guarantee',
              'manifest_sha256': stage['manifest_sha256'],
              'assignment_sha256': stage['assignment_sha256'],
              'terrain_support': terrains, 'repeatability': repeatability_summary(trials),
              'trials': trials,
              'batch_host_seals': {batch['batch']: batch['host_seal_sha256'] for batch in batches}}
    if result['repeatability']['trial_count'] != 48:
        raise ValueError('direct repeat 48-trial completeness')
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = aggregate_root(args.root)
    if args.output.exists():
        raise FileExistsError('analysis output already exists')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    durable_json(args.output, result)
    print(args.output)


if __name__ == '__main__':
    main()
