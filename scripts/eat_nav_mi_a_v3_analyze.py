"""Read-only sealed MI-A v3 audit and seed-clustered motion analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.mi_a_v3 import CONDITIONS, SCHEMA
try:
    from .eat_nav_mi_a_v3_run import MANIFEST, PROTOCOL, load_frozen, row_name, verify_attempt, atomic_json
    from .eat_nav_mi_fi_v1_host import sha
except ImportError:  # direct script execution
    from eat_nav_mi_a_v3_run import MANIFEST, PROTOCOL, load_frozen, row_name, verify_attempt, atomic_json
    from eat_nav_mi_fi_v1_host import sha


def _read_labels(root: Path, row: dict) -> np.ndarray:
    attempt = root/row_name(row)
    verify_attempt(attempt, row)
    labels = json.loads((attempt/'sim-output'/'motion-labels.json').read_text())['labels']
    if len(labels) != 4 or any(not label['available'] for label in labels):
        raise ValueError('four complete motion labels required')
    values = np.asarray([[*label['delta_body_xy_m'], label['delta_body_yaw_rad']]
                         for label in labels], dtype=np.float64)
    if values.shape != (4, 3) or not np.isfinite(values).all():
        raise ValueError('invalid motion label')
    return values


def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = a-b
    result[..., 2] = (result[..., 2]+math.pi) % (2*math.pi)-math.pi
    return result


def _cluster_ci(seed_vectors: np.ndarray, terrain_indices: list[list[int]], rng,
                iterations: int) -> dict:
    values = []
    direction = seed_vectors[:, :2].mean(axis=0)
    norm = float(np.linalg.norm(direction))
    projection = direction/norm if norm else np.array([1., 0.])
    for _ in range(iterations):
        draw = np.concatenate([rng.choice(index, len(index), replace=True)
                               for index in terrain_indices])
        values.append(seed_vectors[draw].mean(axis=0))
    boot = np.asarray(values)
    scalar = boot[:, :2] @ projection
    return {'mean_delta_xy_m': direction.tolist(), 'mean_delta_yaw_rad': float(seed_vectors[:, 2].mean()),
            'mean_xy_norm_m': norm,
            'xy_projected_95_ci_m': np.quantile(scalar, [.025, .975]).tolist(),
            'yaw_95_ci_rad': np.quantile(boot[:, 2], [.025, .975]).tolist(),
            'per_seed_mean': seed_vectors.tolist(),
            'seeds_above_0_03m': int((np.linalg.norm(seed_vectors[:, :2], axis=1) > .03).sum())}


def audit(root: Path, phase: str) -> dict:
    manifest = load_frozen()
    frozen = json.loads((root/'frozen-provenance.json').read_text())
    if frozen['manifest_sha256'] != sha(MANIFEST) or frozen['protocol_sha256'] != sha(PROTOCOL):
        raise ValueError('frozen provenance mismatch')
    selected = [r for r in manifest['assignment'] if phase == 'formal' or
                r['state_group_id'] in manifest['smoke_state_groups']]
    seals, gates = [], []
    labels = {}
    for row in selected:
        attempt = root/row_name(row)
        verify_attempt(attempt, row)
        seals.append({'row_index': row['index'], 'outer_seal_sha256': sha(attempt/'outer-seal.json')})
        gate = json.loads((attempt/'sim-output'/'prepath-pair-gate.json').read_text())
        gates.append({'row_index': row['index'], 'pass': gate['pass']})
        if row['stage'] == 'candidate':
            key = (row['state_group_id'], row['flipper_initialization_mode'], row['candidate_id'])
            labels[key] = _read_labels(root, row)
    result = {'schema': SCHEMA, 'phase': phase, 'gate': 'pass',
              'independent_terrain_seed_count': len({r['terrain_seed'] for r in selected}),
              'state_group_count': len({r['state_group_id'] for r in selected}),
              'verified_processes': len(seals), 'candidate_trials': len(labels)*4,
              'gate_p_passes': sum(row['pass'] for row in gates),
              'seals': seals, 'gate_p': gates}
    if phase == 'smoke':
        return result
    groups = manifest['state_groups']
    terrain_indices = [[i for i, group in enumerate(groups) if group['terrain_type'] == terrain]
                       for terrain in manifest['terrain_types']]
    rng = np.random.default_rng(manifest['bootstrap_seed'])
    effects, interactions = {}, {}
    for condition in CONDITIONS:
        effects[condition] = {}
        for candidate in manifest['candidates']:
            if candidate.endswith('direct'):
                continue
            direct = 'forward_direct' if candidate.startswith('forward') else 'reverse_direct'
            per_group = []
            for group in groups:
                group_id = group['state_group_id']
                delta = _difference(labels[group_id, condition, candidate],
                                    labels[group_id, condition, direct])
                per_group.append(delta.mean(axis=0))
            summary = _cluster_ci(np.asarray(per_group), terrain_indices, rng,
                                  manifest['bootstrap_iterations'])
            summary['per_terrain_mean'] = {
                terrain: np.asarray(per_group)[indices].mean(axis=0).tolist()
                for terrain, indices in zip(manifest['terrain_types'], terrain_indices)}
            effects[condition][candidate] = summary
    for candidate in effects['front_folded']:
        front, rear = [], []
        direct = 'forward_direct' if candidate.startswith('forward') else 'reverse_direct'
        for group in groups:
            group_id = group['state_group_id']
            front.append(_difference(labels[group_id, 'front_folded', candidate],
                                     labels[group_id, 'front_folded', direct]).mean(axis=0))
            rear.append(_difference(labels[group_id, 'rear_folded', candidate],
                                    labels[group_id, 'rear_folded', direct]).mean(axis=0))
        interactions[candidate] = _cluster_ci(_difference(np.asarray(rear), np.asarray(front)),
                                              terrain_indices, rng, manifest['bootstrap_iterations'])
    passed = [f'{condition}:{candidate}' for condition, rows in effects.items()
              for candidate, summary in rows.items()
              if summary['xy_projected_95_ci_m'][0] > manifest['primary_xy_threshold_m']
              and summary['seeds_above_0_03m'] >= manifest['min_consistent_seeds']]
    forward = any(':forward_' in key for key in passed)
    reverse = any(':reverse_' in key for key in passed)
    if forward and reverse:
        decision = 'pass'
    elif all(summary['xy_projected_95_ci_m'][1] < manifest['primary_xy_threshold_m']
             for rows in effects.values() for summary in rows.values()):
        decision = 'fail'
    else:
        decision = 'inconclusive'
    result.update({'motion_gate': decision, 'identifiable_candidates': passed,
                   'arc_minus_direct': effects, 'initialization_interaction': interactions,
                   'primary_xy_threshold_m': manifest['primary_xy_threshold_m'],
                   'interpretation': 'seed-clustered terrain-stratified bootstrap; env tiles are paired replicates'})
    return result


def save(root: Path, phase: str, result: dict) -> None:
    name = 'mi_a_v3_smoke_test_results.json' if phase == 'smoke' else 'mi_a_v3_motion_results.json'
    if (root/name).exists():
        raise FileExistsError('analysis result already exists')
    atomic_json(root/name, result)
    if phase == 'formal':
        decision = {'schema': SCHEMA, 'decision': result['motion_gate'],
                    'identifiable_candidates': result['identifiable_candidates'],
                    'independent_seed_count': result['independent_terrain_seed_count'],
                    'generalization_limit': 'four selected terrains and twelve terrain seeds only'}
        atomic_json(root/'mi_a_v3_gate_decision.json', decision)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--phase', choices=('smoke', 'formal'), required=True)
    args = parser.parse_args()
    save(args.root, args.phase, audit(args.root, args.phase))
