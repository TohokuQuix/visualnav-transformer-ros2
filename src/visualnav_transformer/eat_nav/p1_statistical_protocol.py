"""Prospective assignment and pre-Path gates for P1 statistical comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import random


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_assignments(source_contract: dict, batch_seeds: list[int], random_seed: int) -> dict:
    if len(batch_seeds) != 2 or len(set(batch_seeds)) != 2:
        raise ValueError('pilot requires two independent batch seeds')
    anchors, paths = source_contract['anchors'], source_contract['paths']
    if len(anchors) != 4 or len(paths) != 5 or len({a['anchor_id'] for a in anchors}) != 4:
        raise ValueError('frozen P1 StateGroup/Path count changed')
    direct = [p for p in paths if p['bearing_deg'] == 0]
    if len(direct) != 1 or len({p['path_sha256'] for p in paths}) != 5:
        raise ValueError('direct reference or Path identity ambiguous')
    choices = [p for p in paths if p is not direct[0]] + direct * 3
    trials = []
    for group_index, anchor in enumerate(anchors):
        previous = None
        for batch_index, batch_seed in enumerate(batch_seeds):
            rng = random.Random(_digest([random_seed, group_index, batch_index, batch_seed]))
            for _ in range(1000):
                allocated = choices.copy()
                rng.shuffle(allocated)
                positions = {p['path_sha256']: slot for slot, p in enumerate(allocated)
                             if p['path_sha256'] != direct[0]['path_sha256']}
                direct_slots = {i for i, p in enumerate(allocated)
                                if p['path_sha256'] == direct[0]['path_sha256']}
                if previous is None or (all(positions[k] != previous[0][k] for k in positions)
                                        and direct_slots != previous[1]):
                    break
            else:
                raise RuntimeError('cannot rotate Path assignments')
            previous = positions, direct_slots
            for slot, path in enumerate(allocated):
                identity = 'direct_reference' if path['path_sha256'] == direct[0]['path_sha256'] else f"bearing_{path['bearing_deg']:+g}"
                trials.append({
                    'trial_id': f"{anchor['anchor_id']}-b{batch_index}-s{slot}",
                    'anchor_id': anchor['anchor_id'], 'source_pose_id': anchor['pose_id'],
                    'terrain_type': anchor['terrain'], 'terrain_seed': anchor['terrain_seed'],
                    'batch_index': batch_index, 'batch_seed': batch_seed,
                    'observation_randomization_seed': batch_seed,
                    'initial_state_replicate': batch_index,
                    'policy_action_mode': 'act_inference', 'policy_noise_seed': None,
                    'slot': slot, 'env_id': slot, 'tile_coordinate': [0, slot],
                    'path_identity': identity, 'path_sha256': path['path_sha256'],
                    'path_bearing_deg': path['bearing_deg'],
                })
    manifest = {'schema': 'p1_statistical_assignment_v1',
                'source_contract_sha256': _digest(source_contract),
                'randomization_seed': random_seed, 'batch_seeds': batch_seeds,
                'independent_unit': 'StateGroup x batch_seed', 'paired_policy_noise': False,
                'trials': trials}
    validate_assignment(manifest)
    return manifest


def validate_assignment(manifest: dict) -> None:
    if manifest['schema'] != 'p1_statistical_assignment_v1' or manifest['paired_policy_noise']:
        raise ValueError('assignment schema or pairing contract')
    trials = manifest['trials']
    if len(trials) != 56 or len({t['trial_id'] for t in trials}) != 56:
        raise ValueError('trial count or duplicate identity')
    groups = sorted({t['anchor_id'] for t in trials})
    if len(groups) != 4 or len(set(manifest['batch_seeds'])) != 2:
        raise ValueError('group/batch count')
    for group in groups:
        previous = None
        for batch in (0, 1):
            rows = [t for t in trials if t['anchor_id'] == group and t['batch_index'] == batch]
            if len(rows) != 7 or {r['slot'] for r in rows} != set(range(7)):
                raise ValueError('slot coverage')
            if any(r['batch_seed'] != manifest['batch_seeds'][batch] or
                   r['policy_action_mode'] != 'act_inference' or r['policy_noise_seed'] is not None
                   for r in rows):
                raise ValueError('seed/action contract')
            direct = [r for r in rows if r['path_identity'] == 'direct_reference']
            alternatives = [r for r in rows if r['path_identity'] != 'direct_reference']
            if len(direct) != 3 or len(alternatives) != 4 or len({r['path_sha256'] for r in alternatives}) != 4 or len({r['path_sha256'] for r in direct}) != 1:
                raise ValueError('Path coverage')
            positions = {r['path_sha256']: r['slot'] for r in alternatives}
            direct_slots = {r['slot'] for r in direct}
            if previous is not None and (any(positions[k] == previous[0][k] for k in positions)
                                         or direct_slots == previous[1]):
                raise ValueError('slot_rotation')
            previous = positions, direct_slots


def terrain_identity_checks(records: list[dict]) -> dict:
    fields = ('canonical_mesh_sha256', 'collision_sha256', 'material_sha256')
    failures = [field for field in fields if len(records) != 7 or
                any(not row.get(field) for row in records) or
                len({row[field] for row in records}) != 1]
    return {'pass': not failures, 'failures': failures,
            'fingerprints': {field: records[0].get(field) if records else None for field in fields}}


def _numbers(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from _numbers(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def initialization_checks(records: list[dict], limits: dict) -> dict:
    fields = ('root_state_w', 'joint_position', 'joint_velocity', 'flipper_latch',
              'flipper_previous_targets', 'flipper_preclip_targets', 'action_current',
              'action_previous', 'actuator_targets', 'latency_substeps',
              'latency_buffers', 'observation_history', 'tile_origin_xy')
    failures = []
    if len(records) != 7 or any(any(field not in row for field in fields) for row in records):
        return {'pass': False, 'failures': ['missing_initial_state_field']}
    if any(not all(math.isfinite(x) for x in _numbers(row)) for row in records):
        failures.append('nonfinite')
    def yaw(row):
        _, _, _, w, x, y, z = row['root_state_w'][:7]
        return math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    def local_xy(row):
        return [row['root_state_w'][i] - row['tile_origin_xy'][i] for i in range(2)]
    def spread(values):
        return max(max(abs(v[i] - values[0][i]) for i in range(len(values[0]))) for v in values)
    metrics = {
        'root_xy_m': spread([local_xy(r) for r in records]),
        'root_z_m': spread([[r['root_state_w'][2]] for r in records]),
        'root_yaw_rad': spread([[yaw(r)] for r in records]),
        'root_velocity': spread([r['root_state_w'][7:13] for r in records]),
        'joint_position_rad': spread([r['joint_position'] for r in records]),
        'joint_velocity_radps': spread([r['joint_velocity'] for r in records]),
    }
    failures.extend(key for key, value in metrics.items() if not math.isfinite(value) or value > limits[key])
    exact_fields = ('flipper_latch', 'flipper_previous_targets', 'flipper_preclip_targets',
                    'action_current', 'action_previous', 'actuator_targets',
                    'latency_substeps', 'latency_buffers', 'observation_history')
    failures.extend(field for field in exact_fields if any(row[field] != records[0][field] for row in records[1:]))
    return {'pass': not failures, 'failures': failures, 'maximum_spread': metrics}
