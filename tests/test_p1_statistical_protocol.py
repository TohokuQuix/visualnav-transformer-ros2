import copy
import json
from pathlib import Path

import pytest

from visualnav_transformer.eat_nav.p1_statistical_protocol import (
    build_assignments, validate_assignment, initialization_checks, terrain_identity_checks,
)


SOURCE = json.loads((Path(__file__).parents[1] / 'config/eat_nav_p1_batch_source_contract.json').read_text())


def test_two_batches_rotate_frozen_paths_and_repeat_direct():
    manifest = build_assignments(SOURCE, [2026100901, 2026100902], 123)
    validate_assignment(manifest)
    assert len(manifest['trials']) == 56
    for group in SOURCE['anchors']:
        aid = group['anchor_id']
        blocks = [[t for t in manifest['trials'] if t['anchor_id'] == aid and t['batch_index'] == b]
                  for b in (0, 1)]
        assert all(len(block) == 7 for block in blocks)
        assert all(sum(t['path_identity'] == 'direct_reference' for t in block) == 3 for block in blocks)
        for path in SOURCE['paths']:
            if path['bearing_deg'] == 0:
                continue
            slots = [[t['slot'] for t in block if t['path_sha256'] == path['path_sha256']][0]
                     for block in blocks]
            assert slots[0] != slots[1]
        assert {t['slot'] for t in blocks[0] if t['path_identity'] == 'direct_reference'} != {
            t['slot'] for t in blocks[1] if t['path_identity'] == 'direct_reference'}
    assert all(t['policy_action_mode'] == 'act_inference' and t['policy_noise_seed'] is None
               for t in manifest['trials'])


def test_rejects_permanently_confounded_slot():
    manifest = build_assignments(SOURCE, [2026100901, 2026100902], 123)
    altered = copy.deepcopy(manifest)
    group = SOURCE['anchors'][0]['anchor_id']
    first = next(t for t in altered['trials'] if t['anchor_id'] == group and t['batch_index'] == 0
                 and t['path_identity'] != 'direct_reference')
    second = next(t for t in altered['trials'] if t['anchor_id'] == group and t['batch_index'] == 1
                  and t['path_sha256'] == first['path_sha256'])
    swap = next(t for t in altered['trials'] if t['anchor_id'] == group and t['batch_index'] == 1
                and t['slot'] == first['slot'])
    second['slot'], swap['slot'] = swap['slot'], second['slot']
    with pytest.raises(ValueError, match='slot_rotation'):
        validate_assignment(altered)


def snapshot(slot, *, mesh='mesh', flipper=(0, 0, 3.14, 3.14)):
    return {
        'slot': slot, 'tile_origin_xy': [0, slot * 8],
        'root_state_w': [.6, slot * 8, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'joint_position': [0, 0], 'joint_velocity': [0, 0],
        'flipper_latch': 1, 'flipper_previous_targets': list(flipper),
        'flipper_preclip_targets': list(flipper),
        'action_current': [0] * 6, 'action_previous': [0] * 6,
        'actuator_targets': [0] * 6,
        'latency_substeps': 5, 'latency_buffers': {'ring': [0, 0]},
        'observation_history': {'ring': [0, 0]},
        'canonical_mesh_sha256': mesh, 'collision_sha256': 'collision',
        'material_sha256': 'material',
    }


def test_geometry_and_control_history_are_prepath_gates():
    records = [snapshot(i) for i in range(7)]
    limits = {'root_xy_m': .005, 'root_z_m': .002, 'root_yaw_rad': .01,
              'root_velocity': .01, 'joint_position_rad': .005,
              'joint_velocity_radps': .02}
    assert terrain_identity_checks(records)['pass']
    assert initialization_checks(records, limits)['pass']
    changed = copy.deepcopy(records)
    changed[3]['canonical_mesh_sha256'] = 'other'
    assert not terrain_identity_checks(changed)['pass']
    changed = copy.deepcopy(records)
    changed[3]['flipper_previous_targets'][0] = 3.14
    assert not initialization_checks(changed, limits)['pass']
    changed = copy.deepcopy(records)
    changed[3]['root_state_w'][2] += .003
    assert not initialization_checks(changed, limits)['pass']
