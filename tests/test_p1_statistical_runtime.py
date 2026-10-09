import copy

import pytest


def test_candidate_free_trace_requires_120_finite_contiguous_steps():
    from visualnav_transformer.eat_nav.p1_statistical_runtime import validate_candidate_free_trace
    rows=[{'control_step':i,'actor_input':[0.0]*22,'critic_observation':[0.0]*267,
           'actor_action':[0.0]*6,'command':[0.0]*3,
           'root_state':[0.0]*13,'post_root_state':[0.0]*13,
           'reward':0.0,'terminal':None} for i in range(120)]
    assert validate_candidate_free_trace(rows)['control_steps']==120
    with pytest.raises(ValueError):
        validate_candidate_free_trace(rows[:80])
    rows[119]['actor_input'][0]=float('nan')
    with pytest.raises(ValueError):
        validate_candidate_free_trace(rows)

from visualnav_transformer.eat_nav.p1_statistical_runtime import (
    CachedTerrainGeometry, require_prepath_gate, jsonable_values,
)


class FakeMesh:
    def __init__(self):
        self.vertices = [[0, 0, 0], [1, 0, 0]]

    def copy(self):
        return copy.deepcopy(self)


def test_geometry_generated_once_and_each_slot_gets_independent_copy():
    calls = []
    def generate():
        calls.append(1)
        return FakeMesh(), [0, 0, .1]
    cache = CachedTerrainGeometry(generate)
    first, origin = cache.copy_for_slot()
    first.vertices[0][0] = 100
    origin[0] = 100
    second, second_origin = cache.copy_for_slot()
    assert len(calls) == 1
    assert second.vertices[0][0] == 0
    assert second_origin == [0, 0, .1]
    assert cache.canonical_sha256


def test_numpy_tile_origin_is_serializable_before_sealing():
    import numpy as np
    assert jsonable_values(np.asarray([1.0,2.0]))==[1.0,2.0]


def test_prepath_gate_refuses_path_publication_on_collision_or_history_mismatch():
    records = [{
        'slot': i, 'tile_origin_xy': [0, i*8],
        'root_state_w': [.6, i*8, .5, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'joint_position': [0, 0], 'joint_velocity': [0, 0],
        'flipper_latch': 1, 'flipper_previous_targets': [0]*4,
        'flipper_preclip_targets': [0]*4, 'action_current': [0]*6,
        'action_previous': [0]*6, 'actuator_targets': [0]*6,
        'latency_substeps': 5, 'latency_buffers': {'ring':[0]},
        'observation_history': {'ring':[0]},
        'canonical_mesh_sha256': 'mesh', 'collision_sha256': 'collision',
        'material_sha256': 'material',
    } for i in range(7)]
    limits = {'root_xy_m': .005, 'root_z_m': .002, 'root_yaw_rad': .01,
              'root_velocity': .01, 'joint_position_rad': .005,
              'joint_velocity_radps': .02}
    assert require_prepath_gate(records, limits)['pass']
    changed = copy.deepcopy(records)
    changed[2]['collision_sha256'] = 'different'
    with pytest.raises(ValueError, match='terrain_identity'):
        require_prepath_gate(changed, limits)
    changed = copy.deepcopy(records)
    changed[2]['observation_history']['ring'] = [1]
    with pytest.raises(ValueError, match='controlled_initialization'):
        require_prepath_gate(changed, limits)
