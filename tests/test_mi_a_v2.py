import numpy as np
import pytest


def _prepath():
    root = np.zeros((4, 13), dtype=np.float32)
    root[:, 3] = 1
    root[:, 1] = .08  # An asymmetric, nonzero settled position is valid.
    dem = np.zeros((4, 4, 2, 20, 20), dtype=np.float32)
    dem[:, :, 1, :, 2:] = 1  # The mask is intentionally not mirror symmetric.
    return {
        'identity': {'seed': 2026101305, 'terrain_sha256': 'terrain',
                     'material_sha256': 'material', 'episode_length_s': 40.0,
                     'dem_stamps_ns': [250000000, 500000000, 750000000, 1000000000],
                     'joint_names': ['joint_0'], 'dem_frame': 'robot_base',
                     'dem_resolution_m': 0.1, 'python_rng_sha256': 'py',
                     'numpy_rng_sha256': 'np', 'torch_cpu_rng_sha256': 'cpu',
                     'torch_cuda_rng_sha256': 'cuda'},
        'arrays': {'root': root, 'joint': np.zeros((4, 100), np.float32),
                   'actor_pre_noise': np.zeros((4, 22), np.float32),
                   'actor_post_noise': np.zeros((4, 22), np.float32),
                   'actor_input': np.zeros((4, 22), np.float32),
                   'dem': dem, 'command': np.zeros((4, 3), np.float32),
                   'latency': np.zeros((4, 2), np.float32),
                   'contact_force': np.zeros((4, 2, 3), np.float32)},
    }


def test_gate_p_accepts_same_asymmetric_settled_state():
    from visualnav_transformer.eat_nav.mi_a_v2 import compare_gate_p
    first = _prepath()
    assert compare_gate_p(first, _prepath())['pass']


def test_gate_p_rejects_missing_required_provenance():
    from visualnav_transformer.eat_nav.mi_a_v2 import compare_gate_p
    first, second = _prepath(), _prepath()
    del second['identity']['material_sha256']
    assert not compare_gate_p(first, second)['pass']
    del first['identity']['material_sha256']
    assert not compare_gate_p(first, second)['pass']


@pytest.mark.parametrize('field', ['root', 'dem', 'actor_input', 'latency', 'contact_force'])
def test_gate_p_rejects_a_candidate_specific_prepath_difference(field):
    from visualnav_transformer.eat_nav.mi_a_v2 import compare_gate_p
    first, second = _prepath(), _prepath()
    second['arrays'][field].flat[0] += .001
    result = compare_gate_p(first, second)
    assert not result['pass']
    assert not result['fields'][field]['pass']


def test_stage_plan_stops_after_one_seed_and_one_direct_repeat():
    from visualnav_transformer.eat_nav.mi_a_v2 import pilot_plan
    plan = pilot_plan(2026101305)
    assert len(plan) == 9
    assert [x['stage'] for x in plan] == ['prepath'] * 2 + ['candidate'] * 6 + ['repeat']
    assert all(x['seed'] == 2026101305 for x in plan)
    assert plan[-1]['candidate'] == 'forward_direct'


def test_body_path_uses_measured_nonzero_initial_yaw():
    from visualnav_transformer.eat_nav.motion_collector import body_path_to_tile_local
    path = np.array([[0., 0.], [.75, 0.]])
    tile = body_path_to_tile_local(path, (1., 2., np.pi / 2), (0., 0.))
    np.testing.assert_allclose(tile, [[1., 2.], [1., 2.75]], atol=1e-12)


def test_prepath_only_seal_requires_zero_trace_and_no_path(tmp_path):
    import hashlib
    import json
    from scripts.eat_nav_mi_a_v2_host import verify_sim_seal
    output = tmp_path/'output'
    output.mkdir()
    complete = {'status': 'complete', 'mode': 'prepath_only', 'env_count': 4,
                'control_steps': 0, 'trace_sha256': []}
    (output/'complete.json').write_text(json.dumps(complete))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    ledger = {'complete.json': digest(output/'complete.json')}
    (output/'sha-ledger.json').write_text(json.dumps(ledger))
    (output/'host-seal.json').write_text(json.dumps({'ledger_sha256': digest(output/'sha-ledger.json'),
                                                    'file_count': 1}))
    assert verify_sim_seal(output)['trace_count'] == 0
    (output/'path-activation.json').write_text('{}')
    with pytest.raises(ValueError):
        verify_sim_seal(output)


def test_v2_manifest_limits_runtime_to_nine_frozen_processes():
    import json
    from pathlib import Path
    from scripts.eat_nav_mi_a_v2_pilot import validate_pilot_manifest
    manifest = json.loads((Path(__file__).parents[1]/'docs/eat-nav/motion-predictor-vnext/'
                           'mi_a_v2_frozen_manifest.json').read_text())
    assert len(validate_pilot_manifest(manifest)) == 9
    changed = json.loads(json.dumps(manifest))
    changed['pilot_order'].append(changed['pilot_order'][-1])
    with pytest.raises(ValueError):
        validate_pilot_manifest(changed)
