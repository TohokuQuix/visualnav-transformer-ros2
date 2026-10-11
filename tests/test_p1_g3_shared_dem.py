"""Synthetic contract checks for the shared experimental DEM input."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import eat_nav_p1_g3_shared_dem_host as host
from eat_nav_p1_g3_shared_dem_stage import stage_bundle, verify_stage
from visualnav_transformer.eat_nav.p1_g3_shared_dem import (
    build_shared_history, diagnose_branch_inputs, persist_shared_history,
    validate_shared_history, verify_shared_artifacts,
)
from visualnav_transformer.eat_nav.p1_dem_capture import persist_dem_evidence

STEPS = (4, 9, 14, 19)


def native_history():
    result = []
    for env in range(4):
        history = []
        for step in STEPS:
            dem = np.zeros((2, 20, 20), dtype=np.float32)
            dem[1] = 1
            dem[0, 0, 0] = env * .001
            history.append({
                'identity': f'settling_step_{step}',
                'timestamp_ns': (step + 1) * 50_000_000,
                'shape': [2, 20, 20], 'frame': 'robot_base',
                'resolution_m': .1, 'sensor_update_timestamp_s': (step + 1) * .05,
                'transformed': dem,
                'raw_sha256': hashlib.sha256(dem.tobytes()).hexdigest(),
            })
        result.append(history)
    return result


def frozen_gate(frames):
    assert len(frames) == 4
    assert all(frames[0][i]['raw_sha256'] == frames[1][i]['raw_sha256']
               for i in range(4))
    return [{'height_rms_m': float(np.sqrt(np.mean(
        (frames[0][i]['transformed'][0] - frames[1][i]['transformed'][0]) ** 2)))}
        for i in range(4)]


def test_immutable_four_frame_copy_and_native_mismatch_are_distinct(tmp_path):
    native = native_history()
    shared = build_shared_history(native)
    assert shared['source_env'] == 0
    assert shared['history_role'] == 'shared_experimental_history'
    for env in range(4):
        for idx in range(4):
            frame = shared['branches'][env][idx]
            assert frame['transformed'].tobytes() == native[0][idx]['transformed'].tobytes()
            assert frame['transformed'] is not native[0][idx]['transformed']
            assert not frame['transformed'].flags.writeable
            assert frame['source_frame_sha256'] == shared['branches'][0][idx]['source_frame_sha256']
    native[0][0]['transformed'][0, 0, 0] = 99
    assert shared['branches'][1][0]['transformed'][0, 0, 0] == 0
    result = validate_shared_history(shared, frozen_gate)
    assert result['pass'] and result['meaning'] == 'shared_experimental_input_equality'
    assert native[1][0]['transformed'][0, 0, 0] == pytest.approx(.001)
    persist_shared_history(tmp_path, shared, native_diagnostic={'frames': [{'height_rms_m': .001}]})
    assert verify_shared_artifacts(tmp_path)['pass']
    assert json.loads((tmp_path/'prepath-native-dem-diagnostic.json').read_text())['frames'][0]['height_rms_m'] == .001


def test_host_can_bind_saved_shared_input_to_actual_native_source(tmp_path):
    native = native_history()
    for env, history in enumerate(native):
        for frame in history:
            hits = np.zeros((400, 3), dtype=np.float32)
            starts = np.zeros_like(hits)
            hits[:, 0] = env * 8
            starts[:, 0] = env * 8
            source_height = hits[:, 2].reshape(20, 20).copy()
            frame.update({'ray_hits_w': hits, 'ray_starts_w': starts,
                'source_height': source_height, 'source_valid': np.ones((20, 20), bool),
                'env_origin_w': np.array([env * 8, 0, 0], np.float32),
                'robot_pose_w': np.zeros(13, np.float32),
                'root_state_w': np.zeros(13, np.float32),
                'joint_pos': np.zeros(6, np.float32), 'joint_vel': np.zeros(6, np.float32),
                'sensor_timestamp_s': frame['sensor_update_timestamp_s'],
                'sensor_is_outdated': False, 'sensor_update_period_s': .25,
                'raw_sha256': hashlib.sha256(source_height.tobytes()).hexdigest()})
    persist_dem_evidence(tmp_path, native)
    shared = build_shared_history(native)
    persist_shared_history(tmp_path, shared, native_diagnostic={'frames': []})
    assert verify_shared_artifacts(tmp_path, require_native=True)['pass']
    native_file = tmp_path/'prepath-dem-env-0-frame-0.npz'
    native_file.write_bytes(b'tampered')
    with pytest.raises(ValueError, match='native source SHA'):
        verify_shared_artifacts(tmp_path, require_native=True)


def test_shared_metadata_age_and_byte_tampering_fail_closed(tmp_path):
    shared = build_shared_history(native_history())
    assert [f['timestamp_ns'] for f in shared['branches'][0]] == [250_000_000, 500_000_000, 750_000_000, 1_000_000_000]
    assert validate_shared_history(shared, frozen_gate)['sensor_update_age_s'] == [0., 0., 0., 0.]
    shared['branches'][2][1]['timestamp_ns'] -= 50_000_000
    with pytest.raises(ValueError, match='timestamp'):
        validate_shared_history(shared, frozen_gate)
    shared = build_shared_history(native_history())
    shared['branches'][2][1]['transformed'].setflags(write=True)
    shared['branches'][2][1]['transformed'][0, 0, 0] = 7
    with pytest.raises(ValueError, match='byte integrity'):
        validate_shared_history(shared, frozen_gate)
    native = native_history()
    native[0][0]['sensor_update_timestamp_s'] = .60
    shared = build_shared_history(native)
    with pytest.raises(ValueError, match='sensor update age'):
        validate_shared_history(shared, frozen_gate)
    shared = build_shared_history(native_history())
    persist_shared_history(tmp_path, shared, native_diagnostic={'frames': []})
    (tmp_path/'prepath-shared-dem-env-1-frame-0.npy').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='SHA'):
        verify_shared_artifacts(tmp_path)


def test_actor_critic_contact_diagnostics_are_separate():
    actor = np.zeros((4, 22), dtype=np.float32)
    critic = np.zeros((4, 267), dtype=np.float32)
    contact = np.zeros((4, 3, 3), dtype=np.float32)
    actor[1, 0] = 2
    critic[1, 0] = 3
    contact[1, 0, 0] = 4
    diag = diagnose_branch_inputs(actor, actor, actor, critic, contact, contact[:, None])
    assert diag['actor_pre_noise']['direct_max_abs'] == 2
    assert diag['critic_observation']['direct_max_abs'] == 3
    assert diag['contact_current']['direct_max_abs'] == 4
    assert diag['contact_history']['direct_max_abs'] == 4
    critic[2, 2] = np.nan
    with pytest.raises(ValueError, match='critic_observation shape/nonfinite'):
        diagnose_branch_inputs(actor, actor, actor, critic, contact, contact[:, None])


def test_frozen_dem_gate_source_is_identical():
    repo = Path(__file__).resolve().parents[1]
    old = ast.parse((repo/'scripts/eat_nav_p1_g3_pilot_sim.py').read_text())
    new = ast.parse((repo/'scripts/eat_nav_p1_g3_shared_dem_sim.py').read_text())
    def node(tree):
        return next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == 'direct_dem_gate')
    assert ast.dump(node(old), include_attributes=False) == ast.dump(node(new), include_attributes=False)
    text = (repo/'scripts/eat_nav_p1_g3_shared_dem_sim.py').read_text()
    between = text.split('clone=clone_internal_branch_state', 1)[1].split(
        "bridge.request('reset'", 1)[0]
    assert 'env.step(' not in between
    assert 'persist_dem_evidence(args.output,dem_history)' in between
    assert 'persist_shared_history(args.output,shared' in between
    assert 'validate_shared_history(shared,direct_dem_gate)' in between
    assert "'native_sensor_history':{'pass':native_pass,'diagnostic_only':True}" in between
    namespace = {'np': np}
    executable = ast.Module(body=[node(new)], type_ignores=[])
    exec(compile(executable, '<frozen-direct-dem-gate>', 'exec'), namespace)
    result = validate_shared_history(build_shared_history(native_history()),
                                     namespace['direct_dem_gate'])
    assert result['pass'] and len(result['frozen_numeric_diagnostics']) == 4


def test_diagnostic_error_propagates_even_if_native_exit_is_zero(tmp_path):
    output = tmp_path/'output'; output.mkdir()
    (output/'diagnostic-error.json').write_text(json.dumps({
        'type': 'ValueError', 'message': 'shared DEM byte integrity mismatch'}))
    assert 'shared DEM byte integrity' in host.simulator_artifact_failure(
        output, 0, diagnostic=True)
    assert 'shared DEM byte integrity' in host.classify_monitor_violation(
        'gpu_attribution:simulator_disappeared_in_bracket', 0, output, diagnostic=True)
    (output/'diagnostic-error.json').unlink()
    assert host.simulator_artifact_failure(output, 0, diagnostic=True) == \
           'simulator_completion_marker_missing'
    (output/'gate1-probe-complete.json').write_text('{}')
    assert host.simulator_artifact_failure(output, 0, diagnostic=True) is None


def test_staged_source_and_sha_ledger(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    g2, old = tmp_path/'g2', tmp_path/'old'
    for path in (g2/'src/visualnav_transformer/eat_nav', g2/'input', old):
        path.mkdir(parents=True)
    for name in ('assignment.json', 'freeze.json', 'statistical-config.json',
                 'source-contract.json', 'anchor-fixtures.json', 'eat_nav.yaml'):
        (g2/'input'/name).write_text('{}')
    (old/'g3-assignment.json').write_text('{"trials":[]}' )
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {'schema': 'eatnav_p1_longhorizon_g3_shared_dem_v2',
                'execution_mode': 'candidate_free_20_step_dem',
                'g3_simulator_source_sha256': digest(repo/'scripts/eat_nav_p1_g3_shared_dem_sim.py'),
                'g3_host_source_sha256': digest(repo/'scripts/eat_nav_p1_g3_shared_dem_host.py'),
                'shared_dem_module_sha256': digest(repo/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py'),
                'assignment_sha256': digest(old/'g3-assignment.json')}
    root = tmp_path/'new'
    stage_bundle(repo, g2, old, root, manifest)
    batch = root/'batch-0'
    assert verify_stage(batch, manifest)['pass']
    assert (batch/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py').exists()
    (batch/'input/assignment.json').write_text('tamper')
    with pytest.raises(ValueError, match='SHA'):
        verify_stage(batch, manifest)


def test_committed_manifest_freezes_new_sources_and_original_assignment():
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo/'docs/eat-nav/p1-g3-shared-dem-v2-manifest.json').read_text())
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert manifest['schema'] == 'eatnav_p1_longhorizon_g3_shared_dem_v2'
    assert manifest['execution_mode'] == 'candidate_free_20_step_dem'
    assert manifest['assignment_sha256'] == \
           'f5ce459bf26ffe0a2316e37f046f2105d0637060ac17758bacf6da46fe49608f'
    assert manifest['g3_simulator_source_sha256'] == digest(repo/'scripts/eat_nav_p1_g3_shared_dem_sim.py')
    assert manifest['g3_host_source_sha256'] == digest(repo/'scripts/eat_nav_p1_g3_shared_dem_host.py')
    assert manifest['shared_dem_module_sha256'] == digest(
        repo/'src/visualnav_transformer/eat_nav/p1_g3_shared_dem.py')
    assert manifest['direct_duplicate_gates']['dem_height_rms_m'] == 1e-5
    assert manifest['direct_duplicate_gates']['dem_height_max_abs_m'] == 1e-4
    assert manifest['direct_duplicate_gates']['dem_mask_iou_min'] == 1.0
