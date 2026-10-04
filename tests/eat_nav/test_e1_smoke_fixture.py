import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from eat_nav_e1_smoke_fixture import build_alias_state, extract_frames, path_digest, prepare_fixture, validate_fixture
from eat_nav_e1_lifecycle_smoke_retry import preflight as retry_preflight


READINESS = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
RAW = Path('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json')


def test_alias_preserves_every_physical_readiness_field():
    source = json.loads((READINESS/'selected-states.json').read_text())['train'][0]
    alias = build_alias_state(source)
    assert alias['state_id'] == 'g0e-train-readiness-000'
    assert alias['original_state_id'] == source['state_id']
    assert alias['global_step'] == source['step'] == 75
    assert alias['logged_goal'] == source['goal']
    for name, value in source.items():
        if name != 'state_id':
            assert alias[name] == value


def test_derived_four_frame_dem_is_exact_existing_readiness_window():
    state = json.loads((READINESS/'selected-states.json').read_text())['train'][0]
    frames = extract_frames(RAW, state)
    assert [row['step'] for row in frames] == [60, 65, 70, 75]
    assert all(row['episode_id'] == 0 and row['env_id'] == 52 and not row['pre_reset'] for row in frames)
    dem = np.asarray([row['dem']['transformed'] for row in frames], dtype=np.float32)
    assert dem.shape == (4, 2, 20, 20)
    assert hashlib.sha256(dem.tobytes()).hexdigest() == state['dem_sha256']


def test_nominal_path_digest_uses_same_eight_float32_points():
    source = READINESS/'state-000'
    base = json.loads((source/'bank-nominal-base-path.json').read_text())
    mapped = json.loads((source/'bank-nominal-map-path.json').read_text())
    assert path_digest(base) == 'bf25f9513b8d6077c9f86f6a180c15c99b0e80522fce38d476f4df96c80957ad'
    assert path_digest(mapped) == '1be13e59162dc857a3590746e55754da00c61bf07790adde3784101d95047267'


def test_readiness_alias_rejects_changed_episode_or_step():
    state = json.loads((READINESS/'selected-states.json').read_text())['train'][0]
    with pytest.raises(ValueError, match='frozen readiness identity'):
        build_alias_state({**state, 'step': 76})


def test_prepared_fixture_only_adds_diagnostic_identity_and_verification_sidecars(tmp_path):
    entry = prepare_fixture(tmp_path)
    state = tmp_path/'inputs/state-000'
    original = READINESS/'state-000'
    for name in ('prefix-actions.npz', 'suffix-actions-120.npz', 'logged-anchor.json'):
        assert (state/name).read_bytes() == (original/name).read_bytes()
    assert (state/f"{entry['candidate_id']}-base-path.json").read_bytes() == (original/'bank-nominal-base-path.json').read_bytes()
    assert (state/f"{entry['candidate_id']}-map-path.json").read_bytes() == (original/'bank-nominal-map-path.json').read_bytes()
    with np.load(tmp_path/'inputs/g0e-selected-states.npz') as selected:
        assert selected['dem'].shape == (1, 4, 2, 20, 20)
        assert selected['goal'][0].tolist() == pytest.approx(entry['goal'])
    assert entry['episode_key'] == [0, 52, 0]
    assert entry['anchor_step'] == 75
    assert entry['logical_t0_ns'] == 1020000000000
    assert entry['kind'] == 'duplicate'
    assert entry['local_path_sha256'] == path_digest(json.loads((original/'bank-nominal-base-path.json').read_text()))
    assert validate_fixture(tmp_path, entry) == state
    with pytest.raises(ValueError, match='diagnostic run entry'):
        validate_fixture(tmp_path, {**entry, 'map_path_file_sha256':'0'*64})


def test_retry_preflight_rejects_changed_production_source(tmp_path):
    entry = prepare_fixture(tmp_path)
    source = Path('/tmp/eatnav-vnt-e1-feasibility/scripts/eat_nav_e1_sim.py')
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert retry_preflight(tmp_path, entry, expected)['status'] == 'pass'
    with pytest.raises(ValueError, match='production simulator source SHA'):
        retry_preflight(tmp_path, entry, '0'*64)


def test_e1_runner_uses_remote_input_root_for_fixed_map_path(monkeypatch):
    import eat_nav_e1_run_one as runner
    monkeypatch.setattr(runner, 'ROOT', Path('/tmp/diagnostic-output'))
    monkeypatch.setattr(runner, 'REMOTE_INPUT_ROOT', Path('/tmp/frozen-input'))
    assert runner.fixed_map_path(0, 'diagnostic-readiness-002') == (
        '/tmp/frozen-input/inputs/state-000/diagnostic-readiness-002-map-path.json')
