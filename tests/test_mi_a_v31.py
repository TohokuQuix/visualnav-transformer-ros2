import json
import numpy as np
import pytest
from pathlib import Path

from scripts.eat_nav_mi_a_v31_run import COLLECTION, load_frozen, selected_rows, verify_prepath_snapshot


def test_preflight_runs_only_first_smoke_seed_prepath_and_smoke_has_32_rows():
    manifest = load_frozen()
    preflight = selected_rows(manifest, 'preflight')
    smoke = selected_rows(manifest, 'smoke')
    assert len(preflight) == 1
    assert preflight[0]['stage'] == 'prepath'
    assert preflight[0]['flipper_initialization_mode'] == 'front_folded'
    assert preflight[0] == smoke[0]
    assert len(smoke) == 32
    assert len({row['terrain_seed'] for row in smoke}) == 2
    assert len({row['terrain_type'] for row in smoke}) == 2
    assert len(manifest['assignment']) == 192
    assert len({row['terrain_seed'] for row in manifest['assignment']}) == 12
    assert all(0 <= row['terrain_seed'] <= 2**32-1 for row in manifest['assignment'])


def test_seed_formula_is_reproducible_and_preserves_balanced_collection():
    manifest = load_frozen()
    assert [g['terrain_seed'] for g in manifest['state_groups']] == [
        2026101401, 2026101402, 2026101403,
        2026101501, 2026101502, 2026101503,
        2026101601, 2026101602, 2026101603,
        2026101701, 2026101702, 2026101703,
    ]
    collection = json.loads(COLLECTION.read_text())
    assert list(collection['assignment'].values()).count('front_folded') == 6
    assert list(collection['assignment'].values()).count('rear_folded') == 6


def test_v31_changes_only_seed_dependent_assignment_fields():
    manifest = load_frozen()
    old_path = Path(__file__).resolve().parents[1]/'docs/eat-nav/motion-predictor-vnext/mi_a_v3_assignment_manifest.json'
    old = json.loads(old_path.read_text())
    unchanged = ('num_envs', 'settling_steps', 'candidate_control_steps', 'step_seconds',
                 'flipper_fold_mode', 'terrain_types', 'num_processes', 'candidate_trials',
                 'candidates', 'resource_caps', 'primary_xy_threshold_m',
                 'min_consistent_seeds', 'bootstrap_seed', 'bootstrap_iterations', 'gate_p_abs')
    assert all(manifest[key] == old[key] for key in unchanged)
    for current, previous in zip(manifest['assignment'], old['assignment']):
        for key in previous:
            if key not in ('state_group_id', 'terrain_seed', 'flipper_initialization_seed'):
                assert current[key] == previous[key]


def test_preflight_snapshot_requires_complete_finite_four_env_history(tmp_path):
    output = tmp_path/'sim-output'
    output.mkdir()
    (output/'prepath-identity.json').write_text(json.dumps({
        'seed': 2026101401, 'condition': 'front_folded',
        'dem_frame': 'robot_base', 'dem_resolution_m': .1,
        'dem_stamps_ns': [250_000_000, 500_000_000, 750_000_000, 1_000_000_000]}))
    arrays = {'root': np.zeros((4, 13)), 'joint': np.zeros((4, 20)),
              'actor_pre_noise': np.zeros((4, 22)), 'actor_input': np.zeros((4, 22)),
              'critic': np.zeros((4, 4)), 'dem': np.zeros((4, 4, 2, 20, 20)),
              'contact_force': np.zeros((4, 1, 3)),
              'contact_history': np.zeros((4, 1, 1, 3)),
              'latency': np.zeros((4, 2)), 'command': np.zeros((4, 3)),
              'flipper_reset_joint_pos': np.zeros((4, 4)),
              'flipper_prepath_joint_pos': np.zeros((4, 4))}
    np.savez_compressed(output/'prepath.npz', **arrays)
    result = verify_prepath_snapshot(output, seed=2026101401, condition='front_folded')
    assert result['dem_shape'] == [4, 4, 2, 20, 20]
    arrays['dem'][0, 0, 0, 0, 0] = np.nan
    np.savez_compressed(output/'prepath.npz', **arrays)
    with pytest.raises(ValueError, match='nonfinite'):
        verify_prepath_snapshot(output, seed=2026101401, condition='front_folded')
