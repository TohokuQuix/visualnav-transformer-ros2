from pathlib import Path

import pytest

from scripts.eat_nav_goal_bank_review import load_anchor_inputs, run_review, _decision, _verify_prior_file


BACKUP = Path('/home/kojima/eatnav-backups/2026-10-08-pre-shutdown')


@pytest.mark.skipif(not BACKUP.exists(), reason='frozen R3 backup unavailable')
def test_eight_anchor_freeze_reads_sealed_inputs_and_exact_dem():
    anchors, provenance = load_anchor_inputs(BACKUP)
    assert len(anchors) == 8
    assert [a['pose_id'] for a in anchors] == [
        'b20261007-k_rail_field-r0.60-a04',
        'b20261007-boxes-r0.60-a01',
        'b20261007-k_rail_step_field-r0.30-a06',
        'b20261007-k_rail_field-r0.50-a04',
        'b3773891777-boxes-r0.50-a12',
        'b3773891777-k_rail_step_field-r0.60-a00',
        'b3773891777-k_rail_step_field-r0.30-a06',
        'b3773891777-pyramid_stairs_inv-r0.60-a00',
    ]
    assert all(a['dem'].shape == (2, 20, 20) for a in anchors)
    assert all(len(a['dem_history_sha256']) == 4 for a in anchors)
    assert len({a['source_family'] for a in anchors}) == 6
    assert provenance['source_file_count'] >= 8 * 6


@pytest.mark.skipif(not BACKUP.exists(), reason='frozen R3 backup unavailable')
def test_full_offline_review_is_deterministic_and_produces_560_paths(tmp_path):
    one = run_review(BACKUP, tmp_path / 'one', figures=False)
    two = run_review(BACKUP, tmp_path / 'two', figures=False)
    assert one['candidate_counts'] == {'A': 280, 'B': 280}
    assert two['candidate_counts'] == one['candidate_counts']
    assert one['path_collection_sha256'] == two['path_collection_sha256']
    assert one['unique_path_sha256_count'] == 65
    assert len((tmp_path / 'one' / 'state-groups.json').read_text()) > 100
    assert (tmp_path / 'one' / 'sha-ledger.json').exists()
    assert (tmp_path / 'one' / 'bank-B-terminal-mode-comparison.json').exists()


def test_bank_readiness_is_independent_and_requires_per_group_supply():
    good = {'hard_valid_state_groups': 40, 'seven_geometry_mode_supply_groups': 40,
            'groups_meeting_follower_minimum': 40, 'state_groups': 40,
            'terrain_separated_known_pairs': 1, 'follower_compatible_candidate_count': 160}
    bad = dict(good, hard_valid_state_groups=0, groups_meeting_follower_minimum=0)
    decision, _ = _decision(good, bad)
    assert decision == 'candidate_bank_common_endpoint_ready_for_rollout_review'
    sparse = dict(good, groups_meeting_follower_minimum=39)
    decision, _ = _decision(sparse, bad)
    assert decision == 'candidate_bank_follower_compatibility_insufficient'


def test_prior_source_ledger_rejects_changed_frozen_input(tmp_path):
    import hashlib
    import json
    source=tmp_path/'source.json'
    source.write_text('{"rank":1}')
    ledger=tmp_path/'final-output-sha-ledger.json'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    ledger.write_text(json.dumps({'files':[{'path':'source.json','sha256':digest}]}))
    provenance={'sha256':{}}
    _verify_prior_file(source,ledger,provenance)
    source.write_text('{"rank":2}')
    with pytest.raises(ValueError,match='frozen source ledger mismatch'):
        _verify_prior_file(source,ledger,provenance)
