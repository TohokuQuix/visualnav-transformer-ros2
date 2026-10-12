"""V4 preparation must preserve the frozen schedule and prohibit runtime."""
from pathlib import Path
import json

from eat_nav_p1_g3_crossover_v4_prepare import make_manifest


REPO = Path(__file__).resolve().parents[1]


def test_v4_manifest_preserves_18_assignments_and_lists_only_14_unrun():
    manifest = make_manifest(REPO)
    assert manifest['host_protocol_version'] == 'eatnav_p1_g3_path_crossover_v4_router_lifecycle'
    assert len(manifest['processes']) == 18
    assert len(manifest['remaining_process_ids']) == 14
    assert manifest['remaining_process_ids'][0] == 'batch-first-differential-1-period-1'
    assert manifest['resume_blocker']['required_predecessor'] == 'batch-first-differential-1-period-0'
    assert manifest['resume_blocker']['predecessor_status'] == 'protocol_invalid'
    assert manifest['runtime_authorized'] is False
    assert manifest['authorized_process_ids'] == []
    assert manifest['resource_cap']['gpu_device_total_mib'] == 7680
    assert manifest['g3_simulator_source_sha256'] == manifest['parent_v3_simulator_source_sha256']
    assert manifest['assignment_sha256'] == manifest['parent_v3_assignment_sha256']


def test_v4_manifest_contains_all_new_source_and_contract_hashes():
    manifest = make_manifest(REPO)
    for name in ('router_observer_source_sha256', 'runtime_evidence_source_sha256',
                 'v4_outer_host_source_sha256', 'g3_host_source_sha256',
                 'monitoring_contract_sha256', 'parent_v3_manifest_sha256'):
        assert len(manifest[name]) == 64


def test_checked_in_manifest_matches_frozen_source_bytes():
    saved = json.loads((REPO/'docs/eat-nav/p1-g3-crossover-v4-manifest.json').read_text())
    assert saved == make_manifest(REPO)
