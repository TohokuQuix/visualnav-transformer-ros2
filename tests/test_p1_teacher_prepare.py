import hashlib
import json
from pathlib import Path

import pytest

from eat_nav_p1_teacher_prepare import build_teacher_assignment, build_teacher_manifest, stage_teacher


REPO = Path(__file__).resolve().parents[1]


def test_direct_only_assignment_is_new_protocol_and_not_old_assignment():
    original = json.loads((REPO / 'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = build_teacher_assignment(original)
    first = assignment['processes'][0]['trials']
    assert [row['path_role'] for row in first] == ['direct'] * 4
    assert len({row['path_sha256'] for row in first}) == 1
    assert first[2]['slot'] == 2
    assert original['processes'][0]['trials'][2]['path_role'] != 'direct'


def test_teacher_manifest_has_new_source_and_single_process_authorization(tmp_path):
    original = json.loads((REPO / 'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = build_teacher_assignment(original)
    manifest = build_teacher_manifest(REPO, assignment, '.eatnav-p1-teacher-capture-v1-attempt-test')
    assert manifest['teacher_capture_schema'] == 'eatnav_p1_teacher_capture_v1'
    assert manifest['authorized_process_ids'] == ['batch-first-differential-0-period-0']
    assert manifest['teacher_runtime_authorized'] is True
    assert manifest['authorized_period_index'] == 0
    assert manifest['g3_simulator_source_sha256'] == hashlib.sha256((REPO/'scripts/eat_nav_p1_teacher_sim.py').read_bytes()).hexdigest()
    assert manifest['attempt_host_root'].endswith('/.eatnav-p1-teacher-capture-v1-attempt-test')
    assert manifest['assignment_sha256'] != original.get('assignment_sha256')


def test_stage_rejects_existing_root_and_stages_matching_sha(tmp_path):
    original = json.loads((REPO / 'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())
    assignment = build_teacher_assignment(original)
    manifest = build_teacher_manifest(REPO, assignment, '.eatnav-p1-teacher-capture-v1-attempt-test')
    root = tmp_path/'.eatnav-p1-teacher-capture-v1-attempt-test'
    stage_teacher(REPO, root, assignment, manifest)
    batch = root/'batch-first-differential-0-period-0'
    ledger = json.loads((batch/'input-sha-ledger.json').read_text())
    assert hashlib.sha256((batch/'source.py').read_bytes()).hexdigest() == manifest['g3_simulator_source_sha256']
    assert hashlib.sha256((batch/'host-runner.py').read_bytes()).hexdigest() == manifest['g3_host_source_sha256']
    assert 'src/visualnav_transformer/eat_nav/p1_teacher_capture.py' in ledger
    with pytest.raises(FileExistsError):
        stage_teacher(REPO, root, assignment, manifest)
