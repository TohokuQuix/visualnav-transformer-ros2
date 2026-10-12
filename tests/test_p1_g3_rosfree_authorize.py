import json
from pathlib import Path

import pytest

from eat_nav_p1_g3_rosfree_authorize import create_attempt, verify_preparation, PROCESS_ID, sha
from eat_nav_p1_g3_rosfree_host import validate_staged
from eat_nav_p1_g3_rosfree_prepare import (
    REFERENCE, build_assignment, build_manifest, stage_one, write_new,
)


def test_attempt_only_authorizes_one_process_and_preserves_preparation(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    source = json.loads((REFERENCE/'batch-first-differential-0-repeat-0/input/source-contract.json').read_text())
    assignment = build_assignment(source, 2026101101)
    assignment_file = tmp_path/'assignment.json'
    write_new(assignment_file, assignment)
    manifest = build_manifest(repo, REFERENCE, sha(assignment_file), assignment)
    preparation = tmp_path/'preparation'
    stage_one(repo, REFERENCE, preparation, assignment, manifest)
    before = sha(preparation/'stage-complete.json')
    host_bind = tmp_path/'host-bind'
    host_bind.mkdir()
    destination = host_bind/'.eatnav-p1-g3-rosfree-direct-core-v3-attempt-2'
    report = create_attempt(preparation, destination,
                            host_bind_source=str(host_bind),
                            container_bind_destination=str(tmp_path/'container-bind'))
    assert report['source_input_file_count'] >= 197
    assert report['authorized_process_ids'] == [PROCESS_ID]
    assert sha(preparation/'stage-complete.json') == before
    assert verify_preparation(preparation)[1]['runtime_authorized'] is False
    manifest = json.loads((destination/'manifest.json').read_text())
    assert manifest['runtime_authorized'] is True
    assert manifest['g3_simulator_source_sha256'] == sha(destination/PROCESS_ID/'source.py')
    assert manifest['g3_host_source_sha256'] == sha(destination/PROCESS_ID/'host-runner.py')
    assert manifest['attempt_host_root'] == str(destination)
    class Args:
        anchor_id = 'first-differential'
        batch_index = 0
        execution_index = 0
    validated, process, assigned = validate_staged(destination/PROCESS_ID, Args())
    assert validated['authorized_process_ids'] == [PROCESS_ID]
    assert len(assigned) == 4 and all(row['path_role'] == 'direct' for row in assigned)
    with pytest.raises(FileExistsError):
        create_attempt(preparation, destination,
                       host_bind_source=str(host_bind),
                       container_bind_destination=str(tmp_path/'container-bind'))
