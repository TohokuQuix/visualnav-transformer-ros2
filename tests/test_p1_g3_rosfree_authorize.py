import json
from pathlib import Path

import pytest

from eat_nav_p1_g3_rosfree_authorize import create_attempt, verify_preparation, PROCESS_ID, sha
from eat_nav_p1_g3_rosfree_host import validate_staged


PREPARATION = Path('/home/kojima/eatnav-experiments/p1-g3-rosfree-direct-v1-prep')


def test_attempt_only_authorizes_one_process_and_preserves_preparation(tmp_path):
    before = sha(PREPARATION/'stage-complete.json')
    destination = tmp_path/'attempt'
    report = create_attempt(PREPARATION, destination)
    assert report['source_input_file_count'] >= 197
    assert report['authorized_process_ids'] == [PROCESS_ID]
    assert sha(PREPARATION/'stage-complete.json') == before
    assert verify_preparation(PREPARATION)[1]['runtime_authorized'] is False
    manifest = json.loads((destination/'manifest.json').read_text())
    assert manifest['runtime_authorized'] is True
    assert manifest['g3_simulator_source_sha256'] == sha(destination/PROCESS_ID/'source.py')
    assert manifest['g3_host_source_sha256'] == sha(destination/PROCESS_ID/'host-runner.py')
    class Args:
        anchor_id = 'first-differential'
        batch_index = 0
        execution_index = 0
    validated, process, assigned = validate_staged(destination/PROCESS_ID, Args())
    assert validated['authorized_process_ids'] == [PROCESS_ID]
    assert len(assigned) == 4 and all(row['path_role'] == 'direct' for row in assigned)
    with pytest.raises(FileExistsError):
        create_attempt(PREPARATION, destination)
