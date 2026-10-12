"""One-process v3 freeze/stage verification; no Isaac runtime."""

import json
from pathlib import Path

import pytest

from eat_nav_p1_g3_crossover_v3_stage import make_manifest, stage_one, sha
from visualnav_transformer.eat_nav.p1_g3_assignment import validate_frozen_assignment


REPO = Path(__file__).resolve().parents[1]
OLD = Path('/home/kojima/eatnav-experiments/p1-long-horizon-v1/g3-direct-repeat-v1-attempt-0')
ASSIGNMENT = REPO/'docs/eat-nav/p1-g3-crossover-v2-assignment.json'


def test_v3_manifest_freezes_only_first_remaining_process(tmp_path):
    manifest = make_manifest(REPO, OLD, ASSIGNMENT)
    assert manifest['host_protocol_version']=='eatnav_p1_g3_path_crossover_v3_assignment_validator'
    assert manifest['authorized_process_id']=='batch-first-differential-1-period-0'
    assert manifest['authorized_process_count']==1
    assert manifest['g3_host_source_sha256']==sha(REPO/'scripts/eat_nav_p1_g3_crossover_v3_host.py')
    assert manifest['g3_assignment_module_sha256']==sha(
        REPO/'src/visualnav_transformer/eat_nav/p1_g3_assignment.py')
    assert manifest['g3_simulator_source_sha256']==json.loads(
        (REPO/'docs/eat-nav/p1-g3-crossover-v2-manifest.json').read_text())['g3_simulator_source_sha256']
    root=tmp_path/'fresh'
    staged=stage_one(REPO,OLD,ASSIGNMENT,manifest,root)
    assert staged['batch_name']=='batch-first-differential-1-period-0'
    batch=root/staged['batch_name']
    assert len(list(root.glob('batch-*')))==1
    assert sha(batch/'host-runner.py')==manifest['g3_host_source_sha256']
    assert sha(batch/'src/visualnav_transformer/eat_nav/p1_g3_assignment.py')==manifest['g3_assignment_module_sha256']
    frozen=next(p['trials'] for p in manifest['processes'] if p['anchor_id']=='first-differential'
                and p['batch_index']==1 and p['execution_index']==0)
    assigned=next(p['trials'] for p in json.loads((batch/'input/g3-assignment.json').read_text())['processes']
                  if p['anchor_id']=='first-differential' and p['batch_index']==1
                  and p['execution_index']==0)
    validate_frozen_assignment(frozen,assigned,stage='prelaunch')
    assert not (batch/'output').exists()
    with pytest.raises(FileExistsError):
        stage_one(REPO,OLD,ASSIGNMENT,manifest,root)


def test_v3_stage_rejects_host_or_assignment_drift(tmp_path):
    manifest=make_manifest(REPO,OLD,ASSIGNMENT)
    with pytest.raises(ValueError,match='host'):
        stage_one(REPO,OLD,ASSIGNMENT,{**manifest,'g3_host_source_sha256':'0'*64},tmp_path/'bad')
    with pytest.raises(ValueError,match='assignment'):
        stage_one(REPO,OLD,ASSIGNMENT,{**manifest,'assignment_sha256':'0'*64},tmp_path/'bad2')
